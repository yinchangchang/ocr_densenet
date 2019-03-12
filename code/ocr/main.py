# coding=utf8
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""ResNet Train/Eval module.
"""
import time
import sys
import os

import numpy as np
import dataloader
import json
from tqdm import tqdm

import densenet
import resnet
from PIL import Image

import torchvision

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score

from tools import parse
from glob import glob
from skimage import measure
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import traceback

args = parse.args
# anchor大小
args.anchors = [8, 12, 18, 27, 40, 60]
args.stride = 8
args.image_size = [512,64]


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.inplanes = 1024
        self.densenet121 = densenet.densenet121(pretrained=True, small=args.small)
        num_ftrs = self.densenet121.classifier.in_features
        self.classifier_font = nn.Sequential(
                # 这里可以用fc做分类
                # nn.Linear(num_ftrs, out_size)
                # 这里可以用1×1卷积做分类
                nn.Conv2d(num_ftrs, out_size, kernel_size=1, bias=False)
        )
        self.train_params = []
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, phase='train'):
        feats = self.densenet121(x)     # (32, 1024, 2, 16)
        if not args.small:
            feats = F.max_pool2d(feats, kernel_size=2, stride=2) # (32, 1024, 1, 8)
        out = self.classifier_font(feats) # (32, 1824, 1, 8)
        out_size = out.size()
        # print out.size()
        out = out.view(out.size(0),out.size(1),-1) # (32, 1824, 8)
        # print out.size()
        if phase == 'train':
            out = F.adaptive_max_pool1d(out, output_size=(1)).view(out.size(0),-1) # (32, 1824)
            return out
        else:
            out = out.transpose(1,2).contiguous()
            out = out.view(out_size[0],out_size[2], out_size[3], out_size[1]) # (32, 1, 8, 1824)
            return out, feats

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.classify_loss = nn.BCELoss()
        self.sigmoid = nn.Sigmoid()
        self.regress_loss = nn.SmoothL1Loss()

    def forward(self, font_output, font_target, weight=None, use_hard_mining=False):
        font_output = self.sigmoid(font_output)
        font_loss = F.binary_cross_entropy(font_output, font_target, weight)

        # hard_mining 
        if use_hard_mining:
            font_output = font_output.view(-1)
            font_target = font_target.view(-1)
            pos_index = font_target > 0.5
            neg_index = font_target == 0

            # pos
            pos_output = font_output[pos_index]
            pos_target = font_target[pos_index]
            num_hard_pos = max(len(pos_output)/4, min(5, len(pos_output)))
            if len(pos_output) > 5:
                pos_output, pos_target = hard_mining(pos_output, pos_target, num_hard_pos, largest=False)
            pos_loss = self.classify_loss(pos_output, pos_target) * 0.5


            # neg
            num_hard_neg = len(pos_output) * 2
            neg_output = font_output[neg_index]
            neg_target = font_target[neg_index]
            neg_output, neg_target = hard_mining(neg_output, neg_target, num_hard_neg, largest=True)
            neg_loss = self.classify_loss(neg_output, neg_target) * 0.5

            font_loss += pos_loss + neg_loss

        else:
            pos_loss, neg_loss = font_loss, font_loss
        return [font_loss, pos_loss, neg_loss]

    def _forward(self, font_output, font_target, weight, bbox_output=None, bbox_label=None, seg_output=None, seg_labels=None):
        font_output = self.sigmoid(font_output)
        font_loss = F.binary_cross_entropy(font_output, font_target, weight)

        acc = []
        if bbox_output is not None:
            # bbox_loss = 0
            bbox_output = bbox_output.view((-1, 4))
            bbox_label = bbox_label.view((-1, 4))
            pos_index = bbox_label[:,-1] >= 0.5
            pos_index = pos_index.unsqueeze(1).expand(pos_index.size(0), 4)
            neg_index = bbox_label[:,-1] <= -0.5
            neg_index = neg_index.unsqueeze(1).expand(neg_index.size(0), 4)

            # 正例
            pos_label = bbox_label[pos_index].view((-1,4))
            pos_output = bbox_output[pos_index].view((-1,4))
            lx,ly,ld,lc = pos_label[:,0],pos_label[:,1],pos_label[:,2],pos_label[:,3]
            ox,oy,od,oc = pos_output[:,0],pos_output[:,1],pos_output[:,2],pos_output[:,3]
            regress_loss = [
                    self.regress_loss(ox, lx),
                    self.regress_loss(oy, ly),
                    self.regress_loss(od, ld),
                    ]
            pc = self.sigmoid(oc)
            acc.append((pc>=0.5).data.cpu().numpy().astype(np.float32).sum())
            acc.append(len(pc))
            # print pc.size(), lc.size()
            classify_loss = self.classify_loss(pc, lc) * 0.5

            # 负例
            neg_label = bbox_label[neg_index].view((-1,4))
            neg_output = bbox_output[neg_index].view((-1,4))
            lc = neg_label[:, 3]
            oc = neg_output[:, 3]
            pc = self.sigmoid(oc)
            acc.append((pc<=0.5).data.cpu().numpy().astype(np.float32).sum())
            acc.append(len(pc))
            # print pc.size(), lc.size()
            classify_loss += self.classify_loss(pc, lc+1) * 0.5

            # seg_loss
            seg_output = seg_output.view(-1)
            seg_labels = seg_labels.view(-1)
            pos_index = seg_labels > 0.5
            neg_index = seg_labels < 0.5
            seg_loss = 0.5 * self.classify_loss(seg_output[pos_index], seg_labels[pos_index]) + \
                       0.5 * self.classify_loss(seg_output[neg_index], seg_labels[neg_index])
            seg_tpr = (seg_output[pos_index] > 0.5).data.cpu().numpy().astype(np.float32).sum() / len(seg_labels[pos_index])
            seg_tnr = (seg_output[neg_index] < 0.5).data.cpu().numpy().astype(np.float32).sum() / len(seg_labels[neg_index])
            # print seg_output[neg_index]
            # print seg_labels[neg_index]




        else:
            return font_loss

        if args.model == 'resnet':
            loss = font_loss + classify_loss + seg_loss
        else:
            loss = font_loss + classify_loss + seg_loss
        for reg in regress_loss:
            loss += reg
        # if args.model == 'resnet':
        #     loss = seg_loss

        return [loss, font_loss, seg_loss, classify_loss] + regress_loss + acc + [seg_tpr, seg_tnr]

        font_num = font_target.sum(0).data.cpu().numpy()
        font_loss = 0
        for di in range(font_num.shape[0]):
            if font_num[di] > 0:
                font_output_i = font_output[:,di]
                font_target_i = font_target[:,di]
                pos_font_index = font_target_i > 0.5
                font_loss  += 0.5 * self.classify_loss(font_output_i[pos_font_index], font_target_i[pos_font_index])
                neg_font_index = font_target_i < 0.5
                if len(font_target_i[neg_font_index]) > 0:
                    font_loss  += 0.5 * self.classify_loss(font_output_i[neg_font_index], font_target_i[neg_font_index])
        font_loss = font_loss / (font_num>0).sum()

        return font_loss
        # '''

def hard_mining(neg_output, neg_labels, num_hard, largest=True):
    num_hard = min(max(num_hard, 10), len(neg_output))
    _, idcs = torch.topk(neg_output, min(num_hard, len(neg_output)), largest=largest)
    neg_output = torch.index_select(neg_output, 0, idcs)
    neg_labels = torch.index_select(neg_labels, 0, idcs)
    return neg_output, neg_labels

def save_model(save_dir, phase, name, epoch, f1score, model):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_dir = os.path.join(save_dir, phase)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    state_dict = model.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].cpu()
    state_dict_all = {
            'state_dict': state_dict,
            'epoch': epoch,
            'f1score': f1score,
            }
    torch.save( state_dict_all , os.path.join(save_dir, '{:s}.ckpt'.format(name)))
    if 'best' in name and f1score > 0.3:
        torch.save( state_dict_all , os.path.join(save_dir, '{:s}_{:s}.ckpt'.format(name, str(epoch))))

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def test(epoch, model, train_loader, phase='test'):
    print '\ntest {:s}_files, epoch: {:d}'.format(phase, epoch)
    mkdir('../../data/result')
    model.eval()
    f1score_list = []
    recall_list = []
    precision_list = []
    word_index_dict = json.load(open(args.word_index_json))
    index_word_dict = { v:k for k,v in word_index_dict.items() }
    result_file = open('../../data/result/{:d}_{:s}_result.csv'.format(epoch, phase), 'w')
    result_file.write('name,content\n')
    name_f1score_dict = dict()

    # 保存densenet生成的feature
    feat_dir = args.data_dir.replace('dataset', 'feats')
    mkdir(feat_dir)
    feat_dir = os.path.join(feat_dir, phase)
    print feat_dir
    mkdir(feat_dir)

    names = []
    if phase != 'test':
        gt_file = open('../../data/result/{:d}_{:s}_gt.csv'.format(epoch, phase), 'w')
        gt_file.write('name,content\n')
        analysis_file = open('../../data/result/{:s}_{:s}_gt.csv'.format('analysis', phase), 'w')
        os.system('rm -r ../../data/analysis/{:s}'.format(phase))
        labels_all = []
    probs_all = []
    for i,data in enumerate(tqdm(train_loader)):
        name = data[0][0].split('/')[-1].split('.seg')[0]
        names.append(name)
        images, labels = [Variable(x.cuda(async=True)) for x in data[1:3]]
        if len(images.size()) == 5:
            images = images[0]

        probs, feats = model(images, 'test')
        probs_all.append(probs.data.cpu().numpy().max(2).max(1).max(0))

        preds = probs.data.cpu().numpy() > 0.5 # (-1, 8, 1824)

        # result_file.write(name+',')
        result = u''
        last_set = set()
        all_set = set()

        if args.feat:
            # 保存所有的feat
            feats = feats.data.cpu().numpy()
            if i == 0:
                print feats.shape
            np.save(os.path.join(feat_dir, name.replace('.png','.npy')), feats)
            if len(feats) > 1: # feats: [-1, 1024, 1, 8]
                # 多个patch
                new_feats = []
                for i,feat in enumerate(feats):
                    if i == 0:
                        # 第一个patch,保存前6个
                        new_feats.append(feat[:,:,:6])
                    elif i == len(feats) - 1:
                        # 最后一个patch,保存后6个
                        new_feats.append(feat[:,:,2:])
                    else:
                        # 保存中间4个
                        new_feats.append(feat[:,:,2:6])
                feats = np.concatenate(new_feats, 2)

        # 这种方法用于检测不同区域的同一个字，当同一个字同一个区域出现时，可能检测不到多次
        preds = preds.max(1) # 沿着竖直方向pooling
        # if len(preds) > 1:
        #     print name
        for patch_i, patch_pred in enumerate(preds):
            for part_i, part_pred in enumerate(patch_pred):
                new_set = set()
                for idx,p in enumerate(part_pred):
                    if p:
                        # 出现了这个字
                        w = index_word_dict[idx]
                        new_set.add(w)
                        if w not in all_set:
                            # 从没见过的字
                            all_set.add(w)
                            result += w
                        elif w not in last_set:
                            # 以前出现过
                            if patch_i == 0:
                                # 第一个patch # 上一个部分没有这个字
                                result += w
                            elif part_i >= preds.shape[1]/2 :
                                # 后续patch的后一半，不写 # 上一个部分没有这个字
                                result += w
                last_set = new_set
        # if len(result) > len(set(result)):
        #     print name




        '''
        for idx,p in enumerate(preds.reshape(-1)):
            if p:
                # result_file.write(index_word_dict[idx])
                result = result + index_word_dict[idx]
        '''

        result = result.replace(u'"', u'')
        if u','  in result:
            result = '"' + result + '"'
        if len(result) == 0:
            global_prob = probs.data.cpu().numpy().max(0).max(0).max(0)
            max_index = global_prob.argmax()
            result = index_word_dict[max_index]
            print name

        result_file.write(name+','+result+'\n')
        # result_file.write('\n')

        if phase == 'test':
            continue
        labels = labels.data.cpu().numpy()
        gt_file.write(name+',')
        gt = u''
        for idx,l in enumerate(labels.reshape(-1)):
            if l:
                gt = gt + index_word_dict[idx]
                gt_file.write(index_word_dict[idx])
        gt_file.write('\n')

        
        labels_all.append(labels[0])
        # 全局pooling
        preds = np.array([preds.max(1).max(0)])
        # print preds.shape
        for pred, label in zip(preds, labels):
            tp = (pred + label == 2).sum()
            tn = (pred + label == 0).sum()
            fp = (pred - label == 1).sum()
            fn = (pred - label ==-1).sum()
            precision = 1.0 * tp / max(tp + fp , 10e-20)
            recall   = 1.0 * tp / max(tp + fn , 10e-20)
            f1score = 2. * precision * recall / max(precision + recall , 10e-20)
            precision_list.append(precision)
            recall_list.append(recall)
            f1score_list.append(f1score)
            name_f1score_dict[name] = f1score

            # 分析不好的结果
            if phase == 'train_val':
                th = 0.8
            elif phase == 'train':
                th = 0.95
            else:
                th = 0.6
            if f1score < th:
                save_dir = '../../data/analysis'
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                save_dir = os.path.join(save_dir, phase)
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                os.system('cp ../../data/dataset/train/{:s} {:s}/{:d}_{:s}'.format(name, save_dir, 100000+i, name))
                analysis_file.write(name+'\t\t')
                gt = set(gt)
                result = set(result.strip('"'))
                analysis_file.write(''.join(sorted(gt - result))+'\t\t')
                analysis_file.write(''.join(sorted(result - gt))+'\t\n')
                
            
    
    if phase != 'test':
        # f1score = np.mean(f1score_list)
        # print 'f1score all', f1score
        # f1score_list = sorted(f1score_list)[500:]
        f1score = np.mean(f1score_list)
        recall = np.mean(recall_list)
        precision = np.mean(precision_list)
        print 'f1score', f1score
        print 'recall', recall
        print 'precision', precision
        gt_file.write('f1score,' +  str(f1score))
        gt_file.write('recall,' +  str(recall))
        gt_file.write('precision,' +  str(precision))
        gt_file.close()
        result_file.write('f1score,' +  str(f1score))
        result_file.write('recall,' +  str(recall))
        result_file.write('precision,' +  str(precision))
        with open('../../data/result/name_f1score_dict.json','w') as f:
            f.write(json.dumps(name_f1score_dict, indent=4))
        np.save('../../data/result/{:d}_{:s}_labels.npy'.format(epoch, phase), labels_all)
    result_file.close()
    os.system('cp ../../data/result/{:d}_{:s}_result.csv ../../data/result/{:s}_result.csv'.format(epoch, phase, phase))

    np.save('../../data/result/{:d}_{:s}_probs.npy'.format(epoch, phase), probs_all)
    with open('../../data/result/{:s}_names.json'.format(phase), 'w') as f:
        f.write(json.dumps(names, indent=4))

def get_weight(labels):
    labels = labels.data.cpu().numpy()
    weights = np.zeros_like(labels)
    # weight_false = 1.0 / ((labels<0.5).sum() + 10e-20)
    # weight_true  = 1.0 / ((labels>0.5).sum() + 10e-20)
    weight_false = 1.0 / ((labels<0.5).sum(0) + 10e-20)
    label_true = (labels>0.5).sum(0)
    for i in range(labels.shape[1]):
        label_i = labels[:,i]
        weight_i = np.ones(labels.shape[0]) * weight_false[i]
        # weight_i = np.ones(labels.shape[0]) * weight_false
        if label_true[i] > 0:
            weight_i[label_i>0.5] = 1.0 / label_true[i]
        weights[:,i] = weight_i
    weights *= np.ones_like(labels).sum() / (weights.sum() + 10e-20)
    weights[labels<-0.5] = 0
    return weights

def train_eval(epoch, model, train_loader, loss, optimizer, best_f1score=0, phase='train'):
    print '\n',epoch, phase
    if 'train' in phase:
        model.train()
    else:
        model.eval()
    loss_list = []
    f1score_list = []
    recall_list = []
    precision_list = []
    for i,data in enumerate(tqdm(train_loader)):
        images, labels = [Variable(x.cuda(async=True)) for x in data[1:3]]
        weights = torch.from_numpy(get_weight(labels)).cuda(async=True)
        probs = model(images)

        # 训练阶段
        if 'train' in phase:
            loss_output = loss(probs, labels, weights, args.hard_mining)
            try:
                optimizer.zero_grad()
                loss_output[0].backward()
                optimizer.step()
                loss_list.append([x.data.cpu().numpy()[0] for x in loss_output])
            except:
                # pass
                traceback.print_exc()


        # 计算 f1score, recall, precision
        '''
        x = probs.data.cpu().numpy() 
        l = labels.data.cpu().numpy()
        print (get_weight(labels) * l).sum()
        l = 1 - l
        print (get_weight(labels) * l).sum()
        print x.max()
        print x.min()
        print x.mean()
        print
        # '''
        preds = probs.data.cpu().numpy() > 0
        labels = labels.data.cpu().numpy()
        for pred, label in zip(preds, labels):
            pred[label<0] = -1
            if label.sum() < 0.5:
                continue
            tp = (pred + label == 2).sum()
            tn = (pred + label == 0).sum()
            fp = (pred - label == 1).sum()
            fn = (pred - label ==-1).sum()
            precision = 1.0 * tp / (tp + fp + 10e-20)
            recall   = 1.0 * tp / (tp + fn + 10e-20)
            f1score = 2. * precision * recall / (precision + recall + 10e-20)
            precision_list.append(precision)
            recall_list.append(recall)
            f1score_list.append(f1score)
    
            
        # 保存中间结果到 data/middle_result，用于分析
        if i == 0:
            images = images.data.cpu().numpy() * 128 + 128
            if phase == 'pretrain':
                bbox_labels = bbox_labels.data.cpu().numpy()
                seg_labels = seg_labels.data.cpu().numpy()
                seg_output = seg_output.data.cpu().numpy()
            for ii in range(len(images)):
                middle_dir = os.path.join(args.save_dir, 'middle_result')
                if not os.path.exists(middle_dir):
                    os.mkdir(middle_dir)
                middle_dir = os.path.join(middle_dir, phase)
                if not os.path.exists(middle_dir):
                    os.mkdir(middle_dir)
                Image.fromarray(images[ii].astype(np.uint8).transpose(1,2,0)).save(os.path.join(middle_dir, str(ii)+'.image.png'))
                if phase == 'pretrain':
                    segi = seg_labels[ii]
                    _segi = np.array([segi, segi, segi]) * 255
                    segi = np.zeros([3, _segi.shape[1]*2, _segi.shape[2]*2])
                    for si in range(segi.shape[1]):
                        for sj in range(segi.shape[2]):
                            segi[:,si,sj] = _segi[:,si/2,sj/2]
                    Image.fromarray(segi.transpose(1,2,0).astype(np.uint8)).save(os.path.join(middle_dir, str(ii)+'.seg.png'))
                    segi = seg_output[ii]
                    _segi = np.array([segi, segi, segi]) * 255
                    segi = np.zeros([3, _segi.shape[1]*2, _segi.shape[2]*2])
                    for si in range(segi.shape[1]):
                        for sj in range(segi.shape[2]):
                            segi[:,si,sj] = _segi[:,si/2,sj/2]
                    Image.fromarray(segi.transpose(1,2,0).astype(np.uint8)).save(os.path.join(middle_dir, str(ii)+'.seg.out.png'))

    f1score = np.mean(f1score_list)
    print 'f1score', f1score
    print 'recall', np.mean(recall_list)
    print 'precision', np.mean(precision_list)
    if 'train' in phase:
        loss_mean = np.array(loss_list).mean(0)
        print 'loss: {:3.4f}    pos loss: {:3.4f}   neg loss: {:3.4f}'.format(loss_mean[0], loss_mean[1], loss_mean[2])

    # 保存模型
    if ('eval' in phase or 'pretrain' in phase)and best_f1score < 2: 
        if args.small:
            save_dir = os.path.join(args.save_dir, 'models-small')
        else:
            save_dir = os.path.join(args.save_dir, 'models')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        if epoch % 5 == 0:
            save_model(save_dir, phase, str(epoch), epoch, f1score, model)
        if f1score > best_f1score:
            save_model(save_dir, phase, 'best_f1score', epoch, f1score, model)
        if args.model == 'resnet':
            tpnr = loss[11] + loss[12]
            # 这里用 best_f1score 也当tpnr好了，懒得改
            if tpnr > best_f1score:
                best_f1score = tpnr
                save_model(save_dir, phase, 'best_tpnr', epoch, f1score, model)
            print 'best tpnr', best_f1score
        else:
            best_f1score = max(best_f1score, f1score)
            if best_f1score < 1:
                print '\n\t{:s}\tbest f1score {:3.4f}\n'.format(phase, best_f1score)
        return best_f1score


def main():
    word_index_dict = json.load(open(args.word_index_json))
    num_classes = len(word_index_dict)
    image_label_dict = json.load(open(args.image_label_json))

    cudnn.benchmark = True
    if args.model == 'densenet':
        # 两千多种字符，multi-label分类
        model = DenseNet121(num_classes).cuda()
    elif args.model == 'resnet':
        # resnet主要用于文字区域的segmentation以及object detection操作
        model = resnet.ResNet(num_classes=num_classes, args=args).cuda()
    else:
        return
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # model = torch.nn.DataParallel(model).cuda()
    loss = Loss().cuda()

    if args.resume:
        state_dict = torch.load(args.resume)
        model.load_state_dict(state_dict['state_dict'])
        best_f1score = state_dict['f1score']
        start_epoch = state_dict['epoch'] + 1
    else:
        best_f1score = 0
        if args.model == 'resnet':
            start_epoch = 100
        else:
            start_epoch = 1
    args.epoch = start_epoch
    print 'best_f1score', best_f1score


    # 划分数据集
    test_filelist = sorted(glob(os.path.join(args.data_dir,'test','*')))
    trainval_filelist = sorted(glob(os.path.join(args.data_dir,'train','*')))

    # 两种输入size训练
    # train_filelist1: 长宽比小于8:1的图片，经过padding后变成 64*512 的输入
    # train_filelist2: 长宽比大于8:1的图片，经过padding,crop后变成 64*1024的输入
    train_filelist1, train_filelist2 = [],[]

    # 黑名单，这些图片的label是有问题的
    black_list = set(json.load(open(args.black_json))['black_list'])
    image_hw_ratio_dict = json.load(open(args.image_hw_ratio_json))
    for f in trainval_filelist:
        image = f.split('/')[-1]
        if image in black_list:
            continue
        r = image_hw_ratio_dict[image]
        if r == 0:
            train_filelist1.append(f)
        else:
            train_filelist2.append(f)
    train_val_filelist = train_filelist1 + train_filelist2
    val_filelist = train_filelist1[-2048:]
    train_filelist1 = train_filelist1[:-2048]

    train_filelist2 = train_filelist2
    image_size = [512, 64]

    if args.phase in ['test', 'val', 'train_val']:
        # 测试输出文字检测结果
        test_dataset = dataloader.DataSet(
                test_filelist, 
                image_label_dict,
                num_classes, 
                # transform=train_transform, 
                args=args,
                image_size=image_size,
                phase='test')
        test_loader = DataLoader(
                dataset=test_dataset, 
                batch_size=1, 
                shuffle=False, 
                num_workers=8, 
                pin_memory=True)
        train_filelist = train_filelist1[-2048:]
        train_dataset  = dataloader.DataSet(
                train_filelist, 
                image_label_dict, 
                num_classes, 
                image_size=image_size,
                args=args,
                phase='test')
        train_loader = DataLoader(
                dataset=train_dataset, 
                batch_size=1,
                shuffle=False, 
                num_workers=8, 
                pin_memory=True)

        val_dataset  = dataloader.DataSet(
                val_filelist, 
                image_label_dict, 
                num_classes, 
                image_size=image_size,
                args=args,
                phase='test')
        val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=1,
                shuffle=False, 
                num_workers=8, 
                pin_memory=True)

        train_val_dataset  = dataloader.DataSet(
                train_val_filelist, 
                image_label_dict, 
                num_classes, 
                image_size=image_size,
                args=args,
                phase='test')
        train_val_loader= DataLoader(
                dataset=train_val_dataset, 
                batch_size=1,
                shuffle=False, 
                num_workers=8, 
                pin_memory=True)

        if args.phase == 'test':
            test(start_epoch - 1, model, val_loader, 'val')
            test(start_epoch - 1, model, test_loader, 'test')
            # test(start_epoch - 1, model, train_val_loader, 'train_val')
        elif args.phase == 'val':
            test(start_epoch - 1, model, train_loader, 'train')
            test(start_epoch - 1, model, val_loader, 'val')
        elif args.phase == 'train_val':
            test(start_epoch - 1, model, train_val_loader, 'train_val')
        return

    elif args.phase == 'train':

        train_dataset1 = dataloader.DataSet(
                train_filelist1,
                image_label_dict,
                num_classes, 
                image_size=image_size,
                args=args,
                phase='train')
        train_loader1 = DataLoader(
                dataset=train_dataset1, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)
        train_dataset2 = dataloader.DataSet(
                train_filelist2, 
                image_label_dict,
                num_classes, 
                image_size=(1024,64),
                args=args,
                phase='train')
        train_loader2 = DataLoader(
                dataset=train_dataset2, 
                batch_size=args.batch_size / 2, 
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)
        val_dataset  = dataloader.DataSet(
                val_filelist, 
                image_label_dict, 
                num_classes, 
                image_size=image_size,
                args=args,
                phase='val')
        val_loader = DataLoader(
                dataset=val_dataset, 
                batch_size=min(8,args.batch_size),
                shuffle=False, 
                num_workers=8, 
                pin_memory=True)
        filelist = glob(os.path.join(args.bg_dir,'*'))
        pretrain_dataset1 = dataloader.DataSet(
                filelist, 
                image_label_dict,
                num_classes, 
                image_size=args.image_size,
                word_index_dict = word_index_dict,
                args=args,
                font_range=[8,32],
                margin=10,
                rotate_range=[-10., 10. ],
                phase='pretrain')
        pretrain_loader1 = DataLoader(
                dataset=pretrain_dataset1, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)
        pretrain_dataset2 = dataloader.DataSet(
                filelist, 
                image_label_dict,
                num_classes, 
                image_size=(256, 128),
                word_index_dict = word_index_dict,
                args=args,
                font_range=[24,64],
                margin=20,
                rotate_range=[-20., 20.],
                phase='pretrain')
        pretrain_loader2 = DataLoader(
                dataset=pretrain_dataset2, 
                batch_size=args.batch_size, 
                shuffle=True, 
                num_workers=8, 
                pin_memory=True)
    
        best_f1score = 0
        # eval_mode = 'pretrain-2'
        eval_mode = 'eval'
        for epoch in range(start_epoch, args.epochs):

            args.epoch = epoch

            if eval_mode == 'eval':
                if best_f1score > 0.9:
                    args.lr = 0.0001
                if best_f1score > 0.9:
                    args.hard_mining = 1

            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

            train_eval(epoch, model, train_loader1, loss, optimizer, 2., 'train-1')
            if best_f1score > 0.9:
                train_eval(epoch, model, train_loader2, loss, optimizer, 2., 'train-2')
            best_f1score = train_eval(epoch, model, val_loader, loss, optimizer, best_f1score, 'eval-{:d}-{:d}'.format(args.batch_size, args.hard_mining))
            continue
            '''

            if eval_mode == 'pretrain-2':
                args.epoch = 1
                best_f1score = train_eval(epoch, model, pretrain_loader2, loss, optimizer, best_f1score, 'pretrain-2')
                if best_f1score > 0.8:
                    eval_mode = 'pretrain-1'
                    best_f1score = 0
            elif eval_mode == 'pretrain-1':
                args.epoch = max(100, epoch)
                train_eval(epoch, model, pretrain_loader2, loss, optimizer, 2.0 , 'pretrain-2')
                best_f1score = train_eval(epoch, model, pretrain_loader1, loss, optimizer, best_f1score, 'pretrain-1')
                if best_f1score > 0.5:
                    eval_mode = 'eval'
                    best_f1score = 0
            else:
                train_eval(epoch, model, train_loader1, loss, optimizer, 2., 'train-1')
                train_eval(epoch, model, train_loader2, loss, optimizer, 2., 'train-2')
                best_f1score = train_eval(epoch, model, val_loader, loss, optimizer, best_f1score, 'eval-{:d}-{:d}'.format(args.batch_size, args.hard_mining))

            '''


    



if __name__ == '__main__':
    main()
