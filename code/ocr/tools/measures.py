# coding=utf8
import os
import numpy as np
from sklearn import metrics
from PIL import Image
import traceback

def stati_class_number_true_flase(label, pred):
    label = np.array(label)
    pred = np.array(pred)

    cls_list = set(label) | set(pred)
    d = dict()
    for cls in cls_list:
        d[cls] = dict()
        d[cls]['number'] = np.sum(label==cls)
        d[cls]['true'] = np.sum(label[label==cls]==pred[label==cls])
        d[cls]['pred'] = np.sum(pred==cls)
    return d

def stati_class_number_true_flase_multi_label_margin(labels, preds):

    d = dict()
    for label, pred in zip(labels, preds):
        label = set(label[label>=0])
        for cls in range(len(pred)):
            if cls not in d:
                d[cls] = dict()
                d[cls]['number'] = 0
                d[cls]['true'] = 0
                d[cls]['pred'] = 0
            if cls in label:
                d[cls]['number'] += 1
                if pred[cls] > 0.5:
                    d[cls]['true'] += 1
            if pred[cls] > 0.5:
                d[cls]['pred'] += 1
    return d

def stati_class_number_true_flase_bce(labels, preds):
    d = dict()
    labels = labels.astype(np.int64).reshape(-1)
    preds = preds.reshape(-1) > 0
    index = labels >= 0
    labels = labels[index]
    preds = preds[index]

    preds_num = preds.sum(0)
    true_num = (labels+preds==2).sum(0)
    for cls in range(2):
        d[cls] = dict()
        d[cls]['number'] = (labels==cls).sum()
        d[cls]['true'] = (labels+preds==2*cls).sum()
        d[cls]['pred'] = (labels==cls).sum()
    return d

def measures(d_list):
    # 合并每一个预测的结果
    d_all = dict()
    for d in d_list:
        for cls in d.keys():
            if cls not in d_all:
                d_all[cls] = dict()
            for k in d[cls].keys():
                if k not in d_all[cls]:
                    d_all[cls][k] = 0
                d_all[cls][k] += d[cls][k]
    m = dict()
    number = sum([d_all[cls]['number'] for cls in d_all.keys()])
    for cls in d_all:
        m[cls] = dict()
        m[cls]['number'] = d_all[cls]['number']
        m[cls]['true'] = d_all[cls]['true']
        m[cls]['pred'] = d_all[cls]['pred']
        m[cls]['ratio'] = d_all[cls]['number'] / (float(number) + 10e-10)
        m[cls]['accuracy'] = d_all[cls]['true'] / (float(d_all[cls]['number']) + 10e-10)
        m[cls]['precision'] = d_all[cls]['true'] /(float(d_all[cls]['pred']) + 10e-10) 
    return m

def print_measures(m, s = 'measures'):
    print s
    accuracy = 0
    for cls in sorted(m.keys()):
        print '\tclass: {:d}\taccuracy:{:.6f}\tprecision:{:.6f}\tratio:{:.6f}\t\tN/T/P:{:d}/{:d}/{:d}\
            '.format(cls, m[cls]['accuracy'],m[cls]['precision'],m[cls]['ratio'],m[cls]['number'],m[cls]['true'],m[cls]['pred'])
	accuracy += m[cls]['accuracy'] * m[cls]['ratio']
    print '\tacc:{:.6f}'.format(accuracy)
    return accuracy

def mse(pred_image, image):
    pred_image = pred_image.reshape(-1).astype(np.float32)
    image = image.reshape(-1).astype(np.float32)
    mse_err = metrics.mean_squared_error(pred_image,image)
    return mse_err

def psnr(pred_image, image):
    return 10 * np.log10(255*255/mse(pred_image,image))


def psnr_pred(stain_vis=20, end= 10000):
    clean_dir = '../../data/AI/testB/'
    psnr_list = []
    f = open('../../data/result.csv','w')
    for i,clean in enumerate(os.listdir(clean_dir)):
        clean = os.path.join(clean_dir, clean)
        clean_file = clean
        pred = clean.replace('.jpg','.png').replace('data','data/test_clean')
        stain = clean.replace('trainB','trainA').replace('testB','testA').replace('.jpg','_.jpg')

        try:
            pred = np.array(Image.open(pred).resize((250,250))).astype(np.float32)
            clean = np.array(Image.open(clean).resize((250,250))).astype(np.float32)
            stain = np.array(Image.open(stain).resize((250,250))).astype(np.float32)

            # diff = np.abs(stain - pred)
            # vis = 20
            # pred[diff<vis] = stain[diff<vis]

            # gray_vis = 240
            # pred[stain>gray_vis] = stain[stain>gray_vis]

            if end < 1000:
                diff = np.abs(clean - stain)
                # stain[diff>stain_vis] = pred[diff>stain_vis]
                stain[diff>stain_vis] = clean[diff>stain_vis]

            psnr_pred  = psnr(clean, pred)
            psnr_stain = psnr(clean, stain)
            psnr_list.append([psnr_stain, psnr_pred])
        except:
            continue
        if i>end:
            break
        print i, min(end, 1000)

        f.write(clean_file.split('/')[-1].split('.')[0])
        f.write(',')
        f.write(str(psnr_stain))
        f.write(',')
        f.write(str(psnr_pred))
        f.write(',')
        f.write(str(psnr_pred/psnr_stain - 1))
        f.write('\n')
    # print '预测',np.mean(psnr_list)
    psnr_list = np.array(psnr_list)
    psnr_mean = ((psnr_list[:,1] - psnr_list[:,0]) / psnr_list[:,0]).mean()
    if end > 1000:
        print '网纹图PSNR', psnr_list[:,0].mean()
        print '预测图PSNR', psnr_list[:,1].mean()
        print '增益率', psnr_mean
    f.write(str(psnr_mean))
    f.close()
    return psnr_list[:,0].mean()

def main():
    pmax = [0.,0.]
    for vis in range(1, 30):
        p = psnr_pred(vis, 10)
        print vis, p
        if p > pmax[1]:
            pmax = [vis, p]
    print '...'
    # print 256,psnr_pred(256)
    print pmax
    # print 10 * np.log10(255*255/metrics.mean_squared_error([3],[9]))


if __name__ == '__main__':
    psnr_pred(4000)
    # main()
    # for v in range(1,10):
    #     print v, 10 * np.log10(255*255/v/v)
