# encoding: utf-8

"""
Read images and corresponding labels.
"""

import numpy as np
import os
import json
# import skimage
# from skimage import io
from PIL import Image,ImageDraw,ImageFont,ImageFilter
from torch.utils.data import Dataset
import time

filters = [
            ImageFilter.SMOOTH,                 # 平滑，大于16可以用
            ImageFilter.SMOOTH_MORE,            # 平滑，大于16可以用
            ImageFilter.GaussianBlur(radius=1), # 大于16可以用

            ImageFilter.GaussianBlur(radius=2), # 大于32可以用
            ImageFilter.BLUR,                   # 大于32可以用
        ]

def histeq (im,nbr_bins =256):  
    # 对一副灰度图像进行直方图均衡化  
    #该函数有两个输入参数，一个是灰度图像，一个是直方图中使用小区间的数目  
    #函数返回直方图均衡化后的图像，以及用来做像素值映射的累计分布函数  
    # 计算图像的直方图  
    imhist,bins =np.histogram(im.flatten(),nbr_bins,normed=True)  
    cdf =imhist.cumsum() #cumulative distribution function  
    cdf =255*cdf/cdf[-1] #归一化，函数中使用累计分布函数的最后一个元素（下标为-1，目标是  
    # 将其归一化到0-1范围 ）  
    # 使用累计分布函数的线性插值，计算新的像素值  
    im2=np.interp(im.flatten(),bins[:-1],cdf) # im2 is an array  
    return im2.reshape(im.shape),cdf  


class DataSet(Dataset):
    def __init__(self, 
            image_names, 
            image_label_dict, 
            class_num, 
            transform=None, 
            image_size=None,        # 最后生成的图片大小
            word_index_dict=None,   # 字符与index的对应
            phase='train',          # phase
            args=None,              # 全局参数
            font_range=None,        # 生成字符大小范围
            rotate_range=None,      # 图片旋转范围
            margin=None             # 图片边缘不覆盖字符，以免旋转时候丢失
            ):

        self.font_range = font_range
        self.rotate_range = rotate_range
        self.margin = margin
        self.image_names = image_names
        self.image_label_dict = image_label_dict
        self.transform = transform
        self.phase = phase
        self.class_num = class_num
        self.word_labels = { }
        self.image_size = image_size
        self.word_index_dict = word_index_dict
        self.args = args
        if self.phase != 'pretrain':
            for image_name in image_names:
                image_name = image_name.split('/')[-1]
                if image_name not in image_label_dict:
                    try:
                        image_label_dict[image_name] = image_label_dict[image_name.replace('seg.','').split('.png')[0]+'.png']
                    except:
                        image_label_dict[image_name] = ''
                word_label = np.zeros(class_num)
                label = image_label_dict[image_name]
                for l in label.split():
                    word_label[int(l)] = 1
                self.word_labels[image_name] = word_label.astype(np.float32)

    def __getitem__(self, index):
        image_name = self.image_names[index]
        # print self.image_size
        if self.phase == 'pretrain':
            image = Image.open(image_name).convert('RGB')
            # 改变灰度
            image = np.array(image)
            r = get_random(index) 
            # 通常背景为高亮度颜色
            if r < 0.3:
                min_rgb = 192.
            elif r < 0.7:
                min_rgb = 128.
            else:
                min_rgb = 64.
            if self.args.model == 'resnet':
                pass
            elif index % 2 == 0:
                image = image / (255. - min_rgb) + min_rgb
            else:
                image[image<min_rgb] = min_rgb
            image = Image.fromarray(image.astype(np.uint8))
            no_aug = get_random(index+1000) < 0.1
            if self.args.epoch < 40:
                no_aug = 1
            image, label, bbox_label, seg_label, font_size = generate_image( index, image, no_aug, self)

            # 转化为numpy数组之后，增加一些其他的augmentation
            image = np.transpose(np.array(image), [2,0,1]).astype(np.float32)

            if get_random(index+1) < 0.2:
                # 灰度反向，变成黑底，白字
                image = 255. - image

            if not no_aug:
                # 每一列灰度有所改变
                if get_random(index + 3) < 0.3:
                    change_level = 256. / image.shape[1]
                    gray_change = 0 
                    for j in range(image.shape[1]):
                        gray_change += change_level * get_random(j+index) - change_level / 2
                        image[:,j,:] += int(gray_change)
                # 每一行灰度有所改变
                if get_random(index + 4) < 0.3:
                    change_level = 256. / image.shape[2]
                    gray_change = 0
                    for k in range(image.shape[2]):
                        gray_change += change_level * get_random(10+k+index) - change_level / 2
                        image[:,:,k] += int(gray_change)
                image_name = image_name.split('/')[-1]
            '''
            # 增加噪声
            if get_random(index+5) > 0.5 and self.args.epoch > 35:
                noise_level = 10
                noise = np.random.random(image.shape) * noise_level - noise_level / 2.
                image = image + noise
            '''
            image = (image / 128. - 1).astype(np.float32)

            if font_size > 32:
                size_label = 1
            elif font_size < 16:
                size_label = 0
            else:
                size_label = 11
            size_label = np.array([size_label]).astype(np.float32)

            return image_name, image.astype(np.float32), label, bbox_label, seg_label, size_label

        elif self.phase == 'seg':
				# 保持和原图相同的分辨率
                image = Image.open(image_name).convert('RGB')
                # image_name = image_name.split('/')[-1]
                # image = image.resize(self.image_size)
                image = np.transpose(np.array(image), [2,0,1]).astype(np.float32)
                min_size = 32
                shape = (np.array(image.shape).astype(np.int32) / min_size) * min_size + min_size # * 2
                new_image = np.zeros([3, shape[1], shape[2]], dtype=np.float32) 
                '''
                for i in range(3):
                    gray = sorted(image[i].reshape(-1))
                    gray = gray[len(gray)/2]
                    new_image[i] = gray
                '''
                # new_image[:, min_size/2:image.shape[1]+min_size/2, min_size/2:image.shape[2]+min_size/2] = image
                new_image[:, :image.shape[1], :image.shape[2]] = image
                image = new_image
                # word_label = self.word_labels[image_name]
                image = (image / 128. - 1).astype(np.float32)
                return image_name, image, np.zeros(self.class_num, dtype=np.float32)
        else:
            seg_name = image_name.replace('train','seg.train').replace('test','seg.test') + '.seg.crop.png'
            no_aug = self.args.no_aug
            if os.path.exists(seg_name):
                # image, word_label = random_crop_image(seg_name, self.image_label_dict[image_name.split('/')[-1]], self.image_size, self.class_num, self.phase, index, no_aug)
                image, word_label = random_crop_image(image_name, self.image_label_dict[image_name.split('/')[-1]], self.image_size, self.class_num, self.phase, index, no_aug, self.args)
            else:
                image, word_label = random_crop_image(image_name, self.image_label_dict[image_name.split('/')[-1]], self.image_size, self.class_num, self.phase, index, no_aug, self.args)

            # 灰度反向翻转，变成黑底，白字
            if self.phase == 'train':
                r = get_random(index+111) 
                if r < 0.1:
                    image[0,:,:] = 255 - image[0,:,:]
                elif r < 0.2:
                    image[1,:,:] = 255 - image[1,:,:]
                elif r < 0.3:
                    image[2,:,:] = 255 - image[2,:,:]
                if get_random(index+112) < 0.2:
                    image = 255. - image

            image = (image / 128. - 1).astype(np.float32)
            return image_name, image, word_label

    def __len__(self):
        return len(self.image_names) 

last_random = 10
def get_random(idx):
    global last_random
    if last_random < 1:
        np.random.seed(int(last_random * 1000000 + time.time()) + idx)
    else:
        np.random.seed(int((time.time())))
    x = np.random.random()
    while np.abs(last_random - x) < 0.1:
        x = np.random.random()
    last_random = x
    return x

def comput_iou(font, proposal):
    fx,fy,fh,fw = font
    px,py,pd = proposal
    overlap_x =  max(min(pd, fh) - np.abs(fx - px), 0)
    overlap_y =  max(min(pd, fw) - np.abs(fy - py), 0)
    # 面积
    sf = fh * fw
    sp = pd * pd
    so = overlap_x * overlap_y
    iou = float(so) / (sf + sp - so)
    return iou

def generate_bbox_label(image, font_place, font_size, font_num, args, image_size):
    imgh,imgw = image.size
    seg_label = np.zeros((image_size[0]/2, image_size[1]/2), dtype=np.float32)
    sx = float(font_place[0]) / image.size[0] * image_size[0]
    ex = sx + float(font_size) / image.size[0] * image_size[0] * font_num
    sy = float(font_place[1]) / image.size[1] * image_size[1]
    ey = sy + float(font_size) / image.size[1] * image_size[1]
    seg_label[int(sx)/2:int(ex)/2, int(sy)/2:int(ey)/2] = 1
    seg_label = seg_label.transpose((1,0))

    bbox_label = np.zeros((
        image_size[0]/args.stride,  # 16
        image_size[1]/args.stride,  # 16
        len(args.anchors),          # 4
        4                           # dx,dy,dd,c
        ), dtype=np.float32)
    fonts= []
    for i in range(font_num):
        x = font_place[0] + font_size/2. + i * font_size
        y = font_place[1] + font_size/2.
        h = font_size
        w = font_size

        x = float(x) * image_size[0] / imgh
        h = float(h) * image_size[0] / imgh
        y = float(y) * image_size[1] / imgw
        w = float(w) * image_size[1] / imgw
        fonts.append([x,y,h,w])

    # print bbox_label.shape
    for ix in range(bbox_label.shape[0]):
        for iy in range(bbox_label.shape[1]):
            for ia in range(bbox_label.shape[2]):
                proposal = [ix*args.stride + args.stride/2, iy*args.stride + args.stride/2, args.anchors[ia]]
                iou_fi = []
                for fi, font in enumerate(fonts):
                    iou = comput_iou(font, proposal)
                    iou_fi.append((iou, fi))
                max_iou, max_fi = sorted(iou_fi)[-1]
                if max_iou > 0.5:
                    # 正例
                    dx = (font[0] - proposal[0]) / float(proposal[2])
                    dy = (font[1] - proposal[1]) / float(proposal[2])
                    fd = max(font[2:])
                    dd = np.log(fd / float(proposal[2]))
                    # bbox_label[ix,iy,ia] = [dx, dy, dd, 1]
                    bbox_label[ix,iy,ia] = [dx, dy, dd, 1]
                elif max_iou > 0.25:
                    # 忽略
                    bbox_label[ix,iy,ia,3] = 0
                else:
                    # 负例
                    bbox_label[ix,iy,ia,3] = -1
    # 这里有一个transpose操作
    bbox_label = bbox_label.transpose((1,0,2,3))


                # 计算anchor信息
    return bbox_label, seg_label

def get_resize_para(size, idx):
    if size > 48:
        rh, rw = 4,4
    elif size > 32:
        if idx % 2:
            rh, rw = 2,4
        else:
            rh, rw = 4,2
    elif size > 16:
        if idx % 2:
            rh, rw = 1,2
        else:
            rh, rw = 2,1
    else:
        return 1,1

    rhs = range(rh)
    np.random.seed(int(time.time()) + idx + 1)
    np.random.shuffle(rhs)
    rh = rhs[0] + 1

    rws = range(rw)
    np.random.seed(int(time.time()) + idx + 2)
    np.random.shuffle(rws)
    rw = rws[0] + 1

    return rh, rw

# def generate_image(idx, image, word_index_dict, class_num, args, image_size, no_aug, epoch):
def generate_image( idx, image, no_aug, dataset):
    '''
    args.model == 'resnet' 的时候只是用于训练分割网络，大部分augmentation都不用
    这里的注释，默认参数是
        image_size [512, 64]
        rotate_range [-5, 5]
        font_range [8,32]
    '''

    word_index_dict = dataset.word_index_dict
    class_num = dataset.class_num
    args = dataset.args
    image_size = dataset.image_size
    font_range = dataset.font_range
    rotate_range = dataset.rotate_range 
    epoch = args.epoch
    margin = dataset.margin

    # 选择文字背景
    image = image.resize((1024,1024))
    h,w = image.size
    # 随机crop一个部分，resize成固定大小，会对文字有一定的水平竖直方向拉伸
    h_crop = int(get_random(idx + 10) * image_size[0] * 2 / 8) + image_size[0] * 6 / 8 # 长度范围 [374, 512]
    w_crop = int(get_random(idx + 11) * image_size[1] * 2 / 8) + image_size[1] * 6 / 8 # 宽度范围 [48, 64]
    if args.model == 'resnet' or no_aug or epoch < 60:
        # resnet: 分割网络采用固定大小crop
        # epoch<60: 网络训练初期采用固定大小，加速收敛
        h_crop = image_size[0]
        w_crop = image_size[1]
    # 选择文字背景，随机选择crop起始位置
    x = int(get_random(idx+12) * (h - h_crop))
    y = int(get_random(idx+13) * (w - w_crop))
    image = image.crop((x,y,x+h_crop,y+w_crop))


    # 字体大小是最容易引起错误的变量，字体大小不能超出图片中心区域大小
    size = font_range[0] + int(get_random(idx+20) * (font_range[1] - font_range[0]))
    size = min(size, h_crop - 2*margin - 2, w_crop - 2*margin - 2)

    # 字体数量，超过可容纳数量的一半以上，至少包含一个字符
    large_num = max(0, (h_crop - 2 * margin)/ size - 1)     
    word_num = int(min(large_num / 2, 5) + get_random(idx+21) * large_num / 2) + 1
    # word_num = int(large_num / 2 + get_random(idx+21) * large_num / 2) + 1
    word_num = max(1, word_num)

    # 添加字体位置，并生成label信息
    place_x = int(get_random(idx+22) * (h_crop - word_num * size - margin)) + margin
    if margin == 0:
        # 用于添加两排文字
        place_y = int(get_random(idx+23) * (w_crop/2 - size - margin)) + margin
    else:
        place_y = int(get_random(idx+23) * (w_crop - size - margin)) + margin
    place = (place_x, place_y)
    label = np.zeros(class_num).astype(np.float32)

    text = u''
    words = word_index_dict.keys()

    if margin == 0:
        # 两排文字
        word_num *= 2
    while len(text) < word_num:
        np.random.shuffle(words)
        w = words[len(text)]
        if w in u'"(),':
            # 部分字符不建议生成
            continue
        text = text + w
        index = word_index_dict[w]
        label[index] = 1

    # 得到bbox_label
    if args.model == 'resnet':
        bbox_label, seg_label = generate_bbox_label(image, place, size, word_num, args, image_size)
    else:
        bbox_label, seg_label = 0,0

    # 字体，可以添加其他字体
    fonts = ['../../files/ttf/simsun.ttf']
    np.random.shuffle(fonts)
    font = fonts[0]

    # 颜色
    r = get_random(idx+24)
    if no_aug or r < 0.7:
        # 选择不同程度的黑色
        if r < 0.3:
            c = int(get_random(idx + 25) * 64)
            color = (c,c,c)
        else:
            rgb = 64
            r = int(get_random(idx + 27) * rgb)
            g = int(get_random(idx + 28) * rgb)
            b = int(get_random(idx + 29) * rgb)
            color = (r,g,b)
    else:
        # 随机颜色，但是选择较暗的颜色
        rgb = 256
        r = int(get_random(idx + 27) * rgb)
        g = int(get_random(idx + 28) * rgb)
        b = int(get_random(idx + 29) * rgb)
        ra = get_random(idx + 30)
        if ra < 0.5:
            ra = int(1000 * ra) % 3
            if ra == 0:
                r = 0
            elif ra == 1:
                g = 0
            else:
                b = 0
        color = (r,g,b)

    # 增加文字到图片
    if margin == 0:
        image = add_text_to_img(image, text[:word_num/2], size, font, color, place)
        image = add_text_to_img(image, text[word_num/2:], size, font, color, (place[0], place[1]+image_size[1]/2))
    else:
        image = add_text_to_img(image, text, size, font, color, place)

    '''
    # 随机翻转，增加泛化程度
    if args.model != 'resnet':
        if get_random(idx+130) < 0.3:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        if get_random(idx+131) < 0.3:
            image = image.transpose(Image.FLIP_TOP_BOTTOM)

    # 先做旋转，然后在拉伸图片
    h,w = image.size
    max_hw, min_hw = float(max(h,w)), float(min(h,w))
    if max_hw / min_hw >= 5:
        rotate_size = 5
    elif max_hw / min_hw >= 3:
        rotate_size = 10
    elif max_hw / min_hw >= 1.5:
        rotate_size = 30
    else:
        rotate_size = 50
    if args.model != 'resnet' and not no_aug and epoch>70 and get_random(idx+50) < 0.8:
        theta = int(rotate_size * 2 * get_random(idx+32)) - rotate_size
        image = image.rotate(theta)
    else:
        theta = 0
    '''


    # 还原成 [512, 64] 的大小
    image = image.resize(image_size)


    # 最后生成图片后再一次旋转，图片模糊化
    if args.model == 'resnet' or (get_random(idx+50) < 0.8 and not no_aug):

        # 旋转
        if args.model == 'resnet' :
            rotate_size = 10
        else:
            rotate_size = rotate_range[0] + int(get_random(idx+32) * (rotate_range[1] - rotate_range[0]))
        theta = int(rotate_size * 2 * get_random(idx+33)) - rotate_size
        image = image.rotate(theta)
        if args.model == 'resnet':
            # 作分割的时候，标签信息也需要一起旋转
            seg_label = np.array([seg_label, seg_label, seg_label]) * 255
            seg_label = np.array(Image.fromarray(seg_label.transpose([1,2,0]).astype(np.uint8)).rotate(theta))
            seg_label = (seg_label[:,:,0] > 128).astype(np.float32)

    filters = [
            ImageFilter.SMOOTH,                 # 平滑，大于16可以用
            ImageFilter.SMOOTH_MORE,            # 平滑，大于16可以用
            ImageFilter.GaussianBlur(radius=1), # 大于16可以用

            ImageFilter.GaussianBlur(radius=2), # 大于32可以用
            ImageFilter.BLUR,                   # 大于32可以用
            ImageFilter.GaussianBlur(radius=2), # 多来两次
            ImageFilter.BLUR,                   # 多来两次
            ]

    # 当文字比较大的时候，增加一些模糊
    if size > 16:
        if size < 32:
            filters = filters[:3]
        np.random.shuffle(filters)
        image = image.filter(filters[idx % len(filters)])

    if args.model == 'resnet':
        # add noise
        noise_level = 32
        image = np.array(image)
        noise = np.random.random(image.shape) * noise_level - noise_level / 2.
        image = image + noise
        image = image.astype(np.uint8)
        image = Image.fromarray(image)


    # 有时候需要低分辨率的图片
    resize_0, resize_1 = get_resize_para(size, idx)
    image = image.resize([image_size[0]/resize_0, image_size[1]/resize_1])

    # 还原成 [512, 64] 的大小
    image = image.resize(image_size)

    return image, label, bbox_label, seg_label, size

def add_text_to_img(img, text, size, font, color, place):
    imgdraw = ImageDraw.Draw(img)
    imgfont = ImageFont.truetype(font,size=size)
    imgdraw.text(place, text, fill=color, font=imgfont)
    return img

def random_crop_image(image_name, text, image_size, class_num, phase, idx, no_aug, args):
    # label
    text = text.split()
    word_label = np.zeros(class_num, dtype=np.float32)

    
    if args.hist:
        if get_random(idx+34) < 0.4 and phase == 'train':
            image = Image.open(image_name).convert('RGB')
        else:
            # 直方图均衡化
            image = Image.open(image_name).convert('YCbCr')
            image = np.array(image)
            imy = image[:,:,0]
            imy,_ = histeq(imy)
            image[:,:,0] = imy
            image = Image.fromarray(image, mode='YCbCr').convert('RGB')
    else:
        image = Image.open(image_name).convert('RGB')
    x = np.array(image)
    assert x.min() >= 0
    assert x.max() < 256

    if phase == 'train' and not no_aug:
        # 旋转
        if get_random(idx+11) < 0.8:
            theta = int(6 * get_random(idx+1)) - 3
            image = image.rotate(theta)

        # 模糊处理
        if get_random(idx+2) < 0.3:
            np.random.shuffle(filters)
            image = image.filter(filters[0])

        # 短边小于64， 直接填0
        h,w = image.size
        if w < image_size[1] and h > 64:
            if get_random(idx+3) < 0.3:
                image = np.array(image)
                start_index = (image_size[1] - w)/2
                new_image = np.zeros((image_size[1], h, 3), dtype=np.uint8)
                new_image[start_index:start_index+w, :, :] = image
                image = Image.fromarray(new_image)


    # 先处理成 X * 64 的图片
    h,w = image.size
    h = int(float(h) * image_size[1] / w)
    image = image.resize((h, image_size[1]))

    if phase == 'train' and not no_aug:

        # 放缩 0.8~1.2
        h,w = image.size
        r = get_random(idx+4) / 4. + 0.8
        image = image.resize((int(h*r), int(w*r)))

        # crop
        if min(h,w) > 32:
            crop_size = 20
            x = int((crop_size * get_random(idx+5) - crop_size/2) * r)
            y = int((crop_size * get_random(idx+6) - crop_size/2) * r)
            image = image.crop((max(0,x),max(0,y),min(0,x)+h,min(0,y)+w))

        # 有时需要生成一些低分辨率的图片
        h,w = image.size
        r = get_random(idx+7)
        
        '''
        if r < 0.01 and min(h,w) > 64:
            image = image.resize((h/8, w/8))
        elif r < 0.1 and min(h,w) > 64:
            image = image.resize((h/4, w/4))
        elif r < 0.3 and min(h,w) > 32:
            image = image.resize((h/2, w/2))
        '''

        # 从新变为 X * 64 的图片
        h = int(float(h) * image_size[1] / w)
        image = image.resize((h, image_size[1]))

    # 填充成固定大小
    image = np.transpose(np.array(image), [2,0,1]).astype(np.float32)
    if image.shape[2] < image_size[0]:
        # 长宽比例小于8(16)，直接填充
        if phase == 'test':
            # 正中间
            start = np.abs(image_size[0] - image.shape[2])/2
        else:
            start = int(np.random.random() * np.abs(image_size[0] - image.shape[2]))
        new_image = np.zeros((3, image_size[1], image_size[0]), dtype=np.float32)
        new_image[:,:,start:start+image.shape[2]] = image
        if phase == 'test':
            new_image = np.array([new_image]).astype(np.float32)
        for w in text:
            word_label[int(w)] = 1
    else:
        # 长宽比例大于16，随机截取
        if phase == 'test':
            # 测试阶段直接合并
            crop_num = image.shape[2] * 2 / image_size[0] + 1
            new_image = np.zeros((crop_num, 3, image_size[1], image_size[0]), dtype=np.float32)
            for i in range(crop_num):
                start_index = i * image_size[0] / 2
                end_index = start_index + image_size[0]
                if end_index > image.shape[2]:
                    new_image[i,:,:,:image.shape[2] - start_index] = image[:,:,start_index:end_index]
                else:
                    new_image[i] = image[:,:,start_index:end_index]
            for w in text:
                word_label[int(w)] = 1
        else:
            # 训练阶段不算负例loss
            start = int(np.random.random() * np.abs(image_size[0] - image.shape[2]))
            new_image = image[:,:,start:start+image_size[0]]
            for w in text:
                word_label[int(w)] = -1

    image = new_image
    if phase == 'train':
        image = image.astype(np.float32)
        '''
        # 每一列灰度有所改变
        if get_random(idx+9) < 0.3:
            change_level = 256. / image.shape[1]
            gray_change = 0 
            for j in range(image.shape[1]):
                gray_change += change_level * get_random(j+idx) - change_level / 2
                image[:,j,:] += gray_change
        # 每一行灰度有所改变
        if get_random(idx+10) < 0.3:
            change_level = 256. / image.shape[2]
            gray_change = 0
            for k in range(image.shape[2]):
                gray_change += change_level * get_random(10+k+idx) - change_level / 2
                image[:,:,k] += gray_change
        '''
        # 增加噪声
        if get_random(idx+8) < 0.1:
            noise_level = 64
            noise = np.random.random(image.shape) * noise_level - noise_level / 2.
            image = image + noise 
            # noise = np.random.random(image.shape[1:]) * noise_level - noise_level / 2.
            # image = image + np.array([noise, noise, noise])
            image = image.astype(np.float32)

    return image, word_label
