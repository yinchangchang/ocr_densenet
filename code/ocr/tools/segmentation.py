# coding=utf8
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage import morphology,color,data
from skimage import filters
import numpy as np
import skimage 
import os
from skimage import measure



def watershed(image, label=None):
    denoised = filters.rank.median(image, morphology.disk(2)) #过滤噪声
    #将梯度值低于10的作为开始标记点
    markers = filters.rank.gradient(denoised, morphology.disk(5)) < 10
    markers = ndi.label(markers)[0]

    gradient = filters.rank.gradient(denoised, morphology.disk(2)) #计算梯度
    labels =morphology.watershed(gradient, markers, mask=image) #基于梯度的分水岭算法

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    axes = axes.ravel()
    ax0, ax1, ax2, ax3 = axes

    ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax0.set_title("Original")
    # ax1.imshow(gradient, cmap=plt.cm.spectral, interpolation='nearest')
    ax1.imshow(gradient, cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title("Gradient")
    if label is not None:
        # ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
        ax2.imshow(label, cmap=plt.cm.gray, interpolation='nearest')
    else:
        ax2.imshow(markers, cmap=plt.cm.spectral, interpolation='nearest')
    ax2.set_title("Markers")
    ax3.imshow(labels, cmap=plt.cm.spectral, interpolation='nearest')
    ax3.set_title("Segmented")

    for ax in axes:
        ax.axis('off')

    fig.tight_layout()
    plt.show()

def plot_4(image, gradient,label,segmentation, save_path=None):
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6, 6))
    axes = axes.ravel()
    ax0, ax1, ax2, ax3 = axes
    ax0.imshow(image, cmap=plt.cm.gray, interpolation='nearest')
    ax0.set_title("Original")
    ax1.imshow(gradient, cmap=plt.cm.gray, interpolation='nearest')
    ax1.set_title("Gradient")
    ax2.imshow(label, cmap=plt.cm.gray, interpolation='nearest')
    ax2.set_title("label")
    ax3.imshow(segmentation, cmap=plt.cm.spectral, interpolation='nearest')
    ax3.set_title("Segmented")

    for ax in axes:
        ax.axis('off')

    fig.tight_layout()
    if save_path:
        print save_path
        plt.savefig(save_path)
    else:
        plt.show()

def fill(image):
    '''
    填充图片内部空白
    临时写的函数
    建议后期替换
    '''
    label_img = measure.label(image, background=1)
    props = measure.regionprops(label_img)
    max_area = np.array([p.area for p in props]).max()
    for i,prop in enumerate(props):
        if prop.area < max_area:
            image[prop.coords[:,0],prop.coords[:,1]] = 1
    return image



def my_watershed(image, label=None, min_gray=480, max_gray=708, min_gradient=5, show=False, save_path='/tmp/x.jpg'):
    image = image - min_gray
    image[image>max_gray] = 0
    image[image< 10]  = 0
    image = image * 5

    denoised = filters.rank.median(image, morphology.disk(2)) #过滤噪声
    #将梯度值低于10的作为开始标记点
    markers = filters.rank.gradient(denoised, morphology.disk(5)) < 10
    markers = ndi.label(markers)[0]

    gradient = filters.rank.gradient(denoised, morphology.disk(2)) #计算梯度
    labels = gradient > min_gradient

    mask = gradient > min_gradient
    label_img = measure.label(mask, background=0)
    props = measure.regionprops(label_img)
    pred = np.zeros_like(gradient)
    for i,prop in enumerate(props):
        if prop.area > 50:
            region = np.array(prop.coords)
            vx,vy = region.var(0)
            v = vx + vy
            if v < 200:
                pred[prop.coords[:,0],prop.coords[:,1]] = 1

    # 填充边缘内部空白
    pred = fill(pred)

    if show:
        plot_4(image, gradient, label, pred)
    else:
        plot_4(image, gradient, label, pred, save_path)

    return pred

def segmentation(image_npy, label_npy,save_path):
    print image_npy
    image = np.load(image_npy)
    label = np.load(label_npy)
    if np.sum(label) == 0:
        return
    min_gray,max_gray = 480, 708
    my_watershed(image,label,min_gray, max_gray,show=False, save_path=save_path)

def main():
    data_dir = '/home/yin/all/PVL_DATA/preprocessed/2D/'
    save_dir = '/home/yin/all/PVL_DATA/tool_result/'
    os.system('rm -r ' + save_dir)
    os.system('mkdir ' + save_dir)
    for patient in os.listdir(data_dir):
        patient_dir = os.path.join(data_dir, patient)
        for f in os.listdir(patient_dir):
            if 'roi.npy' in f:
                label_npy = os.path.join(patient_dir,f)
                image_npy = label_npy.replace('.roi.npy','.npy')
                segmentation(image_npy,label_npy, os.path.join(save_dir,label_npy.strip('/').replace('/','.').replace('npy','jpg')))

if __name__ == '__main__':
    # image =color.rgb2gray(data.camera())
    # watershed(image)
    main()
    image_npy = '/home/yin/all/PVL_DATA/preprocessed/2D/JD_chen_xi/23.npy'
    image_npy = '/home/yin/all/PVL_DATA/preprocessed/2D/JD_chen_xi/14.npy' 
    image_npy = '/home/yin/all/PVL_DATA/preprocessed/2D/JD_zhang_yu_chen/23.npy'
    label_npy = image_npy.replace('.npy','.roi.npy')
    segmentation(image_npy,label_npy)


