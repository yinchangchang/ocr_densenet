# coding=utf8
import matplotlib.pyplot as plt
import numpy as np

def plot_multi_graph(image_list, name_list, save_path=None, show=False):
    graph_place = int(np.sqrt(len(name_list) - 1)) + 1
    for i, (image, name) in enumerate(zip(image_list, name_list)):
        ax1 = plt.subplot(graph_place,graph_place,i+1)
        ax1.set_title(name)
        # plt.imshow(image,cmap='gray')
        plt.imshow(image)
        plt.axis('off')
    if save_path:
        plt.savefig(save_path)
        pass
    if show:
        plt.show()

def plot_multi_line(x_list, y_list, name_list, save_path=None, show=False):
    graph_place = int(np.sqrt(len(name_list) - 1)) + 1
    for i, (x, y, name) in enumerate(zip(x_list, y_list, name_list)):
        ax1 = plt.subplot(graph_place,graph_place,i+1)
        ax1.set_title(name)
        plt.plot(x,y)
        # plt.imshow(image,cmap='gray')
    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()


