# coding: utf-8
import tensorflow as tf
import os
import numpy as np


def Smooth_L1_Loss(labels, predictions, name, is_weights):
    with tf.variable_scope(name):
        diff = tf.abs(labels-predictions)
        less_than_one = tf.cast(tf.less(diff, 1.0), tf.float32)  # Bool to float32
        smooth_l1_loss = (less_than_one*0.5*diff**2)+(1.0-less_than_one)*(diff-0.5)
        return tf.reduce_mean(is_weights * smooth_l1_loss)  # get the average


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        os.makedirs(path)  # makedirs 创建文件时如果路径不存在会创建这个路径
        print("--- new folder... ---")
        print("--- OK ---")
    else:
        print("---  The folder is already exist  ---")
        raise FileExistsError


def check_dir(path):
    folder = os.path.exists(path)
    if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
        print("---  The folder does not exist  ---")
        raise FileExistsError


def fig2data(fig):
    """
    fig = plt.figure()
    image = fig2data(fig)
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    import PIL.Image as Image
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    image = Image.frombytes("RGBA", (w, h), buf.tostring())
    image = np.asarray(image)
    return image
