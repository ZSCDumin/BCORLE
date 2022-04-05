# # coding: utf-8
#
# # save plot figures, if exists, overwrite
# import seaborn as sns
# import pandas as pd
# import itertools
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# import scipy
# #from scipy.misc import imsave
# from utils.tf_utils import fig2data
#
#
# def write_figure_to_oss(path):
#     # === define paras ==================
#     para_names = ['layer_n', 'activition', 'seed']
#
#     layer_n = [1, 2, 3, 4, 5, 6]
#     activition = ['tanh', 'sigmod', 'relu']
#     seed = [11, 17, 19]
#
#     # 创建 dataframe
#     iris = pd.DataFrame([], columns=para_names)
#     for values in itertools.product(layer_n, activition, seed):
#         newline = pd.DataFrame(list(values), index=para_names)
#         iris = iris.append(newline.T, ignore_index=True)
#     activ_dict = {'tanh': 2, 'sigmod': 4, 'relu': 6}  # 也可以直接定义字典，然后replace
#     iris['results'] = iris['layer_n'] + iris['activition'].replace(activ_dict) + iris[
#         'seed'] * 0.1 + np.random.random(
#         (54,))
#     iris['results'] = iris['results'].astype('float')  # 转换成浮点类型
#     fig = plt.figure(figsize=(8, 6))
#     sns.lineplot(x='layer_n', y='results', hue='activition', style='activition',
#                  markers=True, data=iris)
#     plt.grid(linestyle=':')
#     image = fig2data(fig)
#     scipy.misc.imsave('im.png', image)  # 保存图片
#
#     from scipy.misc import imsave, imread
#     oss_file = path + 'Bonus2.png'
#     with tf.afile.AFile(oss_file, "w") as f:
#         imsave(f, image)
#         print("figure saved in {}...".format(oss_file))
#         print("...OK")