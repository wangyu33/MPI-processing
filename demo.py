#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : demo.py
from utils import  *

if __name__ == '__main__':
    filename = './igg0.75.dcm'
    img = read_MPI_img(filename)
    center, img_flag = get_cluster_centers(img,2,4.5)
    ret = get_local_maximum_point(img, center,80)
    img_final = threshold_expansion(img, img_flag, 0.7, ret)
    fig = plt.figure()
    plt.imshow(img_final, vmax=np.max(img_final), vmin=0)
    plt.axis('off')
    fig.patch.set_alpha(0)
    plt.show()
    print(1)