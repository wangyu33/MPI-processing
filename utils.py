#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File  : utils.py

import matplotlib.pyplot as plt
import numpy as np
import os
import SimpleITK as sitk
import cv2
from sklearn.cluster import KMeans
from collections import defaultdict

def read_MPI_img(path):
    """
    Reading MPI files
    :param path: MPI image file path
    :return: img Array
    """
    file_reader = sitk.ImageFileReader()
    file_reader.SetFileName(path)
    data = file_reader.Execute()
    data_np = sitk.GetArrayFromImage(data)
    img = data_np[0]
    img[img < 0] = 0
    return img

def get_cluster_centers(img, nums, Threshold):
    """
    For obtaining cluster centres
    :param img: MPI array
    :param nums: Number of clustering centres
    :param Threshold: Threshold to distinguish signal from background
    :return: cluster centers, MPI foreground signal splitting matrix
    """
    nrow = img.shape[0]
    ncol = img.shape[1]
    tt = []
    index = defaultdict(tuple)
    for i in range(nrow):
        for j in range(ncol):
            if img[i][j] > Threshold:
                tt.append([i, j])
                index[len(tt) - 1] = (i, j)

    kms = KMeans(n_clusters = nums)
    pre = kms.fit_predict(tt)
    img_flag = np.where(img > Threshold, 1, 0)
    center = []
    for x,y in kms.cluster_centers_:
        center.append((round(x),round(y)))
    return center, img_flag

def get_local_maximum_point(img,center,boxing_len):
    """
    Obtaining local maxima based on cluster centres
    :param img: MPI img
    :param center: cluster centres
    :param boxing_len: Box length for calculating local extremes
    :return: Local maximum point coordinates
    """
    nrow = img.shape[0]
    ncol = img.shape[1]
    ret = []
    for i in range(len(center)):
        x,y = center[i]
        maxn = 0
        x_max, y_max= x, y
        # print(x,y)
        for j in range(max(0,x-boxing_len//2),min(nrow, x+boxing_len//2)):
            for k in range(max(0,y-boxing_len//2),min(ncol, y+boxing_len//2)):
                if img[j][k] > maxn:
                    maxn = img[j][k]
                    x_max, y_max= j, k
        ret.append((x_max, y_max))
    return ret

def union_mpi_img(img_list,img):
    '''
    MPI img merge
    :param img_list: img after threshold_expansion
    :param img:MPI img
    :return:
    '''
    img1 = img_list[0]
    for i in range(len(img_list)-1):
        img2 = img_list[i+1]
        d = np.array((img1,img2))
        img1 = d.max(axis=0)
    return img1 * img

def threshold_expansion(img,img_flag, Threshold, center):
    '''
    Threshold expansion based on local maximum points
    :param img:MPI img
    :param img:MPI foreground signal splitting matrix
    :param Threshold: Expansion Threshold  0-1
    :param center:Local maximum point list
    :return:
    '''
    img_list = []
    nrow = img.shape[0]
    ncol = img.shape[1]
    for x, y in center:
        maxn = img[x][y]
        dir = [(-1,0),(1,0),(0,1),(0,-1)]
        point = [(x,y)]
        img_new = np.zeros([nrow, ncol], dtype=int)
        d = defaultdict(int)
        d[(x,y)] = 1
        while len(point) > 0:
            tmp_point = []
            for x1,y1 in point:
                img_new[x1][y1] = 1
                for dir_x,dir_y in dir:
                    x2 = x1 + dir_x
                    y2 = y1 + dir_y
                    try:
                        if img_flag[x2][y2] == 1 and img[x2][y2] >= maxn*Threshold \
                                and d[(x2,y2)] !=1:
                            tmp_point.append((x2,y2))
                            d[(x2, y2)] = 1
                    except:
                        pass
            point = tmp_point
    return union_mpi_img(img_list,img)



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
