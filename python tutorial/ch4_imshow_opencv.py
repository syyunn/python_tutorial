# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 10:55:42 2017
Chapter 4 imshowh with opencv
download : installation with conda
https://anaconda.org/menpo/opencv
conda install -c menpo opencv
http://opencv-python.readthedocs.io/en/latest/doc/01.imageStart/imageStart.html
cv 사용법 한글
@author: gihun
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import cv2
import re

from skimage.filters.rank import entropy
from skimage.morphology import disk
##%% single image show example
##imshow
#img = np.reshape(np.arange(0,24,1),[6,4])
#plt.imshow(img,cmap="gray")
#img0 = np.hamming(64)[:,None]@np.hamming(64)[None,:]
#plt.imshow(img0, cmap="jet")
#plt.imshow(img0, cmap="magma")
#plt.imshow(img0, cmap="gray")
#
##import os
##cd "C:\Users\gihun\Tensorflow\STUDY"
##os.mkdir("IMGSAVE")
#img0 = (img0-np.min(img0))/(np.max(img0)-np.min(img0))*255.0
#path = "IMGSAVE\hamming0.jpg"
#cv2.imwrite(path,img0)
##imread
#img1=cv2.imread(path)
#img1
#plt.imshow(img1, cmap="gray")




#%% multiple image show
def unstack(img, axis):
    d =img.shape[axis]
    arr = [np.squeeze(a,axis=axis) for a in np.split(img, d, axis=axis)]
    return arr
def make(img, nx, ny):
    '''
    img:[n,h,w] --> img:[ny*h, nx*w]
    '''
    n,h,w = img.shape
    black = np.zeros([nx*ny-n,h,w],dtype=img.dtype)
    img = np.concatenate([img, black],axis=0) #[nx*ny, h,w]
    img = np.reshape(img, [ny,nx,h,w]) #[ny,nx,h,w]
    img = unstack(img,axis=0) # ny *[nx,h,w]
    img = np.concatenate(img, axis=1) # [nx, ny*h,w]
    img = unstack(img, axis=0) # nx*[ny*h, w]
    img = np.concatenate(img, axis=1) # [ny*h, nx*w]
    return img

def make3d(img, nx,ny):
    '''
    img:[b,h,w,c]--> img:[ny*h,nx*w,c]
    '''
    b,h,w,c= img.shape
    black = np.zeros([nx*ny-b,h,w,c],dtype=img.dtype)
    img = np.concatenate([img,black],axis=0)
    img = np.reshape(img, [ny,nx,h,w,c])
    img = unstack(img,axis=0) # ny *[nx,h,w,c]
    img = np.concatenate(img, axis=1) # [nx, ny*h,w,c]
    img = unstack(img, axis=0) # nx*[ny*h, w,c]
    img = np.concatenate(img, axis=1) # [ny*h, nx*w,c]
    return img

def show(img, vmin=None, vmax=None, figsize=(10,200), path=None):
    if path==None:
        plt.figure(figsize=figsize)
        fig =plt.imshow(img, cmap="gray",interpolation="none",vmin=vmin,vmax=vmax)
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)
        plt.show()
    else:
        vmin=vmin if vmin!=None else np.min(img)
        vmax=vmax if vmax!=None else np.max(img)
        img = np.clip(img, vmin, vmax)
        img = (img-vmin)/(vmax-vmin)*255.0
        if len(img.shape)==3 and img.shape[2]==3:
            b, g, r = np.split(img, 3, axis=2)
            img =np.concatenate([r,g,b],axis=2)
            cv2.imwrite(path, img)
        else:    
            cv2.imwrite(path, img)
        
#%%
def imshow3d(img, nx=10, vmin=None, vmax=None, figsize=(10,200),path=None):
    '''
    x : [b, h, w]
    '''
    b,h,w = img.shape
    ny = ((b-1)//nx)+1
    img = make(img, nx,ny)
    show(img, vmin, vmax, figsize, path)
def imshow4d(img, nx=10, vmin=None, vmax=None, figsize=(10,200),path=None):
    '''
    x : [b,h,w,c] (1) c is 3 RGB
    (2) c is not 3 -->  2d image channel out
    '''
    b,h,w,c = img.shape
    if c==3:
        ny = ((b-1)//nx)+1
        img = make3d(img,nx,ny)
        show(img, vmin, vmax, figsize,path)
    else:
        '''
        [b,h,w,c] ---> [b*c, w,c]
        '''
        ny = ((b*c-1)//nx)+1
        img = np.transpose(img, axes=[0,3,1,2])
        img = np.reshape(img, [b*c,h,w])
        img = make(img, nx,ny)
        show(img, vmin, vmax, figsize, path)

def imread(path):
    img = cv2.imread(path)
    if img.shape[2]==3:
        b,g,r = np.split(img,3,axis=2)
        img = np.concatenate([r,g,b],axis=2)
    else:
        img = img
    return img
    

#%%
if __name__ =="__main__":
#%% signle image
    img0 = np.hamming(64)[:,None]@np.hamming(64)[None,:]
    zer0 = np.zeros_like(img0)
    img1 = np.stack([img0,zer0,zer0],axis=2)       
    show(img0)
    show(img1)
#%% multiple image
    imgs = np.stack([img0]*10, axis=0)
    path = "IMGSAVE\hammings.jpg"
    imshow3d(imgs,path=path)
    imgcs = np.stack([img1]*10, axis=0)
    path = "IMGSAVE\hammingcs.jpg"
    imshow4d(imgcs)
#%% real image example
    path= "IMGSAVE\example1.jpg"
    path= "IMGSAVE\A.png"
    
    pika =imread(path)
    show(pika)
#%% real image resize
    path= "IMGSAVE\example2.jpg"
    pika2 = imread(path)
    show(pika2)
    pika2.shape
    pika2 = cv2.resize(pika2, (225,225))
    pika2.shape

    #%% read multiple images and then multiple show
    path2="IMGSAVE"    
    import os
    import glob
    datapaths=glob.glob(os.path.join(path2,"example*"))
    datapaths
    img=np.zeros([0,225,225,3])   
    i=0
    img=[]
    for path in datapaths:
        ing = imread(path)
        ing = cv2.resize(ing,(225,225))              
        img.append(ing)
    # 주의 : concatenate 하면 깨짐?
    img2 = np.stack(img,axis=0)
    imshow4d(img2)
    
    
#%%
    path = r"D:\Dropbox\ecube@study\python tutorial\IMGSAVE\face.jpg"
    faceimg = imread(path)
    type(faceimg)
    show(faceimg,path = r"D:\Dropbox\ecube@study\python tutorial\IMGSAVE\face2.jpg")
    path = r"D:\Dropbox\ecube@study\python tutorial\IMGSAVE\face2.raw"
    import sys
    _path = r"D:\Dropbox\ZOOMFACE\SR"
    sys.path.append(_path)
    from util import *
    writeraw(path,faceimg,dtype=np.uint8)

