import pandas as pd
import numpy as np
import random
import operator
import math
import matplotlib.pyplot as plt 
from scipy.stats import multivariate_normal
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from numpy import unique
from scipy.stats import entropy as scipy_entropy
from scipy import ndimage as ndi
import openslide
from openslide import OpenSlide, OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
import cv2
import time
import pyopencl as cl

def get_subimages(img,cord):    
    subimages = []
    for r in cord:
        row_subimages = (img.read_region(r,0,(300,300)).convert('RGB'))
        row_subimages = cv2.cvtColor(np.array(row_subimages), cv2.COLOR_RGB2GRAY)
        subimages.append(row_subimages)
        
    return subimages

def shannon_entropy(image, base=2):
    _, counts = unique(image, return_counts=True)
    return scipy_entropy(counts, base=base)

def platformslist():
    return [platform.name for platform in cl.get_platforms()]

def platformselect(ind):
    return [platform for platform in cl.get_platforms()][ind]

def convolve(image, kernels, context, prgs):
    batch=512
    tmpkrn=np.array(kernels)
    krn=np.zeros((ksize, ksize, 16), dtype=np.float32)
    for i in range(ksize):
        for j in range(ksize):
            for k in range(16):
                krn[i,j,k]=tmpkrn[k,i,j]
    #print(tmpkrn[0,:,:])
    print(krn.shape, tmpkrn.shape)
    print(image.shape[0])
    queue=cl.CommandQueue(context)
    for ibatch in range(image.shape[0]//batch+1):
        print(f"\nbatch number {ibatch}")
        img=np.array(image[ibatch*batch:(ibatch+1)*batch])
        res=np.zeros((img.shape[0],img.shape[1],img.shape[2],16), dtype=np.float32)
        img_buff=cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=img.nbytes)
        res_buff=cl.Buffer(context, flags=cl.mem_flags.READ_WRITE, size=res.nbytes)
        krn_buff=cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=krn.nbytes)
        begin=time.perf_counter()
        inp=[(img,img_buff),(res,res_buff),(krn, krn_buff)]
        out=[(res,res_buff)]
        for (arr, buff) in inp:
            cl.enqueue_copy(queue, src=arr, dest=buff)
        patchsz=16
        krn_args=[img_buff, res_buff, krn_buff, np.int32(img.shape[1]),np.int32(img.shape[2]), np.int32(31), np.int32(16), np.int32(img.shape[0]), np.int32(patchsz)]
        completedEvent=prgs.convolvecut(queue, (img.shape[1]//patchsz+1, img.shape[2]//patchsz+1), None, *krn_args)
        #completedEvent=prgs.convolvecut(queue, (img.shape[0],), None, *krn_args)
        #completedEvent.wait()
        for (arr, buff) in out:
            cl.enqueue_copy(queue, src=buff, dest=arr)
        img_buff.release()
        res_buff.release()
        krn_buff.release()
        np.save(f"ibatch{ibatch}.npy", res)

    queue.finish()
    return filtered
    


def compute_feats(image,filtered):
    accum = np.zeros_like(image, dtype=np.float32)
    feats = np.zeros((len(kernels), 3), dtype=np.double)
    beginall=time.perf_counter()
    for k in range(16):
        #filtered = ndi.convolve(image, kernel, mode='wrap')
        """mn=res[k].min(axis=None)
        mx=res[k].max(axis=None)
        res[k]=res[k]-mn
        mn=res[k].min(axis=None)
        mx=res[k].max(axis=None)
        vpp=256/((mx-mn))#value per pixel
        for i in range(res[k].shape[0]):
            for j in range(res[k].shape[1]):
                res[k,i,j]=round(res[i,j,k]*vpp)"""
        np.maximum(accum, res[k], accum)
        x = accum.mean()
        y = np.std(accum)
        z = shannon_entropy(accum)
        return x,y,z

if __name__ == "__main__":
    image1=openslide.OpenSlide(r'ABCD3.svs')
    #image2=openslide.OpenSlide(r'WLS.svs')

    src=open("src-3darray-multiimg-working2dglobal size.cl", "r").read()
    print(platformslist())
    platform = platformselect(0)
    devices=platform.get_devices()

    context=cl.Context(devices=devices)
    prgs_src=cl.Program(context, src)
    prgs=prgs_src.build()


    df_full = pd.read_csv('Haralick_Features_QuPath_measurements_swap.tsv', sep='\t', header=0)
    df = df_full[['Centroid X µm','Centroid Y µm']].div(0.50119999999999998) 
    df = df.sub(150) 
    df = df.round(0).astype(int) 
    cord1= df.loc[0:28459]         
    cord2 = df.loc[28460::]

    cord1 = list(cord1[['Centroid X µm', 'Centroid Y µm']].itertuples(index=False, name=None))   
    cord2 = list(cord2[['Centroid X µm', 'Centroid Y µm']].itertuples(index=False, name=None))

    print("Getting ABCD3 subimages")
    ABCD3=np.load("ABCD3.npy")
    #print("File read. Coverting to a list")
    #ABCD3 = get_subimages(image1,cord1)
    #print("Getting WLS subimages")
    #WLS = get_subimages(image2,cord2)
    #np.save("ABCD3.npy", np.array(ABCD3))
    result = []

    kernel1 = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel((ksize, ksize), 0.84, theta, 2, 0.5, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        kernel1.append(kernel)

    kernel2 = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel((ksize, ksize), 1.68, theta, 4, 0.5, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        kernel2.append(kernel)
    
    kernel3 = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel((ksize, ksize), 3.36, theta, 8, 0.5, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        kernel3.append(kernel)

    kernel4 = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel((ksize, ksize), 7.72, theta, 16, 0.5, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        kernel4.append(kernel)

    kernel5 = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 16):
        kernel = cv2.getGaborKernel((ksize, ksize), 15.42, theta, 32, 0.5, 0, ktype=cv2.CV_32F)
        kernel /= 1.5*kernel.sum()
        kernel5.append(kernel)

    try:
        print("Starting filtering")
        begin=time.perf_counter()
        filtered=convolve(ABCD3, kernel1,context, prgs)
        print(f"Filtered all images in {time.perf_counter()-begin}")
        featsABCD1=[compute_feats(image) for image in filtered]
    except KeyboardInterrupt:
        exit()