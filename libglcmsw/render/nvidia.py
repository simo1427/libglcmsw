from numba import njit, cuda, float64, uint8, uint16, int8, void
import numpy as np
import math
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from skimage import io as si
import time
import concurrent.futures

@cuda.jit("void(float32[:,:], uint8[:,:], uint8, uint8, float32[:])")
def glcmgen_gpu(glcm,img, x_neighbour,y_neighbour, sum):
    xdims, ydims = img.shape
    xstart = 0
    xend = xdims
    ystart = 0
    yend = ydims
    i, j = cuda.grid(2)
    if x_neighbour < 0:
        xstart += -x_neighbour
    elif x_neighbour >= 0:
        xend = xdims - x_neighbour
    if y_neighbour < 0:
        ystart += -y_neighbour
    elif y_neighbour >= 0:
        yend = ydims - y_neighbour
    if (i>=xstart and i<xend) and ((j>=ystart and j<yend)):
        ref=img[i, j]
        val=img[i + x_neighbour, j + y_neighbour]
        cuda.syncthreads()
        cuda.atomic.add(glcm, (ref, val),1)
        cuda.atomic.add(glcm, (val, ref), 1)
        #glcm[i,j]+=1
        cuda.atomic.add(sum,0,2)
        #cuda.syncthreads()
    else:
        return

@cuda.jit("void(float32[:,:],float32[:], float32[:,:])")
def normalize(glcm, div, glcm_norm):
    i,j = cuda.grid(2)
    glcm_norm[i,j]=glcm[i,j]/div[0]

@cuda.jit("void(float32[:,:],uint8, float32[:,:])")
def feature_gpu(glcm, prop, glcm_edit):
    i,j = cuda.grid(2)
    if prop == 0: #dissimilarity
        glcm_edit[i,j]=glcm[i,j]*abs(i-j)
    elif prop == 1:#contrast
        glcm_edit[i, j] = glcm[i, j] * (i - j)**2
    elif prop == 2:#homogeneity
        glcm_edit[i,j] = glcm[i,j]/(1+(i-j)**2)
    elif prop == 3 or prop==4:#ASM, energy
        glcm_edit[i,j]=glcm[i,j]**2
    elif prop == 5:#entropy
        if glcm[i,j]==0:
            glcm_edit[i,j]=0
        else:
            glcm_edit[i,j]= glcm[i,j]*-math.log(glcm[i,j])


def singleval(img, prop, dist, angle, bitdepth):
    #print(img.shape)
    x_neighbour = round(dist * np.sin(angle))
    y_neighbour = round(dist * np.cos(angle))

    threadsperblock=(32,32)
    blockspergrid_x=math.ceil(img.shape[0]/threadsperblock[0])
    blockspergrid_y=math.ceil(img.shape[1]/threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    glcm=np.zeros((bitdepth, bitdepth), dtype=np.float32)
    glcm_dev = cuda.to_device(glcm)
    img_dev = cuda.to_device(img)
    sum=np.zeros((1,), dtype=np.float32)
    sum_dev=cuda.to_device(sum)
    glcmgen_gpu[blockspergrid, threadsperblock](glcm_dev, img_dev, x_neighbour, y_neighbour, sum_dev)
    threadsperblock = (256, 1)
    blockspergrid_x = math.ceil(bitdepth / threadsperblock[0])
    blockspergrid_y = math.ceil(bitdepth / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    glcm_norm_dev=cuda.device_array_like(glcm)
    normalize[blockspergrid, threadsperblock](glcm_dev, sum_dev, glcm_norm_dev)
    val=np.zeros((bitdepth,bitdepth), dtype=np.float32)
    val_dev=cuda.to_device(val)
    feature_gpu[blockspergrid, threadsperblock](glcm_norm_dev,prop,val_dev)
    val=val_dev.copy_to_host()
    return np.sum(val, axis=None)
    #print("Sum of weights:",)