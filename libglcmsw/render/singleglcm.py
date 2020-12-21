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
    #i, j = cuda.grid(2)
    if x_neighbour < 0:
        xstart += -x_neighbour
    elif x_neighbour >= 0:
        xend = xdims - x_neighbour
    if y_neighbour < 0:
        ystart += -y_neighbour
    elif y_neighbour >= 0:
        yend = ydims - y_neighbour
    for i in range(xstart, xend, 1):
        for j in range(ystart, yend, 1):
            #if (i>=xstart and i<xend) and ((j>=ystart and j<yend)):
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

@cuda.jit("void(float32[:,:],float32[:])")
def normalize(glcm, div):
    #i,j = cuda.grid(2)
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            tmp=glcm[i,j]
            glcm[i,j]=tmp/div[0]

@cuda.jit("void(float32[:,:],uint8)")
def feature_gpu(glcm, prop):
    #i,j = cuda.grid(2)
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            tmp=glcm[i,j]
            if prop == 0: #dissimilarity
                glcm[i,j]=tmp*abs(i-j)
            elif prop == 1:#contrast
                glcm[i, j] = tmp * (i - j)**2
            elif prop == 2:#homogeneity
                glcm[i,j] = tmp/(1+(i-j)**2)
            elif prop == 3 or prop==4:#ASM, energy
                glcm[i,j]=tmp**2
            elif prop == 5:#entropy
                if tmp==0:
                    glcm[i,j]=0
                else:
                    glcm[i,j]= tmp*-math.log(glcm[i,j])

def singlerungpu(img):
    #print(img.shape)
    dist=1
    angle=0
    bitdepth=256
    windowsz = 13
    x_neighbour = round(dist * np.sin(angle))
    y_neighbour = round(dist * np.cos(angle))

    threadsperblock=(1,1)
    blockspergrid_x=1
    blockspergrid_y=1
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    glcm=np.zeros((256, 256), dtype=np.float32)
    glcm_dev = cuda.to_device(glcm)
    img_dev = cuda.to_device(img)
    sum=np.zeros((1,), dtype=np.float32)
    sum_dev=cuda.to_device(sum)
    glcmgen_gpu[blockspergrid, threadsperblock](glcm_dev, img_dev, x_neighbour, y_neighbour, sum_dev)
    normalize[blockspergrid, threadsperblock](glcm_dev, sum_dev)
    feature_gpu[blockspergrid, threadsperblock](glcm_dev,3)
    val=glcm_dev.copy_to_host()
    print("Sum of weights:",np.sum(val, axis=None))
    return glcm


@cuda.jit("void(float32[:,:,:], uint8[:,:],uint16, uint16, uint16,uint8, uint8, float32[:,:])", device=True)
def glcmgen_gpu_dev(glcm,img, rownum, colnum,windowsz,x_neighbour,y_neighbour, sum):
    xdims=windowsz
    ydims=windowsz
    xstart = rownum
    xend = xdims
    ystart = colnum
    yend = ydims
    #i, j = cuda.grid(2)
    if x_neighbour < 0:
        xstart += -x_neighbour
    elif x_neighbour >= 0:
        xend = rownum+xdims - x_neighbour
    if y_neighbour < 0:
        ystart += -y_neighbour
    elif y_neighbour >= 0:
        yend = colnum + ydims - y_neighbour
    for i in range(xstart, xend, 1):
        for j in range(ystart, yend, 1):
            #if (i>=xstart and i<xend) and ((j>=ystart and j<yend)):
            ref=img[i, j]
            val=img[i + x_neighbour, j + y_neighbour]
            cuda.syncthreads()
            cuda.atomic.add(sum, (cuda.blockIdx.x, 0), 2)

            cuda.atomic.add(glcm, (cuda.blockIdx.x,ref, val),1)
            cuda.syncthreads()
            cuda.atomic.add(glcm, (cuda.blockIdx.x,val, ref), 1)

    for i in range(glcm.shape[1]):
        for j in range(glcm.shape[2]):
            tmp=glcm[cuda.blockIdx.x,i,j]
            glcm[cuda.blockIdx.x,i,j]=tmp/sum[cuda.blockIdx.x,0]
    #return glcm

@cuda.jit("void(float32[:,:,:],uint8)", device=True)
def feature_gpu_dev(glcm, prop):
    #i,j = cuda.grid(2)
    for i in range(glcm.shape[1]):
        for j in range(glcm.shape[2]):
            tmp=glcm[cuda.blockIdx.x,i,j]
            if prop == 0: #dissimilarity
                glcm[cuda.blockIdx.x,i,j]=tmp*abs(i-j)
            elif prop == 1:#contrast
                glcm[cuda.blockIdx.x,i, j] = tmp * (i - j)**2
            elif prop == 2:#homogeneity
                glcm[cuda.blockIdx.x,i,j] = tmp/(1+(i-j)**2)
            elif prop == 3 or prop==4:#ASM, energy
                glcm[cuda.blockIdx.x,i,j]=tmp**2
            elif prop == 5:#entropy
                if tmp==0:
                    glcm[cuda.blockIdx.x,i,j]=0
                else:
                    glcm[cuda.blockIdx.x,i,j]= tmp*-math.log(glcm[cuda.blockIdx.x,i,j])

@cuda.jit("void(float32[:,:,:], uint8[:,:], uint8,uint8, uint8, float32[:,:],uint8, uint8)")
def swkrn(glcm, img, windowsz, x_neighbour, y_neighbour, sum, prop, rownum):
    glcmgen_gpu_dev(glcm,img,rownum, cuda.blockIdx.x, windowsz,x_neighbour, y_neighbour, sum)
    feature_gpu_dev(glcm, prop)
    sum[cuda.blockIdx.x,0]=0
    for i in range(glcm.shape[1]):
        for j in range(glcm.shape[2]):
            cuda.atomic.add(sum, (cuda.blockIdx.x,0),glcm[cuda.blockIdx.x,i,j])


def singlerungpusw(img, windowsz, batchsz):
    #print(img.shape)
    dist=1
    angle=0
    bitdepth=256
    x_neighbour = round(dist * np.sin(angle))
    y_neighbour = round(dist * np.cos(angle))

    threadsperblock=(1,1,1)
    blockspergrid_x=batchsz
    blockspergrid_y=1
    blockspergrid_z=1
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    glcm=np.zeros((batchsz,256, 256), dtype=np.float32)
    glcm_dev = cuda.to_device(glcm)
    img_dev = cuda.to_device(img)
    sum=np.zeros((batchsz,1), dtype=np.float32)
    sum_dev=cuda.to_device(sum)
    swkrn[blockspergrid, threadsperblock](glcm_dev, img_dev, windowsz, x_neighbour, y_neighbour, sum_dev, 3,0)
    glcm_tmp=glcm_dev.copy_to_host()
    val=sum_dev.copy_to_host()
    return val



def singleruncpu(img):
    #print(img.shape)
    dist=1
    angle=0
    bitdepth=256
    windowsz = 13
    x_neighbour = round(dist * np.sin(angle))
    y_neighbour = round(dist * np.cos(angle))
    begin = time.perf_counter()
    ref=greycomatrix(img, distances=[dist], angles=[angle], levels=256,symmetric=True, normed=False)[:,:,0,0]
    print(greycoprops(greycomatrix(img, distances=[dist], angles=[angle], levels=256,symmetric=True, normed=True), prop="ASM"))
    return ref

def singleruncpusw(img, windowsz, batchsz):
    #print(img.shape)
    dist=1
    angle=0
    bitdepth=256
    x_neighbour = round(dist * np.sin(angle))
    y_neighbour = round(dist * np.cos(angle))
    print(img.shape[1]-windowsz)
    begin = time.perf_counter()
    singleline=np.zeros((img.shape[1]-windowsz,), dtype=np.float32)
    for i in range(1):
        for j in range(batchsz):
            tmp=img[i:i+windowsz, j:j+windowsz]
            ref=greycomatrix(tmp, distances=[dist], angles=[angle], levels=bitdepth,symmetric=True, normed=True)
            singleline[j]=greycoprops(ref, prop="ASM")[0,0]
    return singleline

#img = img_as_ubyte(si.imread("./0_0.tif"))
img = img_as_ubyte(rgb2gray(si.imread("../../examples/input.tif")))

begin = time.perf_counter()
gpu=singlerungpu(img)
print(f"gpu:{time.perf_counter()-begin} sum:{np.sum(gpu, axis=None)}")

begin = time.perf_counter()
cpu=singleruncpu(img)
print(f"cpu:{time.perf_counter() - begin} sum:{np.sum(cpu, axis=None)}")

batch=64
begin = time.perf_counter()
cpu=singleruncpusw(img, 13, batch)
print(f"cpu:{time.perf_counter()-begin} sum:{np.sum(cpu, axis=None)}")

begin = time.perf_counter()
gpu=singlerungpusw(img, 13, batch)
print(f"gpu:{time.perf_counter() - begin} sum:{np.sum(gpu, axis=None)}")