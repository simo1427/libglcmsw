from numba import njit, cuda, float64, uint8, uint16, int8, void
import numpy as np
import math
from skimage import io as si
import time

@cuda.jit("void(float32[:,:,:], uint8[:,:],uint16, uint16, uint16,int8, int8, float32[:,:])", device=True)
def glcmgen_gpu_dev(glcm,img, rownum, colnum,windowsz,x_neighbour,y_neighbour, sum):
    xdims=windowsz
    ydims=windowsz
    xstart = rownum
    xend = xdims+rownum
    ystart = colnum
    yend = ydims+colnum
    if x_neighbour < 0:
        xstart += -x_neighbour
    elif x_neighbour >= 0:
        xend = rownum+xdims - x_neighbour
    if y_neighbour < 0:
        ystart += -y_neighbour
    elif y_neighbour >= 0:
        yend = colnum + ydims - y_neighbour
    tmp=0
    j = ystart + cuda.threadIdx.x
    stridej = windowsz // cuda.blockDim.x if windowsz % cuda.blockDim.x == 0 else windowsz // cuda.blockDim.x + 1
    yrangestart=ystart+stridej*cuda.threadIdx.x
    yrangeend=ystart+stridej*(cuda.threadIdx.x+1)
    for i in range(xstart, xend, 1):
        for j in range(yrangestart, yrangeend, 1):
            if (i>=xstart and i<xend) and ((j>=ystart and j<yend)):
                ref=img[i, j]
                val=img[i + x_neighbour, j + y_neighbour]
                cuda.atomic.add(sum, (rownum, colnum), 2)
                cuda.atomic.add(glcm, (cuda.blockIdx.x,ref, val),1)
                cuda.atomic.add(glcm, (cuda.blockIdx.x,val, ref), 1)

    #return glcm
@cuda.jit("void(float32[:,:,:],uint8)", device=True)
def feature_gpu_dev(glcm, prop):
    #i,j = cuda.grid(2)
    stridej = glcm.shape[2] // cuda.blockDim.x if glcm.shape[2] % cuda.blockDim.x == 0 else glcm.shape[2] // cuda.blockDim.x + 1
    for i in range(glcm.shape[1]):
        for j in range(cuda.threadIdx.x * stridej, (cuda.threadIdx.x + 1) * stridej):
            if j < glcm.shape[2]:
                if prop == 0: #dissimilarity
                    glcm[cuda.blockIdx.x,i,j]=glcm[cuda.blockIdx.x,i,j]*abs(i-j)
                elif prop == 1:#contrast
                    glcm[cuda.blockIdx.x,i, j] = glcm[cuda.blockIdx.x,i,j] * (i - j)**2
                elif prop == 2:#homogeneity
                    glcm[cuda.blockIdx.x,i,j] = glcm[cuda.blockIdx.x,i,j]/(1+(i-j)*(i-j))
                elif prop == 3 or prop==4:#ASM, energy
                    glcm[cuda.blockIdx.x,i,j]=glcm[cuda.blockIdx.x,i,j]*glcm[cuda.blockIdx.x,i,j]
                elif prop == 5:#entropy
                    if glcm[cuda.blockIdx.x,i,j]==0:
                        glcm[cuda.blockIdx.x,i,j]=0
                    else:
                        glcm[cuda.blockIdx.x,i,j]= glcm[cuda.blockIdx.x,i,j]*-math.log(glcm[cuda.blockIdx.x,i,j])

@cuda.jit("void(float32[:,:,:], uint8[:,:], uint8,int8, int8, float32[:,:],uint8)")
def swkrn(glcm, img, windowsz, x_neighbour, y_neighbour, sum, prop):
    nrows=sum.shape[0]
    ncols=sum.shape[1]
    stridey=nrows//cuda.gridDim.y if nrows%cuda.gridDim.y == 0 else nrows//cuda.gridDim.y+1
    #stridex = ncols // cuda.gridDim.x if ncols % cuda.gridDim.x == 0 else ncols // cuda.gridDim.x + 1
    colnum=cuda.blockIdx.x
    #print(stridex, stridey, cuda.blockIdx.x, cuda.blockIdx.y)
    #sum[9,9]=glcm.shape[0]
    for rownum in range(cuda.blockIdx.y*stridey, (cuda.blockIdx.y+1)*stridey):
        if rownum<nrows and colnum < ncols:
            glcmgen_gpu_dev(glcm,img,rownum, colnum, windowsz,x_neighbour, y_neighbour, sum)
            stridej=glcm.shape[2]//cuda.blockDim.x if glcm.shape[2]%cuda.blockDim.x==0 else glcm.shape[2]//cuda.blockDim.x+1
            for i in range(glcm.shape[1]):
                for j in range(cuda.threadIdx.x*stridej,(cuda.threadIdx.x+1)*stridej):
                    if j<glcm.shape[2]:
                        glcm[cuda.blockIdx.x, i, j] = glcm[cuda.blockIdx.x, i, j] / sum[rownum, colnum]
            feature_gpu_dev(glcm, prop)
            sum[rownum,colnum]=0
            for i in range(glcm.shape[1]):
                for j in range(cuda.threadIdx.x*stridej,(cuda.threadIdx.x+1)*stridej):
                    if j<glcm.shape[2]:
                        cuda.atomic.add(sum, (rownum,colnum),glcm[cuda.blockIdx.x,i,j])
                        glcm[cuda.blockIdx.x, i, j]=0
            if prop==4:
                sum[rownum, colnum]=math.sqrt(sum[rownum, colnum])
            #print(rownum, colnum, sum[rownum, colnum])
        #print(rownum)

@cuda.jit("void(float32[:,:,:], uint8[:,:],uint8[:,:], uint8,int8, int8, float32[:,:],uint8)")
def swkrnmask(glcm, img, mask, windowsz, x_neighbour, y_neighbour, sum, prop):
    nrows=sum.shape[0]
    ncols=sum.shape[1]
    stridey=nrows//cuda.gridDim.y if nrows%cuda.gridDim.y == 0 else nrows//cuda.gridDim.y+1
    #stridex = ncols // cuda.gridDim.x if ncols % cuda.gridDim.x == 0 else ncols // cuda.gridDim.x + 1
    colnum=cuda.blockIdx.x
    #print(stridex, stridey, cuda.blockIdx.x, cuda.blockIdx.y)
    #sum[9,9]=glcm.shape[0]
    for rownum in range(cuda.blockIdx.y*stridey, (cuda.blockIdx.y+1)*stridey):
        if rownum<nrows and colnum < ncols:
            if mask[rownum+windowsz//2, colnum+windowsz//2]:
                glcmgen_gpu_dev(glcm,img,rownum, colnum, windowsz,x_neighbour, y_neighbour, sum)
                stridej=glcm.shape[2]//cuda.blockDim.x if glcm.shape[2]%cuda.blockDim.x==0 else glcm.shape[2]//cuda.blockDim.x+1
                for i in range(glcm.shape[1]):
                    for j in range(cuda.threadIdx.x*stridej,(cuda.threadIdx.x+1)*stridej):
                        if j<glcm.shape[2]:
                            glcm[cuda.blockIdx.x, i, j] = glcm[cuda.blockIdx.x, i, j] / sum[rownum, colnum]
                feature_gpu_dev(glcm, prop)
                sum[rownum,colnum]=0
                for i in range(glcm.shape[1]):
                    for j in range(cuda.threadIdx.x*stridej,(cuda.threadIdx.x+1)*stridej):
                        if j<glcm.shape[2]:
                            cuda.atomic.add(sum, (rownum,colnum),glcm[cuda.blockIdx.x,i,j])
                            glcm[cuda.blockIdx.x, i, j]=0
                if prop==4:
                    sum[rownum, colnum]=math.sqrt(sum[rownum, colnum])
            else:
                sum[rownum, colnum]=-1
            #print(rownum, colnum, sum[rownum, colnum])
        #print(rownum)

def singletilegpusw(img, windowsz,prop, dist, angle, bitdepth=256, threads=32, vertworkers=1):
    x_neighbour = round(dist * np.sin(angle))
    y_neighbour = round(dist * np.cos(angle))

    batchsz=img.shape[1]
    threadsperblock=(threads,1,1)
    blockspergrid_x=batchsz
    blockspergrid_y=vertworkers
    blockspergrid_z=1
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    glcm=np.zeros((batchsz,bitdepth, bitdepth), dtype=np.float32)
    glcm_dev = cuda.to_device(glcm)
    img_dev = cuda.to_device(img)
    sum=np.zeros((img.shape[0]-windowsz,img.shape[1]-windowsz), dtype=np.float32)
    sum_dev=cuda.to_device(sum)
    swkrn[blockspergrid, threadsperblock](glcm_dev, img_dev, windowsz, x_neighbour, y_neighbour, sum_dev, prop)
    sum=sum_dev.copy_to_host()
    return sum

def masked(img, mask, windowsz,prop, dist, angle, bitdepth=256, threads=32, vertworkers=1):
    x_neighbour = round(dist * np.sin(angle))
    y_neighbour = round(dist * np.cos(angle))

    #batchsz=img.shape[1]-windowsz
    batchsz=img.shape[1]
    threadsperblock=(threads,1,1)
    blockspergrid_x=batchsz
    blockspergrid_y=vertworkers
    blockspergrid_z=1
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    glcm=np.zeros((batchsz,bitdepth, bitdepth), dtype=np.float32)
    glcm_dev = cuda.to_device(glcm)
    img_dev = cuda.to_device(img)
    sum=np.zeros((img.shape[0]-windowsz,img.shape[1]-windowsz), dtype=np.float32)
    sum_dev=cuda.to_device(sum)
    mask_dev = cuda.to_device(mask)
    swkrnmask[blockspergrid, threadsperblock](glcm_dev, img_dev,mask_dev, windowsz, x_neighbour, y_neighbour, sum_dev, prop)
    sum=sum_dev.copy_to_host()
    return sum