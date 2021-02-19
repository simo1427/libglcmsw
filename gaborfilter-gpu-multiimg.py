import numpy as np
from scipy import ndimage as ndi
import cv2
from skimage import io as si
from skimage.util import img_as_float, img_as_ubyte
from skimage.color import rgb2gray
from skimage.filters import gabor_kernel
import pyopencl as cl
import time
import openslide
import os

#os.chdir(".\gabor")

#img = img_as_ubyte(rgb2gray(si.imread("E:\Monkey\MkyISH_SVZi1\ABCD3 - Mky506 - SVZi - isch downsample 4x.png")))
#img = img_as_ubyte(rgb2gray(si.imread("E:\Monkey\MkyISH_SVZi1\ischdown8x.png")))
img =np.array([img_as_ubyte(rgb2gray(si.imread("..\examples\input.tif"))),img_as_ubyte(rgb2gray(si.imread("..\examples\input.tif")))])
#img =np.array([img_as_ubyte(rgb2gray(si.imread("E:\Monkey\MkyISH_SVZi1\ABCD3 - Mky506 - SVZi - isch (1, x=18140, y=12784, w=2049, h=2049).png"))),img_as_ubyte(rgb2gray(si.imread("E:\Monkey\MkyISH_SVZi1\ABCD3 - Mky506 - SVZi - isch (1, x=27815, y=19608, w=2049, h=2049).png")))])
print(img.shape)
kernel5 = []
ksize = 31
for theta in np.arange(0, np.pi, np.pi / 16):
    kernel = cv2.getGaborKernel((ksize, ksize), 15.42, theta, 32, 0.5, 0, ktype=cv2.CV_32F)
    kernel /= 1.5*kernel.sum()
    kernel5.append(kernel)

print(kernel5[0].shape)
tmpkrn=np.array(kernel5)
krn=np.zeros((ksize, ksize, 16), dtype=np.float32)
for i in range(ksize):
    for j in range(ksize):
        for k in range(16):
            krn[i,j,k]=tmpkrn[k,i,j]
res=np.zeros((2,img.shape[1],img.shape[2],16), dtype=np.float32)
print(krn.shape)

begin=time.perf_counter()
si.imsave("test.tif",ndi.convolve(img[0], krn[:,:,0], mode='wrap'))
print(f"CPU convolution: {time.perf_counter()-begin}")

#si.imsave("test1.tif",res)
src=open("src-3darray-multiimg-working2dglobal size.cl", "r").read()
def platformslist():
    return [platform.name for platform in cl.get_platforms()]

def platformselect(ind):
    return [platform for platform in cl.get_platforms()][ind]

groupsize=8
items=16
dims=((img.shape[0]//groupsize+1)*groupsize,(img.shape[1]//groupsize+1)*groupsize)
print(platformslist())
platform = platformselect(0)
devices=platform.get_devices()

context=cl.Context(devices=devices)
prgs_src=cl.Program(context, src)
prgs=prgs_src.build()

img_buff=cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=img.nbytes)
res_buff=cl.Buffer(context, flags=cl.mem_flags.READ_WRITE, size=res.nbytes)
krn_buff=cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=krn.nbytes)
print(img.nbytes+res.nbytes+krn.nbytes)
queue=cl.CommandQueue(context)
begin=time.perf_counter()
inp=[(img,img_buff),(res,res_buff),(krn, krn_buff)]
out=[(res,res_buff)]
for (arr, buff) in inp:
    cl.enqueue_copy(queue, src=arr, dest=buff)
patchsz=32
krn_args=[img_buff, res_buff, krn_buff, np.int32(img.shape[1]), np.int32(img.shape[2]), np.int32(ksize), np.int32(16), np.int32(img.shape[0]), np.int32(patchsz)]
completedEvent=prgs.convolvecut(queue, (img.shape[1]//patchsz+1, img.shape[2]//patchsz+1), (1,1), *krn_args)
completedEvent.wait()
for (arr, buff) in out:
    cl.enqueue_copy(queue, src=buff, dest=arr)
queue.finish()
print(f"GPU convolution of 2 images:{time.perf_counter()-begin}")
print(res.shape)
for iimg in range(2):
    for i in range(16):
        """mn=res[iimg,:,:,i].min(axis=None)
        mx=res[iimg,:,:,i].max(axis=None)
        res[iimg,:,:,i]=res[iimg,:,:,i]-mn
        mn=res[iimg,:,:,i].min(axis=None)
        mx=res[iimg,:,:,i].max(axis=None)
        vpp=256/((mx-mn))#value per pixel
        for j in range(res[iimg,:,:,i].shape[0]):
            for k in range(res[iimg,:,:,i].shape[1]):
                try:
                    res[iimg,j,k,i]=round(res[iimg,j,k,i]*vpp)
                except:
                    print(iimg, i, j, k)"""
        si.imsave(f"vector-testimg-down30x-{iimg}-{i}-mod.tif", res[iimg,:,:,i])
        #print((img[iimg,:,:]).shape, (krn[:,:,i]).shape)
        #si.imsave(f"vector-testimg-down30x-{iimg}-{i}-mod-cmp.tif", np.convolve(img[iimg,:,:], krn[:,:,i]))
print(time.perf_counter()-begin)
