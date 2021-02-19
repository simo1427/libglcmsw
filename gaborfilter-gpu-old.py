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

#img = img_as_ubyte(rgb2gray(si.imread("E:\Monkey\MkyISH_SVZi1\ABCD3 - Mky506 - SVZi - isch downsample 4x.png")))
#img = img_as_ubyte(rgb2gray(si.imread("E:\Monkey\MkyISH_SVZi1\ischdown8x.png")))
img = img_as_ubyte(rgb2gray(si.imread("..\examples\input.tif")))
kernel5 = []
ksize = 31
for theta in np.arange(0, np.pi, np.pi / 16):
    kernel = cv2.getGaborKernel((ksize, ksize), 15.42, theta, 32, 0.5, 0, ktype=cv2.CV_32F)
    kernel /= 1.5*kernel.sum()
    kernel5.append(kernel)

print(kernel5[0].shape)
krn=np.array(kernel5)
res=np.zeros((16,img.shape[0],img.shape[1]), dtype=np.float32)
print(krn.shape)
"""begin=time.perf_counter()
si.imsave("test.tif",ndi.convolve(img, krn, mode='wrap'))
print(time.perf_counter()-begin)"""

#si.imsave("test1.tif",res)
src=open("src-3darray-multiimg.cl", "r").read()
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
queue=cl.CommandQueue(context)
begin=time.perf_counter()
inp=[(img,img_buff),(res,res_buff),(krn, krn_buff)]
out=[(res,res_buff)]
for (arr, buff) in inp:
    cl.enqueue_copy(queue, src=arr, dest=buff)

krn_args=[img_buff, res_buff, krn_buff, np.int32(img.shape[0]),np.int32(img.shape[1]), np.int32(ksize), np.int32(16)]
prgs.convolvecut(queue, (img.shape[0], img.shape[1]), (1,1), *krn_args)
for (arr, buff) in out:
    cl.enqueue_copy(queue, src=buff, dest=arr)
queue.finish()
print(time.perf_counter()-begin)
for i in range(16):
    si.imsave(f"testimg-down30x-{i}-vector.tif", res[i])
"""for i in range(16):
    mn=res[i].min(axis=None)
    mx=res[i].max(axis=None)
    res[i]=res[i]-mn
    mn=res[i].min(axis=None)
    mx=res[i].max(axis=None)
    vpp=256/((mx-mn))#value per pixel
    for j in range(res[i].shape[0]):
        for k in range(res[i].shape[1]):
            res[i,j,k]=round(res[i,j,k]*vpp)
    print(i)
    si.imsave(f"testimg-down8x-{i}-mod.png", res[i].astype(np.uint8))"""
print(time.perf_counter()-begin)
