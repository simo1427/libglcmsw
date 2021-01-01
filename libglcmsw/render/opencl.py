import pyopencl as cl
import numpy as np
import math
from skimage.color import rgb2gray
from skimage.util import img_as_ubyte
from skimage.feature import greycomatrix, greycoprops
from skimage import io as si
import time

src=open("src-gpu-new.cl", "r").read()
def platformslist():
    return [platform.name for platform in cl.get_platforms()]

def platformselect(ind):
    return [platform for platform in cl.get_platforms()][ind]

print(platformslist())
platform = platformselect(0)
devices=platform.get_devices()

context=cl.Context(devices=devices)
prgs_src=cl.Program(context, src)
prgs=prgs_src.build(options="-cl-opt-disable")
print(f"Kernel Names: {prgs.get_info(cl.program_info.KERNEL_NAMES)}")

"""N=(5,5,4)
a=np.random.rand(100).astype(np.float32).reshape(N)
b=np.random.rand(100).astype(np.float32).reshape(N)
c=np.empty_like(a)
"""

windowsz=13
dist=1
angle=0
img = img_as_ubyte(rgb2gray(si.imread("../../examples/input.tif")))
res=np.zeros((img.shape[0]-windowsz//2*2,img.shape[1]-windowsz//2*2), dtype=np.float32)
x_neighbour = round(dist * np.cos(angle))
y_neighbour = round(dist * np.sin(angle))
prop=np.uint8(2)
workitems=(16,16)
blocksize=64
workgroups=(math.ceil(res.shape[0]/blocksize)*workitems[0],math.ceil(res.shape[1]/blocksize)*workitems[1])
print(workgroups[0]/workitems[0], workgroups[1]/workitems[1])
print(workgroups)
glcm=np.zeros(((workgroups[0]//workitems[0]*workgroups[1]//workitems[1]), 256, 256), dtype=np.float32)


img_buff=cl.Buffer(context, flags=cl.mem_flags.READ_ONLY, size=img.nbytes)
res_buff=cl.Buffer(context, flags=cl.mem_flags.READ_WRITE, size=res.nbytes)
glcm_buff=cl.Buffer(context, flags=cl.mem_flags.READ_WRITE, size=glcm.nbytes)

queue=cl.CommandQueue(context)
input=[(img,img_buff),(glcm,glcm_buff),(res,res_buff)]
output=[(glcm,glcm_buff),(res,res_buff)]
for (arr, buff) in input:
    cl.enqueue_copy(queue, src=arr, dest=buff)

#krn_args=[glcm_buff, img_buff, np.uint8(windowsz), np.int32(x_neighbour), np.int32(y_neighbour), res_buff, np.uint8(prop), np.int32(res.shape[0]), np.int32(res.shape[1])]
#prgs.swkrn(queue, (res.shape[1], 1), (1,1), *krn_args)
print(x_neighbour, y_neighbour)
krn_args=[glcm_buff, img_buff,res_buff, np.int32(img.shape[0]), np.int32(img.shape[1]), np.int32(windowsz), np.int32(x_neighbour), np.int32(y_neighbour), np.int32(prop), np.int32(blocksize)]
begin=time.perf_counter()
prgs.swkrn_debug(queue, workgroups, workitems, *krn_args)
for (arr, buff) in output:
    cl.enqueue_copy(queue, src=buff, dest=arr)
queue.finish()
print(time.perf_counter()-begin)
si.imsave(f"newgpu-threads-11.tif", res)
si.imsave(f"orig.tif", img)


assert True

"""def singletilegpusw(img, windowsz,prop, dist, angle, platform, bitdepth=256, threads=32, vertworkers=1):
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
    sum=np.zeros((img.shape[0]-windowsz//2*2,img.shape[1]-windowsz//2*2), dtype=np.float32)
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
    sum=np.zeros((img.shape[0]-windowsz//2*2,img.shape[1]-windowsz//2*2), dtype=np.float32)
    sum_dev=cuda.to_device(sum)
    mask_dev = cuda.to_device(mask)
    swkrnmask[blockspergrid, threadsperblock](glcm_dev, img_dev,mask_dev, windowsz, x_neighbour, y_neighbour, sum_dev, prop)
    sum=sum_dev.copy_to_host()"
    return sum"""