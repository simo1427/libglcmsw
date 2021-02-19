#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

__kernel void convolvecut(__global uchar *img, __global float *res, __constant float *krn, int sz0, int sz1, int krnsz, int groupsz)
{
    int wid0=get_group_id(0), wid1=get_group_id(1);
    int lid0=get_local_id(0), lid1=get_local_id(1);
    int gid0=get_global_id(0), gid1=get_global_id(1);
    float sum=0;
    int hfs=krnsz/2;
    int addr0, addr1;
    for(int i=-hfs;i<hfs; i++)
    {
        for(int j=-hfs;j<hfs; j++)
        {
            addr0=i<0?gid0+i+sz0:gid0+i;
            if(addr0>sz0)addr0-=sz0;
            addr1=j<0?gid1+i+sz1:gid1+j;
            if(addr1>sz1)addr1-=sz1;
            sum+=img[addr0*sz1+addr1]*krn[i*krnsz+j];
        }
    }
    res[gid0*sz1+gid1]=sum;
}