#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

__kernel void convolvecut(__global uchar *img, __global float16 *res, __constant float16 *krn, int sz0, int sz1, int krnsz, int numkrn, int numimg, int patchsz)
{
    int wid0=get_group_id(0), wid1=get_group_id(1);
    int lid0=get_local_id(0), lid1=get_local_id(1);
    int gid0=get_global_id(0), gid1=get_global_id(1);
    float16 sum,tmp;
    int hfs=krnsz/2;
    int addr0, addr1;
    //for(int iimg=0;iimg<numimg;iimg++)
    //{
        int iimg=get_global_id(2);
        for(int imaddr0=gid0*patchsz;imaddr0<(gid0+1)*patchsz;imaddr0++)
        {
            for(int imaddr1=gid1*patchsz;imaddr1<(gid1+1)*patchsz;imaddr1++)
            {
                if(imaddr0<sz0 && imaddr1<sz1)
                {
                    sum=(float16)(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
                    for(int i=-hfs;i<hfs; i++)
                    {
                        for(int j=-hfs;j<hfs; j++)
                        {
                            addr0=i<0?imaddr0+i+sz0:imaddr0+i;
                            if(addr0>sz0)addr0-=sz0;
                            addr1=j<0?imaddr1+i+sz1:imaddr1+j;
                            if(addr1>sz1)addr1-=sz1;
                            tmp=img[iimg*sz0*sz1+addr0*sz1+addr1]*krn[(i+hfs)*krnsz+j+hfs];
                            sum+=tmp;
                            if(gid0==0 && gid1==0&&i==-hfs&&j==-hfs)
                            {
                                ;//printf("%d %d ", i,j);
                                //printf("%v16f\n", krn[(i+hfs)*krnsz+j+hfs]);
                            }
                        }
                    }
                    res[iimg*sz1*sz0+imaddr0*sz1+imaddr1]=sum;
                }
            }
        }
    //}
    
}