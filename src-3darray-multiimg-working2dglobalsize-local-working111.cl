#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
#define patchsz 100

__kernel void convolvecut(__global uchar *img, __global float16 *res, __constant float16 *krn, int sz0, int sz1, int krnsz, int numkrn, int numimg)
{
    int wid0=get_group_id(0), wid1=get_group_id(1);
    int lid0=get_local_id(0), lid1=get_local_id(1);
    int gid0=get_global_id(0), gid1=get_global_id(1);
    float16 tmp,sum;
    __local float16 locsum;
    //if(gid0==0 && gid1==0 && get_global_id(2)==0)printf("%d %d %d: %d %d %d\n", get_global_size(0), get_global_size(1), get_global_size(2), get_local_size(0), get_local_size(1), get_local_size(2));
    //printf("%d %d %d\n",gid0, gid1, get_global_id(2));
    int hfs=krnsz/2;
    int addr0, addr1;
    __local uchar cached[patchsz*patchsz];
    //if(gid0==0 && gid1==0)printf("%v16f\n",krn[30*31+30]);
    //for(int iimg=0;iimg<numimg;iimg++)
    //{
        int iimg=get_group_id(2);
        //printf("iimg: %d\n", iimg);
        for(int imaddr0=0;imaddr0<patchsz;imaddr0++)
        {
            for(int imaddr1=0;imaddr1<patchsz;imaddr1++)
            {
                addr0=gid0*(patchsz-krnsz)-hfs+imaddr0;
                if(addr0<0)addr0+=sz0;
                else if(addr0>sz0)addr0-=sz0;
                addr1=gid1*(patchsz-krnsz)-hfs+imaddr1;
                if(addr1<0)addr1+=sz1;
                else if(addr1>sz1)addr1-=sz1;
                //if((addr0)<sz0&&(addr1)<sz1)
                //mem_fence(CLK_GLOBAL_MEM_FENCE)
                cached[imaddr0*patchsz+imaddr1]=img[iimg*sz0*sz1+addr0*sz1+addr1];

            }
        }

        //barrier(CLK_GLOBAL_MEM_FENCE);
        for(int imaddr0=hfs;imaddr0<patchsz-hfs;imaddr0++)
        {
            for(int imaddr1=hfs;imaddr1<patchsz-hfs;imaddr1++)
            {
                if((imaddr0+gid0*(patchsz-krnsz)-hfs)<sz0&&(imaddr1+gid1*(patchsz-krnsz)-hfs)<sz1)
                {
                    sum=(float16)(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);
                    for(int i=-hfs;i<hfs; i++)
                    {
                        for(int j=-hfs;j<hfs; j++)
                        {
                            addr0=imaddr0+i;
                            addr1=imaddr1+j;
                            tmp=cached[addr0*patchsz+addr1]*krn[(i+hfs)*krnsz+j+hfs];
                            //res[iimg*sz0*sz1+(imaddr0+gid0*(patchsz-krnsz)-hfs)*sz1+(imaddr1+gid1*(patchsz-krnsz)-hfs)]=cached[imaddr0*patchsz+imaddr1];//(float16)(cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1],cached[addr0*patchsz+addr1]);

                            //barrier(CLK_LOCAL_MEM_FENCE);
                            sum=sum+tmp;
                            //barrier(CLK_LOCAL_MEM_FENCE);
                        }
                    }
                    //mem_fence(CLK_GLOBAL_MEM_FENCE);
                    res[iimg*sz0*sz1+(imaddr0+gid0*(patchsz-krnsz)-hfs)*sz1+(imaddr1+gid1*(patchsz-krnsz)-hfs)]=sum;
                }
            }
        }
    //}
        
}