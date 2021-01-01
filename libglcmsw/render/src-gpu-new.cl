#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

void glcmgen(volatile __global float *glcm, __global uchar *img, int rownum, int colnum, int windowsz, int x_neighbour, int y_neighbour, int nrows, int ncols, int loc0, int loc1, int locsz0, int locsz1, int wid0, int wid1, __global float *sum, __local int *tmpsum, __local int *mutex)
{
    int xstart=rownum, xend=windowsz+rownum, ystart=colnum,yend=windowsz+colnum;
    if(x_neighbour<0)xstart +=-x_neighbour;
    else xend=rownum+windowsz-x_neighbour;
    if(y_neighbour<0)ystart+=-y_neighbour;
    else if(y_neighbour>=0)yend=colnum+windowsz-y_neighbour;
    int stride0=windowsz%locsz0?windowsz/locsz0+1:windowsz/locsz0;
    int stride1=windowsz%locsz1?windowsz/locsz1+1:windowsz/locsz1;
    int sumcols=ncols-windowsz/2*2;
    float tmpsum_private=0;
    for(int i=xstart+loc0*stride0;i<xstart+(loc0+1)*stride0;i++)
    {
        for(int j=ystart+loc1*stride1;j<ystart+(loc1+1)*stride1;j++)
        {
            if(i<xend && j<yend)
            {
                uchar ref=img[i*ncols+j];
                uchar val=img[(i+x_neighbour)*ncols+j+y_neighbour];
                while(atomic_cmpxchg(mutex, 0,1)==1);
                glcm[(wid0*get_num_groups(1)+wid1)*65536+ref*256+val]+=1;
                glcm[(wid0*get_num_groups(1)+wid1)*65536+val*256+ref]+=1;
                atomic_xchg(mutex, 0);
                tmpsum_private+=2;
            }
        }
    }
    atomic_add(tmpsum, tmpsum_private);
}

__kernel void swkrn_debug(__global float *glcm, __global uchar *img, __global float *sum,  int nrows, int ncols, int windowsz, int x_neighbour, int y_neighbour, int prop, int blocksz)
{
    int loc0=get_local_id(0), loc1=get_local_id(1), locsz0=get_local_size(0), locsz1=get_local_size(1);
    int wgs0=get_num_groups(0), wgs1=get_num_groups(1), wid0=get_group_id(0), wid1=get_group_id(1);
    int sumcols=ncols-windowsz/2*2, sumrows=nrows-windowsz/2*2;
    int imgstride0=blocksz/get_local_size(0);
    //if(wid0!=0 && wid1!=0)return;
    //printf("%d %d %d %d\n", wgs0, wgs1, wid0, wid1);
    __local int tmpsum, mutex;
    __local float accum;
    accum=0;
    int imgxstart=wid0*blocksz,imgxend =(wid0+1)*blocksz, imgystart=wid1*blocksz, imgyend=(wid1+1)*blocksz;
    int locstride=256/get_local_size(0);
    for(int i=imgxstart;i<imgxend;i++)
    {
        for(int j=imgystart;j<imgyend;j++)
        {
            if(i<sumrows && j<sumcols)
            {
                //sum[i*sumcols+j]=wid0*wid1;
                tmpsum=0;
                mutex=0;
                glcmgen(glcm, img, i,j,windowsz, x_neighbour, y_neighbour, nrows, ncols, loc0, loc1, locsz0, locsz1, wid0, wid1, sum, &tmpsum, &mutex);
                barrier(CLK_LOCAL_MEM_FENCE);
                //if(wid0==0 && wid1==0)printf("%d\n", tmpsum);
                //normalize
                for(int gi=loc0*locstride;gi<(loc0+1)*locstride;gi++)
                {
                    for(int gj=loc1*locstride;gj<(loc1+1)*locstride;gj++)
                    {
                        glcm[(wid0*get_num_groups(1)+wid1)*65536+gi*256+gj]/=tmpsum;
                        //while(atomic_cmpxchg(&mutex, 0,1)==1);
                        //accum+=glcm[(wid0*get_num_groups(1)+wid1)*65536+gi*256+gj];
                        //atomic_xchg(&mutex, 0);
                    }
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                //if(wid0==0 && wid1==0)printf("%f\n", accum);
                accum=0;
                float accum_private=0;
                //generate features
                if(prop==2)
                {
                    for(int gi=loc0*locstride;gi<(loc0+1)*locstride;gi++)
                    {
                        for(int gj=loc1*locstride;gj<(loc1+1)*locstride;gj++)
                        {
                            accum_private+=glcm[(wid0*get_num_groups(1)+wid1)*65536+gi*256+gj]/(1+(gi-gj)*(gi-gj));
                            glcm[(wid0*get_num_groups(1)+wid1)*65536+gi*256+gj]=0;
                        }
                    }
                    while(atomic_cmpxchg(&mutex, 0,1)==1);
                    sum[i*sumcols+j]+=accum_private;
                    atomic_xchg(&mutex, 0);
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
            }
        }
    }
}