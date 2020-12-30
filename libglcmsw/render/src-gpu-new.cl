#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

void glcmgen(volatile __global float *glcm, __global uchar *img, int rownum, int colnum, int windowsz, int x_neighbour, int y_neighbour, int nrows, int ncols, int loc0, int loc1, int locsz0, int locsz1, int wid1, __global float *sum, __local int *tmpsum)
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
                sum[i*sumcols+j]=img[i*ncols+j];
            }
        }
    }
}

__kernel void swkrn_debug(__global float *glcm, __global uchar *img, __global float *sum,  int nrows, int ncols, int windowsz, int x_neighbour, int y_neighbour, int prop)
{
    int loc0=get_local_id(0), loc1=get_local_id(1), locsz0=get_local_size(0), locsz1=get_local_size(1);
    int wgs0=get_num_groups(0), wgs1=get_num_groups(1), wid0=get_group_id(0), wid1=get_group_id(1);
    int sumcols=ncols-windowsz/2*2;
    //printf("%d %d %d %d\n", wgs0, wgs1, wid0, wid1);
    __local int tmpsum;
    for(int i=wid0*locsz0;i<(wid0+1)*locsz0;i++)
    {
        for(int j=wid1*locsz1;j<(wid1+1)*locsz1;j++)
        {
            if(i<nrows && j<ncols)
            {
                //sum[i*ncols+j]=img[i*(ncols+windowsz/2+windowsz%2)+j];
                //printf("%d %d\n", sumcols)
                //sum[i*sumcols+j]=img[i*ncols+j];
                tmpsum=0;
                glcmgen(glcm, img, i,j, windowsz, x_neighbour, y_neighbour, nrows, ncols, loc0, loc1, locsz0, locsz1, wid1, sum, &tmpsum);
            }
        }
    }
}