#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable

int glcmgen(volatile __global float *glcm, __global uchar *img, int rownum, int colnum, int windowsz, int x_neighbour, int y_neighbour, int nrows, int ncols, int loc0, int loc1, int locsz0, int locsz1, int wid1, __global float *sum)
{
    int xstart=rownum, xend=windowsz+rownum, ystart=colnum,yend=windowsz+colnum;
    if(x_neighbour<0)xstart +=-x_neighbour;
    else xend=rownum+windowsz-x_neighbour;
    if(y_neighbour<0)ystart+=-y_neighbour;
    else if(y_neighbour>=0)yend=colnum+windowsz-y_neighbour;
}

__kernel void swkrn_debug(__global float *glcm, __global uchar *img, __global float *sum,  int nrows, int ncols, int windowsz, int x_neighbour, int y_neighbour, int prop)
{
    int loc0=get_local_id(0), loc1=get_local_id(1), locsz0=get_local_size(0), locsz1=get_local_size(1);
    int wgs0=get_num_groups(0), wgs1=get_num_groups(1), wid0=get_group_id(0), wid1=get_group_id(1);
    int stride0=nrows%wgs0?nrows%wgs0+1:nrows%wgs0;
    int stride1=ncols%wgs1?ncols%wgs1+1:ncols%wgs1;
    for(int i=wid0*stride0;i<(wid0+1)*stride0;i++)
    {
        for(int j=wid1*stride1;j<(wid1+1)*stride1;j++)
        {
            if(i<nrows && j<ncols)
            {
                sum[i*ncols+j]=wid0*wid1;
            }
        }
    }
}