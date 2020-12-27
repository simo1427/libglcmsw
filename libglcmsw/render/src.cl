#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
//#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable

__kernel void sum(__global const float *a_g, __global const float *b_g, __global float *res_g)
{
  int gid0= get_global_id(0);
  int gid1= get_global_id(1);
  int gid2= get_global_id(2);
  int sz0 = get_global_size(0);
  int sz1 = get_global_size(1);
  int sz2 = get_global_size(2);
  int id=gid2+gid1*sz2+gid0*sz1*sz2;
  
  for(int i=0;i<3;i++)
  {
      for(int j=0;j<4;j++)
      {
          for(int k=0;k<5;k++)
          {
            id=i*4*5+j*5+k;
            printf("%d %d %d, %f\n", i,j,k, a_g[id]);
            res_g[id] = a_g[id] + b_g[id];
          }
      }
  }
  
}

int glcmgen(volatile __global float *glcm, __global uchar *img, int rownum, int colnum, int windowsz, int x_neighbour, int y_neighbour, int nrows, int ncols)
{
  int xstart=rownum, xend=windowsz+rownum, ystart=colnum,yend=windowsz+colnum;
  int gid=get_group_id(0);
  int wdim=get_local_size(0);
  int lid=get_local_id(0);
  if(x_neighbour<0)xstart +=-x_neighbour;
  else xend=rownum+windowsz-x_neighbour;
  if(y_neighbour<0)ystart+=-y_neighbour;
  else if(y_neighbour>=0)yend=colnum+windowsz-y_neighbour;
  int stridej=windowsz%wdim?windowsz/wdim+1:windowsz/wdim;
  int yrangestart=ystart+stridej*lid;
  int yrangeend=ystart+stridej*(lid+1);
  //printf("GLCMGEN ATTRIBUTES:%d,%d %d %d %d, %d %d %d %d\n", gid,xstart, xend, ystart, yend, yrangestart, yrangeend, stridej, wdim);
  float tmp1=0, tmp2=0;
  int tmpsum_private=0;
  for(int i=xstart;i<xend;i++)
  {
    for(int j=yrangestart;j<yrangeend;j++)
    {
      if(i>=xstart && i<xend && j>=ystart && j<yend)
      {
        uchar ref=img[i*ncols+j];
        uchar val=img[(i+x_neighbour)*ncols+j+y_neighbour];
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        //if(gid==45)printf("%d %d, %d %d\n", i,j,ref, val);
        unsigned int addr1=gid*65536+ref*256+val;
        unsigned int addr2=gid*65536+val*256+ref;
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        glcm[addr1]=glcm[addr1]+1;
        glcm[addr2]=glcm[addr2]+1;
        mem_fence(CLK_GLOBAL_MEM_FENCE);
        //mem_fence(CLK_LOCAL_MEM_FENCE);
        tmpsum_private+=2;
        //printf("%d %d, %f %f, %d\n", ref, val,glcm[addr1], glcm[addr2], *tmpsum);
      }
      else printf("%d %d. %d %d/%d %d,%d %d\n", i,j,xstart, xend, yrangestart, yrangeend, ystart, yend);
    }
  }
  //printf("%d\n", gid);
  //printf("Returned:%d\n",2*(xend-xstart)*(yend-ystart));
  return tmpsum_private;//2*(xend-xstart)*(yend-ystart);
}

void feature(__global float *glcm, int prop)
{
  int ldim=get_local_size(0);
  int lid=get_local_id(0);
  int gid=get_group_id(0);
  int stridej=256/ldim+256%ldim?1:0;
  if(prop==2)
  {
    for(int i=0;i<256;i++)
    {
      //for(int j=lid*stridej;j<(lid+1)*stridej;j++)
      for(int j=0;j<256;j++)
      {
        if(j<256)
        {
          glcm[gid*65536+i*256+j]=glcm[gid*65536+i*256+j]/(float)(1+((i-j)*(i-j)));
        }
      }
    }
  }
}

__kernel void swkrn_debug(__global float *glcm, __global uchar *img, __global float *sum,  int nrows, int ncols, int windowsz, int x_neighbour, int y_neighbour, int prop)
{
  int stridey=nrows%get_global_size(1)?nrows/get_global_size(1)+1:nrows/get_global_size(1);
  int colnum=get_group_id(0);
  int gid1=get_group_id(1);
  __local int tmpsum;
  //for(int rownum=gid1*stridey; rownum<(gid1+1)*stridey;rownum++)
  for(int rownum=0;rownum<nrows;rownum++)
  {
    //printf("%d %d %d\n", rownum, colnum, stridey);
    if(rownum<=nrows && colnum<=ncols)
    {
      tmpsum=glcmgen(glcm, img, rownum, colnum, windowsz,x_neighbour, y_neighbour, nrows+windowsz-windowsz%2, ncols+windowsz-windowsz%2);
      //printf("%d\n", tmpsum);
      float accum=0;
      //printf("%d", (tmpsum));
      for(int i=0;i<256;i++)
      {
        for(int j=0;j<256;j++)
        {
          glcm[colnum*65536+i*256+j]/=(float)(tmpsum);
          accum+=glcm[colnum*65536+i*256+j];
        }
      }
      //printf("%f\n",accum);
      feature(glcm, prop);
      int lid=get_local_id(0);
      int ldim=get_local_size(0);
      int stridej=256/ldim+256%ldim?1:0;
      accum=0;
      for(int i=0;i<256;i++)
      {
        //for(int j=lid*stridej;j<(lid+1)*stridej;j++)
        for(int j=0;j<256;j++)
        {
          if(j<256)
          {
            accum+=glcm[colnum*65536+i*256+j];
            //printf("%f", accum);
          }
        }
      }
      sum[rownum*ncols+colnum]=accum;
    }
  }
}