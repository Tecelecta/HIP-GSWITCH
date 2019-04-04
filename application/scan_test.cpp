#include <cstdio>
#include "utils/intrinsics.hxx"
#include "utils/utils.hxx"
#include "utils/sorted_search.hxx"
#include <hip/hip_runtime.h>

__global__ void cl_test(int* data){
  const int tid = hipThreadIdx_x;
  
  int sum;
  int thread_in = data[tid];
  int &thread_out = data[tid];
  warpScan(thread_in,thread_out,sum);
}

__global__ void arr_test(int* data){
  //__shared__ int tmp[64];
  int * tmp = data;
  const int tid = hipThreadIdx_x;
  const int phase = tid; 
  //tmp[tid] = data[tid];
  
  int total_warp=0;
  int offset=1;
  //for(int d=WARP_SIZE>>1; d>0; d>>=1){
  unroller_t<LANE_SHFT>::iterate([&](int cycle){  
    if(phase<(WARP_SIZE>>(cycle+1))){
      int ai = offset*(2*phase+1)-1;
      int bi = offset*(2*phase+2)-1;
      tmp[bi] += tmp[ai];
    }
    offset<<=1;
  });

  //data[tid] = tmp[tid];
  //return;

  total_warp = tmp[WARP_SIZE-1];
  if(!phase) tmp[WARP_SIZE-1]=0;

  //for(int d=1; d<WARP_SIZE; d<<=1){
  unroller_t<LANE_SHFT>::iterate([&](int cycle){
    offset >>=1;
    if(phase<(1<<cycle)){
      int ai = offset*(2*phase+1)-1;
      int bi = offset*(2*phase+2)-1;

      int t = tmp[ai];
      tmp[ai]  = tmp[bi];
      tmp[bi] += t;
    }
  });
  //data[tid] = tmp[tid];
}

int main(){
  int *h_data = new int[64];
  int *d_data1, *d_data2;

  for(int i=0; i<64; i++){
    h_data[i] = 1;
    //h_data[i+32] = i;
  }
  hipMalloc(&d_data1, sizeof(int)*64);
  hipMalloc(&d_data2, sizeof(int)*64);
  TODEV(d_data1, h_data, 64);
  TODEV(d_data2, h_data, 64);

  hipLaunchKernelGGL(cl_test, dim3(1), dim3(64), 0, 0, d_data1);
  hipLaunchKernelGGL(arr_test, dim3(1), dim3(64), 0, 0, d_data2);

  dump_arr(d_data1, 64);
  dump_arr(d_data2, 64);

  hipFree(d_data1);
  hipFree(d_data2);
  delete [] h_data;
}

