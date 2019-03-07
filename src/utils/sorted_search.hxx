#ifndef __SORTED_SEARCH_CUH_
#define __SORTED_SEARCH_CUH_


#include <hip/hip_runtime.h>
#include "utils/utils.hxx"

template<typename data_t>
__device__ __tbdinline__ int
__upper_eq_bound(data_t* array, int len, data_t key){
  int s = 0;
  while(len>0){
    int half = len>>1;
    int mid = s + half;
    if(array[mid] >= key){
      len = half;
    }else{
      s = mid+1;
      len = len-half-1;
    }
  }
  return s;
}


template<typename data_t>
__global__ void global_upper(data_t *dg_a, int a_count,
                      data_t *dg_b, int b_count,
              int* dg_idx){

  const int gtid = threadIdx.x + blockIdx.x*blockDim.x;

  data_t a_item;
  int a_res;
  if(gtid < a_count) a_item = dg_a[gtid];
  else return;
  a_res = __upper_eq_bound(dg_b, b_count, a_item);
  __syncthreads();
  dg_idx[gtid] = a_res;
}

/**
 * gladly copied from mgpu!
 */
template<typename data_t, int TILE_SZ>
__global__ void partition(data_t* dg_a, int a_count, 
                      data_t* dg_b, int b_count, 
              int tile_count,
              int* dg_aout, int* dg_bout){

  const int gtid = threadIdx.x + blockIdx.x*blockDim.x;
  int diag  = MIN(gtid * TILE_SZ, a_count + b_count);

  int begin = MAX(0, diag - b_count);
  int end   = MIN(diag, a_count);

  while(begin < end) {
    int mid = (begin + end) / 2;
    data_t a_key = dg_a[mid];
    data_t b_key = dg_b[diag - 1 - mid];
    
  if(a_key <= b_key) begin = mid + 1;
    else end = mid;
  }
  
  if(gtid < tile_count) {
    dg_aout[gtid] = begin;
    dg_bout[gtid] = diag - begin;
  }
}

template<typename data_t, int TILE_SZ>
__global__ void block_upper(data_t* dg_a, int a_count, data_t* dg_b, int b_count,
                            int* a_offset, int* b_offset, int* dg_idx){

  //const int gtid = threadIdx.x + blockIdx.x*blockDim.x;
  const int tid  = threadIdx.x;
  const int Loc_STRIDE = blockDim.x;
  const int bid  = blockIdx.x;

  __shared__ data_t s_b[TILE_SZ];
  __shared__ int s_idx[TILE_SZ];

  //#0 check if there is work to do
  int local_boffset = b_offset[bid];
  int local_bsize   = b_offset[bid+1] - local_boffset;
  int local_aoffset = a_offset[bid];
  int local_asize   = a_offset[bid+1] - local_aoffset;
  
  if(local_asize == 0) return;

  //#1 load g mem into shared 
  for(int i=tid; i<local_bsize; i+=Loc_STRIDE){
    s_b[i] = dg_b[local_boffset+i];
  }
  __syncthreads();
  
  //#2 every thread do upper and write to shared
  for(int i=tid; i<local_asize; i+=Loc_STRIDE){
    data_t a_item = dg_a[local_aoffset+i];
    s_idx[i] = __upper_eq_bound(s_b, local_bsize, a_item);
  }
  
  //#3 put shared into mem
  for(int i=tid; i<local_asize; i+=Loc_STRIDE){
  dg_idx[i+local_aoffset] = s_idx[i] + local_boffset;
  }
}

template<typename data_t>
static void dump_arr(data_t *dg_arr, int len){
  data_t *h_arr = new data_t[len];
  TOHOST(dg_arr, h_arr, len);
  for(int i=0; i<len; i++) std::cout << h_arr[i] << "\t";
  std::cout << std::endl;
  delete [] h_arr;
}

/**
 * happily modified from mgpu! 
 */
template<typename data_t, int THD_NUM = 256>
void sorted_search(data_t* dg_a, int a_count, 
                     data_t* dg_b, int b_count,
           int* dg_idx){

#define TILE_SZ (THD_NUM<<1)
  int tile_count = CEIL(a_count + b_count, TILE_SZ) + 1;
  //int tile_count = CEIL( b_count, TILE_SZ) + 1;
  data_t *dg_atile, *dg_btile;
  H_ERR(hipMalloc((void**)&dg_atile, sizeof(data_t)*tile_count));
  H_ERR(hipMalloc((void**)&dg_btile, sizeof(data_t)*tile_count));
  hipLaunchKernelGGL(partition<data_t, TILE_SZ>, dim3(1 + CEIL(tile_count, THD_NUM)), dim3(THD_NUM), 0, 0, dg_a, a_count, dg_b, b_count, tile_count, dg_atile, dg_btile);
  //partition1<data_t, TILE_SZ><<<1 + CEIL(tile_count, THD_NUM) ,THD_NUM>>>(dg_a, a_count, dg_b, b_count, tile_count, dg_atile, dg_btile);
//  dump_arr(dg_atile, tile_count);
//  dump_arr(dg_btile, tile_count);
  hipLaunchKernelGGL(block_upper<data_t, TILE_SZ>, dim3(tile_count-1), dim3(THD_NUM), 0, 0, dg_a, a_count, dg_b, b_count, dg_atile, dg_btile, dg_idx);
#undef TILE_SZ
}


template<typename data_t, int THD_NUM = 256>
void sorted_search1(data_t* dg_a, int a_count, 
                   data_t* dg_b, int b_count,
                    int* dg_idx){

  int tile_count = CEIL(a_count, THD_NUM);
  hipLaunchKernelGGL(global_upper, dim3(tile_count), dim3(THD_NUM), 0, 0, dg_a, a_count, dg_b, b_count, dg_idx);
}

#endif //__SORTED_SEARCH_CUH_
