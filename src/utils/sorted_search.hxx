#ifndef __SORTED_SEARCH_CUH_
#define __SORTED_SEARCH_CUH_


#include <hip/hip_runtime.h>
#include "utils/utils.hxx"
#include "utils/platform.hxx"

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

  const int gtid = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;

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
template<typename data_t>
__device__ __tbdinline__ 
int merge_step ( data_t* dg_a, int a_count, 
                  data_t* dg_b, int b_count, 
                  int diag){
  
  int begin = MAX(0, diag - b_count);
  int end   = MIN(diag, a_count);

  while(begin < end) {
    int mid = (begin + end) >> 1;
    data_t a_key = dg_a[mid];
    data_t b_key = dg_b[diag - 1 - mid];
    
    if(a_key <= b_key) begin = mid + 1;
    else end = mid;
  }
  return begin;
}

/**
 * gladly copied from mgpu!
 */
template<typename data_t, int THD_NUM, int THD_WORK>
__global__ void 
partition ( data_t* dg_a, int a_count, 
            data_t* dg_b, int b_count, 
            int tile_count, int* dg_aout){
  const int gtid       = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
  const int tile_size  = THD_NUM*THD_WORK;
  const int diag       = MIN(gtid * tile_size, a_count + b_count);

  if(gtid < tile_count) {
    dg_aout[gtid] = merge_step(dg_a, a_count, dg_b, b_count, diag);
  }
  dg_aout[tile_count] = a_count;
}

template<typename data_t, int THD_NUM, int THD_WORK>
__global__ 
void block_sort ( data_t* dg_a, int a_count, 
                  data_t* dg_b, int b_count,
                  int* a_offset, int* dg_idx, int* dg_tmpidx){
  const int tid = hipThreadIdx_x;
  const int bid = hipBlockIdx_x;

  __shared__ int ibuffer[THD_NUM*(THD_WORK+1)];
  __shared__ int obuffer[THD_NUM*THD_WORK];
  //__shared__ int b_aoffset;

  int diag_s = bid*THD_NUM*THD_WORK;
  int diag_e = MIN(a_count+b_count, diag_s+THD_NUM*THD_WORK);
  int b_as = a_offset[bid];
  int b_ae = a_offset[bid+1];
  int b_bs = diag_s-b_as;
  int b_be = diag_e-b_ae;
  //calc b start end
  int b_acnt = b_ae - b_as;
  int b_bcnt = b_be - b_bs;
  if(b_acnt==0) return;
  //load ab into shared mem
  //if(tid==0) printf("bid:%d load ab into shared mem--start -- as:%d, ae:%d, bs:%d, be:%d\n",bid,b_as,b_ae,b_bs,b_be);
  {
    //load into register
    int l_buffer[THD_WORK];
    unroller_t<THD_WORK>::iterate([&](int cycle){
      int sidx  = cycle*THD_NUM + tid;
      data_t *l_dga = dg_a+b_as;
      data_t *l_dgb = dg_b+b_bs;
      l_dgb -= b_acnt+1;
      if(sidx < b_acnt+b_bcnt)
        if(sidx < b_acnt)      l_buffer[cycle] = l_dga[sidx];
    else if(sidx > b_acnt) l_buffer[cycle] = l_dgb[sidx];
    else                   l_buffer[cycle] = MAX_32S;
        //ibuffer[sidx] = (sidx < b_acnt) ? l_dga[sidx]:l_dgb[sidx]; //try this later
    });
  
  //__syncthreads();
  //if(tid==0) printf("bid:%d load ab into shared mem--end\n", bid);
  //__syncthreads(); 
    
  //save into shared (could be fused! @line126)
    unroller_t<THD_WORK>::iterate([&](int cycle){
      int sidx = cycle*THD_NUM + tid;
      ibuffer[sidx] = l_buffer[cycle];
  });
  }

  //each thread partition on its own
  //__syncthreads();
  //if(tid==0) printf("bid:%d each thread partition--start\n",bid);
  //__syncthreads();

  int ta_idx[THD_WORK]; //thread-local index buffer
  {
    int diag_ts = THD_WORK*tid;
  int diag_te = MIN(diag_ts+THD_WORK, b_acnt+b_bcnt);
    int t_as = merge_step(&ibuffer[0], b_acnt, &ibuffer[0]+b_acnt+1, b_bcnt, diag_ts);
    int t_ae = merge_step(&ibuffer[0], b_acnt, &ibuffer[0]+b_acnt+1, b_bcnt, diag_te);
    int t_bs = diag_ts - t_as;
    int t_be = diag_te - t_ae;
    int t_acnt = t_ae - t_as;
    int t_bcnt = t_be - t_bs;
    int* t_a_in = &ibuffer[0]+t_as;
    int* t_b_in = &ibuffer[0]+b_acnt+t_bs+1;
    //do a binary+linear search of t_a_in into t_b_in
    
  //__syncthreads();  
  //if(tid == 0) printf("bid:%d do a binary+linear--start\n", bid);
  //__syncthreads();
  //printf("[%d,%d] __upper_eq_bound(): t_a_in[0]:%d, diag_ts:%d, diag_te:%d, as:%d, ae:%d, bs:%d, be:%d\n", 
  //    bid, tid, t_a_in[0], diag_ts,diag_te,t_as,t_ae,t_bs,t_be);
  ta_idx[0] = __upper_eq_bound(t_b_in, t_bcnt, t_a_in[0]);
    //printf("[%d,%d] __upper_eq_bound() --fin\n", bid, tid);

  //__syncthreads();
  //if(tid == 0) printf("bid:%d upper bound --passed\n",bid);
  //__syncthreads();

  int ta_cursor=1;
    for(int i=ta_idx[0]; i<t_bcnt&&ta_cursor<t_acnt; i++){
    while(t_b_in[i]>=t_a_in[ta_cursor]){ //insert a after b when equal
        ta_idx[ta_cursor] = i;
        ta_cursor++;
      }
    }
  
  //__syncthreads();
  //if(tid==0) printf("bid: %d linear search --passed\n", bid);
    //__syncthreads();

  while(ta_cursor<t_acnt){
      ta_idx[ta_cursor] = ta_cursor+t_bcnt;
      ta_cursor++;
    }

  //__syncthreads();
  //if(tid==0) printf("bid:%d save to shared --passed\n", bid);
  //__syncthreads();

    for(int i=0; i<t_acnt; i++){
      obuffer[t_as+i] = t_bs+ta_idx[i];
    }
  __syncthreads();
  }
  //if(bid==0 && tid==0) printf("each thread partition--end\n");

  //convert shared index into global index and save into global mem
  for(int i=tid; i<b_acnt; i+=THD_NUM){
    dg_idx[b_as+i] = b_bs+obuffer[i];
  }
  __syncthreads();
}


template<typename data_t, int TILE_SZ>
__global__ void block_upper(data_t* dg_a, int a_count, data_t* dg_b, int b_count,
                            int* a_offset, int* b_offset, int* dg_idx){

  //const int gtid = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
  const int tid  = hipThreadIdx_x;
  const int Loc_STRIDE = hipBlockDim_x;
  const int bid  = hipBlockIdx_x;

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
template<typename data_t, int THD_NUM = 128, int THD_WORK=16>
void sorted_search(data_t* dg_a, int a_count, 
                     data_t* dg_b, int b_count,
           int* dg_idx){

#define TILE_SZ (THD_NUM*THD_WORK)
  auto partition_kernel = partition<data_t, THD_NUM, THD_WORK>;
  auto block_upper_kernel = block_sort<data_t, THD_NUM, THD_WORK>;
  
  int tile_count = CEIL(a_count + b_count, TILE_SZ) + 1;
  //int tile_count = CEIL( b_count, TILE_SZ) + 1;
  data_t *dg_atile, *dg_btile, *dg_tmpidx;
  H_ERR(hipMalloc((void**)&dg_atile, sizeof(data_t)*tile_count));
  //H_ERR(hipMalloc((void**)&dg_btile, sizeof(data_t)*tile_count));
  H_ERR(hipMalloc(&dg_tmpidx, sizeof(int)));
  hipLaunchKernelGGL(partition_kernel, dim3(1 + CEIL(tile_count, THD_NUM)), dim3(THD_NUM), 0, 0, dg_a, a_count, dg_b, b_count, tile_count, dg_atile);
//  dump_arr(dg_atile, tile_count);
//  dump_arr(dg_btile, tile_count);
  hipLaunchKernelGGL(block_upper_kernel, dim3(tile_count-1), dim3(THD_NUM), 0, 0, dg_a, a_count, dg_b, b_count, dg_atile, dg_idx, dg_tmpidx);
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