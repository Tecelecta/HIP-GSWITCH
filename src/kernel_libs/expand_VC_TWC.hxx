#ifndef __expand_VC_TWC_CUH
#define __expand_VC_TWC_CUH


#include <hip/hip_runtime.h>
#include "utils/utils.hxx"
#include "utils/intrinsics.hxx"
#include "data_structures/graph.hxx"
#include "data_structures/active_set.hxx"
#include "data_structures/functor.hxx"
#include "abstraction/config.hxx"

// This optimization variant has been move to expand_VC_TM.hxx

//__global__ void 
//__expand_TWC(active_set_t as, graph_t g, int lvl, int mode){
//  const index_t* __restrict__ strict_adj_list = g.dg_adj_list; 
//  
//  int STRIDE,gtid,phase,cosize,qsize;
//  if(mode==0){
//    STRIDE = hipBlockDim_x*hipGridDim_x;
//    gtid   = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
//    cosize = 1;
//    phase  = 0;
//    qsize  = as.small.get_qsize();
//  }else if(mode==1){
//    STRIDE = (hipBlockDim_x*hipGridDim_x)>>5;
//    gtid   = (hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x)>>5;
//    cosize = 32;
//    phase  = (hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x) & (cosize-1);
//    qsize  = as.medium.get_qsize();
//  }else{
//    STRIDE = hipGridDim_x;
//    gtid   = hipBlockIdx_x;
//    cosize = hipBlockDim_x;
//    phase  = hipThreadIdx_x;
//    qsize  = as.large.get_qsize();
//  }
//
//
//  for(int idx=gtid; idx<qsize; idx+=STRIDE){
//    int v;
//    if(mode==0) v = tex1Dfetch<int>(as.small.dt_queue, idx);
//    else if(mode==1)  v = tex1Dfetch<int>(as.medium.dt_queue, idx);
//    else v = tex1Dfetch<int>(as.large.dt_queue, idx);
//    int end = g.get_out_degree(v);
//    int start = g.get_out_start_pos(v);
//    end += start;
//
//    for(int i=start+phase; i<end; i+=cosize){
//      int u = __ldg(strict_adj_list+i);
//      int u_s = as.bitmap.get_state(u);
//      if(u_s==-1) as.bitmap.mark(u, lvl);
//    }
//    if(mode==1) while(!__all(1));
//    else if(mode==2) __syncthreads();
//  }
//}
//
//__host__ void expand(active_set_t as, graph_t g, int lvl){
//  for(int i=0;i<3;++i){
//    __vexpand<<<CTANUM,THDNUM,0,as.streams[i]>>>(as,g,lvl,i);
//  }
//  for(int i=0;i<3;i++) cudaStreamSynchronize(as.streams[i]);
//}


#endif
