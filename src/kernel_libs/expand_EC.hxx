#ifndef __EXPNAD_EC_CUH
#define __EXPNAD_EC_CUH


#include <hip/hip_runtime.h>
#include "utils/utils.hxx"
#include "utils/intrinsics.hxx"
#include "kernel_libs/kernel_fusion.hxx"
#include "data_structures/graph.hxx"
#include "data_structures/active_set.hxx"
#include "data_structures/functor.hxx"

#include "utils/tempkernel.h"

template<typename G, typename F>
__global__ void
__expand_EC(active_set_t as, G g, F f){
  int STRIDE = hipBlockDim_x*hipGridDim_x;
  int gtid   = hipThreadIdx_x + hipBlockIdx_x*hipBlockDim_x;
  int assize = as.get_size();

  for(int idx=gtid; idx<assize; idx+=STRIDE){
    int e = as.fetch(idx, Active);
    if(e<0) continue; // not Active;

    packed_t vv = g.dg_coo[e];
    int v0 = vv>>32;
    int v1 = vv&((1ll<<32)-1);

    // TODO: monotonous w/o atomic
    f.update(v0, v1, NULL);
  }
}

template<LB S, Direction D>
struct ExpandProxy<EC,S,D>{
  template<typename E, typename F>
  static void expand(active_set_t as, device_graph_t<COO,E> g, F f, config_t conf){
    hipLaunchKernelGGL(__expand_EC, dim3(CTANUM), dim3(THDNUM), 0, 0, as, g, f);
  }

  // why I have to write this ??? C++ (or I) is realy a ps of shit
  template<typename E, typename F>
  static void expand(active_set_t as, device_graph_t<CSR,E> g, F f, config_t conf){}
};

#endif
