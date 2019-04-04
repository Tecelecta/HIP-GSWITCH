#ifndef _PLATFORM_H__
#define _PLATFORM_H__
#include <hip/hip_runtime.h>
#include <assert.h>
/**
 * this file specifies key variables differ from platforms
 */
//adapt for variant warp size --lmy
#ifndef LANE_SHFT
#define LANE_SHFT 6
#endif 
#define WARP_SIZE (1<<LANE_SHFT)
#define LANE_MASK (WARP_SIZE-1)

#if (LANE_SHFT == 5)
typedef uint32_t ballot_t;
#elif (LANE_SHFT == 6)
typedef uint64_t ballot_t;
#else
#error no valid platform specified!
#endif

void check_warpsize(hipDeviceProp_t &prop){
  // this one checks the platform info against compiler configs --lmy
  int warp_sz = prop.warpSize;
  int warp_shft = 0;
  int tmp = warp_sz-1;
  for(; tmp; ++warp_shft) tmp&=(tmp-1);
  assert(warp_sz == WARP_SIZE && warp_shft == LANE_SHFT && "Warp Size doesn't match current platform!");
}

/*this unfolder struct generates different warp scans
template<typename T, unsigned step>
struct __warpScanUnfolder{
  static __device__ __forceinline__ 
  void warp_upsweep(const int lane_id, T& lane_recv, T& lane_local){
    __warpScanUnfolder<T,(step>>1)>::warp_upsweep(lane_id, lane_recv, lane_local);
    if ((lane_id & (step-1)) == 0){
      lane_local += lane_recv;
      lane_recv = __shfl_xor(lane_local, step);
    }
  }
  static __device__ __forceinline__ 
  void warp_downsweep(const int lane_id, T& lane_recv, T& lane_local){
    lane_recv = __shfl_up(lane_local, (step>>1));
    if ((lane_id & step-1) == (step>>1))
      lane_local += lane_recv;
    __warpScanUnfolder<T,(step>>1)>::warp_downsweep(lane_id, lane_recv, lane_local);
  }
};


template<typename T>
struct __warpScanUnfolder<T,1>{
  static __device__ __forceinline__ 
  void warp_upsweep(const int lane_id, T& lane_recv, T& lane_local){
    lane_recv = __shfl_xor(lane_local, 1);
  }
  static __device__ __forceinline__ 
  void warp_downsweep(const int lane_id, T& lane_recv, T& lane_local){}
};
*/
template<int step>
struct unroller_t{
  template<typename func_t> static __device__ __forceinline__
  void iterate(func_t stat){
    unroller_t<step-1>::iterate(stat);
    stat(step-1);
  }
};

template<>
struct unroller_t<0>{
  template<typename func_t> static __device__ __forceinline__ 
  void iterate(func_t stat){}
};

// this constant is reserved due to the current hip bug!
#define DEFUALT_REG_LIM 128


// the following functions acquires tex size, and tries to use dynamic numbered texture objects
hipError_t store_texture_max_dim(int devid);
#ifdef __USE_TEXTURE__
int __tex_sz;
__constant__ unsigned int __d_tex_sz;

#define MAXTEX ((const int)__tex_sz)
#define DMAXTEX __d_tex_sz

#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>
hipError_t store_texture_max_dim(int devid){
  cudaDeviceProp cuda_prop;
  cudaError_t err = cudaGetDeviceProperties(&cuda_prop, devid);
  if(err != cudaSuccess){
  	return hipCUDAErrorTohipError(err);
  }
  __tex_sz = cuda_prop.maxTexture1D;
  err = hipMemcpyToSymbol(&__d_tex_sz, __tex_sz, sizeof(unsigned int));
  return err;
}
#endif

#ifdef __HIP_PLATFORM_HCC__
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
//solution 1
// copied from hip_hcc_internal.h
class ihipDevice_t {
public:
  ihipDevice_t(unsigned deviceId, unsigned deviceCnt, hc::accelerator& acc);
  ~ihipDevice_t();
  ihipCtx_t* getPrimaryCtx() const { return _primaryCtx; };
  void locked_removeContext(ihipCtx_t* c);
  void locked_reset();
  ihipDeviceCritical_t& criticalData() { return _criticalData; };
public:
  unsigned _deviceId;  // device ID
  hc::accelerator _acc;
  hsa_agent_t _hsaAgent;  // hsa agent handle
  unsigned _computeUnits;
  hipDeviceProp_t _props;  // saved device properties.
  int _isLargeBar;
  ihipCtx_t* _primaryCtx;
  int _state;  // 1 if device is set otherwise 0
private:
  hipError_t initProperties(hipDeviceProp_t* prop);
private:
  ihipDeviceCritical_t _criticalData;
};
extern ihipDevice_t* ihipGetDevice(int);

hipError_t store_texture_max_dim(int devid){
  size_t res_buffer;
  hipError_t err;
  hsa_agent_t* agent = &(ihipGetDevice(devid)->_hsaAgent);
  hsa_status_t status = hsa_amd_image_get_info_max_dim(*agent, HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS, &res_buffer);
  if(status != HSA_STATUS_SUCCESS){
  	return hipErrorRuntimeOther;
  }
  __tex_sz = static_cast<int>(res_buffer);
  err = hipMemcpyToSymbol(&__d_tex_sz, __tex_sz, sizeof(unsigned int));
  return err;
}
#endif //hip platform hcc

#else // use texure
hipError_t store_texture_max_dim(int devid){return hipSuccess;}
#endif // not use texure

#endif // _PLATFORM_H__
