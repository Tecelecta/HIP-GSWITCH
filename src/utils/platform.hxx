#ifndef _PLATFORM_H__
#define _PLATFORM_H__
#include <hip/hip_runtime.h>
/**
 * this file specifies key variables differ from platforms
 */
//adapt for variant warp size --lmy
int __warp_sz;
int __warp_shft;
//makes this one a rvalue to prevent it from modifications
#define WARP_SIZE ((const int)__warp_sz)
#define LANE_MASK (__warp_sz-1)
#define LANE_SHFT ((const int)__warp_shft)

void store_warpsize(hipDeviceProp_t &prop){
  // this one deals the warp size problem --lmy
  __warp_sz = prop.warpSize;
  int tmp = __warp_sz-1;
  for(__warp_shft = 0; tmp; ++__warp_shft) tmp&=(tmp-1);
}
// this constant is reserved due to the current hip bug!
#define DEFUALT_REG_LIM 128

hipError_t store_texture_max_dim(int devid);
#ifdef __USE_TEXTURE__
int __tex_sz;
#define MAXTEX ((const int)__tex_sz)

#ifdef __HIP_PLATFORM_NVCC__
#include <cuda.h>
hipError_t store_texture_max_dim(int devid){
  cudaDeviceProp cuda_prop;
  cudaError_t err = cudaGetDeviceProperties(&cuda_prop, devid);
  if(err != cudaSuccess){
  	return hipCUDAErrorTohipError(err);
  }
  __tex_sz = cuda_prop.maxTexture1D;
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
  hsa_agent_t* agent = &(ihipGetDevice(devid)->_hsaAgent);
  hsa_status_t status = hsa_amd_image_get_info_max_dim(*agent, HSA_EXT_AGENT_INFO_IMAGE_1D_MAX_ELEMENTS, &res_buffer);
  if(status != HSA_STATUS_SUCCESS){
  	return hipErrorRuntimeOther;
  }
  __tex_sz = static_cast<int>(res_buffer);
  return hipSucess;
}
#endif //hip platform hcc

#else // use texure
hipError_t store_texture_max_dim(int devid){}
#endif // use texure

#endif // _PLATFORM_H__