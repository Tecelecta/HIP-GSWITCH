#ifndef __UTILS_H
#define __UTILS_H

#include <iostream>
#include <random>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <sys/time.h>
#include <string>
#include <type_traits>
#include <hip/hip_runtime.h>

#include "utils/cmdline.hxx"
#include "utils/common.hxx"
#include "utils/json.hxx"
#include "utils/filesystem.hxx"

#include "utils/platform.hxx"

#define __tbdinline__ __forceinline__
#define LOG(...) if(cmd_opt.verbose) fprintf(stdout, __VA_ARGS__);
#define TRACE() //fprintf(stdout, "\033[34m TRACE: %s in %s at line %d \033[0m\n",__FUNCTION__, __FILE__, __LINE__);
#define FLAG(f) fprintf(stdout, "\033[31m FLAG %d \033[0m\n",f);
#define CUBARRIER() hipDeviceSynchronize();

#define MAX_32S 2147483647 
#define MAX_32U 4294967295
#define H2D hipMemcpyHostToDevice
#define D2H hipMemcpyDeviceToHost
#define MIN(a,b) ((a)<(b)?(a):(b))
#define MAX(a,b) ((a)>(b)?(a):(b))

#define TOHOST(pdev, phost, num) hipMemcpy((phost), (pdev), sizeof(*(pdev))*(num), D2H);
#define TODEV(pdev, phost, num) hipMemcpy((pdev), (phost), sizeof(*(phost))*(num), H2D);
#define CLEAN(pdev, num) hipMemset((pdev), 0, sizeof(*(pdev))*num);
#define CudaCheckError() __CudaCheckError(__FILE__, __LINE__);

#define HASH(v) (-v-2)
#define CEIL(x,b) ((x+b-1)/b)
#define ALIGN(x,b) ( (x)%(b)==0?(x):(x+b-(x)%(b)) )

#define TODEVICE(ptr, size, hptr) \
    H_ERR(hipMalloc((void**)&ptr, sizeof(int)*(size))); \
    H_ERR(hipMemcpy(ptr, hptr, sizeof(int)*(size), H2D));

#define TODEVICE_FLOAT(ptr, size, hptr) \
    H_ERR(hipMalloc((void**)&ptr, sizeof(float)*(size))); \
    H_ERR(hipMemcpy(ptr, hptr, sizeof(float)*(size), H2D));

#define TODEVICE_SHORT(ptr, size, hptr) \
    H_ERR(hipMalloc((void**)&ptr, sizeof(short)*(size))); \
    H_ERR(hipMemcpy(ptr, hptr, sizeof(short)*(size), H2D));



static void 
__CudaCheckError(const char* file, const int line){
  hipError_t err = hipGetLastError();
  if(hipSuccess != err){
    fprintf(stderr, "cudaCHeckError() failed at %s:%i : %s\n",file,line,hipGetErrorString(err));
    exit(-1);
  }
}


// HandleError
static void 
HandleError( hipError_t err, const char *file, int line ) {
  if (err != hipSuccess) {
    printf( "%s in %s at line %d\n", \
    hipGetErrorString( err ), file, line );
    exit( EXIT_FAILURE );
  }
}

#define H_ERR( err ) (HandleError( err, __FILE__, __LINE__ ))


#define ASSERT( Predicate, Err_msg ) \
if(true){                            \
  if( !(Predicate) ){                \
    std::cerr << "CHECK failed :"    \
      << Err_msg  << " at ("         \
      << __FILE__ << ", "            \
      << __LINE__ << ")"             \
      << std::endl;                  \
		exit(1);						             \
  }                                  \
}

template<typename data_t>
struct rand_device{};

template<>
struct rand_device<float>{
  static inline float rand_weight(int lim){
    return (float)drand48()*lim;
  }
};

template<>
struct rand_device<int>{
  static inline int rand_weight(int lim){
    return (int)(1+rand()%lim);
  }
};


template<typename data_t>
void build_tex(hipTextureObject_t &tex_obj, data_t* buf, int N){
  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeLinear;
  resDesc.res.linear.devPtr = buf;
#ifdef __HIP_PLATFORM_NVCC__
  if(std::is_same<int,data_t>::value) resDesc.res.linear.desc.f = 
    hipChannelFormatKindToCudaChannelFormatKind(hipChannelFormatKindSigned);
  else if(std::is_same<float, data_t>::value) resDesc.res.linear.desc.f = 
    hipChannelFormatKindToCudaChannelFormatKind(hipChannelFormatKindFloat);
#else
  if(std::is_same<int,data_t>::value) resDesc.res.linear.desc.f = hipChannelFormatKindSigned;
  else if(std::is_same<float, data_t>::value) resDesc.res.linear.desc.f = hipChannelFormatKindFloat;
#endif
  else ASSERT(false, "build texture w/ bad data type");
  resDesc.res.linear.desc.x = 32;
  resDesc.res.linear.sizeInBytes = N*sizeof(data_t);

  hipTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.readMode = hipReadModeElementType;
  H_ERR(hipCreateTextureObject(&tex_obj, &resDesc, &texDesc, NULL));
}

void query_device_prop(){
  int nDevices;
  hipGetDeviceCount(&nDevices);
  for(int i = 0; i < nDevices; i++){
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, i);
    printf("Device Number: %d\n", i);
    printf(" Device name: %s\n", prop.name);
    printf(" Device Capability: %d.%d\n", prop.major, prop.minor);
    //this attrib is not supported by rocm
    //printf(" Device Overlap: %s\n", (prop.deviceOverlap ? "yes":"no"));
    printf(" Device canMapHostMemory: %s\n", (prop.canMapHostMemory ? "yes":"no"));
    printf(" Memory Detils\n");
    printf("  - registers per Block (KB): %d\n", (prop.regsPerBlock));
    printf("  - registers per Thread (1024): %d\n", (prop.regsPerBlock/1024));
    printf("  - Share Memory per Block (KB): %.2f\n", (prop.sharedMemPerBlock+.0)/(1<<10));
    printf("  - Total Global Memory (GB): %.2f\n", (prop.totalGlobalMem+.0)/(1<<30));
    printf("  - Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    printf("  - Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    printf("  - Peak Memory Bandwidth (GB/s): %f\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
    printf(" Thread Detils\n");
    printf("  - max threads per Block: %d\n", prop.maxThreadsPerBlock);
    printf("  - processor Count: %d\n", prop.multiProcessorCount);
    //this is acturally very important --lmy
    printf("  - warp size: %d\n", prop.warpSize);
    printf("\n");
  }
}


// timming
#define RDTSC(val) do {			                          \
    uint64_t __a,__d;					                        \
    asm volatile("rdtsc" : "=a" (__a), "=d" (__d));		\
    (val) = ((uint64_t)__a) | (((uint64_t)__d)<<32);	\
  } while(0)

static inline uint64_t rdtsc() {
  uint64_t val;
  RDTSC(val);
  return val;
}

inline double wtime(){
	double time[2];	
	struct timeval time1;
	gettimeofday(&time1, NULL);

	time[0]=time1.tv_sec;
	time[1]=time1.tv_usec;

	return time[0]+time[1]*1.0e-6;
}

inline double mwtime(){
  return 1000*wtime();
}

template<typename T>
bool is_nan(const T& val){return val!=val;}

template<typename T>
bool is_inf(const T& val){
  T _max = std::numeric_limits<T>::max();
  T _min = - _max;
  return !(_min <= val && val <= _max);
}

inline bool isAcceptable(float a, float b){
  if( is_nan(a) || is_inf(a)) return true;
  if(fabs(a) > 1) return fabs(b/a-1)<1e-2;
  else return fabs(b-a)<1e-2;
}


// data type 
//typedef int     index_t;
typedef int64_t packed_t;

struct helper_t{
  hipStream_t stream[3];
  void build(){ for(int i=0;i<3;++i) hipStreamCreate(&stream[i]); }
};
helper_t global_helper;


struct Empty{char placeholder;};
std::istream& operator>>(std::istream& in, Empty& e){
  in>>e.placeholder;
  return in;
}
std::ostream& operator<<(std::ostream& out, Empty& e){
  out<<e.placeholder;
  return out;
}


int* global_dg_blk_sum=NULL;


#endif
