
#include <hip/hip_runtime.h>
#include "utils/utils.cuh"
#include "utils/intrinsics.cuh"

#include <cstdio>
#include <vector>
#include <iostream>

#include <moderngpu/kernel_sortedsearch.hxx>
#include <moderngpu/kernel_scan.hxx>

#include "utils/sorted_search.cuh"

void Print(std::vector<int> &vec){
  for(auto x: vec){
    printf("%4d ",x);
  }
  puts("\n");
}

__global__ void
my_memsetIdx(int* dg_array, int size, int scale){
  const int gtid=hipBlockIdx_x*hipBlockDim_x + hipThreadIdx_x;
  if(gtid < size){
    dg_array[gtid] = gtid*scale;
  }
}

void test_mgpu_scan(mgpu::standard_context_t* context){
  printf("\n\nScan:\n\n");
  const int size = 10;
  mgpu::mem_t<int> A = mgpu::fill_random(0, 299, size, true, *context);
  mgpu::mem_t<int> B(size, *context);
  //mgpu::mem_t<int> total(1, *context, mgpu::memory_space_host);
  mgpu::scan<mgpu::scan_type_exc>(A.data(), 10, B.data(), *context);
  std::vector<int> A_host = mgpu::from_mem(A);
  std::vector<int> B_host = mgpu::from_mem(B);

  //printf("TOTAL = %d\n", total.data()[0]);
  printf("A:\n  ");
  Print(A_host);
  printf("B:\n  ");
  Print(B_host);
}

void test_mgpu_sortedsearch(mgpu::standard_context_t* context){
  printf("\n\nSORTED SEARCH:\n\n");
  const int vertexsize = 25600;
  const int blksize = 25600/16 - 1;
  mgpu::mem_t<int> A = mgpu::fill_random(0, 299, blksize, true, *context);
  mgpu::mem_t<int> B = mgpu::fill_random(0, 50000, vertexsize, true, *context);
  mgpu::mem_t<int> idx(blksize, *context);
  
  mgpu::mem_t<int> my_idx(blksize, *context); 
  mgpu::mem_t<int> my_idx1(blksize, *context);

  hipLaunchKernelGGL(my_memsetIdx, dim3(256), dim3(256), 0, 0, (int*)A.data(), blksize, 16);
  hipLaunchKernelGGL(my_memsetIdx, dim3(256), dim3(256), 0, 0, (int*)B.data(), 1, 0);
  
  double t1 = wtime();
  mgpu::sorted_search<mgpu::bounds_lower>((int*)A.data(), blksize, 
      (int*)B.data(), vertexsize, idx.data(), mgpu::less_t<int>(), *context);
  double t2 = wtime();

  std::vector<int> A_host = mgpu::from_mem(A);
  std::vector<int> B_host = mgpu::from_mem(B);
  std::vector<int> idx_host = mgpu::from_mem(idx);
  
  double t3 = wtime();
  //sorted_search<int,256>((int*)A.data(), blksize, (int*)B.data(), vertexsize, my_idx.data());
  double t4 = wtime();
  
  std::vector<int> myh_idx = mgpu::from_mem(my_idx);

  double t5 = wtime();
  sorted_search1<int,256>((int*)A.data(), blksize, (int*)B.data(), vertexsize, my_idx1.data());
  double t6 = wtime();

  printf("A:\n  ");
//  Print(A_host);
  printf("B:\n  ");
//  Print(B_host);
  printf("Idx: %f s\n  ", t2-t1);
//  Print(idx_host);
  printf("My Idx: %f s\n ", t4-t3);
  printf("Mine Idx: %f s\n", t6-t5);
//  Print(myh_idx);
}

/*
void test_scan(int n){
  int *dg_input, *dg_output, *dg_index;
  cudaMalloc((void**)&dg_input, n*sizeof(int));
  cudaMalloc((void**)&dg_output, n*sizeof(int));
  cudaMalloc((void**)&dg_index, n*sizeof(int));
  
  int* h_index  = (int*)malloc(sizeof(int)*n);
  int* h_input  = (int*)malloc(sizeof(int)*n);
  int* h_output = (int*)malloc(sizeof(int)*n);
  int* h_answer = (int*)malloc(sizeof(int)*n);

  for(int i=0; i < n; i++){
    h_index[i] = random()%n;
    h_input[i] = i;
  }


  h_answer[0] = 0;
  for(int i = 1 ; i < n; i ++){
    h_answer[i] = h_input[h_index[i-1]] + h_answer[i-1];
  }

  cudaMemcpy(dg_input, h_input, sizeof(int)*n, H2D);
  cudaMemcpy(dg_index, h_index, sizeof(int)*n, H2D);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  scan<128,128>(dg_index, dg_input, dg_output, n, stream);
  cudaMemcpy(h_output, dg_output, sizeof(int)*n, D2H);

  bool check = true;
  for(int i = 0; i < n; i ++){
    if(h_output[i] != h_answer[i]) check = false;
  }

  if(check){
    std::cout << "passed" << std::endl;
  }else {
    std::cout << "failed" << std::endl;
    for(int i=0; i<n; ++i){
      std::cout << h_index[i] << " ";
    }puts("");

    for(int i=0; i<n; ++i){
      std::cout << h_output[i] << " ";
    }puts("");

    for(int i=0; i<n; ++i){
      std::cout << h_answer[i] << " ";
    }puts("");
  }

  free(h_index);
  free(h_input);
  free(h_output);
  free(h_answer);

  cudaFree(dg_input);
  cudaFree(dg_output);
}

*/
struct GPU_device{
  GPU_device(){
    context = new mgpu::standard_context_t(false);
  }

  mgpu::standard_context_t* context;
};

__global__ void
inspect(int* dg_data, int N, int* dg_output){
  const int STRIDE = hipBlockDim_x*hipGridDim_x;
  const int gtid = hipThreadIdx_x+hipBlockIdx_x*hipBlockDim_x;

  int min=MAX_32S, max=0;
  for(int idx=gtid; idx<N; idx+=STRIDE){
    min = MIN(dg_data[idx], min);
    max = MAX(dg_data[idx], max);
  }
  __syncthreads();
  max = blockReduceMax(max);
  //if(!hipThreadIdx_x) atomicMax(&dg_output[0], max);
  if(!hipThreadIdx_x) dg_output[0] = max;

  __syncthreads();
  min = blockReduceMin(min);
  //if(!hipThreadIdx_x) atomicMin(&dg_output[1], min);
  if(!hipThreadIdx_x) dg_output[1] = min;
}

void test_max_min(){
  const int SZ=256;
  GPU_device gdev;
  mgpu::mem_t<int> A = mgpu::fill_random(0, 1024, SZ, false, *gdev.context);
  int* data = A.data();
  int* output;
  hipMalloc((void**)&output, sizeof(int)*2);
  hipMemset(output,0, sizeof(int));
  hipMemset(&output[1], 0x7f, sizeof(int));
  int houtput[2];
  std::vector<int> hdata = mgpu::from_mem(A);
  int max=0,min=MAX_32S;

  // 1 show the data
  for(size_t i=0; i<hdata.size(); ++i){
    std::cout << hdata[i] << " ";
    max = MAX(max, hdata[i]);
    min = MIN(min, hdata[i]);
  }puts("");

  // 2 inspect
  hipLaunchKernelGGL(inspect, dim3(1), dim3(SZ), 0, 0, data, SZ, output);

  // 3 copy the data
  hipMemcpy(houtput, output, sizeof(int)*2, D2H);

  std::cout << "Correct: " << max << " " << min << std::endl;
  std::cout << "My answer: " << houtput[0] << " " << houtput[1] << std::endl;
  gdev.context->synchronize();
}


int main(int argc, char** argv){
  //query_device_prop();
  //parse_cmd(argc, argv, "test");
  //cmd_opt.output();
  //test_scan(100*256*256);
  GPU_device gdev;
  //test_mgpu_scan(gdev.context);
  test_mgpu_sortedsearch(gdev.context);
  //gdev.context->synchronize();
  //test_max_min();
  return 0;
}
