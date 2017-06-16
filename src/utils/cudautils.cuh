#ifndef CUDAUTILS_H
#define CUDAUTILS_H

#include <cuda_runtime.h>
#include <cufft.h>
#include <cuComplex.h>
#include <math.h>

#define CUDA_BLOCK_SIZE     128
#define CUDA_BLOCK_SIZE_2D  16
#define CUDA_BLOCK_SIZE_3D  8

#define TID_X (blockIdx.x * blockDim.x + threadIdx.x)
#define TID_Y (blockIdx.y * blockDim.y + threadIdx.y)
#define TID_Z (blockIdx.z * blockDim.z + threadIdx.z)

#define GRID_STRIDE_X (blockDim.x * gridDim.x)
#define GRID_STRIDE_Y (blockDim.y * gridDim.y)
#define GRID_STRIDE_Z (blockDim.z * gridDim.z)

#define cudaCheckError(ans) { cudaCheck((ans), __FILE__, __LINE__); }
inline void cudaCheck(cudaError_t code, const char *file, int line) {
   if (code != cudaSuccess) {
      fprintf(stderr,"CUDA error %d: %s %s %d\n", code, cudaGetErrorString(code), file, line);
      exit(code);
   }
}

static const char *cudaGetFFTErrorString(cufftResult error) {
   switch (error) {
   case CUFFT_SUCCESS:
         return "CUFFT_SUCCESS";
   case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";

   case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";

   case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";

   case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";

   case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";

   case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";

   case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";

   case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";

   case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";

   case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";

   default:
      return "<unknown>";
   }
}

#define cudaCheckFFTError(ans) { cudaFFTCheck((ans), __FILE__, __LINE__); }
inline void cudaFFTCheck(cufftResult code, const char *file, int line) {
   if (code != CUFFT_SUCCESS) {
      fprintf(stderr,"cuFFT error: %s %s %d\n", cudaGetFFTErrorString(code), file, line);
      exit(code);
   }
}

__device__ __host__ __forceinline__ double *get_double_matrix_row(struct cudaPitchedPtr cudaptr, long row) {
   return (double *)((char *)cudaptr.ptr + row * cudaptr.pitch);
}

__device__ __host__ __forceinline__ cuDoubleComplex *get_complex_matrix_row(struct cudaPitchedPtr cudaptr, long row) {
   return (cuDoubleComplex *)((char *)cudaptr.ptr + row * cudaptr.pitch);
}

__device__ __host__ __forceinline__ double *get_double_tensor_row(struct cudaPitchedPtr cudaptr, long slice, long row) {
   return (double *)((((char *)cudaptr.ptr) + slice * cudaptr.pitch * cudaptr.ysize) + row * cudaptr.pitch);
}

__device__ __host__ __forceinline__ cuDoubleComplex *get_complex_tensor_row(struct cudaPitchedPtr cudaptr, long slice, long row) {
   return (cuDoubleComplex *)((((char *)cudaptr.ptr) + slice * cudaptr.pitch * cudaptr.ysize) + row * cudaptr.pitch);
}

__device__ __inline__ cuDoubleComplex cuCexp(double arg) {
   cuDoubleComplex res;

   sincos(arg, &res.y, &res.x);

   return res;

   //double factor = exp(arg);

   //return make_cuDoubleComplex(factor * cos(0.0), factor * sin(0.0));
}

#endif /* CUDAUTILS_H */
