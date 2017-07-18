extern "C" {
#include "cudamem.cuh"
}

/* Pinned memory functions */

/**
 *    Pin allocated double vector
 */
void pin_double_vector(double *vector, long Nx) {
   if(cudaHostRegister((void *) vector, Nx * sizeof(double), cudaHostRegisterPortable) != cudaSuccess) {
      fprintf(stderr, "Failed to pin memory for double vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Pin allocated complex vector
 */
void pin_complex_vector(cuDoubleComplex *vector, long Nx) {
   if(cudaHostRegister((void *) vector, Nx * sizeof(cuDoubleComplex), cudaHostRegisterPortable) != cudaSuccess) {
      fprintf(stderr, "Failed to pin memory for complex vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Pin allocated double matrix
 */
void pin_double_matrix(double **matrix, long Nx, long Ny) {
   if(cudaHostRegister((void *) matrix[0], Nx * Ny * sizeof(double), cudaHostRegisterPortable) != cudaSuccess) {
      fprintf(stderr, "Failed to pin memory for double matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Pin allocated complex matrix
 */
void pin_complex_matrix(cuDoubleComplex **matrix, long Nx, long Ny) {
   if(cudaHostRegister((void *) matrix[0], Nx * Ny * sizeof(cuDoubleComplex), cudaHostRegisterPortable) != cudaSuccess) {
      fprintf(stderr, "Failed to pin memory for complex matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Pin allocated double tensor
 */
void pin_double_tensor(double ***tensor, long Nx, long Ny, long Nz) {
   if(cudaHostRegister((void *) tensor[0][0], Nx * Ny * Nz * sizeof(double), cudaHostRegisterPortable) != cudaSuccess) {
      fprintf(stderr, "Failed to pin memory for double tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Pin allocated complex tensor
 */
void pin_complex_tensor(cuDoubleComplex ***tensor, long Nx, long Ny, long Nz) {
   if(cudaHostRegister((void *) tensor[0][0], Nx * Ny * Nz * sizeof(cuDoubleComplex), cudaHostRegisterPortable) != cudaSuccess) {
      fprintf(stderr, "Failed to pin memory for complex tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/* Mapped memory functions */

/**
 *    Map pinned double matrix
 */
struct cudaPitchedPtr map_double_matrix(double **matrix, long Nx, long Ny) {
   double *d_matrix;

   if(cudaHostGetDevicePointer((void **) &d_matrix, matrix[0], 0) != cudaSuccess) {
      fprintf(stderr, "Failed to get device pointer for double matrix in mapped memory.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return make_cudaPitchedPtr(d_matrix, Ny * sizeof(double), Ny * sizeof(double), Nx);
}

/**
 *    Map pinned double matrix
 */
struct cudaPitchedPtr map_double_matrix2(double *matrix, long Nx, long Ny) {
   double *d_matrix;

   if(cudaHostGetDevicePointer((void **) &d_matrix, matrix, 0) != cudaSuccess) {
      fprintf(stderr, "Failed to get device pointer for double matrix in mapped memory.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return make_cudaPitchedPtr(d_matrix, Ny * sizeof(double), Ny * sizeof(double), Nx);
}

/**
 *    Map pinned complex matrix
 */
struct cudaPitchedPtr map_complex_matrix(cuDoubleComplex **matrix, long Nx, long Ny) {
   cuDoubleComplex *d_matrix;

   if(cudaHostGetDevicePointer((void **) &d_matrix, matrix[0], 0) != cudaSuccess) {
      fprintf(stderr, "Failed to get device pointer for complex matrix in mapped memory.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return make_cudaPitchedPtr(d_matrix, Ny * sizeof(cuDoubleComplex), Ny * sizeof(cuDoubleComplex), Nx);
}

/**
 *    Map pinned double tensor
 */
struct cudaPitchedPtr map_double_tensor(double ***tensor, long Nx, long Ny, long Nz) {
   double *d_tensor;

   if(cudaHostGetDevicePointer((void **) &d_tensor, tensor[0][0], 0) != cudaSuccess) {
      fprintf(stderr, "Failed to get device pointer for double tensor in mapped memory.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return make_cudaPitchedPtr(d_tensor, Nz * sizeof(double), Nz * sizeof(double), Ny);
}

/**
 *    Map pinned complex tensor
 */
struct cudaPitchedPtr map_complex_tensor(cuDoubleComplex ***tensor, long Nx, long Ny, long Nz) {
   cuDoubleComplex *d_tensor;

   if(cudaHostGetDevicePointer((void **) &d_tensor, tensor[0][0], 0) != cudaSuccess) {
      fprintf(stderr, "Failed to get device pointer for complex tensor in mapped memory.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return make_cudaPitchedPtr(d_tensor, Nz * sizeof(cuDoubleComplex), Nz * sizeof(cuDoubleComplex), Ny);
}

/* Pinned/mapped memory release */

/**
 *    Free pinned/mapped double vector
 */
void free_pinned_double_vector(double *vector) {
   if (cudaHostUnregister(vector) != cudaSuccess) {
      fprintf(stderr, "Failed to unregister pinned memory for double vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   //free_double_vector(vector);
}

/**
 *    Free pinned/mapped complex vector
 */
void free_pinned_complex_vector(cuDoubleComplex *vector) {
   if (cudaHostUnregister(vector) != cudaSuccess) {
      fprintf(stderr, "Failed to unregister pinned memory for complex vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   //free_complex_vector(vector);
}

/**
 *    Free pinned/mapped double matrix
 */
void free_pinned_double_matrix(double **matrix) {
   if (cudaHostUnregister(matrix[0]) != cudaSuccess) {
      fprintf(stderr, "Failed to unregister pinned memory for double matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   //free_double_matrix(matrix);
}

/**
 *    Free pinned/mapped complex matrix
 */
void free_pinned_complex_matrix(cuDoubleComplex **matrix) {
   if (cudaHostUnregister(matrix[0]) != cudaSuccess) {
      fprintf(stderr, "Failed to unregister pinned memory for complex matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   //free_complex_matrix(matrix);
}

/**
 *    Free pinned/mapped double tensor
 */
void free_pinned_double_tensor(double ***tensor) {
   if (cudaHostUnregister(tensor[0][0]) != cudaSuccess) {
      fprintf(stderr, "Failed to unregister pinned memory for double tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   //free_double_tensor(tensor);
}

/**
 *    Free pinned/mapped complex tensor
 */
void free_pinned_complex_tensor(cuDoubleComplex ***tensor) {
   if (cudaHostUnregister(tensor[0][0]) != cudaSuccess) {
      fprintf(stderr, "Failed to unregister pinned memory for complex tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   //free_complex_tensor(tensor);
}

/* CUDA memory functions */

/**
 *    Double vector allocation on CUDA device
 */
double *alloc_double_vector_device(long Nx) {
   double *vector;

   if(cudaMalloc((void**) &vector, Nx * sizeof(double)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA double vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return vector;
}

/**
 *    Complex vector allocation on CUDA device
 */
cuDoubleComplex *alloc_complex_vector_device(long Nx) {
   cuDoubleComplex *vector;

   if(cudaMalloc((void**) &vector, Nx * sizeof(cuDoubleComplex)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA complex vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return vector;
}

/**
 *    Double matrix allocation on CUDA device
 */
struct cudaPitchedPtr alloc_double_matrix_device(long Nx, long Ny) {
   struct cudaPitchedPtr matrix;

   if(cudaMalloc3D(&matrix, make_cudaExtent(Ny * sizeof(double), Nx, 1)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA double matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return matrix;
}

/**
 *    Complex matrix allocation on CUDA device
 */
struct cudaPitchedPtr alloc_complex_matrix_device(long Nx, long Ny) {
   struct cudaPitchedPtr matrix;

   if(cudaMalloc3D(&matrix, make_cudaExtent(Ny * sizeof(cuDoubleComplex), Nx, 1)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA complex matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return matrix;
}

/**
 *    Double tensor allocation on CUDA device
 */
struct cudaPitchedPtr alloc_double_tensor_device(long Nx, long Ny, long Nz) {
   struct cudaPitchedPtr tensor;

   if(cudaMalloc3D(&tensor, make_cudaExtent(Nz * sizeof(double), Ny, Nx)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA double tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return tensor;
}

/**
 *    Complex tensor allocation on CUDA device
 */
struct cudaPitchedPtr alloc_complex_tensor_device(long Nx, long Ny, long Nz) {
   struct cudaPitchedPtr tensor;

   if(cudaMalloc3D(&tensor, make_cudaExtent(Nz * sizeof(cuDoubleComplex), Ny, Nx)) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate device memory for the CUDA complex tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   return tensor;
}

/**
 *    Free double vector on CUDA device
 */
void free_double_vector_device(double *vector) {
   if (cudaFree(vector) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for double vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Free complex vector on CUDA device
 */
void free_complex_vector_device(cuDoubleComplex *vector) {
   if (cudaFree(vector) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for complex vector.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Free double matrix on CUDA device
 */
void free_double_matrix_device(struct cudaPitchedPtr matrix) {
   if (cudaFree(matrix.ptr) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for double matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Free complex matrix on CUDA device
 */
void free_complex_matrix_device(struct cudaPitchedPtr matrix) {
   if (cudaFree(matrix.ptr) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for complex matrix.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Free double tensor on CUDA device
 */
void free_double_tensor_device(struct cudaPitchedPtr tensor) {
   if (cudaFree(tensor.ptr) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for double tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}

/**
 *    Free complex tensor on CUDA device
 */
void free_complex_tensor_device(struct cudaPitchedPtr tensor) {
   if (cudaFree(tensor.ptr) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory for complex tensor.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }
}




static char *d_pool;
static size_t pool_size = 0;
static size_t offset = 0;
void init_mem_device(size_t size) {

   if(cudaMalloc((void**) &d_pool, size) != cudaSuccess) {
      fprintf(stderr, "Failed to allocate %zu of device memory.\n", size);
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   pool_size = size;
   offset = 0;
}

void *alloc_mem_device(size_t size) {
   void *d_ptr;

   d_ptr = d_pool + offset;
   offset += size;

   if (offset > pool_size) {
      fprintf(stderr, "Failed to allocate %zu of device memory.\n", size);
      exit(EXIT_FAILURE);
   }

   return d_ptr;
}

void reset_mem_device() {
   offset = 0;
}

void free_mem_device() {
   if (cudaFree(d_pool) != cudaSuccess) {
      fprintf(stderr, "Failed to free device memory.\n");
      cudaError_t error = cudaGetLastError();
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
      exit(EXIT_FAILURE);
   }

   offset = 0;
}
