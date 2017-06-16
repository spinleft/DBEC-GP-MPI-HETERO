#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

/* Pinned memory functions */
void pin_double_vector(double *, long);
void pin_complex_vector(cuDoubleComplex *, long);
void pin_double_matrix(double **, long, long);
void pin_complex_matrix(cuDoubleComplex **, long, long);
void pin_double_tensor(double ***, long, long, long);
void pin_complex_tensor(cuDoubleComplex ***, long, long, long);

/* Mapped memory functions */
struct cudaPitchedPtr map_double_matrix(double **, long, long);
struct cudaPitchedPtr map_double_matrix2(double *, long, long);
struct cudaPitchedPtr map_complex_matrix(cuDoubleComplex **, long, long);
struct cudaPitchedPtr map_double_tensor(double ***, long, long, long);
struct cudaPitchedPtr map_complex_tensor(cuDoubleComplex ***, long, long, long);

/* Pinned/mapped memory release */
void free_pinned_double_vector(double *);
void free_pinned_complex_vector(cuDoubleComplex *);
void free_pinned_double_matrix(double **);
void free_pinned_complex_matrix(cuDoubleComplex **);
void free_pinned_double_tensor(double ***);
void free_pinned_complex_tensor(cuDoubleComplex ***);

/* CUDA memory allocation */
double *alloc_double_vector_device(long);
cuDoubleComplex *alloc_complex_vector_device(long);
struct cudaPitchedPtr alloc_double_matrix_device(long, long);
struct cudaPitchedPtr alloc_complex_matrix_device(long, long);
struct cudaPitchedPtr alloc_double_tensor_device(long, long, long);
struct cudaPitchedPtr alloc_complex_tensor_device(long, long, long);

/* CUDA memory release */
void free_double_vector_device(double *);
void free_complex_vector_device(cuDoubleComplex *);
void free_double_matrix_device(struct cudaPitchedPtr);
void free_complex_matrix_device(struct cudaPitchedPtr);
void free_double_tensor_device(struct cudaPitchedPtr);
void free_complex_tensor_device(struct cudaPitchedPtr);

/* CUDA memory pooling */
void init_mem_device(size_t);
void *alloc_mem_device(size_t);
void reset_mem_device();
void free_mem_device();
