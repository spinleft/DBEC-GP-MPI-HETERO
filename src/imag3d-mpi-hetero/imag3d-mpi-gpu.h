/**
 * DBEC-GP-MPI-HETERO programs are developed by:
 *
 * Vladimir Loncar, Antun Balaz
 * (Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)
 *
 * Srdjan Skrbic
 * (Department of Mathematics and Informatics, Faculty of Sciences, University of Novi Sad, Serbia)
 *
 * Paulsamy Muruganandam
 * (Bharathidasan University, Tamil Nadu, India)
 *
 * Luis E. Young-S, Sadhan K. Adhikari
 * (UNESP - Sao Paulo State University, Brazil)
 *
 *
 * Public use and modification of these codes are allowed provided that the
 * following papers are cited:
 * [1] V. Loncar et al., Comput. Phys. Commun. 209 (2016) 190.
 * [2] V. Loncar et al., Comput. Phys. Commun. 200 (2016) 406.
 * [3] R. Kishor Kumar et al., Comput. Phys. Commun. 195 (2015) 117.
 *
 * The authors would be grateful for all information and/or comments
 * regarding the use of the programs.
 */

#include <math.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftw.h>
#include "../utils/cudamem.cuh"
#include "../utils/cudautils.cuh"

#define MAX(a, b)             (a > b) ? a : b

extern int nprocs;
extern long Nx, Ny, Nz;
extern long Nx2, Ny2, Nz2;
extern long cpuNx_fft, gpuNx_fft, cpuNx_nyz, gpuNx_nyz, cpuNy_fft, gpuNy_fft, cpuNy_lux, gpuNy_lux;
extern long localNx, localNy;
extern int block3d_dd2[3], block2d_dd2[2], block2d_luy[2], block2d_luz[2], block3d_nu[3], block2d_lux[2], block3d_potdd[3];
extern int grid3d_dd2, grid2d_dd2, grid2d_luy, grid2d_luz, grid3d_nu, grid2d_lux, grid3d_potdd;
extern int Ngpu, offload;
extern int chunksNx_fft, chunksNx_nyz, chunksNy_fft, chunksNy_lux;

extern double g, gd, dt;
extern double Ax0r, Ay0r, Az0r, Ax, Ay, Az;
extern double *calphax, *calphay, *calphaz;
extern double *cgammax, *cgammay, *cgammaz;
extern double ***psi, ***psi_t;
extern double ***psidd2, ***pot, ***potdd;
extern double *psidd2tmp;
extern fftw_complex *psidd2fft, *psidd2fft_t;

__constant__ long d_Nx, d_Ny, d_Nz;
__constant__ long d_chunkNx_fft, d_chunkNx_nyz, d_chunkNy_fft, d_chunkNy_lux;
__constant__ double d_dt;
__constant__ double d_Ax0r, d_Ax, d_Ay0r, d_Ay, d_Az0r, d_Az;

long chunkNx_fft, chunkNx_nyz, chunkNy_fft, chunkNy_lux;

struct gpux_t {
   char *ptr1, *ptr2, *ptr3;
   struct cudaPitchedPtr *psix, *psidd2x, *pot, *cbetax;
   struct cudaMemcpy3DParms *psix_h2d, *psix_d2h, *pot_h2d, *psidd2_d2h, *psidd2fftx_h2d, *psidd2fftx_d2h;
   double *calphay, *calphaz, *cgammay, *cgammaz;
   struct cudaPitchedPtr psidd2tmp;
   struct cudaMemcpy3DParms psidd2tmp_h2d;
   cufftHandle *plan_forward_row, *plan_backward_row;
   dim3 dimGrid3d_dd2, dimBlock3d_dd2;
   dim3 dimGrid2d_dd2, dimBlock2d_dd2;
   dim3 dimGrid2d_luy, dimBlock2d_luy;
   dim3 dimGrid2d_luz, dimBlock2d_luz;
   dim3 dimGrid3d_nu, dimBlock3d_nu;
};

struct gpux_t *gpux_fft, *gpux_nyz;

struct gpuy_t {
   char *ptr1, *ptr2, *ptr3;
   struct cudaPitchedPtr *psiy, *psidd2y_orig, *psidd2y_tran, *potdd, *cbetay;
   struct cudaMemcpy3DParms *psiy_h2d, *psiy_d2h, *potdd_h2d, *psidd2ffty_h2d, *psidd2ffty_d2h;
   double *calphax, *cgammax;
   cufftHandle *plan_forward_col, *plan_backward_col;
   dim3 dimGrid2d_lux, dimBlock2d_lux;
   dim3 dimGrid3d_potdd, dimBlock3d_potdd;
};

struct gpuy_t *gpuy_fft, *gpuy_lux;

struct gpu_t {
   void *memptr;
   size_t memsize;
   cudaStream_t *exec_stream;
   int nstreams;
   cudaEvent_t syncEvent;
};

struct gpu_t *gpu;

void init_gpu();
void init_gpux_fft();
void init_gpux_nyz();
void init_gpuy_fft();
void init_gpuy_lux();
void free_gpu();

void copy_psidd2_d2h();
void sync_with_gpu();

void calcnorm_gpu_p1();
void calcnorm_gpu_p2(double);
void (*calcpsidd2_gpu_p1)();
void calcpsidd2_gpu_p1_1ce();
void calcpsidd2_gpu_p1_2ce();
void (*calcpsidd2_gpu_p2)();
void calcpsidd2_gpu_p2_1ce();
void calcpsidd2_gpu_p2_2ce();
void (*calcpsidd2_gpu_p3)();
void calcpsidd2_gpu_p3_1ce();
void calcpsidd2_gpu_p3_2ce();
void calcpsidd2_gpu_p4(int);
void (*calcnu_gpu)();
void calcnu_gpu_1ce();
void calcnu_gpu_2ce();
void (*calclux_gpu)();
void calclux_gpu_1ce();
void calclux_gpu_2ce();
void (*calcluy_gpu)();
void calcluy_gpu_1ce();
void calcluy_gpu_2ce();
void (*calcluz_gpu)();
void calcluz_gpu_1ce();
void calcluz_gpu_2ce();

__global__ void calcpsi2_kernel(struct cudaPitchedPtr, struct cudaPitchedPtr);
__global__ void calcnorm_kernel(struct cudaPitchedPtr, double);
__global__ void calcpsidd2_kernel1(cudaPitchedPtr, cudaPitchedPtr);
__global__ void calcpsidd2_kernel2(cudaPitchedPtr);
__global__ void calcpsidd2_kernel3(cudaPitchedPtr, cudaPitchedPtr);
__global__ void calcpsidd2_kernel4(cudaPitchedPtr, int);
__global__ void calcnu_kernel(cudaPitchedPtr, cudaPitchedPtr, cudaPitchedPtr, double, double);
__global__ void calclux_kernel(cudaPitchedPtr, cudaPitchedPtr, double *, double *);
__global__ void calcluy_kernel(cudaPitchedPtr, cudaPitchedPtr, double *, double *);
__global__ void calcluz_kernel(cudaPitchedPtr, cudaPitchedPtr, double *, double *);
