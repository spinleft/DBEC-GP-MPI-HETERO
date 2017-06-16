#ifndef R3D_CPU_H_
#define R3D_CPU_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>
#include <fftw3.h>
#include <complex.h>
#include <omp.h>
#include <mpi.h>
#include <sys/time.h>
#ifdef FFTW_TRAN
#include <fftw3-mpi.h>
#else
#include "../utils/tran.h"
#endif

#define MAX(a, b, c)       (a > b) ? ((a > c) ? a : c) : ((b > c) ? b : c)
#define MAX_FILENAME_SIZE  256
#define RMS_ARRAY_SIZE     4

#define BOHR_RADIUS        5.2917720859e-11

#define FFT_FLAG FFTW_MEASURE
/* #define FFT_FLAG FFTW_ESTIMATE */

#ifdef FFTW_TRAN
#define TRAN_PSI_X fftw_execute(plan_tr_psi_x);
#define TRAN_PSI_Y fftw_execute(plan_tr_psi_y);
#define TRAN_PSIDD2_X fftw_execute(plan_tr_psidd2_x);
#define TRAN_PSIDD2_Y fftw_execute(plan_tr_psidd2_y);
#define TRAN_DPSI fftw_execute(plan_tr_dpsi);
#else
#define TRAN_PSI_X transpose(tran_psi);
#define TRAN_PSI_Y transpose_back(tran_psi);
#define TRAN_PSIDD2_X transpose(tran_dd2);
#define TRAN_PSIDD2_Y transpose_back(tran_dd2);
#define TRAN_DPSI transpose_back(tran_dpsi);
#endif

char *output, *rmsout, *initout, *Nstpout, *Npasout, *Nrunout, *dynaout;
long outstpx, outstpy, outstpz, outstpt;

int nthreads, nthreads_luy, nthreads_luz;
int nprocs;
int opt;
long Na;
long Nstp, Npas, Nrun;
long Nx, Ny, Nz;
long cpuNx_fft, gpuNx_fft, cpuNx_nyz, gpuNx_nyz, cpuNy_fft, gpuNy_fft, cpuNy_lux, gpuNy_lux;
int chunksNx_fft, chunksNx_nyz, chunksNy_fft, chunksNy_lux;
int block3d_dd2[3], block2d_dd2[2], block2d_luy[2], block2d_luz[2], block3d_nu[3], block2d_lux[2], block3d_potdd[3];
int grid3d_dd2, grid2d_dd2, grid2d_luy, grid2d_luz, grid3d_nu, grid2d_lux, grid3d_potdd;
int Ngpu, offload;
long cpuNx_luy, cpuNx_luz;
int rank;
long localNx, localNy, offsetNx, offsetNy;
long Nx2, Ny2, Nz2;
double g, g0, gpar, gd, gd0, gdpar;
double aho, as, add;
double dx, dy, dz;
double dx2, dy2, dz2;
double dt;
double vnu, vlambda, vgamma;
double par;
double pi, cutoff;

double *x, *y, *z;
double *x2, *y2, *z2;
double complex ***psi, ***psi_t;
double ***psidd2;
double ***pot;
double ***potdd;
fftw_complex *psidd2fft, *psidd2fft_t;
double *psidd2tmp;

double complex Ax0, Ay0, Az0, Ax0r, Ay0r, Az0r, Ax, Ay, Az;
double complex *calphax, *calphay, *calphaz;
double complex *cgammax, *cgammay, *cgammaz;
double complex **cbeta;

fftw_plan plan_forward_row, plan_forward_col;
fftw_plan plan_backward_row, plan_backward_col;

#ifdef FFTW_TRAN
fftw_plan plan_tr_psi_x, plan_tr_psi_y, plan_tr_psidd2_x, plan_tr_psidd2_y, plan_tr_dpsi;
#else
struct tran_params tran_psi, tran_dd2, tran_dpsi;
#endif

void readpar(void);
void initpsi(double complex ***, double *);
void initpot(double ***);
void gencoef(void);
void initpotdd(double *, double *, double *, double *, double *, double *);
void calcnorm(double *, double complex ***, double **, double **, double **);
void calcmuen(double *, double *, double complex ***, double complex ***, double ***, double ***, double ***, fftw_complex *, fftw_complex *, double **, double **, double **, double **, double **, double **);
void calcpsidd2(double complex ***, double ***, fftw_complex *, fftw_complex *);
void calcrms(double *, double complex ***, double complex ***, double **, double **, double **);
void calcnu(double complex ***, double ***, double ***);
void calclux(double complex ***, double complex **);
void calcluy(double complex ***, double complex **);
void calcluz(double complex ***, double complex **);

void (*calcpsi)(double complex ***, double complex ***, double ***, double ***, double complex **);
void calcpsi_o1(double complex ***, double complex ***, double ***, double ***, double complex **);
void calcpsi_o2(double complex ***, double complex ***, double ***, double ***, double complex **);
void calcpsi_o3(double complex ***, double complex ***, double ***, double ***, double complex **);

void calcnuluy(double complex ***, double ***, double ***, double complex **);
void calcnuluyluz(double complex ***, double ***, double ***, double complex **);

void calcpsidd2_cpu_p1(double complex ***, double ***, fftw_complex *);
void calcpsidd2_cpu_p2(fftw_complex *);
void calcpsidd2_cpu_p3(double ***, fftw_complex *);
void calcpsidd2_cpu_p4(double ***);

void calcnu_cpu(double complex ***, double ***, double ***);
void calclux_cpu(double complex ***, double complex **);
void calcluy_cpu(double complex ***, double complex **);
void calcluz_cpu(double complex ***, double complex **);

void outdenx(double complex ***, double **, double *, double *, MPI_File);
void outdeny(double complex ***, double **, double *, double *, MPI_File);
void outdenz(double complex ***, double **, double *, double *, MPI_File);
void outdenxy(double complex ***, double ***, double *, MPI_File);
void outdenxz(double complex ***, double ***, double *, MPI_File);
void outdenyz(double complex ***, double ***, double *, MPI_File);
void outpsi2xy(double complex ***, double ***, MPI_File);
void outpsi2xz(double complex ***, double ***, MPI_File);
void outpsi2yz(double complex ***, double ***, MPI_File);
void outdenxyz(double complex ***, double ***, MPI_File);

extern double simpint(double, double *, long);
extern void diff(double, double *, double *, long);

extern int cfg_init(char *);
extern char *cfg_read(char *);

extern double *alloc_double_vector(long);
extern double complex *alloc_complex_vector(long);
extern double **alloc_double_matrix(long, long);
extern double complex **alloc_complex_matrix(long, long);
extern double ***alloc_double_tensor(long, long, long);
extern double complex ***alloc_complex_tensor(long, long, long);
extern fftw_complex *alloc_fftw_complex_vector(long);
extern void free_double_vector(double *);
extern void free_complex_vector(double complex *);
extern void free_double_matrix(double **);
extern void free_complex_matrix(double complex **);
extern void free_double_tensor(double ***);
extern void free_complex_tensor(double complex ***);
extern void free_fftw_complex_vector(fftw_complex *);

extern void pin_double_vector(double *, long);
extern void pin_double_tensor(double ***, long, long, long);
extern void pin_complex_tensor(double complex ***, long, long, long);

extern void init_gpu();
extern void free_gpu();

extern void copy_psidd2_d2h();
extern void sync_with_gpu();

extern void (*calcpsidd2_gpu_p1)();
extern void (*calcpsidd2_gpu_p2)();
extern void (*calcpsidd2_gpu_p3)();
extern void calcpsidd2_gpu_p4(int);
extern void (*calcnu_gpu)();
extern void (*calclux_gpu)();
extern void (*calcluy_gpu)();
extern void (*calcluz_gpu)();

#endif /* R3D_CPU_H_ */
