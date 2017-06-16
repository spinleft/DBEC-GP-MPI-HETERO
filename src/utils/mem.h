/**
 * DBEC-GP codes are developed by:
 *
 * R. Kishor Kumar, Paulsamy Muruganandam
 * (Bharathidasan University, Tamil Nadu, India)
 *
 * Luis E. Young, Sadhan K. Adhikari
 * (UNESP - Sao Paulo State University, Brazil)
 *
 * Dusan Vudragovic, Antun Balaz
 * (Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)
 *
 * Public use and modification of this code are allowed provided that the
 * following paper is cited:
 * R. Kishor Kumar et al., Comput. Phys. Commun. NN (2014) NNNN.
 *
 * The authors would be grateful for all information and/or comments
 * regarding the use of the code.
 */

#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <fftw3.h>

double *alloc_double_vector(long);
double complex *alloc_complex_vector(long);
double **alloc_double_matrix(long, long);
double complex **alloc_complex_matrix(long, long);
double ***alloc_double_tensor(long, long, long);
double complex ***alloc_complex_tensor(long, long, long);
fftw_complex *alloc_fftw_complex_vector(long);

void free_double_vector(double *);
void free_complex_vector(double complex *);
void free_double_matrix(double **);
void free_complex_matrix(double complex **);
void free_double_tensor(double ***);
void free_complex_tensor(double complex ***);
void free_fftw_complex_vector(fftw_complex *);
