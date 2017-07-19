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
