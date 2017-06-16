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
#include <math.h>

#define GAULEG_EPS   1.e-12

double simpint(double, double *, long);
void diff(double, double *, double *, long);
void gauleg(double, double, double *, double *, long);
