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

#include "diffint.h"

/**
 *    Spatial 1D integration with Simpson's rule.
 *    h - space step
 *    f - array with the function values
 *    N - number of integration points
 */
double simpint(double h, double *f, long N) {
   int c;
   long cnti;
   double sum;

   sum = f[0];
   for (cnti = 1; cnti < N - 1; cnti ++) {
      c = 2 + 2 * (cnti % 2);
      sum += c * f[cnti];
   }
   sum += f[N - 1];

   return sum * h / 3.;
}

/**
 *    Richardson extrapolation formula for calculation of space
 *    derivatives.
 *    h  - space step
 *    f  - array with the function values
 *    df - array with the first derivatives of the function
 *    N  - number of space mesh points
 */
void diff(double h, double *f, double *df, long N) {
   long cnti;

   df[0] = 0.;
   df[1] = (f[2] - f[0]) / (2. * h);

   for (cnti = 2; cnti < N - 2; cnti ++) {
      df[cnti] = (f[cnti - 2] - 8. * f[cnti - 1] + 8. * f[cnti + 1] - f[cnti + 2]) / (12. * h);
   }

   df[N - 2] = (f[N - 1] - f[N - 3]) / (2. * h);
   df[N - 1] = 0.;

   return;
}

void gauleg(double x1, double x2, double *x, double *w, long N) {
   long m, j, i;
   double z1, z, xm, xl, pp, p3, p2, p1;

   m = (N + 1) / 2;
   xm = 0.5 * (x2 + x1);
   xl = 0.5 * (x2 - x1);
   for (i = 1; i <= m; i ++) {
      z = cos(4. * atan(1.) * (i - 0.25) / (N + 0.5));
      do {
         p1 = 1.;
         p2 = 0.;
         for (j = 1; j <= N; j ++) {
            p3 = p2;
            p2 = p1;
            p1 = ((2. * j - 1.) * z * p2 - (j - 1.) * p3) / j;
         }
         pp = N * (z * p1 - p2) / (z * z - 1.);
         z1 = z;
         z = z1 - p1 / pp;
      } while (fabs(z - z1) > GAULEG_EPS);
      x[i] = xm - xl * z;
      x[N + 1 - i] = xm + xl * z;
      w[i] = 2. * xl / ((1. - z * z) * pp * pp);
      w[N + 1 - i] = w[i];
   }

   return;
}
