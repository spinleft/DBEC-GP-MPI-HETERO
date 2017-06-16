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
#include <string.h>

int cfg_size;
char cfg_key[256][256], cfg_val[256][256];

int cfg_init(char *);
char *cfg_read(char *);
