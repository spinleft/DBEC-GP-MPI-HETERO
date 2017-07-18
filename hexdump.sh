#!/bin/bash
# DBEC-GP-OMP-CUDA-MPI programs are developed by:
#
# Vladimir Loncar, Antun Balaz
# (Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)
#
# Srdjan Skrbic
# (Department of Mathematics and Informatics, Faculty of Sciences, University of Novi Sad, Serbia)
#
# Paulsamy Muruganandam
# (Bharathidasan University, Tamil Nadu, India)
#
# Luis E. Young-S, Sadhan K. Adhikari
# (UNESP - Sao Paulo State University, Brazil)
#
#
# Public use and modification of these codes are allowed provided that the
# following papers are cited:
# [1] V. Loncar et al., Comput. Phys. Commun. 209 (2016) 190.      
# [2] V. Loncar et al., Comput. Phys. Commun. 200 (2016) 406.      
# [3] R. Kishor Kumar et al., Comput. Phys. Commun. 195 (2015) 117.
#
# The authors would be grateful for all information and/or comments
# regarding the use of the programs.

# This script will convert binary outputs of DBEC-GP-MPI and DBEC-GP-MPI-CUDA
# programs to textual format and removee the original binary files.

# Place the script in the directory with the output files, modify the PREFIX
# variable to match the prefix of the output files, and execute the script.

PREFIX=real3d-den

hexdump -v -e '1/8 "%8e" " %8e\n"' ${PREFIX}1d_x.bin > ${PREFIX}1d_x.txt && sed -i 's/,/./g' ${PREFIX}1d_x.txt
hexdump -v -e '1/8 "%8e" " %8e\n"' ${PREFIX}1d_y.bin > ${PREFIX}1d_y.txt && sed -i 's/,/./g' ${PREFIX}1d_y.txt
hexdump -v -e '1/8 "%8e" " %8e\n"' ${PREFIX}1d_z.bin > ${PREFIX}1d_z.txt && sed -i 's/,/./g' ${PREFIX}1d_z.txt

hexdump -v -e '1/8 "%8e" " %8e" " %8e\n"' ${PREFIX}2d_xy.bin > ${PREFIX}2d_xy.txt && sed -i 's/,/./g' ${PREFIX}2d_xy.txt
hexdump -v -e '1/8 "%8e" " %8e" " %8e\n"' ${PREFIX}2d_xz.bin > ${PREFIX}2d_xz.txt && sed -i 's/,/./g' ${PREFIX}2d_xz.txt
hexdump -v -e '1/8 "%8e" " %8e" " %8e\n"' ${PREFIX}2d_yz.bin > ${PREFIX}2d_yz.txt && sed -i 's/,/./g' ${PREFIX}2d_yz.txt

hexdump -v -e '1/8 "%8e" " %8e" " %8e\n"' ${PREFIX}3d_x0z.bin > ${PREFIX}3d_x0z.txt && sed -i 's/,/./g' ${PREFIX}3d_x0z.txt
hexdump -v -e '1/8 "%8e" " %8e" " %8e\n"' ${PREFIX}3d_xy0.bin > ${PREFIX}3d_xy0.txt && sed -i 's/,/./g' ${PREFIX}3d_xy0.txt
hexdump -v -e '1/8 "%8e" " %8e" " %8e\n"' ${PREFIX}3d_0yz.bin > ${PREFIX}3d_0yz.txt && sed -i 's/,/./g' ${PREFIX}3d_0yz.txt

hexdump -v -e '1/8 "%8e\n"' ${PREFIX}.bin > ${PREFIX}.txt && sed -i 's/,/./g' ${PREFIX}.txt

# Comment out this line if you want to preserve binary outputs
rm ${PREFIX}*.bin
