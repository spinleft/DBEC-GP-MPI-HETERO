## Instructions on compiling and installing FFTW library for the use with DBEC-GP-MPI-HETERO programs

These instructions should be followed only if FFTW is not installed on the target computer where you plan to run DBEC-GP-MPI-HETERO programs. If FFTW is installed on the target computer and you plan to use it for data transposes, make sure that it has support for MPI enabled, as MPI support is not enabled by default. Look for `fftw3-mpi.h` in the installation folder. MPI is often installed in a non-standard location, especially on computer clusters. In this case, location of the library and MPI header files need to be passed to `./configure` script. See [here](http://www.fftw.org/doc/FFTW-MPI-Installation.html) for details.

1. Download the latest source of [FFTW](http://www.fftw.org/)

2. Unpack the source tarball and go to the corresponding directory

3. a) If you plan to use GNU compiler suite, proceed by typing the following commands:

        export CC=gcc
        export CFLAGS=-O3
        export F77=gfortran
        export FFLAGS=-O3
        AR=ar
        ./configure --prefix=$HOME/fftw/gcc --enable-shared --enable-static --enable-fma --enable-sse2 --enable-openmp --enable-threads --enable-mpi
        make
        make install
   This will compile and install FFTW library (including its OpenMP-threaded version with MPI support) into: `$HOME/fftw/gcc`  
   b) If you also plan to use Intel compiler suite, proceed by typing the following commands:

        export CC=icc
        export CFLAGS="-xHOST -O3 -ipo -no-prec-div"
        export AR=xiar
        ./configure --prefix=$HOME/fftw/icc --enable-shared --disable-static --enable-fma --enable-sse2 --enable-openmp --enable-threads --enable-mpi
        make
        make install
   or

        export CC=icc
        export CFLAGS=-fast
        export AR=xiar
        ./configure --prefix=$HOME/fftw/icc --disable-shared --enable-static --enable-fma --enable-sse2 --enable-openmp --enable-threads --enable-mpi
        make
        make install
   This will compile and install FFTW library (including its OpenMP-threaded version with MPI support) into: `$HOME/fftw/icc`. Note that step 3b) does not interfere with step 3a), and you can perform them in arbitrary order.

4. When compiling the dipolar programs, in order to use one of the above-built FFTW libraries, you need to
adjust properly the variable `FFTW_PATH` in the provided `makefile` so that it has the correct value.
