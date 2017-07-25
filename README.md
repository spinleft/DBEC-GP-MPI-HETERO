# DBEC-GP-MPI-HETERO

**DBEC-GP-MPI-HETERO** is a set of hybrid OpenMP/CUDA/MPI programs that solve the time-(in)dependent Gross-Pitaevskii nonlinear partial differential equation for BECs with contact and dipolar interaction in three spatial dimensions in a trap using imaginary-time and real-time propagation. The Gross-Pitaevskii equation describes the properties of a dilute trapped Bose-Einstein condensate. The equation is solved using the split-step Crank-Nicolson method by discretizing space and time, as described in Ref. [R1]. The discretized equation is then propagated in imaginary or real time over small time steps. Additional details, for the case of pure contact interaction, are given in Refs. [R2-R5].

[R1] [R. Kishor Kumar et al., Comput. Phys. Commun. 195 (2015) 117.](https://doi.org/10.1016/j.cpc.2015.03.024)  
[R2] [P. Muruganandam and S. K. Adhikari, Comput. Phys. Commun. 180 (2009) 1888.](https://doi.org/10.1016/j.cpc.2009.04.015)  
[R3] [D. Vudragovic et al., Comput. Phys. Commun. 183 (2012) 2021.](https://doi.org/10.1016/j.cpc.2012.03.022)  
[R4] [B. Sataric et al., Comput. Phys. Commun. 200 (2016) 411.](https://doi.org/10.1016/j.cpc.2015.12.006)  
[R5] [L. E. Young-S. et al., Comput. Phys. Commun. 204 (2016) 209.](https://doi.org/10.1016/j.cpc.2016.03.015)

## Description of DBEC-GP-MPI-HETERO code distribution

### I) Source codes

Programs are written in C programming language with OpenMP parallelization on host and CUDA on the GPU device. MPI is used for communication between processes running on different nodes. Sources are located in the [src](src/) folder, which has the following structure:

 - [src/imag3d-mpi-hetero](src/imag3d-mpi-hetero/) program solves the imaginary-time dipolar Gross-Pitaevskii equation in three spatial dimensions in an anisotropic harmonic trap.
 - [src/real3d-mpi-hetero](src/real3d-mpi-hetero/) program solves the real-time dipolar Gross-Pitaevskii equation in three spatial dimensions in an anisotropic harmonic trap.
 - [src/utils](src/utils/) provides utility functions for transposing data, parsing of configuration files, integration and differentiation, as well as allocation/deallocation of memory.

### II) Input parameters

For each program, a specific parameter input file has to be given on the command line. Examples of input files with characteristic set of options and their detailed descriptions are provided in the [input](input/) folder.

### III) Examples of matching outputs

The [output](output/) folder contains examples of matching outputs for all programs and default inputs available in the DBEC-GP-MPI-HETERO distribution. Some large density files are omitted to save space.

### IV) Compilation

*(In case FFTW library with OpenMP and MPI support is not installed, please follow the
instructions given in the file [readme-fftw.md](readme-fftw.md))*

Programs are compiled via a provided `makefile`.

The use of the makefile:

    make <target> [compiler=icc] [transpose=fftw]

where possible targets are:

    all, clean

as well as program-specific targets, which compile only a specified program:

    imag3d-mpi-hetero, real3d-mpi-hetero

The provided makefile allows compilation of the DBEC-GP-MPI-HETERO programs, which rely on `mpicc` compiler being installed on the system. The `mpicc` compiler is provided with the MPI implementation and does not have to be installed separately. The special `compiler=icc` flag indicates that `mpicc` relies on Intel's C compiler, and should be used if MPI itself was compiled with it.

Programs can perform data transposes in two ways: via built-in transpose routine, or via FFTW transpose interface. If FFTW is used for data transposes, additional `transpose=fftw` parameter should be passed to `make` command. If the built-in transpose routine is used, FFTW library without MPI support may be used.

As already stressed, FFTW library with OpenMP (and optionally MPI) support has to be installed on the target cluster. Before attempting compilation, make sure that the variable `FFTW_PATH` is correctly set in the makefile. Current value corresponds to instructions in the file [readme-fftw.md](readme-fftw.md).

**Examples of compiling:**

1. Compile all DBEC-GP-MPI-HETERO programs:

        make all

2. Compile only imag3d-mpi-hetero DBEC-GP-MPI-HETERO program:

        make imag3d-mpi-hetero

3. Compile only `imag3d-mpi-hetero` with data transpose routines from FFTW:

        make imag3d-mpi-hetero transpose=fftw

### V) Running the compiled programs

To run any of the OpenMP/MPI programs compiled with the make command, you need to use the
syntax:

    mpiexec -np <nprocs> ./<programname> -i <parameterfile>

where `<nprocs>` is the number of MPI processes invoked, `<programname>` is a name of the compiled executable, and `<parameterfile>` is a parameter input file prepared by the user. Examples of parameter input files are described in section II above, and are provided in the folder [input](input/). Matching output of the principal output files are given in folder [output](output/); very large density output files are omitted.

**Example of running a program:**

Run `imag3d-mpi-hetero` compiled program with the parameter input file `input/imag3d-input`, in 4 parallel processes:

    mpiexec -np 4 ./imag3d-mpi-hetero -i input/imag3d-input

**Important:**

You have to ensure that each MPI process is executed on a separate cluster node, since it will use all available CPU cores on it by launching as many OpenMP threads. Usually, this is managed through the submission file to the batch system on the cluster. For example, if OpenMPI is used, this can be achieved by the switches `--map-by ppr:1:node --bind-to none`, and the above example is usually executed as:

    mpiexec -np 4 --map-by ppr:1:node --bind-to none ./imag3d-mpi -i input/imag3d-input

For other implementations of MPI you should consult their manual pages, as well as user guide for your cluster.

### VI) Authors

**DBEC-GP-MPI-HETERO** programs are developed by:

Vladimir Lončar, Antun Balaž *(Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)*  
Srđan Skrbić *(Department of Mathematics and Informatics, Faculty of Sciences, University of Novi Sad, Serbia)*  
Paulsamy Muruganandam *(Bharathidasan University, Tamil Nadu, India)*  
Luis E. Young-S, Sadhan K. Adhikari *(UNESP - Sao Paulo State University, Brazil)*  

Public use and modification of these codes are allowed provided that the following papers are cited:  
[1] [V. Loncar et al., Comput. Phys. Commun. 209 (2016) 190.](https://doi.org/10.1016/j.cpc.2016.07.029)  
[2] [V. Loncar et al., Comput. Phys. Commun. 200 (2016) 406.](https://doi.org/10.1016/j.cpc.2015.11.014)  
[3] [R. Kishor Kumar et al., Comput. Phys. Commun. 195 (2015) 117.](https://doi.org/10.1016/j.cpc.2015.03.024)

The authors would be grateful for all information and/or comments regarding the use of the programs.

### VII) Licence

**DBEC-GP-MPI-HETERO** code distribution is licensed under Apache License, Version 2.0. See [LICENCE](LICENCE) for details.