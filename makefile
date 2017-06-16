#
# DBEC-GP codes are developed by:
#
# R. Kishor Kumar, Paulsamy Muruganandam
# (Bharathidasan University, Tamil Nadu, India)
#
# Luis E. Young, Sadhan K. Adhikari
# (UNESP - Sao Paulo State University, Brazil)
#
# Dusan Vudragovic, Antun Balaz
# (Scientific Computing Laboratory, Institute of Physics Belgrade, Serbia)
#
# Public use and modification of this code are allowed provided that the
# following paper is cited:
# R. Kishor Kumar et al., Comput. Phys. Commun. NN (2014) NNNN.
#
# The authors would be grateful for all information and/or comments
# regarding the use of the code.
#

CC = mpicc
FFTW_PATH=/usr
FFTW_INCS=-I$(FFTW_PATH)/include
FFTW_LIBS=-L$(FFTW_PATH)/lib
CFLAGS = -O3 -Wall
OMPFLAGS = -fopenmp
CLIBS = -lfftw3 -lm $(FFTW_LIBS) -lcudart
OMPLIBS = -lfftw3_omp

NVCC = nvcc
NVCCFLAGS = -m64 -O3 -D_FORCE_INLINES

CUDA_PATH = /usr/local/cuda
CUDA_INCS = -I$(CUDA_PATH)/include
CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcufft

TRANSPOSE =

ifeq ($(transpose), fftw)
	TRANSPOSE = -DFFTW_TRAN
	CLIBS := -lfftw3_mpi $(CLIBS)
endif

TARGETS=imag3d real3d

CPU_OBJECTS=$(TARGETS:=-mpi-cpu.o)
GPU_OBJECTS=$(TARGETS:=-mpi-gpu.o)

UTIL_TARGETS=diffint mem cfg tran
UTIL_OBJECTS=$(UTIL_TARGETS:=.o)

CUDA_UTIL_TARGETS=cudamem
CUDA_UTIL_OBJECTS=$(CUDA_UTIL_TARGETS:=.o)

all: $(TARGETS)
	rm -rf $(CPU_OBJECTS) $(GPU_OBJECTS) $(UTIL_OBJECTS) $(CUDA_UTIL_OBJECTS)

$(TARGETS): % : $(UTIL_OBJECTS) $(CUDA_UTIL_OBJECTS) %-mpi-cpu.o %-mpi-gpu.o
	$(CC) $(CFLAGS) $(OMPFLAGS) $(FFTW_INCS) $(CUDA_INCS) -o $@-mpi-hetero $@-mpi-cpu.o $@-mpi-gpu.o $(UTIL_OBJECTS) $(CUDA_UTIL_OBJECTS) $(CUDA_LIBS) $(CLIBS) $(OMPLIBS)

$(CPU_OBJECTS): %.o:
	$(CC) $(CFLAGS) $(OMPFLAGS) $(FFTW_INCS) $(CUDA_INCS) $(TRANSPOSE) -c src/$(@:cpu.o=hetero)/$*.c -o $@

$(GPU_OBJECTS): %.o:
	$(NVCC) $(NVCCFLAGS) $(FFTW_INCS) $(CUDA_INCS) -c src/$(@:gpu.o=hetero)/$*.cu -o $@

$(UTIL_OBJECTS): %.o:
	$(CC) $(CFLAGS) $(FFTW_INCS) $(CUDA_INCS) -c src/utils/$*.c -o $@

$(CUDA_UTIL_OBJECTS): %.o:
	$(NVCC) $(NVCCFLAGS) $(FFTW_INCS) $(CUDA_INCS) -c src/utils/$*.cu -o $@

clean:
	rm -rf $(TARGETS:=-mpi-hetero) $(CPU_OBJECTS) $(GPU_OBJECTS) $(UTIL_OBJECTS) $(CUDA_UTIL_OBJECTS)
