extern "C" {
#include "real3d-mpi-gpu.h"
}

void init_gpu() {
   int cnti, cntj;
   int Nce = 0;
   int chunksMax = 0;
   size_t free, total;

   chunksMax = MAX(chunksMax, chunksNx_fft);
   chunksMax = MAX(chunksMax, chunksNx_nyz);
   chunksMax = MAX(chunksMax, chunksNy_fft);
   chunksMax = MAX(chunksMax, chunksNy_lux);

   gpu = (gpu_t *) calloc(Ngpu, sizeof(struct gpu_t));

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      cudaCheckError(cudaMemGetInfo(&free, &total));
      free *= (0.8 / nprocs);

      init_mem_device(free);

      cudaCheckError(cudaEventCreate(&(gpu[cnti].syncEvent)));

      gpu[cnti].exec_stream = (cudaStream_t *) calloc(chunksMax, sizeof(cudaStream_t));
      for (cntj = 0; cntj < chunksMax; cntj ++) {
         cudaCheckError(cudaStreamCreate(&(gpu[cnti].exec_stream[cntj])));
      }
      gpu[cnti].nstreams = chunksMax;

      cudaCheckError(cudaMemcpyToSymbol(d_Nx, &Nx, sizeof(Nx)));
      cudaCheckError(cudaMemcpyToSymbol(d_Ny, &Ny, sizeof(Ny)));
      cudaCheckError(cudaMemcpyToSymbol(d_Nz, &Nz, sizeof(Nz)));

      cudaCheckError(cudaMemcpyToSymbol(d_dt, &dt, sizeof(dt)));

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, cnti);

      Nce = MAX(Nce, deviceProp.asyncEngineCount);

      /*switch (Nce) {
      case 0:
         printf("WARN: Device %d cannot execute a kernel and concurrently copy memory between host and device.\n", cnti);
         break;
      case 1:
         printf("INFO: Device %d can execute a kernel and concurrently copy memory between host and device.\n", cnti);
         break;
      case 2:
         printf("INFO: Device %d can execute a kernel and concurrently copy memory between host and device in both directions.\n", cnti);
         break;
      }*/
   }

   init_gpux_fft();
   init_gpux_nyz();
   init_gpuy_fft();
   init_gpuy_lux();

   switch (Nce) {
   case 0:
   case 1:
      calcpsidd2_gpu_p1 = &calcpsidd2_gpu_p1_1ce;
      calcpsidd2_gpu_p2 = &calcpsidd2_gpu_p2_1ce;
      calcpsidd2_gpu_p3 = &calcpsidd2_gpu_p3_1ce;
      calcnu_gpu = &calcnu_gpu_1ce;
      calclux_gpu = &calclux_gpu_1ce;
      calcluy_gpu = &calcluy_gpu_1ce;
      calcluz_gpu = &calcluz_gpu_1ce;
      break;
   case 2:
      calcpsidd2_gpu_p1 = &calcpsidd2_gpu_p1_2ce;
      calcpsidd2_gpu_p2 = &calcpsidd2_gpu_p2_2ce;
      calcpsidd2_gpu_p3 = &calcpsidd2_gpu_p3_2ce;
      calcnu_gpu = &calcnu_gpu_2ce;
      calclux_gpu = &calclux_gpu_2ce;
      calcluy_gpu = &calcluy_gpu_2ce;
      calcluz_gpu = &calcluz_gpu_2ce;
      break;
   }
}

void init_gpux_fft() {
   int cnti, cntj;
   int Nsm;
   size_t pitchX, pitchXPadded;
   long ptrSize, ptrSizePadded;
   cudaPitchedPtr tmpPtr;
   size_t worksize_fr[chunksNx_fft];
   void *fft_workspace[chunksNx_fft];

   chunkNx_fft = gpuNx_fft / chunksNx_fft / Ngpu;

   // psix    (chunkNx_fft, Ny, Nz) // complex
   // psidd2x (chunkNx_fft, Ny, Nz2+1) // complex

   tmpPtr = alloc_complex_tensor_device(chunkNx_fft, Ny, Nz);
   pitchX = tmpPtr.pitch;
   ptrSize = chunksNx_fft * chunkNx_fft * Ny * pitchX;
   free_complex_tensor_device(tmpPtr);

   tmpPtr = alloc_complex_tensor_device(chunkNx_fft, Ny, Nz2 + 1);
   pitchXPadded = tmpPtr.pitch;
   ptrSizePadded = chunksNx_fft * chunkNx_fft * Ny * pitchXPadded;
   free_complex_tensor_device(tmpPtr);

   gpux_fft = (gpux_t *) calloc(Ngpu, sizeof(struct gpux_t));

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      long d_offsetNx = cnti * (gpuNx_fft / Ngpu);

      reset_mem_device();
      gpux_fft[cnti].ptr1 = (char *) alloc_mem_device(ptrSize);
      gpux_fft[cnti].ptr2 = (char *) alloc_mem_device(ptrSizePadded);

      cudaCheckError(cudaMemcpyToSymbol(d_chunkNx_fft, &chunkNx_fft, sizeof(chunkNx_fft)));

      gpux_fft[cnti].psix = (cudaPitchedPtr *) calloc(chunksNx_fft, sizeof(cudaPitchedPtr));
      gpux_fft[cnti].psidd2x = (cudaPitchedPtr *) calloc(chunksNx_fft, sizeof(cudaPitchedPtr));

      gpux_fft[cnti].psix_h2d = (cudaMemcpy3DParms *) calloc(chunksNx_fft, sizeof(cudaMemcpy3DParms));

      gpux_fft[cnti].psidd2fftx_d2h = (cudaMemcpy3DParms *) calloc(chunksNx_fft, sizeof(cudaMemcpy3DParms));

      gpux_fft[cnti].plan_forward_row = (cufftHandle *) calloc(chunksNx_fft, sizeof(cufftHandle));

      for (cntj = 0; cntj < chunksNx_fft; cntj ++) {
         int offsetNx = cntj * chunkNx_fft;

         gpux_fft[cnti].psix[cntj] = make_cudaPitchedPtr(gpux_fft[cnti].ptr1 + (offsetNx * Ny * pitchX), pitchX, Nz, Ny);
         gpux_fft[cnti].psidd2x[cntj] = make_cudaPitchedPtr(gpux_fft[cnti].ptr2 + (offsetNx * Ny * pitchXPadded), pitchXPadded, Nz, Ny);

         gpux_fft[cnti].psix_h2d[cntj].srcPtr = make_cudaPitchedPtr(psi[cpuNx_fft + d_offsetNx + offsetNx][0], Nz * sizeof(cuDoubleComplex), Nz, Ny);
         gpux_fft[cnti].psix_h2d[cntj].dstPtr = gpux_fft[cnti].psix[cntj];
         gpux_fft[cnti].psix_h2d[cntj].extent = make_cudaExtent(Nz * sizeof(cuDoubleComplex), Ny, chunkNx_fft);
         gpux_fft[cnti].psix_h2d[cntj].kind = cudaMemcpyHostToDevice;

         gpux_fft[cnti].psidd2fftx_d2h[cntj].srcPtr = gpux_fft[cnti].psidd2x[cntj];
         gpux_fft[cnti].psidd2fftx_d2h[cntj].dstPtr = make_cudaPitchedPtr(psidd2fft + ((cpuNx_fft + d_offsetNx + offsetNx) * Ny * (Nz2 + 1)), (Nz2 + 1) * sizeof(fftw_complex), Nz2 + 1, Ny);
         gpux_fft[cnti].psidd2fftx_d2h[cntj].extent = make_cudaExtent((Nz2 + 1) * sizeof(fftw_complex), Ny, chunkNx_fft);
         gpux_fft[cnti].psidd2fftx_d2h[cntj].kind = cudaMemcpyDeviceToHost;
      }

      for (cntj = 0; cntj < chunksNx_fft; cntj ++) {
         int fft_rank = 2;
         int n_fr[] = {Ny, Nz};
         int howmany = chunkNx_fft;
         int idist = Ny * (gpux_fft[cnti].psidd2x[cntj].pitch / sizeof(cufftDoubleReal)); int odist = Ny * (gpux_fft[cnti].psidd2x[cntj].pitch / sizeof(cufftDoubleComplex));
         int istride = 1; int ostride = 1;
         int inembed_fr[] = { Ny, gpux_fft[cnti].psidd2x[cntj].pitch / sizeof(cufftDoubleReal) }; int onembed_fr[] = { Ny, gpux_fft[cnti].psidd2x[cntj].pitch / sizeof(cufftDoubleComplex) };

         cudaCheckFFTError(cufftCreate(&(gpux_fft[cnti].plan_forward_row[cntj])));
         cudaCheckFFTError(cufftSetAutoAllocation(gpux_fft[cnti].plan_forward_row[cntj], 0));
         cudaCheckFFTError(cufftSetStream(gpux_fft[cnti].plan_forward_row[cntj], gpu[cnti].exec_stream[cntj]));
         cudaCheckFFTError(cufftMakePlanMany(gpux_fft[cnti].plan_forward_row[cntj], fft_rank, n_fr, inembed_fr, istride, idist, onembed_fr, ostride, odist, CUFFT_D2Z, howmany, &(worksize_fr[cntj])));
      }

      for (cntj = 0; cntj < chunksNx_fft; cntj ++) {
         fft_workspace[cntj] = alloc_mem_device(worksize_fr[cntj]);
         cudaCheckFFTError(cufftSetWorkArea(gpux_fft[cnti].plan_forward_row[cntj], fft_workspace[cntj]));
      }

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, cnti);

      Nsm = deviceProp.multiProcessorCount;

      gpux_fft[cnti].dimBlock3d_dd2.x = block3d_dd2[0]; gpux_fft[cnti].dimBlock3d_dd2.y = block3d_dd2[1]; gpux_fft[cnti].dimBlock3d_dd2.z = block3d_dd2[2];
      gpux_fft[cnti].dimGrid3d_dd2.x = gpux_fft[cnti].dimGrid3d_dd2.y = gpux_fft[cnti].dimGrid3d_dd2.z = Nsm * grid3d_dd2;
   }
}

void init_gpux_nyz() {
   int cnti, cntj;
   int Nsm;
   size_t pitchX, pitchXPadded;
   long ptrSize, ptrSizePadded;
   cudaPitchedPtr tmpPtr;
   cuDoubleComplex minusAy, minusAz;
   size_t worksize_br[chunksNx_nyz];
   void *fft_workspace[chunksNx_nyz];

   chunkNx_nyz = gpuNx_nyz / chunksNx_nyz / Ngpu;

   // psix    (chunkNx, Ny, Nz) // complex
   // pot     (chunkNx, Ny, Nz)
   // psidd2x (chunkNx, Ny, Nz2+1) // complex
   // cbetax  (chunkNx, Ny, Nz) // complex

   tmpPtr = alloc_complex_tensor_device(chunkNx_nyz, Ny, Nz);
   pitchX = tmpPtr.pitch;
   ptrSize = chunksNx_nyz * chunkNx_nyz * Ny * pitchX;
   free_complex_tensor_device(tmpPtr);

   tmpPtr = alloc_complex_tensor_device(chunkNx_nyz, Ny, Nz2 + 1);
   pitchXPadded = tmpPtr.pitch;
   ptrSizePadded = chunksNx_nyz * chunkNx_nyz * Ny * pitchXPadded;
   free_complex_tensor_device(tmpPtr);

   minusAy = make_cuDoubleComplex(0., - Ay.y);
   minusAz = make_cuDoubleComplex(0., - Az.y);

   gpux_nyz = (gpux_t *) calloc(Ngpu, sizeof(struct gpux_t));

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      long d_offsetNx = cnti * (gpuNx_nyz / Ngpu);

      reset_mem_device();
      gpux_nyz[cnti].ptr1 = (char *) alloc_mem_device(ptrSize);
      gpux_nyz[cnti].ptr2 = (char *) alloc_mem_device(ptrSize);
      gpux_nyz[cnti].ptr3 = (char *) alloc_mem_device(MAX(ptrSize, ptrSizePadded));

      gpux_nyz[cnti].calphay = alloc_complex_vector_device(Ny - 1);
      gpux_nyz[cnti].calphaz = alloc_complex_vector_device(Nz - 1);
      gpux_nyz[cnti].cgammay = alloc_complex_vector_device(Ny - 1);
      gpux_nyz[cnti].cgammaz = alloc_complex_vector_device(Nz - 1);

      gpux_nyz[cnti].psidd2tmp = alloc_double_tensor_device(1, Ny, Nz);
      memset(&(gpux_nyz[cnti].psidd2tmp_h2d), 0, sizeof(cudaMemcpy3DParms));
      gpux_nyz[cnti].psidd2tmp_h2d.srcPtr = make_cudaPitchedPtr(psidd2tmp, Nz * sizeof(double), Nz, Ny);
      gpux_nyz[cnti].psidd2tmp_h2d.dstPtr = gpux_nyz[cnti].psidd2tmp;
      gpux_nyz[cnti].psidd2tmp_h2d.extent = make_cudaExtent(Nz * sizeof(double), Ny, 1);
      gpux_nyz[cnti].psidd2tmp_h2d.kind = cudaMemcpyHostToDevice;

      cudaCheckError(cudaMemcpyToSymbol(d_chunkNx_nyz, &chunkNx_nyz, sizeof(chunkNx_nyz)));

      cudaCheckError(cudaMemcpyToSymbol(d_Ay0r, &Ay0r, sizeof(Ay0r)));
      cudaCheckError(cudaMemcpyToSymbol(d_Ay, &Ay, sizeof(Ay)));
      cudaCheckError(cudaMemcpyToSymbol(d_minusAy, &minusAy, sizeof(minusAy)));

      cudaCheckError(cudaMemcpyToSymbol(d_Az0r, &Az0r, sizeof(Az0r)));
      cudaCheckError(cudaMemcpyToSymbol(d_Az, &Az, sizeof(Az)));
      cudaCheckError(cudaMemcpyToSymbol(d_minusAz, &minusAz, sizeof(minusAz)));

      cudaCheckError(cudaMemcpy(gpux_nyz[cnti].calphay, calphay, (Ny - 1) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(gpux_nyz[cnti].calphaz, calphaz, (Nz - 1) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(gpux_nyz[cnti].cgammay, cgammay, (Ny - 1) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(gpux_nyz[cnti].cgammaz, cgammaz, (Nz - 1) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

      gpux_nyz[cnti].psix = (cudaPitchedPtr *) calloc(chunksNx_nyz, sizeof(cudaPitchedPtr));
      gpux_nyz[cnti].pot = (cudaPitchedPtr *) calloc(chunksNx_nyz, sizeof(cudaPitchedPtr));
      gpux_nyz[cnti].psidd2x = (cudaPitchedPtr *) calloc(chunksNx_nyz, sizeof(cudaPitchedPtr));
      gpux_nyz[cnti].cbetax = (cudaPitchedPtr *) calloc(chunksNx_nyz, sizeof(cudaPitchedPtr));

      gpux_nyz[cnti].psix_h2d = (cudaMemcpy3DParms *) calloc(chunksNx_nyz, sizeof(cudaMemcpy3DParms));
      gpux_nyz[cnti].psix_d2h = (cudaMemcpy3DParms *) calloc(chunksNx_nyz, sizeof(cudaMemcpy3DParms));

      gpux_nyz[cnti].pot_h2d = (cudaMemcpy3DParms *) calloc(chunksNx_nyz, sizeof(cudaMemcpy3DParms));

      gpux_nyz[cnti].psidd2_d2h = (cudaMemcpy3DParms *) calloc(chunksNx_nyz, sizeof(cudaMemcpy3DParms));

      gpux_nyz[cnti].psidd2fftx_h2d = (cudaMemcpy3DParms *) calloc(chunksNx_nyz, sizeof(cudaMemcpy3DParms));

      gpux_nyz[cnti].plan_backward_row = (cufftHandle *) calloc(chunksNx_nyz, sizeof(cufftHandle));

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         int offsetNx = cntj * chunkNx_nyz;

         gpux_nyz[cnti].psix[cntj] = make_cudaPitchedPtr(gpux_nyz[cnti].ptr1 + (offsetNx * Ny * pitchX), pitchX, Nz, Ny);
         gpux_nyz[cnti].pot[cntj] = make_cudaPitchedPtr(gpux_nyz[cnti].ptr2 + (offsetNx * Ny * pitchX), pitchX, Nz, Ny);
         gpux_nyz[cnti].psidd2x[cntj] = make_cudaPitchedPtr(gpux_nyz[cnti].ptr3 + (offsetNx * Ny * pitchXPadded), pitchXPadded, Nz, Ny);
         gpux_nyz[cnti].cbetax[cntj] = make_cudaPitchedPtr(gpux_nyz[cnti].ptr3 + (offsetNx * Ny * pitchX), pitchX, Nz, Ny);

         gpux_nyz[cnti].psix_h2d[cntj].srcPtr = make_cudaPitchedPtr(psi[cpuNx_nyz + d_offsetNx + offsetNx][0], Nz * sizeof(cuDoubleComplex), Nz, Ny);
         gpux_nyz[cnti].psix_h2d[cntj].dstPtr = gpux_nyz[cnti].psix[cntj];
         gpux_nyz[cnti].psix_h2d[cntj].extent = make_cudaExtent(Nz * sizeof(cuDoubleComplex), Ny, chunkNx_nyz);
         gpux_nyz[cnti].psix_h2d[cntj].kind = cudaMemcpyHostToDevice;

         gpux_nyz[cnti].psix_d2h[cntj].srcPtr = gpux_nyz[cnti].psix[cntj];
         gpux_nyz[cnti].psix_d2h[cntj].dstPtr = make_cudaPitchedPtr(psi[cpuNx_nyz + d_offsetNx + offsetNx][0], Nz * sizeof(cuDoubleComplex), Nz, Ny);
         gpux_nyz[cnti].psix_d2h[cntj].extent = make_cudaExtent(Nz * sizeof(cuDoubleComplex), Ny, chunkNx_nyz);
         gpux_nyz[cnti].psix_d2h[cntj].kind = cudaMemcpyDeviceToHost;

         gpux_nyz[cnti].pot_h2d[cntj].srcPtr = make_cudaPitchedPtr(pot[cpuNx_nyz + d_offsetNx + offsetNx][0], Nz * sizeof(double), Nz, Ny);
         gpux_nyz[cnti].pot_h2d[cntj].dstPtr = gpux_nyz[cnti].pot[cntj];
         gpux_nyz[cnti].pot_h2d[cntj].extent = make_cudaExtent(Nz * sizeof(double), Ny, chunkNx_nyz);
         gpux_nyz[cnti].pot_h2d[cntj].kind = cudaMemcpyHostToDevice;

         gpux_nyz[cnti].psidd2_d2h[cntj].srcPtr = gpux_nyz[cnti].psidd2x[cntj];
         gpux_nyz[cnti].psidd2_d2h[cntj].dstPtr = make_cudaPitchedPtr(psidd2[cpuNx_nyz + d_offsetNx + offsetNx][0], Nz * sizeof(double), Nz, Ny);
         gpux_nyz[cnti].psidd2_d2h[cntj].extent = make_cudaExtent(Nz * sizeof(double), Ny, chunkNx_nyz);
         gpux_nyz[cnti].psidd2_d2h[cntj].kind = cudaMemcpyDeviceToHost;

         gpux_nyz[cnti].psidd2fftx_h2d[cntj].srcPtr = make_cudaPitchedPtr(psidd2fft + ((cpuNx_nyz + d_offsetNx + offsetNx) * Ny * (Nz2 + 1)), (Nz2 + 1) * sizeof(fftw_complex), Nz2 + 1, Ny);
         gpux_nyz[cnti].psidd2fftx_h2d[cntj].dstPtr = gpux_nyz[cnti].psidd2x[cntj];
         gpux_nyz[cnti].psidd2fftx_h2d[cntj].extent = make_cudaExtent((Nz2 + 1) * sizeof(fftw_complex), Ny, chunkNx_nyz);
         gpux_nyz[cnti].psidd2fftx_h2d[cntj].kind = cudaMemcpyHostToDevice;
      }

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         int fft_rank = 2;
         int n_br[] = {Ny, Nz};
         int howmany = chunkNx_nyz;
         int idist = Ny * (gpux_nyz[cnti].psidd2x[cntj].pitch / sizeof(cufftDoubleComplex)); int odist = Ny * (gpux_nyz[cnti].psidd2x[cntj].pitch / sizeof(cufftDoubleReal));
         int istride = 1; int ostride = 1;
         int inembed_br[] = { Ny, gpux_nyz[cnti].psidd2x[cntj].pitch / sizeof(cufftDoubleComplex) }; int onembed_br[] = { Ny, gpux_nyz[cnti].psidd2x[cntj].pitch / sizeof(cufftDoubleReal) };

         cudaCheckFFTError(cufftCreate(&(gpux_nyz[cnti].plan_backward_row[cntj])));
         cudaCheckFFTError(cufftSetAutoAllocation(gpux_nyz[cnti].plan_backward_row[cntj], 0));
         cudaCheckFFTError(cufftSetStream(gpux_nyz[cnti].plan_backward_row[cntj], gpu[cnti].exec_stream[cntj]));
         cudaCheckFFTError(cufftMakePlanMany(gpux_nyz[cnti].plan_backward_row[cntj], fft_rank, n_br, inembed_br, istride, idist, onembed_br, ostride, odist, CUFFT_Z2D, howmany, &(worksize_br[cntj])));
      }

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         fft_workspace[cntj] = alloc_mem_device(worksize_br[cntj]);
         cudaCheckFFTError(cufftSetWorkArea(gpux_nyz[cnti].plan_backward_row[cntj], fft_workspace[cntj]));
      }

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, cnti);

      Nsm = deviceProp.multiProcessorCount;

      gpux_nyz[cnti].dimBlock3d_nu.x = block3d_nu[0]; gpux_nyz[cnti].dimBlock3d_nu.y = block3d_nu[1]; gpux_nyz[cnti].dimBlock3d_nu.z = block3d_nu[2];
      gpux_nyz[cnti].dimGrid3d_nu.x = gpux_nyz[cnti].dimGrid3d_nu.y = gpux_nyz[cnti].dimGrid3d_nu.z = Nsm * grid3d_nu;

      gpux_nyz[cnti].dimBlock3d_dd2.x = block3d_dd2[0]; gpux_nyz[cnti].dimBlock3d_dd2.y = block3d_dd2[1]; gpux_nyz[cnti].dimBlock3d_dd2.z = block3d_dd2[2];
      gpux_nyz[cnti].dimGrid3d_dd2.x = gpux_nyz[cnti].dimGrid3d_dd2.y = Nsm * grid3d_dd2; gpux_nyz[cnti].dimGrid3d_dd2.z = 1;

      gpux_nyz[cnti].dimBlock2d_dd2.x = block2d_dd2[0]; gpux_nyz[cnti].dimBlock2d_dd2.y = block2d_dd2[1]; gpux_nyz[cnti].dimBlock2d_dd2.z = 1;
      gpux_nyz[cnti].dimGrid2d_dd2.x = gpux_nyz[cnti].dimGrid2d_dd2.y = Nsm * grid2d_dd2; gpux_nyz[cnti].dimGrid2d_dd2.z = 1;

      gpux_nyz[cnti].dimBlock2d_luy.x = block2d_luy[0]; gpux_nyz[cnti].dimBlock2d_luy.y = block2d_luy[1]; gpux_nyz[cnti].dimBlock2d_luy.z = 1;
      gpux_nyz[cnti].dimGrid2d_luy.x = gpux_nyz[cnti].dimGrid2d_luy.y = Nsm * grid2d_luy; gpux_nyz[cnti].dimGrid2d_luy.z = 1;

      gpux_nyz[cnti].dimBlock2d_luz.x = block2d_luz[0]; gpux_nyz[cnti].dimBlock2d_luz.y = block2d_luz[1]; gpux_nyz[cnti].dimBlock2d_luz.z = 1;
      gpux_nyz[cnti].dimGrid2d_luz.x = gpux_nyz[cnti].dimGrid2d_luz.y = Nsm * grid2d_luz; gpux_nyz[cnti].dimGrid2d_luz.z = 1;
   }
}

void init_gpuy_fft() {
   int cnti, cntj;
   int Nsm;
   size_t pitchY, pitchYPadded;
   size_t ptrSize, ptrSizePadded;
   cudaPitchedPtr tmpPtr;
   size_t worksize_fc[chunksNy_fft], worksize_bc[chunksNy_fft];
   void *fft_workspace_fc[chunksNy_fft], *fft_workspace_bc[chunksNy_fft];

   chunkNy_fft = gpuNy_fft / chunksNy_fft / Ngpu;

   // psiy    (Nx, chunkNy_fft, Nz) // complex
   // potdd   (Nx, chunkNy_fft, Nz2 + 1)
   // psidd2y (Nx, chunkNy_fft, Nz2 + 1) // complex

   tmpPtr = alloc_double_tensor_device(Nx, chunkNy_fft, Nz);
   pitchY = tmpPtr.pitch;
   ptrSize = chunksNy_fft * Nx * chunkNy_fft * pitchY;
   free_double_tensor_device(tmpPtr);

   tmpPtr = alloc_complex_tensor_device(Nx, chunkNy_fft, Nz2 + 1);
   pitchYPadded = tmpPtr.pitch;
   ptrSizePadded = chunksNy_fft * Nx * chunkNy_fft * pitchYPadded;
   free_complex_tensor_device(tmpPtr);

   gpuy_fft = (gpuy_t *) calloc(Ngpu, sizeof(struct gpuy_t));

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      long d_offsetNy = cnti * (gpuNy_fft / Ngpu);

      reset_mem_device();
      gpuy_fft[cnti].ptr1 = (char *) alloc_mem_device(ptrSize);
      gpuy_fft[cnti].ptr2 = (char *) alloc_mem_device(ptrSizePadded);
      gpuy_fft[cnti].ptr3 = (char *) alloc_mem_device(ptrSizePadded);

      cudaCheckError(cudaMemcpyToSymbol(d_chunkNy_fft, &chunkNy_fft, sizeof(chunkNy_fft)));

      gpuy_fft[cnti].psidd2y_orig = (cudaPitchedPtr *) calloc(chunksNy_fft, sizeof(cudaPitchedPtr));
      gpuy_fft[cnti].psidd2y_tran = (cudaPitchedPtr *) calloc(chunksNy_fft, sizeof(cudaPitchedPtr));
      gpuy_fft[cnti].potdd = (cudaPitchedPtr *) calloc(chunksNy_fft, sizeof(cudaPitchedPtr));

      gpuy_fft[cnti].potdd_h2d = (cudaMemcpy3DParms *) calloc(chunksNy_fft, sizeof(cudaMemcpy3DParms));

      gpuy_fft[cnti].psidd2ffty_h2d = (cudaMemcpy3DParms *) calloc(chunksNy_fft, sizeof(cudaMemcpy3DParms));
      gpuy_fft[cnti].psidd2ffty_d2h = (cudaMemcpy3DParms *) calloc(chunksNy_fft, sizeof(cudaMemcpy3DParms));

      gpuy_fft[cnti].plan_forward_col = (cufftHandle *) calloc(chunksNy_fft, sizeof(cufftHandle));
      gpuy_fft[cnti].plan_backward_col = (cufftHandle *) calloc(chunksNy_fft, sizeof(cufftHandle));

      for (cntj = 0; cntj < chunksNy_fft; cntj ++) {
         int offsetNy = cntj * chunkNy_fft;

         gpuy_fft[cnti].potdd[cntj] = make_cudaPitchedPtr(gpuy_fft[cnti].ptr1 + (cntj * Nx * chunkNy_fft * pitchY), pitchY, Nz2 + 1, chunkNy_fft);
         gpuy_fft[cnti].psidd2y_orig[cntj] = make_cudaPitchedPtr(gpuy_fft[cnti].ptr2 + (cntj * Nx * chunkNy_fft * pitchYPadded), pitchYPadded, Nz2 + 1, chunkNy_fft);
         gpuy_fft[cnti].psidd2y_tran[cntj] = make_cudaPitchedPtr(gpuy_fft[cnti].ptr3 + (cntj * Nx * chunkNy_fft * pitchYPadded), pitchYPadded, Nz2 + 1, chunkNy_fft);

         gpuy_fft[cnti].potdd_h2d[cntj].srcPtr = make_cudaPitchedPtr(potdd[0][0], Nz * sizeof(double), Nz, localNy);
         gpuy_fft[cnti].potdd_h2d[cntj].srcPos = make_cudaPos(0, cpuNy_fft + d_offsetNy + offsetNy, 0);
         gpuy_fft[cnti].potdd_h2d[cntj].dstPtr = gpuy_fft[cnti].potdd[cntj];
         gpuy_fft[cnti].potdd_h2d[cntj].extent = make_cudaExtent((Nz2 + 1) * sizeof(double), chunkNy_fft, Nx);
         gpuy_fft[cnti].potdd_h2d[cntj].kind = cudaMemcpyHostToDevice;

         gpuy_fft[cnti].psidd2ffty_h2d[cntj].srcPtr = make_cudaPitchedPtr(psidd2fft_t, (Nz2 + 1) * sizeof(fftw_complex), Nz2 + 1, localNy);
         gpuy_fft[cnti].psidd2ffty_h2d[cntj].srcPos = make_cudaPos(0, cpuNy_fft + d_offsetNy + offsetNy, 0);
         gpuy_fft[cnti].psidd2ffty_h2d[cntj].dstPtr = gpuy_fft[cnti].psidd2y_orig[cntj];
         gpuy_fft[cnti].psidd2ffty_h2d[cntj].extent = make_cudaExtent((Nz2 + 1) * sizeof(fftw_complex), chunkNy_fft, Nx);
         gpuy_fft[cnti].psidd2ffty_h2d[cntj].kind = cudaMemcpyHostToDevice;

         gpuy_fft[cnti].psidd2ffty_d2h[cntj].srcPtr = gpuy_fft[cnti].psidd2y_orig[cntj];
         gpuy_fft[cnti].psidd2ffty_d2h[cntj].dstPtr = make_cudaPitchedPtr(psidd2fft_t, (Nz2 + 1) * sizeof(fftw_complex), Nz2 + 1, localNy);
         gpuy_fft[cnti].psidd2ffty_d2h[cntj].dstPos = make_cudaPos(0, cpuNy_fft + d_offsetNy + offsetNy, 0);
         gpuy_fft[cnti].psidd2ffty_d2h[cntj].extent = make_cudaExtent((Nz2 + 1) * sizeof(fftw_complex), chunkNy_fft, Nx);
         gpuy_fft[cnti].psidd2ffty_d2h[cntj].kind = cudaMemcpyDeviceToHost;
      }

      for (cntj = 0; cntj < chunksNy_fft; cntj ++) {
         int fft_rank = 1;
         int n_fc[] = {Nx};
         int howmany = Nz2 + 1;
         int idist = 1; int odist = 1;
         int istride = chunkNy_fft * (gpuy_fft[cnti].psidd2y_orig[cntj].pitch / sizeof(cufftDoubleComplex)); int ostride = chunkNy_fft * (gpuy_fft[cnti].psidd2y_tran[cntj].pitch / sizeof(cufftDoubleComplex));
         int inembed_fc[] = {Nx}; int onembed_fc[] = {Nx};

         cudaCheckFFTError(cufftCreate(&(gpuy_fft[cnti].plan_forward_col[cntj])));
         cudaCheckFFTError(cufftSetAutoAllocation(gpuy_fft[cnti].plan_forward_col[cntj], 0));
         cudaCheckFFTError(cufftSetStream(gpuy_fft[cnti].plan_forward_col[cntj], gpu[cnti].exec_stream[cntj]));
         cudaCheckFFTError(cufftMakePlanMany(gpuy_fft[cnti].plan_forward_col[cntj], fft_rank, n_fc, inembed_fc, istride, idist, onembed_fc, ostride, odist, CUFFT_Z2Z, howmany, &(worksize_fc[cntj])));
      }

      for (cntj = 0; cntj < chunksNy_fft; cntj ++) {
         int fft_rank = 1;
         int n_bc[] = {Nx};
         int howmany = Nz2 + 1;
         int idist = 1; int odist = 1;
         int istride = chunkNy_fft * (gpuy_fft[cnti].psidd2y_tran[cntj].pitch / sizeof(cufftDoubleComplex)); int ostride = chunkNy_fft * (gpuy_fft[cnti].psidd2y_orig[cntj].pitch / sizeof(cufftDoubleComplex));
         int inembed_bc[] = {Nx}; int onembed_bc[] = {Nx};

         cudaCheckFFTError(cufftCreate(&(gpuy_fft[cnti].plan_backward_col[cntj])));
         cudaCheckFFTError(cufftSetAutoAllocation(gpuy_fft[cnti].plan_backward_col[cntj], 0));
         cudaCheckFFTError(cufftSetStream(gpuy_fft[cnti].plan_backward_col[cntj], gpu[cnti].exec_stream[cntj]));
         cudaCheckFFTError(cufftMakePlanMany(gpuy_fft[cnti].plan_backward_col[cntj], fft_rank, n_bc, inembed_bc, istride, idist, onembed_bc, ostride, odist, CUFFT_Z2Z, howmany, &(worksize_bc[cntj])));
      }

      for (cntj = 0; cntj < chunksNy_fft; cntj ++) {
         fft_workspace_fc[cntj] = alloc_mem_device(worksize_fc[cntj]);
         fft_workspace_bc[cntj] = alloc_mem_device(worksize_bc[cntj]);
         cudaCheckFFTError(cufftSetWorkArea(gpuy_fft[cnti].plan_forward_col[cntj], fft_workspace_fc[cntj]));
         cudaCheckFFTError(cufftSetWorkArea(gpuy_fft[cnti].plan_backward_col[cntj], fft_workspace_bc[cntj]));
      }

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, cnti);

      Nsm = deviceProp.multiProcessorCount;

      gpuy_fft[cnti].dimBlock3d_potdd.x = block3d_potdd[0]; gpuy_fft[cnti].dimBlock3d_potdd.y = block3d_potdd[1]; gpuy_fft[cnti].dimBlock3d_potdd.z = block3d_potdd[2];
      gpuy_fft[cnti].dimGrid3d_potdd.x = gpuy_fft[cnti].dimGrid3d_potdd.y = gpuy_fft[cnti].dimGrid3d_potdd.z = Nsm * grid3d_potdd;
   }
}

void init_gpuy_lux() {
   int cnti, cntj;
   int Nsm;
   size_t pitchY;
   size_t ptrSize;
   cudaPitchedPtr tmpPtr;
   cuDoubleComplex minusAx;

   chunkNy_lux = gpuNy_lux / chunksNy_lux / Ngpu;

   // psiy    (Nx, chunkNy_lux, Nz) // complex
   // cbetay  (Nx, chunkNy_lux, Nz) // complex

   tmpPtr = alloc_complex_tensor_device(Nx, chunkNy_lux, Nz);
   pitchY = tmpPtr.pitch;
   ptrSize = chunksNy_lux * Nx * chunkNy_lux * pitchY;
   free_complex_tensor_device(tmpPtr);

   minusAx = make_cuDoubleComplex(0., - Ax.y);

   gpuy_lux = (gpuy_t *) calloc(Ngpu, sizeof(struct gpuy_t));

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      long d_offsetNy = cnti * (gpuNy_lux / Ngpu);

      reset_mem_device();
      gpuy_lux[cnti].ptr1 = (char *) alloc_mem_device(ptrSize);
      gpuy_lux[cnti].ptr2 = (char *) alloc_mem_device(ptrSize);

      gpuy_lux[cnti].calphax = alloc_complex_vector_device(Nx - 1);
      gpuy_lux[cnti].cgammax = alloc_complex_vector_device(Nx - 1);

      cudaCheckError(cudaMemcpyToSymbol(d_chunkNy_lux, &chunkNy_lux, sizeof(chunkNy_lux)));

      cudaCheckError(cudaMemcpyToSymbol(d_Ax0r, &Ax0r, sizeof(Ax0r)));
      cudaCheckError(cudaMemcpyToSymbol(d_Ax, &Ax, sizeof(Ax)));
      cudaCheckError(cudaMemcpyToSymbol(d_minusAx, &minusAx, sizeof(minusAx)));

      cudaCheckError(cudaMemcpy(gpuy_lux[cnti].calphax, calphax, (Nx - 1) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));
      cudaCheckError(cudaMemcpy(gpuy_lux[cnti].cgammax, cgammax, (Nx - 1) * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice));

      gpuy_lux[cnti].psiy = (cudaPitchedPtr *) calloc(chunksNy_lux, sizeof(cudaPitchedPtr));
      gpuy_lux[cnti].cbetay = (cudaPitchedPtr *) calloc(chunksNy_lux, sizeof(cudaPitchedPtr));

      gpuy_lux[cnti].psiy_h2d = (cudaMemcpy3DParms *) calloc(chunksNy_lux, sizeof(cudaMemcpy3DParms));
      gpuy_lux[cnti].psiy_d2h = (cudaMemcpy3DParms *) calloc(chunksNy_lux, sizeof(cudaMemcpy3DParms));

      for (cntj = 0; cntj < chunksNy_lux; cntj ++) {
         int offsetNy = cntj * chunkNy_lux;

         gpuy_lux[cnti].psiy[cntj] = make_cudaPitchedPtr(gpuy_lux[cnti].ptr1 + (cntj * Nx * chunkNy_lux * pitchY), pitchY, Nz, chunkNy_lux);
         gpuy_lux[cnti].cbetay[cntj] = make_cudaPitchedPtr(gpuy_lux[cnti].ptr2 + (cntj * Nx * chunkNy_lux * pitchY), pitchY, Nz, chunkNy_lux);

         gpuy_lux[cnti].psiy_h2d[cntj].srcPtr = make_cudaPitchedPtr(psi_t[0][0], Nz * sizeof(cuDoubleComplex), Nz, localNy);
         gpuy_lux[cnti].psiy_h2d[cntj].srcPos = make_cudaPos(0, cpuNy_lux + d_offsetNy + offsetNy, 0);
         gpuy_lux[cnti].psiy_h2d[cntj].dstPtr = gpuy_lux[cnti].psiy[cntj];
         gpuy_lux[cnti].psiy_h2d[cntj].extent = make_cudaExtent(Nz * sizeof(cuDoubleComplex), chunkNy_lux, Nx);
         gpuy_lux[cnti].psiy_h2d[cntj].kind = cudaMemcpyHostToDevice;

         gpuy_lux[cnti].psiy_d2h[cntj].srcPtr = gpuy_lux[cnti].psiy[cntj];
         gpuy_lux[cnti].psiy_d2h[cntj].dstPtr = make_cudaPitchedPtr(psi_t[0][0], Nz * sizeof(cuDoubleComplex), Nz, localNy);
         gpuy_lux[cnti].psiy_d2h[cntj].dstPos = make_cudaPos(0, cpuNy_lux + d_offsetNy + offsetNy, 0);
         gpuy_lux[cnti].psiy_d2h[cntj].extent = make_cudaExtent(Nz * sizeof(cuDoubleComplex), chunkNy_lux, Nx);
         gpuy_lux[cnti].psiy_d2h[cntj].kind = cudaMemcpyDeviceToHost;
      }

      cudaDeviceProp deviceProp;
      cudaGetDeviceProperties(&deviceProp, cnti);

      Nsm = deviceProp.multiProcessorCount;

      gpuy_lux[cnti].dimBlock2d_lux.x = block2d_lux[0]; gpuy_lux[cnti].dimBlock2d_lux.y = block2d_lux[1]; gpuy_lux[cnti].dimBlock2d_lux.z = 1;
      gpuy_lux[cnti].dimGrid2d_lux.x = gpuy_lux[cnti].dimGrid2d_lux.y = Nsm * grid2d_lux; gpuy_lux[cnti].dimGrid2d_lux.z = 1;
   }
}

void free_gpu() {
   int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNx_fft; cntj ++) {
         cudaCheckFFTError(cufftDestroy(gpux_fft[cnti].plan_forward_row[cntj]));
      }

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         cudaCheckFFTError(cufftDestroy(gpux_nyz[cnti].plan_backward_row[cntj]));
      }

      for (cntj = 0; cntj < chunksNy_fft; cntj ++) {
         cudaCheckFFTError(cufftDestroy(gpuy_fft[cnti].plan_forward_col[cntj]));
         cudaCheckFFTError(cufftDestroy(gpuy_fft[cnti].plan_backward_col[cntj]));
      }

      for (cntj = 0; cntj < gpu[cnti].nstreams; cntj ++) {
         cudaCheckError(cudaStreamDestroy(gpu[cnti].exec_stream[cntj]))
      }

      free_complex_vector_device(gpuy_lux[cnti].calphax);
      free_complex_vector_device(gpux_nyz[cnti].calphay);
      free_complex_vector_device(gpux_nyz[cnti].calphaz);
      free_complex_vector_device(gpuy_lux[cnti].cgammax);
      free_complex_vector_device(gpux_nyz[cnti].cgammay);
      free_complex_vector_device(gpux_nyz[cnti].cgammaz);

      free_double_tensor_device(gpux_nyz[cnti].psidd2tmp);

      free_mem_device();

      free(gpux_fft[cnti].psix_h2d);
      free(gpux_fft[cnti].psidd2fftx_d2h);

      free(gpux_nyz[cnti].psix_h2d);
      free(gpux_nyz[cnti].psix_d2h);
      free(gpux_nyz[cnti].pot_h2d);
      free(gpux_nyz[cnti].psidd2_d2h);
      free(gpux_nyz[cnti].psidd2fftx_h2d);

      free(gpuy_lux[cnti].psiy_h2d);
      free(gpuy_lux[cnti].psiy_d2h);

      free(gpuy_fft[cnti].psiy_h2d);
      free(gpuy_fft[cnti].psiy_d2h);
      free(gpuy_fft[cnti].potdd_h2d);
      free(gpuy_fft[cnti].psidd2ffty_h2d);
      free(gpuy_fft[cnti].psidd2ffty_d2h);

      free(gpu[cnti].exec_stream);

      cudaCheckError(cudaDeviceReset());
   }
}

void copy_psidd2_d2h() {
   int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         cudaCheckError(cudaMemcpy3D(&(gpux_nyz[cnti].psidd2_d2h[cntj])));
      }
   }
}

void sync_with_gpu() {
   /*int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));
      for (cntj = 0; cntj < gpu[cnti].nstreams; cntj ++) {
         cudaCheckError(cudaEventRecord(gpu[cnti].syncEvent, gpu[cnti].exec_stream[cntj]));
      }
      cudaCheckError(cudaEventSynchronize(gpu[cnti].syncEvent));
   }*/
   int cnti;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));
      cudaCheckError(cudaEventRecord(gpu[cnti].syncEvent, 0));
      cudaCheckError(cudaEventSynchronize(gpu[cnti].syncEvent));
   }
   /*int cnti;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));
      cudaCheckError(cudaDeviceSynchronize());
   }*/
}

__global__ void calcpsi2_kernel(cudaPitchedPtr psi, cudaPitchedPtr psi2) {
   long cnti, cntj, cntk;
   cuDoubleComplex *psirow;
   double *psi2row;
   double tmp;

   for (cnti = TID_Z; cnti < d_chunkNx_fft; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_complex_tensor_row(psi, cnti, cntj);
         psi2row = get_double_tensor_row(psi2, cnti, cntj);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            tmp = cuCabs(psirow[cntk]);
            psi2row[cntk] = tmp * tmp;
         }
      }
   }
}

void calcpsidd2_gpu_p1_1ce() {
   int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNx_fft; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpux_fft[cnti].psix_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
      }

      for (cntj = 0; cntj < chunksNx_fft; cntj ++) {
         calcpsi2_kernel<<<gpux_fft[cnti].dimGrid3d_dd2, gpux_fft[cnti].dimBlock3d_dd2, 0, gpu[cnti].exec_stream[cntj]>>>(gpux_fft[cnti].psix[cntj], gpux_fft[cnti].psidd2x[cntj]);
         //cudaCheckError(cudaPeekAtLastError());
      }

      for (cntj = 0; cntj < chunksNx_fft; cntj ++) {
         cudaCheckFFTError(cufftExecD2Z(gpux_fft[cnti].plan_forward_row[cntj], (cufftDoubleReal *) gpux_fft[cnti].psidd2x[cntj].ptr, (cufftDoubleComplex *) gpux_fft[cnti].psidd2x[cntj].ptr));
         cudaCheckError(cudaMemcpy3DAsync(&(gpux_fft[cnti].psidd2fftx_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
      }
   }

   return;
}

void calcpsidd2_gpu_p1_2ce() {
   int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNx_fft; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpux_fft[cnti].psix_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
         calcpsi2_kernel<<<gpux_fft[cnti].dimGrid3d_dd2, gpux_fft[cnti].dimBlock3d_dd2, 0, gpu[cnti].exec_stream[cntj]>>>(gpux_fft[cnti].psix[cntj], gpux_fft[cnti].psidd2x[cntj]);
         //cudaCheckError(cudaPeekAtLastError());
         cudaCheckFFTError(cufftExecD2Z(gpux_fft[cnti].plan_forward_row[cntj], (cufftDoubleReal *) gpux_fft[cnti].psidd2x[cntj].ptr, (cufftDoubleComplex *) gpux_fft[cnti].psidd2x[cntj].ptr));
         cudaCheckError(cudaMemcpy3DAsync(&(gpux_fft[cnti].psidd2fftx_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
      }
   }

   return;
}

void calcpsidd2_gpu_p2_1ce() {
   int cnti, cntj, cntk;
   cufftDoubleComplex *psidd2orig_ptr, *psidd2tran_ptr;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNy_fft; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpuy_fft[cnti].psidd2ffty_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
      }

      for (cntj = 0; cntj < chunksNy_fft; cntj ++) {
         psidd2orig_ptr = (cufftDoubleComplex *) gpuy_fft[cnti].psidd2y_orig[cntj].ptr;
         psidd2tran_ptr = (cufftDoubleComplex *) gpuy_fft[cnti].psidd2y_tran[cntj].ptr;
         for (cntk = 0; cntk < chunkNy_fft; cntk ++) {
            cudaCheckFFTError(cufftExecZ2Z(gpuy_fft[cnti].plan_forward_col[cntj], psidd2orig_ptr, psidd2tran_ptr, CUFFT_FORWARD));
            psidd2orig_ptr += (gpuy_fft[cnti].psidd2y_orig[cntj].pitch / sizeof(cufftDoubleComplex));
            psidd2tran_ptr += (gpuy_fft[cnti].psidd2y_tran[cntj].pitch / sizeof(cufftDoubleComplex));
         }
         cudaCheckError(cudaMemcpy3DAsync(&(gpuy_fft[cnti].potdd_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
      }

      for (cntj = 0; cntj < chunksNy_fft; cntj ++) {
         calcpsidd2_kernel1<<<gpuy_fft[cnti].dimGrid3d_potdd, gpuy_fft[cnti].dimBlock3d_potdd, 0, gpu[cnti].exec_stream[cntj]>>>(gpuy_fft[cnti].psidd2y_tran[cntj], gpuy_fft[cnti].potdd[cntj]);
         //cudaCheckError(cudaPeekAtLastError());
      }

      for (cntj = 0; cntj < chunksNy_fft; cntj ++) {
         psidd2orig_ptr = (cufftDoubleComplex *) gpuy_fft[cnti].psidd2y_orig[cntj].ptr;
         psidd2tran_ptr = (cufftDoubleComplex *) gpuy_fft[cnti].psidd2y_tran[cntj].ptr;
         for (cntk = 0; cntk < chunkNy_fft; cntk ++) {
            cudaCheckFFTError(cufftExecZ2Z(gpuy_fft[cnti].plan_backward_col[cntj], psidd2tran_ptr, psidd2orig_ptr, CUFFT_INVERSE));
            psidd2orig_ptr += (gpuy_fft[cnti].psidd2y_orig[cntj].pitch / sizeof(cufftDoubleComplex));
            psidd2tran_ptr += (gpuy_fft[cnti].psidd2y_tran[cntj].pitch / sizeof(cufftDoubleComplex));
         }
         cudaCheckError(cudaMemcpy3DAsync(&(gpuy_fft[cnti].psidd2ffty_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
      }
   }

   return;
}

void calcpsidd2_gpu_p2_2ce() {
   int cnti, cntj, cntk;
   cufftDoubleComplex *psidd2orig_ptr, *psidd2tran_ptr;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNy_fft; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpuy_fft[cnti].psidd2ffty_h2d[cntj]), gpu[cnti].exec_stream[cntj]));

         psidd2orig_ptr = (cufftDoubleComplex *) gpuy_fft[cnti].psidd2y_orig[cntj].ptr;
         psidd2tran_ptr = (cufftDoubleComplex *) gpuy_fft[cnti].psidd2y_tran[cntj].ptr;
         for (cntk = 0; cntk < chunkNy_fft; cntk ++) {
            cudaCheckFFTError(cufftExecZ2Z(gpuy_fft[cnti].plan_forward_col[cntj], psidd2orig_ptr, psidd2tran_ptr, CUFFT_FORWARD));
            psidd2orig_ptr += (gpuy_fft[cnti].psidd2y_orig[cntj].pitch / sizeof(cufftDoubleComplex));
            psidd2tran_ptr += (gpuy_fft[cnti].psidd2y_tran[cntj].pitch / sizeof(cufftDoubleComplex));
         }
         cudaCheckError(cudaMemcpy3DAsync(&(gpuy_fft[cnti].potdd_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
         calcpsidd2_kernel1<<<gpuy_fft[cnti].dimGrid3d_potdd, gpuy_fft[cnti].dimBlock3d_potdd, 0, gpu[cnti].exec_stream[cntj]>>>(gpuy_fft[cnti].psidd2y_tran[cntj], gpuy_fft[cnti].potdd[cntj]);
         //cudaCheckError(cudaPeekAtLastError());

         psidd2orig_ptr = (cufftDoubleComplex *) gpuy_fft[cnti].psidd2y_orig[cntj].ptr;
         psidd2tran_ptr = (cufftDoubleComplex *) gpuy_fft[cnti].psidd2y_tran[cntj].ptr;
         for (cntk = 0; cntk < chunkNy_fft; cntk ++) {
            cudaCheckFFTError(cufftExecZ2Z(gpuy_fft[cnti].plan_backward_col[cntj], psidd2tran_ptr, psidd2orig_ptr, CUFFT_INVERSE));
            psidd2orig_ptr += (gpuy_fft[cnti].psidd2y_orig[cntj].pitch / sizeof(cufftDoubleComplex));
            psidd2tran_ptr += (gpuy_fft[cnti].psidd2y_tran[cntj].pitch / sizeof(cufftDoubleComplex));
         }
         cudaCheckError(cudaMemcpy3DAsync(&(gpuy_fft[cnti].psidd2ffty_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
      }
   }

   return;
}

void calcpsidd2_gpu_p3_1ce() {
   int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].psidd2fftx_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
      }

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         cudaCheckFFTError(cufftExecZ2D(gpux_nyz[cnti].plan_backward_row[cntj], (cufftDoubleComplex *) gpux_nyz[cnti].psidd2x[cntj].ptr, (cufftDoubleReal *) gpux_nyz[cnti].psidd2x[cntj].ptr));
      }

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].pot_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
      }
   }
}

void calcpsidd2_gpu_p3_2ce() {
   int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].psidd2fftx_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
         cudaCheckFFTError(cufftExecZ2D(gpux_nyz[cnti].plan_backward_row[cntj], (cufftDoubleComplex *) gpux_nyz[cnti].psidd2x[cntj].ptr, (cufftDoubleReal *) gpux_nyz[cnti].psidd2x[cntj].ptr));
         cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].pot_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
      }
   }
}

void calcpsidd2_gpu_p4(int maxrank) {
   int cnti, cntj;
   int lastchunk;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         calcpsidd2_kernel2<<<gpux_nyz[cnti].dimGrid3d_dd2, gpux_nyz[cnti].dimBlock3d_dd2, 0, gpu[cnti].exec_stream[cntj]>>>(gpux_nyz[cnti].psidd2x[cntj]);
         //cudaCheckError(cudaPeekAtLastError());
      }

      if (maxrank) {
         cudaCheckError(cudaMemcpy3D(&(gpux_nyz[cnti].psidd2tmp_h2d)));
         calcpsidd2_kernel3<<<gpux_nyz[cnti].dimGrid2d_dd2, gpux_nyz[cnti].dimBlock2d_dd2>>>(gpux_nyz[cnti].psidd2x[chunksNx_nyz - 1], gpux_nyz[cnti].psidd2tmp);
         //cudaCheckError(cudaPeekAtLastError());
      }

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         lastchunk = (maxrank != 0 && cntj == chunksNx_nyz - 1) ? 1 : 0;
         calcpsidd2_kernel4<<<gpux_nyz[cnti].dimGrid2d_dd2, gpux_nyz[cnti].dimBlock2d_dd2>>>(gpux_nyz[cnti].psidd2x[cntj], lastchunk);
         //cudaCheckError(cudaPeekAtLastError());
      }
   }

   return;
}

__global__ void calcpsidd2_kernel1(cudaPitchedPtr psidd2, cudaPitchedPtr potdd) {
   long cnti, cntj, cntk;
   double *potddrow;
   cuDoubleComplex *psidd2row;
   double tmp;

   for (cnti = TID_Z; cnti < d_Nx; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_chunkNy_fft; cntj += GRID_STRIDE_Y) {
         psidd2row = get_complex_tensor_row(psidd2, cnti, cntj);
         potddrow = get_double_tensor_row(potdd, cnti, cntj);

         for (cntk = TID_X; cntk < (d_Nz / 2) + 1; cntk += GRID_STRIDE_X) {
            tmp = potddrow[cntk];
            psidd2row[cntk].x *= tmp;
            psidd2row[cntk].y *= tmp;
         }
      }
   }
}

__global__ void calcpsidd2_kernel2(cudaPitchedPtr psidd2) {
   long cnti, cntj, cntk;
   double *psidd2row;

   for (cnti = TID_Z; cnti < d_chunkNx_nyz; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psidd2row = get_double_tensor_row(psidd2, cnti, cntj);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            psidd2row[cntk] /= d_Nx * d_Ny * d_Nz;
         }
      }
   }
}

__global__ void calcpsidd2_kernel3(cudaPitchedPtr psidd2, cudaPitchedPtr psidd2_0) {
   long cntj, cntk;
   double *psidd2row, *psidd2_0row;

   for (cntj = TID_Y; cntj < d_Ny - 1; cntj += GRID_STRIDE_Y) {
      psidd2row = get_double_tensor_row(psidd2, d_chunkNx_nyz - 1, cntj);
      psidd2_0row = get_double_tensor_row(psidd2_0, 0, cntj);

      for (cntk = TID_X; cntk < d_Nz - 1; cntk += GRID_STRIDE_X) {
         psidd2row[cntk] = psidd2_0row[cntk];
      }
   }
}

__global__ void calcpsidd2_kernel4(cudaPitchedPtr psidd2, int lastchunk) {
   long cnti, cntj, cntk;
   double *psidd2row, *psidd2firstrow, *psidd2lastrow;

   for (cnti = TID_Y; cnti < d_chunkNx_nyz - lastchunk; cnti += GRID_STRIDE_Y) {
      psidd2firstrow = get_double_tensor_row(psidd2, cnti, 0);
      psidd2lastrow = get_double_tensor_row(psidd2, cnti, d_Ny - 1);

      for (cntk = TID_X; cntk < d_Nz - 1; cntk += GRID_STRIDE_X) {
         psidd2lastrow[cntk] = psidd2firstrow[cntk];
      }
   }

   for (cnti = TID_Y; cnti < d_chunkNx_nyz - lastchunk; cnti += GRID_STRIDE_Y) {
      for (cntj = TID_X; cntj < d_Ny - 1; cntj += GRID_STRIDE_X) {
         psidd2row = get_double_tensor_row(psidd2, cnti, cntj);

         psidd2row[d_Nz - 1] = psidd2row[0];
      }
   }
}

void calcnu_gpu_1ce() {
   int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].psix_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
      }

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         calcnu_kernel<<<gpux_nyz[cnti].dimGrid3d_nu, gpux_nyz[cnti].dimBlock3d_nu, 0, gpu[cnti].exec_stream[cntj]>>>(gpux_nyz[cnti].psix[cntj], gpux_nyz[cnti].psidd2x[cntj], gpux_nyz[cnti].pot[cntj], g, gd);
         //cudaCheckError(cudaPeekAtLastError());
      }

      if (offload == 1) {
         for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
            cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].psix_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
         }
      }
   }

   return;
}

void calcnu_gpu_2ce() {
   int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].psix_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
         calcnu_kernel<<<gpux_nyz[cnti].dimGrid3d_nu, gpux_nyz[cnti].dimBlock3d_nu, 0, gpu[cnti].exec_stream[cntj]>>>(gpux_nyz[cnti].psix[cntj], gpux_nyz[cnti].psidd2x[cntj], gpux_nyz[cnti].pot[cntj], g, gd);
         //cudaCheckError(cudaPeekAtLastError());
         if (offload == 1) {
            cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].psix_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
         }
      }
   }

   return;
}

__global__ void calcnu_kernel(cudaPitchedPtr psi, cudaPitchedPtr psidd2, cudaPitchedPtr pot, double g, double gd) {
   long cnti, cntj, cntk;
   cuDoubleComplex psitmp;
   double psi2lin, psidd2lin, pottmp;
   cuDoubleComplex *psirow;
   double *psidd2row, *potrow;

   for (cnti = TID_Z; cnti < d_chunkNx_nyz; cnti += GRID_STRIDE_Z) {
      for (cntj = TID_Y; cntj < d_Ny; cntj += GRID_STRIDE_Y) {
         psirow = get_complex_tensor_row(psi, cnti, cntj);
         psidd2row = get_double_tensor_row(psidd2, cnti, cntj);
         potrow = get_double_tensor_row(pot, cnti, cntj);

         for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
            psitmp = psirow[cntk];
            psidd2lin = psidd2row[cntk];
            pottmp = potrow[cntk];

            psi2lin = cuCabs(psitmp) * cuCabs(psitmp) * g;
            psidd2lin = psidd2lin  * gd;
            pottmp = d_dt * (pottmp + psi2lin + psidd2lin);

            psirow[cntk] = cuCmul(psitmp, cuCexp(-pottmp));
         }
      }
   }
}

void calclux_gpu_1ce() {
   int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNy_lux; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpuy_lux[cnti].psiy_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
      }

      for (cntj = 0; cntj < chunksNy_lux; cntj ++) {
         calclux_kernel<<<gpuy_lux[cnti].dimGrid2d_lux, gpuy_lux[cnti].dimBlock2d_lux, 0, gpu[cnti].exec_stream[cntj]>>>(gpuy_lux[cnti].psiy[cntj], gpuy_lux[cnti].cbetay[cntj], gpuy_lux[cnti].calphax, gpuy_lux[cnti].cgammax);
         //cudaCheckError(cudaPeekAtLastError());
         cudaCheckError(cudaMemcpy3DAsync(&(gpuy_lux[cnti].psiy_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
      }
   }

   return;
}

void calclux_gpu_2ce() {
   int cnti, cntj;

   for (cnti = 0; cnti < Ngpu; cnti ++) {
      cudaCheckError(cudaSetDevice(cnti));

      for (cntj = 0; cntj < chunksNy_lux; cntj ++) {
         cudaCheckError(cudaMemcpy3DAsync(&(gpuy_lux[cnti].psiy_h2d[cntj]), gpu[cnti].exec_stream[cntj]));
         calclux_kernel<<<gpuy_lux[cnti].dimGrid2d_lux, gpuy_lux[cnti].dimBlock2d_lux, 0, gpu[cnti].exec_stream[cntj]>>>(gpuy_lux[cnti].psiy[cntj], gpuy_lux[cnti].cbetay[cntj], gpuy_lux[cnti].calphax, gpuy_lux[cnti].cgammax);
         //cudaCheckError(cudaPeekAtLastError());
         cudaCheckError(cudaMemcpy3DAsync(&(gpuy_lux[cnti].psiy_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
      }
   }

   return;
}

__global__ void calclux_kernel(cudaPitchedPtr psi, cudaPitchedPtr cbeta, cuDoubleComplex *calphax, cuDoubleComplex *cgammax) {
   long cnti, cntj, cntk;
   cuDoubleComplex c;
   cuDoubleComplex *psirowprev, *psirowcurr, *psirownext, *cbetarowprev, *cbetarowcurr;

   for (cntj = TID_Y; cntj < d_chunkNy_lux; cntj += GRID_STRIDE_Y) {
      for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
         cbetarowcurr = get_complex_tensor_row(cbeta, d_Nx - 2, cntj);
         psirowcurr = get_complex_tensor_row(psi, d_Nx - 1, cntj);

         cbetarowcurr[cntk] = psirowcurr[cntk];

         for(cnti = d_Nx - 2; cnti > 0; cnti --) {
            cbetarowprev = get_complex_tensor_row(cbeta, cnti - 1, cntj);
            cbetarowcurr = get_complex_tensor_row(cbeta, cnti, cntj);

            psirowprev = get_complex_tensor_row(psi, cnti - 1, cntj);
            psirowcurr = get_complex_tensor_row(psi, cnti, cntj);
            psirownext = get_complex_tensor_row(psi, cnti + 1, cntj);

            c = cuCsub(cuCadd(cuCmul(d_minusAx, psirownext[cntk]), cuCmul(d_Ax0r, psirowcurr[cntk])), cuCmul(d_Ax, psirowprev[cntk]));
            cbetarowprev[cntk] = cuCmul(cgammax[cnti], cuCsub(cuCmul(d_Ax, cbetarowcurr[cntk]), c));
         }

         psirowcurr = get_complex_tensor_row(psi, 0, cntj);

         psirowcurr[cntk] = make_cuDoubleComplex(0., 0.);

         for(cnti = 0; cnti < d_Nx - 2; cnti ++) {
            cbetarowcurr = get_complex_tensor_row(cbeta, cnti, cntj);

            psirowcurr = get_complex_tensor_row(psi, cnti, cntj);
            psirownext = get_complex_tensor_row(psi, cnti + 1, cntj);

            psirownext[cntk] = cuCfma(calphax[cnti], psirowcurr[cntk], cbetarowcurr[cntk]);
         }

         psirowcurr = get_complex_tensor_row(psi, d_Nx - 1, cntj);

         psirowcurr[cntk] = make_cuDoubleComplex(0., 0.);
      }
   }
}

void calcluy_gpu_1ce() {
   int cnti, cntj;

   if (offload > 1) {
      for (cnti = 0; cnti < Ngpu; cnti ++) {
         cudaCheckError(cudaSetDevice(cnti));

         for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
            calcluy_kernel<<<gpux_nyz[cnti].dimGrid2d_luy, gpux_nyz[cnti].dimBlock2d_luy, 0, gpu[cnti].exec_stream[cntj]>>>(gpux_nyz[cnti].psix[cntj], gpux_nyz[cnti].cbetax[cntj], gpux_nyz[cnti].calphay, gpux_nyz[cnti].cgammay);
            //cudaCheckError(cudaPeekAtLastError());
         }

         if (offload == 2) {
            for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
               cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].psix_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
            }
         }
      }
   }

   return;
}

void calcluy_gpu_2ce() {
   int cnti, cntj;

   if (offload > 1) {
      for (cnti = 0; cnti < Ngpu; cnti ++) {
         cudaCheckError(cudaSetDevice(cnti));

         for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
            calcluy_kernel<<<gpux_nyz[cnti].dimGrid2d_luy, gpux_nyz[cnti].dimBlock2d_luy, 0, gpu[cnti].exec_stream[cntj]>>>(gpux_nyz[cnti].psix[cntj], gpux_nyz[cnti].cbetax[cntj], gpux_nyz[cnti].calphay, gpux_nyz[cnti].cgammay);
            //cudaCheckError(cudaPeekAtLastError());
            if (offload == 2) {
               cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].psix_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
            }
         }
      }
   }

   return;
}


__global__ void calcluy_kernel(cudaPitchedPtr psi, cudaPitchedPtr cbeta, cuDoubleComplex *calphay, cuDoubleComplex *cgammay) {
   long cnti, cntj, cntk;
   cuDoubleComplex c;
   cuDoubleComplex *psirowprev, *psirowcurr, *psirownext, *cbetarowprev, *cbetarowcurr;

   for (cnti = TID_Y; cnti < d_chunkNx_nyz; cnti += GRID_STRIDE_Y) {
      for (cntk = TID_X; cntk < d_Nz; cntk += GRID_STRIDE_X) {
         cbetarowcurr = get_complex_tensor_row(cbeta, cnti, d_Ny - 2);
         psirowcurr = get_complex_tensor_row(psi, cnti, d_Ny - 1);

         cbetarowcurr[cntk] = psirowcurr[cntk];

         for(cntj = d_Ny - 2; cntj > 0; cntj --) {
            cbetarowprev = get_complex_tensor_row(cbeta, cnti, cntj - 1);
            cbetarowcurr = get_complex_tensor_row(cbeta, cnti, cntj);

            psirowprev = get_complex_tensor_row(psi, cnti, cntj - 1);
            psirowcurr = get_complex_tensor_row(psi, cnti, cntj);
            psirownext = get_complex_tensor_row(psi, cnti, cntj + 1);

            c = cuCsub(cuCadd(cuCmul(d_minusAy, psirownext[cntk]), cuCmul(d_Ay0r, psirowcurr[cntk])), cuCmul(d_Ay, psirowprev[cntk]));
            cbetarowprev[cntk] = cuCmul(cgammay[cntj], cuCsub(cuCmul(d_Ay, cbetarowcurr[cntk]), c));
         }

         psirowcurr = get_complex_tensor_row(psi, cnti, 0);

         psirowcurr[cntk] = make_cuDoubleComplex(0., 0.);

         for(cntj = 0; cntj < d_Ny - 2; cntj ++) {
            cbetarowcurr = get_complex_tensor_row(cbeta, cnti, cntj);

            psirowcurr = get_complex_tensor_row(psi, cnti, cntj);
            psirownext = get_complex_tensor_row(psi, cnti, cntj + 1);

            psirownext[cntk] = cuCfma(calphay[cntj], psirowcurr[cntk], cbetarowcurr[cntk]);
         }

         psirowcurr = get_complex_tensor_row(psi, cnti, d_Ny - 1);

         psirowcurr[cntk] = make_cuDoubleComplex(0., 0.);
      }
   }
}

void calcluz_gpu_1ce() {
   int cnti, cntj;

   if (offload > 2) {
      for (cnti = 0; cnti < Ngpu; cnti ++) {
         cudaCheckError(cudaSetDevice(cnti));

         for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
            calcluz_kernel<<<gpux_nyz[cnti].dimGrid2d_luz, gpux_nyz[cnti].dimBlock2d_luz, 0, gpu[cnti].exec_stream[cntj]>>>(gpux_nyz[cnti].psix[cntj], gpux_nyz[cnti].cbetax[cntj], gpux_nyz[cnti].calphaz, gpux_nyz[cnti].cgammaz);
            //cudaCheckError(cudaPeekAtLastError());
         }

         for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
            cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].psix_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
         }
      }
   }

   return;
}

void calcluz_gpu_2ce() {
   int cnti, cntj;

   if (offload > 2) {
      for (cnti = 0; cnti < Ngpu; cnti ++) {
         cudaCheckError(cudaSetDevice(cnti));

         for (cntj = 0; cntj < chunksNx_nyz; cntj ++) {
            calcluz_kernel<<<gpux_nyz[cnti].dimGrid2d_luz, gpux_nyz[cnti].dimBlock2d_luz, 0, gpu[cnti].exec_stream[cntj]>>>(gpux_nyz[cnti].psix[cntj], gpux_nyz[cnti].cbetax[cntj], gpux_nyz[cnti].calphaz, gpux_nyz[cnti].cgammaz);
            //cudaCheckError(cudaPeekAtLastError());
            cudaCheckError(cudaMemcpy3DAsync(&(gpux_nyz[cnti].psix_d2h[cntj]), gpu[cnti].exec_stream[cntj]));
         }
      }
   }

   return;
}

__global__ void calcluz_kernel(cudaPitchedPtr psi, cudaPitchedPtr cbeta, cuDoubleComplex *calphaz, cuDoubleComplex *cgammaz) {
   long cnti, cntj, cntk;
   cuDoubleComplex c;
   cuDoubleComplex *psirow, *cbetarow;

   for (cnti = TID_Y; cnti < d_chunkNx_nyz; cnti += GRID_STRIDE_Y) {
      for (cntj = TID_X; cntj < d_Ny; cntj += GRID_STRIDE_X) {
         cbetarow = get_complex_tensor_row(cbeta, cnti, cntj);
         psirow = get_complex_tensor_row(psi, cnti, cntj);

         cbetarow[d_Nz - 2] = psirow[d_Nz - 1];

         for(cntk = d_Nz - 2; cntk > 0; cntk --) {
            c = cuCsub(cuCadd(cuCmul(d_minusAz, psirow[cntk + 1]), cuCmul(d_Az0r, psirow[cntk])), cuCmul(d_Az, psirow[cntk - 1]));
            cbetarow[cntk - 1] = cuCmul(cgammaz[cntk], cuCsub(cuCmul(d_Az, cbetarow[cntk]), c));
         }

         psirow[0] = make_cuDoubleComplex(0., 0.);

         for(cntk = 0; cntk < d_Nz - 2; cntk ++) {
            psirow[cntk + 1] = cuCfma(calphaz[cntk], psirow[cntk], cbetarow[cntk]);
         }

         psirow[d_Nz - 1] = make_cuDoubleComplex(0., 0.);
      }
   }
}
