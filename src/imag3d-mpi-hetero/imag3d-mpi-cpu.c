#include "imag3d-mpi-cpu.h"

int main(int argc, char **argv) {
   FILE *out;
   FILE *filerms;
   MPI_File mpifile;
   char filename[MAX_FILENAME_SIZE];
   int rankNx2;
   long offsetNx2;
   long cnti;
   double norm, mu, en;
   double *rms;
   double **cbeta;
   double ***dpsi, ***dpsi_t;
   double **tmpxi, **tmpyi, **tmpzi, **tmpxj, **tmpyj, **tmpzj;
   double **outx, **outy, **outz;
   double ***outxy, ***outxz, ***outyz;
   double ***outxyz;
   double psi2;

   if ((argc != 3) || (strcmp(argv[1], "-i") != 0)) {
      fprintf(stderr, "Usage: %s -i <input parameter file> \n", *argv);
      exit(EXIT_FAILURE);
   }

   if (! cfg_init(argv[2])) {
      fprintf(stderr, "Wrong input parameter file.\n");
      exit(EXIT_FAILURE);
   }

   int provided;
   MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

   #pragma omp parallel
   #pragma omp master
   nthreads = omp_get_num_threads();

   pi = 3.14159265358979;

   struct timeval start, stop, iter_start, iter_stop;
   double wall_time, init_time, iter_time;
   iter_time = 0.;
   gettimeofday(&start, NULL);

   readpar();

   assert(Nx % nprocs == 0);
   assert(Ny % nprocs == 0);
   assert((gpuNx_fft / nprocs) > 0 && chunksNx_fft > 0);
   assert((gpuNx_nyz / nprocs) > 0 && chunksNx_nyz > 0);
   assert((gpuNy_fft / nprocs) > 0 && chunksNy_fft > 0);
   assert((gpuNy_lux / nprocs) > 0 && chunksNy_lux > 0);
   assert((gpuNx_fft / nprocs) % chunksNx_fft == 0);
   assert((gpuNx_nyz / nprocs) % chunksNx_nyz == 0);
   assert((gpuNy_fft / nprocs) % chunksNy_fft == 0);
   assert((gpuNy_lux / nprocs) % chunksNy_lux == 0);
   assert(offload == 1 || offload == 2 || offload == 3);

   localNx = Nx / nprocs;
   localNy = Ny / nprocs;
   offsetNx = rank * localNx;
   offsetNy = rank * localNy;

   Nx2 = Nx / 2; Ny2 = Ny / 2; Nz2 = Nz / 2;
   dx2 = dx * dx; dy2 = dy * dy; dz2 = dz * dz;

   rankNx2 = Nx2 / localNx;
   offsetNx2 = Nx2 % localNx;

   gpuNx_fft /= nprocs;
   gpuNx_nyz /= nprocs;
   gpuNy_fft /= nprocs;
   gpuNy_lux /= nprocs;

   cpuNx_fft = localNx - gpuNx_fft;
   cpuNx_nyz = localNx - gpuNx_nyz;
   cpuNy_fft = localNy - gpuNy_fft;
   cpuNy_lux = localNy - gpuNy_lux;

   if (offload == 1) {
      cpuNx_luy = localNx;
      cpuNx_luz = localNx;
      calcpsi = &calcpsi_o1;
      nthreads_luy = nthreads;
      nthreads_luz = nthreads;
   } else if (offload == 2) {
      cpuNx_luy = cpuNx_nyz;
      cpuNx_luz = localNx;
      calcpsi = &calcpsi_o2;
      nthreads_luy = nthreads - 1;
      nthreads_luz = nthreads;
   } else if (offload == 3) {
      cpuNx_luy = cpuNx_nyz;
      cpuNx_luz = cpuNx_nyz;
      calcpsi = &calcpsi_o3;
      nthreads_luy = nthreads - 1;
      nthreads_luz = nthreads - 1;
   }

   assert(cpuNx_fft > 0);
   assert(cpuNx_nyz > 0);
   assert(cpuNy_fft > 0);
   assert(cpuNy_lux > 0);

   rms = alloc_double_vector(RMS_ARRAY_SIZE);

   x = alloc_double_vector(Nx);
   y = alloc_double_vector(Ny);
   z = alloc_double_vector(Nz);

   x2 = alloc_double_vector(Nx);
   y2 = alloc_double_vector(Ny);
   z2 = alloc_double_vector(Nz);

   pot = alloc_double_tensor(localNx, Ny, Nz);
   potdd = alloc_double_tensor(Nx, localNy, Nz);
   psi = alloc_double_tensor(localNx, Ny, Nz);
   psi_t = alloc_double_tensor(Nx, localNy, Nz);
   psidd2 = alloc_double_tensor(localNx, Ny, Nz);
   psidd2fft = alloc_fftw_complex_vector(localNx * Ny * (Nz2 + 1));
   psidd2fft_t = alloc_fftw_complex_vector(Nx * localNy * (Nz2 + 1));
   psidd2tmp = alloc_double_vector(Ny * Nz);

   dpsi = alloc_double_tensor(localNx, Ny, Nz);
   dpsi_t = psi_t;

   calphax = alloc_double_vector(Nx - 1);
   calphay = alloc_double_vector(Ny - 1);
   calphaz = alloc_double_vector(Nz - 1);
   cbeta = alloc_double_matrix(nthreads, MAX(Nx, Ny, Nz) - 1);
   cgammax = alloc_double_vector(Nx - 1);
   cgammay = alloc_double_vector(Ny - 1);
   cgammaz = alloc_double_vector(Nz - 1);

   tmpxi = alloc_double_matrix(nthreads, Nx);
   tmpyi = alloc_double_matrix(nthreads, Ny);
   tmpzi = alloc_double_matrix(nthreads, Nz);
   tmpxj = alloc_double_matrix(nthreads, Nx);
   tmpyj = alloc_double_matrix(nthreads, Ny);
   tmpzj = alloc_double_matrix(nthreads, Nz);

   outx = alloc_double_matrix(localNx, 2);
   outy = alloc_double_matrix(localNy, 2);
   outz = alloc_double_matrix(Nz, 2); // Because rank 0 will assemble outz
   outxy = alloc_double_tensor(localNx, Ny, 3);
   outxz = alloc_double_tensor(localNx, Nz, 3);
   outyz = alloc_double_tensor((rank == rankNx2) ? Ny : localNy, Nz, 3);
   outxyz = dpsi;

   pin_double_tensor(psi, localNx, Ny, Nz);
   pin_double_tensor(psi_t, Nx, localNy, Nz);
   pin_double_tensor(pot, localNx, Ny, Nz);
   pin_double_tensor(potdd, Nx, localNy, Nz);
   pin_double_tensor(psidd2, localNx, Ny, Nz);
   pin_double_vector((double *) psidd2fft, localNx * Ny * (Nz2 + 1) * 2);
   pin_double_vector((double *) psidd2fft_t, Nx * localNy * (Nz2 + 1) * 2);

   if (rank == 0) {
      if (output != NULL) {
         sprintf(filename, "%s.txt", output);
         out = fopen(filename, "w");
      } else out = stdout;
   } else out = fopen("/dev/null", "w");

   if (rank == 0) {
      if (rmsout != NULL) {
         sprintf(filename, "%s.txt", rmsout);
         filerms = fopen(filename, "w");
      } else filerms = NULL;
   } else filerms = fopen("/dev/null", "w");


   if (opt == 2) par = 2.;
   else par = 1.;

   fprintf(out, " Imaginary time propagation 3D,   OPTION = %d, OMP_NUM_THREADS = %d, MPI_NUM_PROCS = %d\n\n", opt, nthreads, nprocs);
   fprintf(out, "  Number of Atoms N = %li, Unit of length AHO = %.8f m\n", Na, aho);
   fprintf(out, "  Scattering length a = %.2f*a0, Dipolar ADD = %.2f*a0\n", as, add);
   fprintf(out, "  Nonlinearity G_3D = %.4f, Strength of DDI GD_3D = %.5f\n", g0, gd0);
   fprintf(out, "  Parameters of trap: GAMMA = %.2f, NU = %.2f, LAMBDA = %.2f\n\n", vgamma, vnu, vlambda);
   fprintf(out, " # Space Stp: NX = %li, NY = %li, NZ = %li\n", Nx, Ny, Nz);
   fprintf(out, "  Space Step: DX = %.6f, DY = %.6f, DZ = %.6f\n", dx, dy, dz);
   fprintf(out, " # Time Stp : NSTP = %li, NPAS = %li, NRUN = %li\n", Nstp, Npas, Nrun);
   fprintf(out, "   Time Step:   DT = %.6f\n",  dt);
   fprintf(out, "   Dipolar Cut off:   R = %.3f\n\n",  cutoff);
   fprintf(out, "                  --------------------------------------------------------\n");
   fprintf(out, "                    Norm      Chem        Ener/N     <r>     |Psi(0,0,0)|^2\n");
   fprintf(out, "                  --------------------------------------------------------\n");
   fflush(out);

   fftw_init_threads();
#ifdef FFTW_TRAN
   fftw_mpi_init();
#endif
   fftw_plan_with_nthreads(nthreads);

   omp_set_nested(0);

   int fft_rank = 2;
   int nfr[] = {Ny, Nz};
   int howmany = cpuNx_fft;
   int idist = Ny * Nz; int odist = Ny * (Nz2 + 1);
   int istride = 1; int ostride = 1;
   int *inembed = NULL, *onembed = NULL;

   plan_forward_row = fftw_plan_many_dft_r2c(fft_rank, nfr, howmany, **psidd2, inembed, istride, idist, psidd2fft, onembed, ostride, odist, FFT_FLAG);

   fft_rank = 1;
   int nfc[] = {Nx};
   howmany = cpuNy_fft * (Nz2 + 1);
   idist = 1; odist = 1;
   istride = localNy * (Nz2 + 1); ostride = localNy * (Nz2 + 1);

   plan_forward_col = fftw_plan_many_dft(fft_rank, nfc, howmany, psidd2fft_t, inembed, istride, idist, psidd2fft_t, onembed, ostride, odist, FFTW_FORWARD, FFT_FLAG);

   fft_rank = 1;
   int nbc[] = {Nx};
   howmany = cpuNy_fft * (Nz2 + 1);
   idist = 1; odist = 1;
   istride = localNy * (Nz2 + 1); ostride = localNy * (Nz2 + 1);

   plan_backward_col = fftw_plan_many_dft(fft_rank, nbc, howmany, psidd2fft_t, inembed, istride, idist, psidd2fft_t, onembed, ostride, odist, FFTW_BACKWARD, FFT_FLAG);

   fft_rank = 2;
   int nbr[] = {Ny, Nz};
   howmany = cpuNx_nyz;
   idist = Ny * (Nz2 + 1); odist = Ny * Nz;
   istride = 1; ostride = 1;

   plan_backward_row = fftw_plan_many_dft_c2r(fft_rank, nbr, howmany, psidd2fft, inembed, istride, idist, **psidd2, onembed, ostride, odist, FFT_FLAG);

#ifdef FFTW_TRAN
   plan_tr_psi_x = fftw_mpi_plan_many_transpose(Nx, Ny * Nz, 1, localNx, localNy * Nz, **psi, **psi_t, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_OUT);
   plan_tr_psi_y = fftw_mpi_plan_many_transpose(Ny * Nz, Nx, 1, localNy * Nz, localNx, **psi_t, **psi, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_IN);
   plan_tr_psidd2_x = fftw_mpi_plan_many_transpose(Nx, Ny * (Nz2 + 1), 2, localNx, localNy * (Nz2 + 1), (double *) psidd2fft, (double *) psidd2fft_t, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_OUT);
   plan_tr_psidd2_y = fftw_mpi_plan_many_transpose(Ny * (Nz2 + 1), Nx, 2, localNy * (Nz2 + 1), localNx, (double *) psidd2fft_t, (double *) psidd2fft, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_IN);
   plan_tr_dpsi = fftw_mpi_plan_many_transpose(Ny * Nz, Nx, 1, localNy * Nz, localNx, **dpsi_t, **dpsi, MPI_COMM_WORLD, FFT_FLAG | FFTW_MPI_TRANSPOSED_IN);
#else
   tran_psi = init_transpose_double(nprocs, localNx, localNy, Ny, Nz, **psi, Nz * sizeof(double), **psi_t, Nz * sizeof(double));
   tran_dd2 = init_transpose_complex(nprocs, localNx, localNy, Ny, Nz2 + 1, psidd2fft, (Nz2 + 1) * sizeof(fftw_complex), psidd2fft_t, (Nz2 + 1) * sizeof(fftw_complex));
   tran_dpsi = init_transpose_double(nprocs, localNx, localNy, Ny, Nz, **dpsi, Nz * sizeof(double), **dpsi_t, Nz * sizeof(double));
#endif

   omp_set_nested(1);

   initpsi(psi);
   initpot(pot);
   gencoef();
   initpotdd(*tmpxi, *tmpyi, *tmpzi, *tmpxj, *tmpyj, *tmpzj);

   init_gpu();


   calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
   calcmuen(&mu, &en, psi, psi_t, dpsi, dpsi_t, psidd2, psidd2fft, psidd2fft_t, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj);
   calcrms(rms, psi, psi_t, tmpxi, tmpyi, tmpzi);
   psi2 = psi[offsetNx2][Ny2][Nz2] * psi[offsetNx2][Ny2][Nz2];
   MPI_Bcast(&psi2, 1, MPI_DOUBLE, rankNx2, MPI_COMM_WORLD);
   fprintf(out, "Initial : %15.4f %11.5f %11.5f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psi2);
   fflush(out);
   if (initout != NULL) {
      sprintf(filename, "%s.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxyz(psi, outxyz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_x.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenx(psi, outx, *tmpyi, *tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_y.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdeny(psi_t, outy, *tmpxi, *tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_z.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenz(psi, outz, *tmpxi, *tmpyi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xy.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxy(psi, outxy, *tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xz.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxz(psi, outxz, *tmpyi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_yz.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenyz(psi_t, outyz, *tmpxi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_xy0.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xy(psi, outxy, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_x0z.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xz(psi, outxz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_0yz.bin", initout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2yz(psi, outyz, mpifile);
      MPI_File_close(&mpifile);
   }
   if (rmsout != NULL) {
      fprintf(filerms, " Imaginary time propagation 3D,   OPTION = %d\n\n", opt);
      fprintf(filerms, "                  --------------------------------------------------------\n");
      fprintf(filerms, "Values of rms size:     <r>          <x>          <y>          <z>\n");
      fprintf(filerms, "                  --------------------------------------------------------\n");
      fprintf(filerms, "           Initial:%12.5f %12.5f %12.5f %12.5f\n", rms[0], rms[1], rms[2], rms[3]);
      fflush(filerms);
   }

   // ---------------------------------------------------------------- NSTP
   if (Nstp != 0) {
      double g_stp = par * g0 / (double) Nstp;
      double gd_stp = par * gd0 / (double) Nstp;
      g = 0.;
      gd = 0.;
      gettimeofday(&iter_start, NULL);
      for (cnti = 0; cnti < Nstp; cnti ++) {
         g += g_stp;
         gd += gd_stp;
         calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
         calcpsidd2(psi, psidd2, psidd2fft, psidd2fft_t);
         calcpsi(psi, psi_t, psidd2, pot, cbeta);
      }
      gettimeofday(&iter_stop, NULL);
      iter_time += (double) (((iter_stop.tv_sec - iter_start.tv_sec) * 1000 + (iter_stop.tv_usec - iter_start.tv_usec)/1000.0) + 0.5);

      calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
      calcmuen(&mu, &en, psi, psi_t, dpsi, dpsi_t, psidd2, psidd2fft, psidd2fft_t, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj);
      calcrms(rms, psi, psi_t, tmpxi, tmpyi, tmpzi);
      psi2 = psi[offsetNx2][Ny2][Nz2] * psi[offsetNx2][Ny2][Nz2];
      MPI_Bcast(&psi2, 1, MPI_DOUBLE, rankNx2, MPI_COMM_WORLD);
      fprintf(out, "After NSTP iter.:%8.4f %11.5f %11.5f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psi2);
      fflush(out);
      if (rmsout != NULL) {
         fprintf(filerms, "  After NSTP iter.:%12.5f %12.5f %12.5f %12.5f\n", rms[0], rms[1], rms[2], rms[3]);
         fflush(filerms);
      }
      if (Nstpout != NULL) {
         sprintf(filename, "%s.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenxyz(psi, outxyz, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s1d_x.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenx(psi, outx, *tmpyi, *tmpzi, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s1d_y.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdeny(psi_t, outy, *tmpxi, *tmpzi, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s1d_z.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenz(psi, outz, *tmpxi, *tmpyi, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s2d_xy.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenxy(psi, outxy, *tmpzi, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s2d_xz.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenxz(psi, outxz, *tmpyi, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s2d_yz.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outdenyz(psi_t, outyz, *tmpxi, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s3d_xy0.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outpsi2xy(psi, outxy, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s3d_x0z.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outpsi2xz(psi, outxz, mpifile);
         MPI_File_close(&mpifile);

         sprintf(filename, "%s3d_0yz.bin", Nstpout);
         MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
         outpsi2yz(psi, outyz, mpifile);
         MPI_File_close(&mpifile);
      }
   } else {
      g = par * g0;
      gd = par * gd0;
   }

   // ---------------------------------------------------------------- NPAS
   gettimeofday(&iter_start, NULL);
   for (cnti = 0; cnti < Npas; cnti ++) {
      calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
      calcpsidd2(psi, psidd2, psidd2fft, psidd2fft_t);
      calcpsi(psi, psi_t, psidd2, pot, cbeta);
   }
   gettimeofday(&iter_stop, NULL);
   iter_time += (double) (((iter_stop.tv_sec - iter_start.tv_sec) * 1000 + (iter_stop.tv_usec - iter_start.tv_usec)/1000.0) + 0.5);

   calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
   calcmuen(&mu, &en, psi, psi_t, dpsi, dpsi_t, psidd2, psidd2fft, psidd2fft_t, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj);
   calcrms(rms, psi, psi_t, tmpxi, tmpyi, tmpzi);
   psi2 = psi[offsetNx2][Ny2][Nz2] * psi[offsetNx2][Ny2][Nz2];
   MPI_Bcast(&psi2, 1, MPI_DOUBLE, rankNx2, MPI_COMM_WORLD);
   fprintf(out, "After NPAS iter.:%8.4f %11.5f %11.5f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psi2);
   fflush(out);
   if (Npasout != NULL) {
      sprintf(filename, "%s.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxyz(psi, outxyz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_x.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenx(psi, outx, *tmpyi, *tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_y.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdeny(psi_t, outy, *tmpxi, *tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_z.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenz(psi, outz, *tmpxi, *tmpyi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xy.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxy(psi, outxy, *tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xz.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxz(psi, outxz, *tmpyi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_yz.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenyz(psi_t, outyz, *tmpxi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_xy0.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xy(psi, outxy, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_x0z.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xz(psi, outxz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_0yz.bin", Npasout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2yz(psi, outyz, mpifile);
      MPI_File_close(&mpifile);
   }
   if (rmsout != NULL) {
      fprintf(filerms, "  After NPAS iter.:%12.5f %12.5f %12.5f %12.5f\n", rms[0], rms[1], rms[2], rms[3]);
      fflush(filerms);
   }

   // ---------------------------------------------------------------- NRUN
   gettimeofday(&iter_start, NULL);
   for (cnti = 0; cnti < Nrun; cnti ++) {
      calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
      calcpsidd2(psi, psidd2, psidd2fft, psidd2fft_t);
      calcpsi(psi, psi_t, psidd2, pot, cbeta);
   }
   gettimeofday(&iter_stop, NULL);
   iter_time += (double) (((iter_stop.tv_sec - iter_start.tv_sec) * 1000 + (iter_stop.tv_usec - iter_start.tv_usec)/1000.0) + 0.5);

   calcnorm(&norm, psi, tmpxi, tmpyi, tmpzi);
   calcmuen(&mu, &en, psi, psi_t, dpsi, dpsi_t, psidd2, psidd2fft, psidd2fft_t, tmpxi, tmpyi, tmpzi, tmpxj, tmpyj, tmpzj);
   calcrms(rms, psi, psi_t, tmpxi, tmpyi, tmpzi);
   psi2 = psi[offsetNx2][Ny2][Nz2] * psi[offsetNx2][Ny2][Nz2];
   MPI_Bcast(&psi2, 1, MPI_DOUBLE, rankNx2, MPI_COMM_WORLD);
   fprintf(out, "After NRUN iter.:%8.4f %11.5f %11.5f %10.5f %10.5f\n", norm, mu / par, en / par, *rms, psi2);
   fflush(out);
   if (Nrunout != NULL) {
      sprintf(filename, "%s.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxyz(psi, outxyz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_x.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenx(psi, outx, *tmpyi, *tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_y.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdeny(psi_t, outy, *tmpxi, *tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s1d_z.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenz(psi, outz, *tmpxi, *tmpyi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xy.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxy(psi, outxy, *tmpzi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_xz.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenxz(psi, outxz, *tmpyi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s2d_yz.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outdenyz(psi_t, outyz, *tmpxi, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_xy0.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xy(psi, outxy, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_x0z.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2xz(psi, outxz, mpifile);
      MPI_File_close(&mpifile);

      sprintf(filename, "%s3d_0yz.bin", Nrunout);
      MPI_File_open(MPI_COMM_WORLD, filename, MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &mpifile);
      outpsi2yz(psi, outyz, mpifile);
      MPI_File_close(&mpifile);
   }
   if (rmsout != NULL) {
      fprintf(filerms, "  After NRUN iter.:%12.5f %12.5f %12.5f %12.5f\n", rms[0], rms[1], rms[2], rms[3]);
      fprintf(filerms, "                  --------------------------------------------------------\n");
   }

   fprintf(out, "                  --------------------------------------------------------\n\n");

   if (rmsout != NULL) fclose(filerms);

   free_double_vector(rms);

   free_double_vector(x);
   free_double_vector(y);
   free_double_vector(z);

   free_double_vector(x2);
   free_double_vector(y2);
   free_double_vector(z2);

   free_double_tensor(pot);
   free_double_tensor(potdd);
   free_double_tensor(psi);
   free_double_tensor(psi_t);
   free_double_tensor(psidd2);
   free_double_tensor(dpsi);
   free_double_vector(psidd2tmp);
   free_fftw_complex_vector(psidd2fft);
   free_fftw_complex_vector(psidd2fft_t);

   free_double_vector(calphax);
   free_double_vector(calphay);
   free_double_vector(calphaz);
   free_double_matrix(cbeta);
   free_double_vector(cgammax);
   free_double_vector(cgammay);
   free_double_vector(cgammaz);

   free_double_matrix(tmpxi);
   free_double_matrix(tmpyi);
   free_double_matrix(tmpzi);
   free_double_matrix(tmpxj);
   free_double_matrix(tmpyj);
   free_double_matrix(tmpzj);

   fftw_destroy_plan(plan_forward_row);
   fftw_destroy_plan(plan_forward_col);
   fftw_destroy_plan(plan_backward_row);
   fftw_destroy_plan(plan_backward_col);
#ifdef FFTW_TRAN
   fftw_destroy_plan(plan_tr_psi_x);
   fftw_destroy_plan(plan_tr_psi_y);
   fftw_destroy_plan(plan_tr_psidd2_x);
   fftw_destroy_plan(plan_tr_psidd2_y);
   fftw_destroy_plan(plan_tr_dpsi);
#else
   free_transpose(tran_psi);
   free_transpose(tran_dd2);
   free_transpose(tran_dpsi);
#endif

   free_double_matrix(outx);
   free_double_matrix(outy);
   free_double_matrix(outz);
   free_double_tensor(outxy);
   free_double_tensor(outxz);
   free_double_tensor(outyz);

#ifdef FFTW_TRAN
   fftw_mpi_cleanup();
#endif

   MPI_Finalize();

   free_gpu();

   gettimeofday(&stop, NULL);
   wall_time = (double) (((stop.tv_sec - start.tv_sec) * 1000 + (stop.tv_usec - start.tv_usec)/1000.0) + 0.5);
   init_time = wall_time - iter_time;
   fprintf(out, " Initialization/allocation wall-clock time: %.3f seconds\n", init_time / 1000.);
   fprintf(out, " Calculation (iterations) wall-clock time: %.3f seconds\n", iter_time / 1000.);

   if(output != NULL) fclose(out);

   return (EXIT_SUCCESS);
}

/**
 *    Reading input parameters from the configuration file.
 */
void readpar(void) {
   char *cfg_tmp;

   if ((cfg_tmp = cfg_read("OPTION")) == NULL) {
      fprintf(stderr, "OPTION is not defined in the configuration file\n");
      exit(EXIT_FAILURE);
   }
   opt = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NATOMS")) == NULL) {
      fprintf(stderr, "NATOMS is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Na = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("G0")) == NULL) {
      if ((cfg_tmp = cfg_read("AHO")) == NULL) {
         fprintf(stderr, "AHO is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      aho = atof(cfg_tmp);

      if ((cfg_tmp = cfg_read("AS")) == NULL) {
         fprintf(stderr, "AS is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      as = atof(cfg_tmp);

      g0 = 4. * pi * as * Na * BOHR_RADIUS / aho;
   } else {
      g0 = atof(cfg_tmp);
   }

   if ((cfg_tmp = cfg_read("GDD0")) == NULL) {
      if ((cfg_tmp = cfg_read("AHO")) == NULL) {
         fprintf(stderr, "AHO is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      aho = atof(cfg_tmp);

      if ((cfg_tmp = cfg_read("ADD")) == NULL) {
         fprintf(stderr, "ADD is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      add = atof(cfg_tmp);

      gd0 = 3. * add * Na * BOHR_RADIUS / aho;
   } else {
      gd0 = atof(cfg_tmp);
   }

   if ((cfg_tmp = cfg_read("NX")) == NULL) {
      fprintf(stderr, "NX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nx = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NY")) == NULL) {
      fprintf(stderr, "NY is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Ny = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NZ")) == NULL) {
      fprintf(stderr, "Nz is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nz = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NGPU")) == NULL) {
      fprintf(stderr, "NGPU is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Ngpu = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("OFFLOAD")) == NULL) {
      fprintf(stderr, "OFFLOAD is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   offload = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("GPUNX_FFT")) == NULL) {
      fprintf(stderr, "GPUNX_FFT is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   gpuNx_fft = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("CHUNKSX_FFT")) == NULL) {
      fprintf(stderr, "CHUNKSX_FFT is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   chunksNx_fft = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("GPUNX_NYZ")) == NULL) {
      fprintf(stderr, "GPUNX_NYZ is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   gpuNx_nyz = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("CHUNKSX_NYZ")) == NULL) {
      fprintf(stderr, "CHUNKSX_NYZ is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   chunksNx_nyz = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("GPUNY_FFT")) == NULL) {
      fprintf(stderr, "GPUNY_FFT is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   gpuNy_fft = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("CHUNKSY_FFT")) == NULL) {
      fprintf(stderr, "CHUNKSY_FFT is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   chunksNy_fft = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("GPUNY_LUX")) == NULL) {
      fprintf(stderr, "GPUNY_LUX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   gpuNy_lux = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("CHUNKSY_LUX")) == NULL) {
      fprintf(stderr, "CHUNKSY_LUX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   chunksNy_lux = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("BLOCK3D_DD2")) == NULL) {
      fprintf(stderr, "BLOCK3D_DD2 is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   sscanf(cfg_tmp, "%d,%d,%d", &(block3d_dd2[0]), &(block3d_dd2[1]), &(block3d_dd2[2]));

   if ((cfg_tmp = cfg_read("BLOCK2D_DD2")) == NULL) {
      fprintf(stderr, "BLOCK2D_DD2 is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   sscanf(cfg_tmp, "%d,%d", &(block2d_dd2[0]), &(block2d_dd2[1]));

   if ((cfg_tmp = cfg_read("BLOCK2D_LUY")) == NULL) {
      fprintf(stderr, "BLOCK2D_LUY is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   sscanf(cfg_tmp, "%d,%d", &(block2d_luy[0]), &(block2d_luy[1]));

   if ((cfg_tmp = cfg_read("BLOCK2D_LUZ")) == NULL) {
      fprintf(stderr, "BLOCK2D_LUZ is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   sscanf(cfg_tmp, "%d,%d", &(block2d_luz[0]), &(block2d_luz[1]));

   if ((cfg_tmp = cfg_read("BLOCK3D_NU")) == NULL) {
      fprintf(stderr, "BLOCK3D_NU is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   sscanf(cfg_tmp, "%d,%d,%d", &(block3d_nu[0]), &(block3d_nu[1]), &(block3d_nu[2]));

   if ((cfg_tmp = cfg_read("BLOCK2D_LUX")) == NULL) {
      fprintf(stderr, "BLOCK2D_LUX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   sscanf(cfg_tmp, "%d,%d", &(block2d_lux[0]), &(block2d_lux[1]));

   if ((cfg_tmp = cfg_read("BLOCK3D_POTDD")) == NULL) {
      fprintf(stderr, "BLOCK3D_POTDD is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   sscanf(cfg_tmp, "%d,%d,%d", &(block3d_potdd[0]), &(block3d_potdd[1]), &(block3d_potdd[2]));

   if ((cfg_tmp = cfg_read("GRID3D_DD2")) == NULL) {
      fprintf(stderr, "GRID3D_DD2 is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   grid3d_dd2 = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("GRID2D_DD2")) == NULL) {
      fprintf(stderr, "GRID2D_DD2 is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   grid2d_dd2 = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("GRID2D_LUY")) == NULL) {
      fprintf(stderr, "GRID2D_LUY is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   grid2d_luy = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("GRID2D_LUZ")) == NULL) {
      fprintf(stderr, "GRID2D_LUZ is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   grid2d_luz = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("GRID3D_NU")) == NULL) {
      fprintf(stderr, "GRID3D_NU is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   grid3d_nu = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("GRID2D_LUX")) == NULL) {
      fprintf(stderr, "GRID2D_LUX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   grid2d_lux = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("GRID3D_POTDD")) == NULL) {
      fprintf(stderr, "GRID3D_POTDD is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   grid3d_potdd = atoi(cfg_tmp);

   if ((cfg_tmp = cfg_read("DX")) == NULL) {
      fprintf(stderr, "DX is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dx = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("DY")) == NULL) {
      fprintf(stderr, "DY is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dy = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("DZ")) == NULL) {
      fprintf(stderr, "DZ is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dz = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("DT")) == NULL) {
      fprintf(stderr, "DT is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   dt = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("GAMMA")) == NULL) {
      fprintf(stderr, "GAMMA is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vgamma = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("NU")) == NULL) {
      fprintf(stderr, "NU is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vnu = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("LAMBDA")) == NULL) {
      fprintf(stderr, "LAMBDA is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   vlambda = atof(cfg_tmp);

   if ((cfg_tmp = cfg_read("NSTP")) == NULL) {
      fprintf(stderr, "NSTP is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nstp = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NPAS")) == NULL) {
      fprintf(stderr, "NPAS is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Npas = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("NRUN")) == NULL) {
      fprintf(stderr, "NRUN is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   Nrun = atol(cfg_tmp);

   if ((cfg_tmp = cfg_read("CUTOFF")) == NULL) {
      fprintf(stderr, "CUTOFF is not defined in the configuration file.\n");
      exit(EXIT_FAILURE);
   }
   cutoff = atof(cfg_tmp);

   output = cfg_read("OUTPUT");
   rmsout = cfg_read("RMSOUT");
   initout = cfg_read("INITOUT");
   Nstpout = cfg_read("NSTPOUT");
   Npasout = cfg_read("NPASOUT");
   Nrunout = cfg_read("NRUNOUT");

   if ((initout != NULL) || (Nstpout != NULL) || (Npasout != NULL) || (Nrunout != NULL)) {
      if ((cfg_tmp = cfg_read("OUTSTPX")) == NULL) {
         fprintf(stderr, "OUTSTPX is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpx = atol(cfg_tmp);

      if ((cfg_tmp = cfg_read("OUTSTPY")) == NULL) {
         fprintf(stderr, "OUTSTPY is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpy = atol(cfg_tmp);

      if ((cfg_tmp = cfg_read("OUTSTPZ")) == NULL) {
         fprintf(stderr, "OUTSTPZ is not defined in the configuration file.\n");
         exit(EXIT_FAILURE);
      }
      outstpz = atol(cfg_tmp);
   }

   return;
}

/**
 *    Initialization of the space mesh and the initial wave function.
 *    psi - array with the wave function values
 */
void initpsi(double ***psi) {
   long cnti, cntj, cntk;
   double cpsi;
   double tmp;

   cpsi = sqrt(pi * sqrt(pi / (vgamma * vnu * vlambda)));

   for (cnti = 0; cnti < Nx; cnti ++) {
      x[cnti] = (cnti - Nx2) * dx;
      x2[cnti] = x[cnti] * x[cnti];
   }

   for (cntj = 0; cntj < Ny; cntj ++) {
      y[cntj] = (cntj - Ny2) * dy;
      y2[cntj] = y[cntj] * y[cntj];
   }

   for (cntk = 0; cntk < Nz; cntk ++) {
      z[cntk] = (cntk - Nz2) * dz;
      z2[cntk] = z[cntk] * z[cntk];
   }

   #pragma omp parallel for private(cnti, cntj, cntk, tmp)
   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmp = exp(- 0.5 * (vgamma * x2[offsetNx + cnti] + vnu * y2[cntj] + vlambda * z2[cntk]));
            psi[cnti][cntj][cntk] = tmp / cpsi;
         }
      }
   }

   return;
}

/**
 *    Initialization of the potential.
 *    pot - array with the potential
 */
void initpot(double ***pot) {
   long cnti, cntj, cntk;
   double vnu2, vlambda2, vgamma2;

   vnu2 = vnu * vnu;
   vlambda2 = vlambda * vlambda;
   vgamma2 = vgamma * vgamma;

   #pragma omp parallel for private(cnti, cntj, cntk)
   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            pot[cnti][cntj][cntk] = 0.5 * par * (vgamma2 * x2[offsetNx + cnti] + vnu2 * y2[cntj] + vlambda2 * z2[cntk]);
         }
      }
   }

   return;
}

/**
 *    Crank-Nicolson scheme coefficients generation.
 */
void gencoef(void) {
   long cnti;

   Ax0 = 1. + dt / dx2 / (3. - par);
   Ay0 = 1. + dt / dy2 / (3. - par);
   Az0 = 1. + dt / dz2 / (3. - par);

   Ax0r = 1. - dt / dx2 / (3. - par);
   Ay0r = 1. - dt / dy2 / (3. - par);
   Az0r = 1. - dt / dz2 / (3. - par);

   Ax = - 0.5 * dt / dx2 / (3. - par);
   Ay = - 0.5 * dt / dy2 / (3. - par);
   Az = - 0.5 * dt / dz2 / (3. - par);

   calphax[Nx - 2] = 0.;
   cgammax[Nx - 2] = - 1. / Ax0;
   for (cnti = Nx - 2; cnti > 0; cnti --) {
      calphax[cnti - 1] = Ax * cgammax[cnti];
      cgammax[cnti - 1] = - 1. / (Ax0 + Ax * calphax[cnti - 1]);
   }

   calphay[Ny - 2] = 0.;
   cgammay[Ny - 2] = - 1. / Ay0;
   for (cnti = Ny - 2; cnti > 0; cnti --) {
      calphay[cnti - 1] = Ay * cgammay[cnti];
      cgammay[cnti - 1] = - 1. / (Ay0 + Ay * calphay[cnti - 1]);
   }

   calphaz[Nz - 2] = 0.;
   cgammaz[Nz - 2] = - 1. / Az0;
   for (cnti = Nz - 2; cnti > 0; cnti --) {
      calphaz[cnti - 1] = Az * cgammaz[cnti];
      cgammaz[cnti - 1] = - 1. / (Az0 + Az * calphaz[cnti - 1]);
   }

   return;
}

/**
 *    Initialization of the dipolar potential.
 *    kx  - array with the space mesh values in the x-direction in the K-space
 *    ky  - array with the space mesh values in the y-direction in the K-space
 *    kz  - array with the space mesh values in the z-direction in the K-space
 *    kx2 - array with the squared space mesh values in the x-direction in the
 *          K-space
 *    ky2 - array with the squared space mesh values in the y-direction in the
 *          K-space
 *    kz2 - array with the squared space mesh values in the z-direction in the
 *          K-space
 */
void initpotdd(double *kx, double *ky, double *kz, double *kx2, double *ky2, double *kz2) {
   long cnti, cntj, cntk;
   double dkx, dky, dkz, xk, tmp;

   dkx = 2. * pi / (Nx * dx);
   dky = 2. * pi / (Ny * dy);
   dkz = 2. * pi / (Nz * dz);

   for (cnti = 0; cnti < Nx2; cnti ++) kx[cnti] = cnti * dkx;
   for (cnti = 0; cnti < Nx2; cnti ++) kx[cnti + Nx2] = (cnti - Nx2) * dkx;
   for (cntj = 0; cntj < Ny2; cntj ++) ky[cntj] = cntj * dky;
   for (cntj = 0; cntj < Ny2; cntj ++) ky[cntj + Ny2] = (cntj - Ny2) * dky;
   for (cntk = 0; cntk < Nz2; cntk ++) kz[cntk] = cntk * dkz;
   for (cntk = 0; cntk < Nz2; cntk ++) kz[cntk + Nz2] = (cntk - Nz2) * dkz;

   for (cnti = 0; cnti < Nx; cnti ++) kx2[cnti] = kx[cnti] * kx[cnti];
   for (cntj = 0; cntj < localNy; cntj ++) ky2[cntj] = ky[offsetNy + cntj] * ky[offsetNy + cntj];
   for (cntk = 0; cntk < Nz; cntk ++) kz2[cntk] = kz[cntk] * kz[cntk];

   #pragma omp parallel for private(cnti, cntj, cntk, xk, tmp)
   for (cnti = 0; cnti < Nx; cnti ++) {
      for (cntj = 0; cntj < localNy; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            xk = sqrt(kz2[cntk] + kx2[cnti] + ky2[cntj]);
            tmp = 1. + 3. * cos(xk * cutoff) / (xk * xk * cutoff * cutoff) - 3. * sin(xk * cutoff) / (xk * xk * xk * cutoff * cutoff * cutoff);
            potdd[cnti][cntj][cntk] = (4. * pi * (3. * kz2[cntk] / (kx2[cnti] + ky2[cntj] + kz2[cntk]) - 1.) / 3.) * tmp;
         }
      }
   }

   if (rank == 0) {
      potdd[0][0][0] = 0.;
   }

   return;
}

/**
 *    Calculation of the wave function norm and normalization.
 *    norm - wave function norm
 *    psi  - array with the wave function values
 *    tmpx - temporary array
 *    tmpy - temporary array
 *    tmpz - temporary array
 */
/*void calcnorm(double *norm, double ***psi, double **tmpx, double **tmpy, double **tmpz) {
   int threadid;
   long cnti, cntj, cntk;
   double tmp;
   void *sendbuf;

   calcnorm_gpu_p1();

   #pragma omp parallel private(threadid, cnti, cntj, cntk)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               tmpz[threadid][cntk] = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
            }
            tmpy[threadid][cntj] = simpint(dz, tmpz[threadid], Nz);
         }
         (*tmpx)[cnti] = simpint(dy, tmpy[threadid], Ny);
      }
   }

   sync_with_gpu();

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      *norm = sqrt(simpint(dx, *tmpx, Nx));
   }

   MPI_Bcast(norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   tmp = 1. / *norm;

   calcnorm_gpu_p2(tmp);

   #pragma omp for private(cnti, cntj, cntk)
   for (cnti = 0; cnti < localNx; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            psi[cnti][cntj][cntk] *= tmp;
         }
      }
   }

   sync_with_gpu();

   return;
}*/

void calcnorm(double *norm, double ***psi, double **tmpx, double **tmpy, double **tmpz) {
   int threadid;

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcnorm_gpu_p1();
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcnorm_cpu_p1(norm, psi, tmpx, tmpy, tmpz);
      }
   }

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcnorm_gpu_p2(1. / *norm);
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcnorm_cpu_p2(norm, psi);
      }
   }

   return;
}

void calcnorm_cpu_p1(double *norm, double ***psi, double **tmpx, double **tmpy, double **tmpz) {
   int threadid;
   long cnti, cntj, cntk;
   void *sendbuf;

   #pragma omp parallel private(threadid, cnti, cntj, cntk)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               tmpz[threadid][cntk] = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
            }
            tmpy[threadid][cntj] = simpint(dz, tmpz[threadid], Nz);
         }
         (*tmpx)[cnti] = simpint(dy, tmpy[threadid], Ny);
      }
   }

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      *norm = sqrt(simpint(dx, *tmpx, Nx));
   }

   MPI_Bcast(norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
}

void calcnorm_cpu_p2(double *norm, double ***psi) {
   long cnti, cntj, cntk;
   double tmp;

   tmp = 1. / *norm;

   #pragma omp parallel for private(cnti, cntj, cntk) num_threads(nthreads - 1)
   for (cnti = 0; cnti < cpuNx_fft; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            psi[cnti][cntj][cntk] *= tmp;
         }
      }
   }
}

/**
 *    Calculation of the chemical potential and energy.
 *    mu          - chemical potential
 *    en          - energy
 *    psi         - array with the wave function values
 *    psi_t       - array with the transposed wave function values
 *    dpsi        - temporary array
 *    dpsi_t      - temporary array
 *    psidd2      - array with the squared wave function values
 *    psidd2fft   - array with the squared wave function fft values
 *    psidd2fft_t - array with the transposed squared wave function fft values
 *    tmpxi       - temporary array
 *    tmpyi       - temporary array
 *    tmpzi       - temporary array
 *    tmpxj       - temporary array
 *    tmpyj       - temporary array
 *    tmpzj       - temporary array
 */
void calcmuen(double *mu, double *en, double ***psi, double ***psi_t, double ***dpsi, double ***dpsi_t, double ***psidd2, fftw_complex *psidd2fft, fftw_complex *psidd2fft_t, double **tmpxi, double **tmpyi, double **tmpzi, double **tmpxj, double **tmpyj, double **tmpzj) {
   int threadid;
   long cnti, cntj, cntk;
   double psi2, psi2lin, psidd2lin, dpsi2;
   void *sendbuf;

   calcpsidd2(psi, psidd2, psidd2fft, psidd2fft_t);
   copy_psidd2_d2h();

   TRAN_PSI_X

   #pragma omp parallel private(threadid, cnti, cntj, cntk)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cntj = 0; cntj < localNy; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cnti = 0; cnti < Nx; cnti ++) {
               tmpxi[threadid][cnti] = psi_t[cnti][cntj][cntk];
            }
            diff(dx, tmpxi[threadid], tmpxj[threadid], Nx);
            for (cnti = 0; cnti < Nx; cnti ++) {
               dpsi_t[cnti][cntj][cntk] = tmpxj[threadid][cnti] * tmpxj[threadid][cnti];
            }
         }
      }
   }

   TRAN_DPSI

   #pragma omp parallel private(threadid, cnti, cntj, cntk, psi2, psi2lin, psidd2lin, dpsi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cntj = 0; cntj < Ny; cntj ++) {
               tmpyi[threadid][cntj] = psi[cnti][cntj][cntk];
            }
            diff(dy, tmpyi[threadid], tmpyj[threadid], Ny);
            for (cntj = 0; cntj < Ny; cntj ++) {
               dpsi[cnti][cntj][cntk] += tmpyj[threadid][cntj] * tmpyj[threadid][cntj];
            }
         }
      }

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               tmpzi[threadid][cntk] = psi[cnti][cntj][cntk];
            }
            diff(dz, tmpzi[threadid], tmpzj[threadid], Nz);
            for (cntk = 0; cntk < Nz; cntk ++) {
               dpsi[cnti][cntj][cntk] += tmpzj[threadid][cntk] * tmpzj[threadid][cntk];
            }
         }
      }
      #pragma omp barrier

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               psi2 = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
               psi2lin = psi2 * g;
               psidd2lin = psidd2[cnti][cntj][cntk] * gd;
               dpsi2 = dpsi[cnti][cntj][cntk] / (3. - par);
               tmpzi[threadid][cntk] = (pot[cnti][cntj][cntk] + psi2lin + psidd2lin) * psi2 + dpsi2;
               tmpzj[threadid][cntk] = (pot[cnti][cntj][cntk] + 0.5 * psi2lin + 0.5 * psidd2lin) * psi2 + dpsi2;
            }
            tmpyi[threadid][cntj] = simpint(dz, tmpzi[threadid], Nz);
            tmpyj[threadid][cntj] = simpint(dz, tmpzj[threadid], Nz);
         }
         (*tmpxi)[cnti] = simpint(dy, tmpyi[threadid], Ny);
         (*tmpxj)[cnti] = simpint(dy, tmpyj[threadid], Ny);
      }
   }

   MPI_Barrier(MPI_COMM_WORLD);

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpxi;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpxi, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);
   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpxj;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpxj, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      *mu = simpint(dx, *tmpxi, Nx);
      *en = simpint(dx, *tmpxj, Nx);
   }

   return;
}

/**
 *    Calculation of squared wave function values for dipole-dipole
 *    interaction.
 *    psi         - array with the wave function values
 *    psidd2      - array with the squared wave function values
 *    psidd2fft   - array with the squared wave function fft values
 *    psidd2fft_t - array with the transposed squared wave function fft values
 */
/*void calcpsidd2(double ***psi, double ***psidd2, fftw_complex *psidd2fft, fftw_complex *psidd2fft_t) {
   long cnti, cntj, cntk;
   double tmp;
   int ismax = 0;

   calcpsidd2_gpu_p1();

   #pragma omp parallel for private(cnti, cntj, cntk, tmp)
   for (cnti = 0; cnti < cpuNx_fft; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmp = psi[cnti][cntj][cntk];
            psidd2[cnti][cntj][cntk] = tmp * tmp;
         }
      }
   }

   fftw_execute_dft_r2c(plan_forward_row, **psidd2, psidd2fft);
   sync_with_gpu();

   TRAN_PSIDD2_X

   calcpsidd2_gpu_p2();
   fftw_execute_dft(plan_forward_col, psidd2fft_t, psidd2fft_t);

   #pragma omp parallel for private(cnti, cntj, cntk)
   for (cnti = 0; cnti < Nx; cnti ++) {
      for (cntj = 0; cntj < cpuNy_fft; cntj ++) {
         for (cntk = 0; cntk < Nz2 + 1; cntk ++) {
            psidd2fft_t[cnti * localNy * (Nz2 + 1) + cntj * (Nz2 + 1) + cntk][0] *= potdd[cnti][cntj][cntk];
            psidd2fft_t[cnti * localNy * (Nz2 + 1) + cntj * (Nz2 + 1) + cntk][1] *= potdd[cnti][cntj][cntk];
         }
      }
   }

   fftw_execute_dft(plan_backward_col, psidd2fft_t, psidd2fft_t);
   sync_with_gpu();

   TRAN_PSIDD2_Y

   calcpsidd2_gpu_p3();
   fftw_execute_dft_c2r(plan_backward_row, psidd2fft, **psidd2);
   sync_with_gpu();

   if (nprocs > 1) {
      if (rank == 0) {
         MPI_Send(**psidd2, Ny * Nz, MPI_DOUBLE, nprocs - 1, 0, MPI_COMM_WORLD);
      } else if (rank == nprocs - 1) {
         MPI_Recv(psidd2tmp, Ny * Nz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         ismax = 1;
      }
   } else {
      memcpy(psidd2tmp, psidd2[0][0], Ny * Nz * sizeof(double));
      ismax = 1;
   }

   calcpsidd2_gpu_p4(ismax);

   #pragma omp parallel for private(cnti, cntj, cntk)
   for (cnti = 0; cnti < cpuNx_nyz; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            psidd2[cnti][cntj][cntk] /= Nx * Ny * Nz;
         }
      }
   }

   #pragma omp parallel for private(cnti, cntk)
   for (cnti = 0; cnti < cpuNx_nyz; cnti ++) {
      for (cntk = 0; cntk < Nz - 1; cntk ++) {
         psidd2[cnti][Ny - 1][cntk] = psidd2[cnti][0][cntk];
      }
   }

   #pragma omp parallel for private(cnti, cntj)
   for (cnti = 0; cnti < cpuNx_nyz; cnti ++) {
      for (cntj = 0; cntj < Ny - 1; cntj ++) {
         psidd2[cnti][cntj][Nz - 1] = psidd2[cnti][cntj][0];
      }
   }

   sync_with_gpu();

   return;
}*/

void calcpsidd2(double ***psi, double ***psidd2, fftw_complex *psidd2fft, fftw_complex *psidd2fft_t) {
   int threadid;
   int ismax = 0;

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcpsidd2_gpu_p1();
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcpsidd2_cpu_p1(psi, psidd2, psidd2fft);
      }
   }

   TRAN_PSIDD2_X

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcpsidd2_gpu_p2();
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcpsidd2_cpu_p2(psidd2fft_t);
      }
   }

   TRAN_PSIDD2_Y

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcpsidd2_gpu_p3();
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcpsidd2_cpu_p3(psidd2, psidd2fft);
      }
   }

   if (nprocs > 1) {
      if (rank == 0) {
         MPI_Send(**psidd2, Ny * Nz, MPI_DOUBLE, nprocs - 1, 0, MPI_COMM_WORLD);
      } else if (rank == nprocs - 1) {
         MPI_Recv(psidd2tmp, Ny * Nz, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
         ismax = 1;
      }
   } else {
      memcpy(psidd2tmp, psidd2[0][0], Ny * Nz * sizeof(double));
      ismax = 1;
   }

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcpsidd2_gpu_p4(ismax);
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcpsidd2_cpu_p4(psidd2);
      }
   }

   return;
}

void calcpsidd2_cpu_p1(double ***psi, double ***psidd2, fftw_complex *psidd2fft) {
   long cnti, cntj, cntk;
   double tmp;

   #pragma omp parallel for private(cnti, cntj, cntk, tmp) num_threads(nthreads - 1)
   for (cnti = 0; cnti < cpuNx_fft; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmp = psi[cnti][cntj][cntk];
            psidd2[cnti][cntj][cntk] = tmp * tmp;
         }
      }
   }

   fftw_execute_dft_r2c(plan_forward_row, **psidd2, psidd2fft);

   return;
}

void calcpsidd2_cpu_p2(fftw_complex *psidd2fft_t) {
   long cnti, cntj, cntk;

   fftw_execute_dft(plan_forward_col, psidd2fft_t, psidd2fft_t);

   #pragma omp parallel for private(cnti, cntj, cntk) num_threads(nthreads - 1)
   for (cnti = 0; cnti < Nx; cnti ++) {
      for (cntj = 0; cntj < cpuNy_fft; cntj ++) {
         for (cntk = 0; cntk < Nz2 + 1; cntk ++) {
            psidd2fft_t[cnti * localNy * (Nz2 + 1) + cntj * (Nz2 + 1) + cntk][0] *= potdd[cnti][cntj][cntk];
            psidd2fft_t[cnti * localNy * (Nz2 + 1) + cntj * (Nz2 + 1) + cntk][1] *= potdd[cnti][cntj][cntk];
         }
      }
   }

   fftw_execute_dft(plan_backward_col, psidd2fft_t, psidd2fft_t);

   return;
}

void calcpsidd2_cpu_p3(double ***psidd2, fftw_complex *psidd2fft) {

   fftw_execute_dft_c2r(plan_backward_row, psidd2fft, **psidd2);

   return;
}

void calcpsidd2_cpu_p4(double ***psidd2) {
   long cnti, cntj, cntk;

   #pragma omp parallel for private(cnti, cntj, cntk) num_threads(nthreads - 1)
   for (cnti = 0; cnti < cpuNx_nyz; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            psidd2[cnti][cntj][cntk] /= Nx * Ny * Nz;
         }
      }
   }

   #pragma omp parallel for private(cnti, cntk) num_threads(nthreads - 1)
   for (cnti = 0; cnti < cpuNx_nyz; cnti ++) {
      for (cntk = 0; cntk < Nz - 1; cntk ++) {
         psidd2[cnti][Ny - 1][cntk] = psidd2[cnti][0][cntk];
      }
   }

   #pragma omp parallel for private(cnti, cntj) num_threads(nthreads - 1)
   for (cnti = 0; cnti < cpuNx_nyz; cnti ++) {
      for (cntj = 0; cntj < Ny - 1; cntj ++) {
         psidd2[cnti][cntj][Nz - 1] = psidd2[cnti][cntj][0];
      }
   }

   return;
}

/**
 *    Calculation of the root mean square radius.
 *    rms   - root mean square radius
 *    psi   - array with the wave function values
 *    psi_t - array with the transposed wave function values
 *    tmpx  - temporary array
 *    tmpy  - temporary array
 *    tmpz  - temporary array
 */
void calcrms(double *rms, double ***psi, double ***psi_t, double **tmpx, double **tmpy, double **tmpz) {
   int threadid;
   long cnti, cntj, cntk;
   double psi2;
   void *sendbuf;

   TRAN_PSI_X

   #pragma omp parallel private(threadid, cnti, cntj, cntk, psi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cntj = 0; cntj < localNy; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cnti = 0; cnti < Nx; cnti ++) {
               psi2 = psi_t[cnti][cntj][cntk] * psi_t[cnti][cntj][cntk];
               tmpx[threadid][cnti] = x2[cnti] * psi2;
            }
            tmpz[threadid][cntk] = simpint(dx, tmpx[threadid], Nx);
         }
         (*tmpy)[cntj] = simpint(dz, tmpz[threadid], Nz);
      }
   }

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpy;
   MPI_Gather(sendbuf, localNy, MPI_DOUBLE, *tmpy, localNy, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      rms[1] = sqrt(simpint(dy, *tmpy, Ny));
   }

   #pragma omp parallel private(threadid, cnti, cntj, cntk, psi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            for (cntj = 0; cntj < Ny; cntj ++) {
               psi2 = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
               tmpy[threadid][cntj] = y2[cntj] * psi2;
            }
            tmpz[threadid][cntk] = simpint(dy, tmpy[threadid], Ny);
         }
         (*tmpx)[cnti] = simpint(dz, tmpz[threadid], Nz);
      }
   }

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      rms[2] = sqrt(simpint(dx, *tmpx, Nx));
   }

   #pragma omp parallel private(threadid, cnti, cntj, cntk, psi2)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            for (cntk = 0; cntk < Nz; cntk ++) {
               psi2 = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
               tmpz[threadid][cntk] = z2[cntk] * psi2;
            }
            tmpy[threadid][cntj] = simpint(dz, tmpz[threadid], Nz);
         }
         (*tmpx)[cnti] = simpint(dy, tmpy[threadid], Ny);
      }
   }

   sendbuf = (rank == 0) ? MPI_IN_PLACE : *tmpx;
   MPI_Gather(sendbuf, localNx, MPI_DOUBLE, *tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

   if (rank == 0) {
      rms[3] = sqrt(simpint(dx, *tmpx, Nx));
      rms[0] = sqrt(rms[1] * rms[1] + rms[2] * rms[2] + rms[3] * rms[3]);
   }

   return;
}

/**
 *    Time propagation with respect to H1 (part of the Hamiltonian without
 *    spatial derivatives).
 *    psi    - array with the wave function values
 *    psidd2 - array with the squared wave function values
 *    pot    - array with the potential
 */
/*void calcnu(double ***psi, double ***psidd2, double ***pot) {
   long cnti, cntj, cntk;
   double psi2, psi2lin, psidd2lin, tmp;

   calcnu_gpu();

   #pragma omp parallel for private(cnti, cntj, cntk, psi2, psi2lin, psidd2lin, tmp)
   for (cnti = 0; cnti < cpuNx_nyz; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            psi2 = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
            psi2lin = psi2 * g;
            psidd2lin = psidd2[cnti][cntj][cntk] * gd;
            tmp = dt * (pot[cnti][cntj][cntk] + psi2lin + psidd2lin);
            psi[cnti][cntj][cntk] *= exp(- tmp);
         }
      }
   }

   sync_with_gpu();

   return;
}*/

void calcnu(double ***psi, double ***psidd2, double ***pot) {
   int threadid;

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcnu_gpu();
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcnu_cpu(psi, psidd2, pot);
      }
   }

   return;
}

void calcnu_cpu(double ***psi, double ***psidd2, double ***pot) {
   long cnti, cntj, cntk;
   double psi2, psi2lin, psidd2lin, tmp;

   #pragma omp parallel for private(cnti, cntj, cntk, psi2, psi2lin, psidd2lin, tmp) num_threads(nthreads - 1)
   for (cnti = 0; cnti < cpuNx_nyz; cnti ++) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            psi2 = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
            psi2lin = psi2 * g;
            psidd2lin = psidd2[cnti][cntj][cntk] * gd;
            tmp = dt * (pot[cnti][cntj][cntk] + psi2lin + psidd2lin);
            psi[cnti][cntj][cntk] *= exp(- tmp);
         }
      }
   }

   return;
}

/**
 *    Time propagation with respect to H2 (x-part of the Laplacian).
 *    psi   - array with the wave function values (transposed)
 *    cbeta - Crank-Nicolson scheme coefficients
 */
/*void calclux(double ***psi, double **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double c;

   TRAN_PSI_X

   calclux_gpu();

   #pragma omp parallel private(threadid, cnti, cntj, cntk, c)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cntj = 0; cntj < cpuNy_lux; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            cbeta[threadid][Nx - 2] = psi[Nx - 1][cntj][cntk];
            for (cnti = Nx - 2; cnti > 0; cnti --) {
               c = - Ax * psi[cnti + 1][cntj][cntk] + Ax0r * psi[cnti][cntj][cntk] - Ax * psi[cnti - 1][cntj][cntk];
               cbeta[threadid][cnti - 1] =  cgammax[cnti] * (Ax * cbeta[threadid][cnti] - c);
            }
            psi[0][cntj][cntk] = 0.;
            for (cnti = 0; cnti < Nx - 2; cnti ++) {
               psi[cnti + 1][cntj][cntk] = calphax[cnti] * psi[cnti][cntj][cntk] + cbeta[threadid][cnti];
            }
            psi[Nx - 1][cntj][cntk] = 0.;
         }
      }
   }

   sync_with_gpu();

   TRAN_PSI_Y

   return;
}*/

void calclux(double ***psi_t, double **cbeta) {
   int threadid;

   TRAN_PSI_X

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calclux_gpu();
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calclux_cpu(psi_t, cbeta);
      }
   }

   TRAN_PSI_Y

   return;
}

void calclux_cpu(double ***psi, double **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double c;

   #pragma omp parallel private(threadid, cnti, cntj, cntk, c) num_threads(nthreads - 1)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cntj = 0; cntj < cpuNy_lux; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            cbeta[threadid][Nx - 2] = psi[Nx - 1][cntj][cntk];
            for (cnti = Nx - 2; cnti > 0; cnti --) {
               c = - Ax * psi[cnti + 1][cntj][cntk] + Ax0r * psi[cnti][cntj][cntk] - Ax * psi[cnti - 1][cntj][cntk];
               cbeta[threadid][cnti - 1] =  cgammax[cnti] * (Ax * cbeta[threadid][cnti] - c);
            }
            psi[0][cntj][cntk] = 0.;
            for (cnti = 0; cnti < Nx - 2; cnti ++) {
               psi[cnti + 1][cntj][cntk] = calphax[cnti] * psi[cnti][cntj][cntk] + cbeta[threadid][cnti];
            }
            psi[Nx - 1][cntj][cntk] = 0.;
         }
      }
   }

   return;
}

/**
 *    Time propagation with respect to H3 (y-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
/*void calcluy(double ***psi, double **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double c;

   calcluy_gpu();

   #pragma omp parallel private(threadid, cnti, cntj, cntk, c)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < cpuNx_luy; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            cbeta[threadid][Ny - 2] = psi[cnti][Ny - 1][cntk];
            for (cntj = Ny - 2; cntj > 0; cntj --) {
               c = - Ay * psi[cnti][cntj + 1][cntk] + Ay0r * psi[cnti][cntj][cntk] - Ay * psi[cnti][cntj - 1][cntk];
               cbeta[threadid][cntj - 1] =  cgammay[cntj] * (Ay * cbeta[threadid][cntj] - c);
            }
            psi[cnti][0][cntk] = 0.;
            for (cntj = 0; cntj < Ny - 2; cntj ++) {
               psi[cnti][cntj + 1][cntk] = calphay[cntj] * psi[cnti][cntj][cntk] + cbeta[threadid][cntj];
            }
            psi[cnti][Ny - 1][cntk] = 0.;
         }
      }
   }

   sync_with_gpu();

   return;
}*/

void calcluy(double ***psi, double **cbeta) {
   int threadid;

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcluy_gpu();
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcluy_cpu(psi, cbeta);
      }
   }

   return;
}

void calcluy_cpu(double ***psi, double **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double c;

   #pragma omp parallel private(threadid, cnti, cntj, cntk, c) num_threads(nthreads_luy)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < cpuNx_luy; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            cbeta[threadid][Ny - 2] = psi[cnti][Ny - 1][cntk];
            for (cntj = Ny - 2; cntj > 0; cntj --) {
               c = - Ay * psi[cnti][cntj + 1][cntk] + Ay0r * psi[cnti][cntj][cntk] - Ay * psi[cnti][cntj - 1][cntk];
               cbeta[threadid][cntj - 1] =  cgammay[cntj] * (Ay * cbeta[threadid][cntj] - c);
            }
            psi[cnti][0][cntk] = 0.;
            for (cntj = 0; cntj < Ny - 2; cntj ++) {
               psi[cnti][cntj + 1][cntk] = calphay[cntj] * psi[cnti][cntj][cntk] + cbeta[threadid][cntj];
            }
            psi[cnti][Ny - 1][cntk] = 0.;
         }
      }
   }

   return;
}

/**
 *    Time propagation with respect to H4 (z-part of the Laplacian).
 *    psi   - array with the wave function values
 *    cbeta - Crank-Nicolson scheme coefficients
 */
/*void calcluz(double ***psi, double **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double c;

   calcluz_gpu();

   #pragma omp parallel private(threadid, cnti, cntj, cntk, c)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < cpuNx_luz; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            cbeta[threadid][Nz - 2] = psi[cnti][cntj][Nz - 1];
            for (cntk = Nz - 2; cntk > 0; cntk --) {
               c = - Az * psi[cnti][cntj][cntk + 1] + Az0r * psi[cnti][cntj][cntk] - Az * psi[cnti][cntj][cntk - 1];
               cbeta[threadid][cntk - 1] =  cgammaz[cntk] * (Az * cbeta[threadid][cntk] - c);
            }
            psi[cnti][cntj][0] = 0.;
            for (cntk = 0; cntk < Nz - 2; cntk ++) {
               psi[cnti][cntj][cntk + 1] = calphaz[cntk] * psi[cnti][cntj][cntk] + cbeta[threadid][cntk];
            }
            psi[cnti][cntj][Nz - 1] = 0.;
         }
      }
   }

   sync_with_gpu();

   return;
}*/

void calcluz(double ***psi, double **cbeta) {
   int threadid;

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcluz_gpu();
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcluz_cpu(psi, cbeta);
      }
   }

   return;
}

void calcluz_cpu(double ***psi, double **cbeta) {
   int threadid;
   long cnti, cntj, cntk;
   double c;

   #pragma omp parallel private(threadid, cnti, cntj, cntk, c) num_threads(nthreads_luz)
   {
      threadid = omp_get_thread_num();

      #pragma omp for
      for (cnti = 0; cnti < cpuNx_luz; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            cbeta[threadid][Nz - 2] = psi[cnti][cntj][Nz - 1];
            for (cntk = Nz - 2; cntk > 0; cntk --) {
               c = - Az * psi[cnti][cntj][cntk + 1] + Az0r * psi[cnti][cntj][cntk] - Az * psi[cnti][cntj][cntk - 1];
               cbeta[threadid][cntk - 1] =  cgammaz[cntk] * (Az * cbeta[threadid][cntk] - c);
            }
            psi[cnti][cntj][0] = 0.;
            for (cntk = 0; cntk < Nz - 2; cntk ++) {
               psi[cnti][cntj][cntk + 1] = calphaz[cntk] * psi[cnti][cntj][cntk] + cbeta[threadid][cntk];
            }
            psi[cnti][cntj][Nz - 1] = 0.;
         }
      }
   }

   return;
}

void calcpsi_o1(double ***psi, double ***psi_t, double ***psidd2, double ***pot, double **cbeta) {

   calcnu(psi, psidd2, pot);
   calcluy_cpu(psi, cbeta);
   calcluz_cpu(psi, cbeta);
   calclux(psi_t, cbeta);

   return;
}

void calcpsi_o2(double ***psi, double ***psi_t, double ***psidd2, double ***pot, double **cbeta) {

   calcnuluy(psi, psidd2, pot, cbeta);
   calcluz_cpu(psi, cbeta);
   calclux(psi_t, cbeta);

   return;
}

void calcpsi_o3(double ***psi, double ***psi_t, double ***psidd2, double ***pot, double **cbeta) {

   calcnuluyluz(psi, psidd2, pot, cbeta);
   calclux(psi_t, cbeta);

   return;
}

void calcnuluy(double ***psi, double ***psidd2, double ***pot, double **cbeta) {
   int threadid;

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcnu_gpu();
         calcluy_gpu();
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcnu_cpu(psi, psidd2, pot);
         calcluy_cpu(psi, cbeta);
      }
   }

   return;
}

void calcnuluyluz(double ***psi, double ***psidd2, double ***pot, double **cbeta) {
   int threadid;

   #pragma omp parallel private(threadid) num_threads(2)
   {
      threadid = omp_get_thread_num();

      if (threadid == 0) {
         calcnu_gpu();
         calcluy_gpu();
         calcluz_gpu();
         sync_with_gpu();
      }

      if (threadid == 1 || omp_get_num_threads() != 2) {
         calcnu_cpu(psi, psidd2, pot);
         calcluy_cpu(psi, cbeta);
         calcluz_cpu(psi, cbeta);
      }
   }

   return;
}

void outdenx(double ***psi, double **outx, double *tmpy, double *tmpz, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 2 * sizeof(double) * (localNx / outstpx);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmpz[cntk] = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
         }
         tmpy[cntj] = simpint(dz, tmpz, Nz);
      }
      outx[cnti / outstpx][0] = x[offsetNx + cnti];
      outx[cnti / outstpx][1] = simpint(dy, tmpy, Ny);
   }

   MPI_File_write_at_all(file, fileoffset, *outx, (localNx / outstpx) * 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
}

void outdeny(double ***psi_t, double **outy, double *tmpx, double *tmpz, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 2 * sizeof(double) * (localNy / outstpy);

   TRAN_PSI_X

   for (cntj = 0; cntj < localNy; cntj += outstpy) {
      for (cnti = 0; cnti < Nx; cnti ++) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmpz[cntk] = psi_t[cnti][cntj][cntk] * psi_t[cnti][cntj][cntk];
         }
         tmpx[cnti] = simpint(dz, tmpz, Nz);
      }
      outy[cntj / outstpy][0] = y[offsetNy + cntj];
      outy[cntj / outstpy][1] = simpint(dx, tmpx, Nx);
   }

   MPI_File_write_at_all(file, fileoffset, *outy, (localNy / outstpy) * 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
}

void outdenz(double ***psi, double **outz, double *tmpx, double *tmpy, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;
   void *sendbuf;

   sendbuf = (rank == 0) ? MPI_IN_PLACE : tmpx;

   for (cntk = 0; cntk < Nz; cntk += outstpz) {
      for (cnti = 0; cnti < localNx; cnti ++) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            tmpy[cntj] = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
         }
         tmpx[cnti] = simpint(dy, tmpy, Ny);
      }

      MPI_Gather(sendbuf, localNx, MPI_DOUBLE, tmpx, localNx, MPI_DOUBLE, 0, MPI_COMM_WORLD);

      if (rank == 0) {
         outz[cntk / outstpz][0] = z[cntk];
         outz[cntk / outstpz][1] = simpint(dx, tmpx, Nx);
      }
   }

   if (rank == 0) {
      fileoffset = 0;
      MPI_File_write_at(file, fileoffset, *outz, (Nz / outstpz) * 2, MPI_DOUBLE, MPI_STATUS_IGNORE);
   }
}

void outdenxy(double ***psi, double ***outxy, double *tmpz, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Ny / outstpy);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         for (cntk = 0; cntk < Nz; cntk ++) {
            tmpz[cntk] = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
         }
         outxy[cnti / outstpx][cntj / outstpy][0] = x[offsetNx + cnti];
         outxy[cnti / outstpx][cntj / outstpy][1] = y[cntj];
         outxy[cnti / outstpx][cntj / outstpy][2] = simpint(dz, tmpz, Nz);
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outxy, (localNx / outstpx) * (Ny / outstpy) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outdenxz(double ***psi, double ***outxz, double *tmpy, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Nz / outstpz);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         for (cntj = 0; cntj < Ny; cntj ++) {
            tmpy[cntj] = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
         }
         outxz[cnti / outstpx][cntk / outstpz][0] = x[offsetNx + cnti];
         outxz[cnti / outstpx][cntk / outstpz][1] = z[cntk];
         outxz[cnti / outstpx][cntk / outstpz][2] = simpint(dy, tmpy, Ny);
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outxz, (localNx / outstpx) * (Nz / outstpz) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outdenyz(double ***psi_t, double ***outyz, double *tmpx, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNy / outstpy) * (Nz / outstpz);

   TRAN_PSI_X

   for (cntj = 0; cntj < localNy; cntj += outstpy) {
      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         for (cnti = 0; cnti < Nx; cnti ++) {
            tmpx[cnti] = psi_t[cnti][cntj][cntk] * psi_t[cnti][cntj][cntk];
         }
         outyz[cntj / outstpy][cntk / outstpz][0] = y[offsetNy + cntj];
         outyz[cntj / outstpy][cntk / outstpz][1] = z[cntk];
         outyz[cntj / outstpy][cntk / outstpz][2] = simpint(dx, tmpx, Nx);
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outyz, (localNy / outstpy) * (Nz / outstpz) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
}

void outpsi2xy(double ***psi, double ***outxy, MPI_File file) {
   long cnti, cntj;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Ny / outstpy);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         outxy[cnti / outstpx][cntj / outstpy][0] = x[offsetNx + cnti];
         outxy[cnti / outstpx][cntj / outstpy][1] = y[cntj];
         outxy[cnti / outstpx][cntj / outstpy][2] = psi[cnti][cntj][Nz2] * psi[cnti][cntj][Nz2];
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outxy, (localNx / outstpx) * (Ny / outstpy) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outpsi2xz(double ***psi, double ***outxz, MPI_File file) {
   long cnti, cntk;
   MPI_Offset fileoffset;

   fileoffset = rank * 3 * sizeof(double) * (localNx / outstpx) * (Nz / outstpz);

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntk = 0; cntk < Nz; cntk += outstpz) {
         outxz[cnti / outstpx][cntk / outstpz][0] = x[offsetNx + cnti];
         outxz[cnti / outstpx][cntk / outstpz][1] = z[cntk];
         outxz[cnti / outstpx][cntk / outstpz][2] = psi[cnti][Ny2][cntk] * psi[cnti][Ny2][cntk];
      }
   }

   MPI_File_write_at_all(file, fileoffset, **outxz, (localNx / outstpx) * (Nz / outstpz) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}

void outpsi2yz(double ***psi, double ***outyz, MPI_File file) {
   long cntj, cntk;
   int rankNx2, offsetNx2;
   MPI_Offset fileoffset;

   rankNx2 = Nx2 / localNx;
   offsetNx2 = Nx2 % localNx;

   fileoffset = 0;

   if (rank == rankNx2) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         for (cntk = 0; cntk < Nz; cntk += outstpz) {
            outyz[cntj / outstpy][cntk / outstpz][0] = y[cntj];
            outyz[cntj / outstpy][cntk / outstpz][1] = z[cntk];
            outyz[cntj / outstpy][cntk / outstpz][2] = psi[offsetNx2][cntj][cntk] * psi[offsetNx2][cntj][cntk];
         }
      }

      MPI_File_write_at(file, fileoffset, **outyz, (Ny / outstpy) * (Nz / outstpz) * 3, MPI_DOUBLE, MPI_STATUS_IGNORE);
   }
}

void outdenxyz(double ***psi, double ***outxyz, MPI_File file) {
   long cnti, cntj, cntk;
   MPI_Offset fileoffset;

   // MPI IO returns error if the array is too large. As a workaround, we write just Ny * Nz at a time.

   fileoffset = rank * sizeof(double) * localNx * Ny * Nz;

   for (cnti = 0; cnti < localNx; cnti += outstpx) {
      for (cntj = 0; cntj < Ny; cntj += outstpy) {
         for (cntk = 0; cntk < Nz; cntk += outstpz) {
            //outxyz[cnti][cntj][cntk] = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
            outxyz[0][cntj][cntk] = psi[cnti][cntj][cntk] * psi[cnti][cntj][cntk];
         }
      }

      MPI_File_write_at_all(file, fileoffset, **outxyz, (Ny / outstpy) * (Nz / outstpz), MPI_DOUBLE, MPI_STATUS_IGNORE);
      fileoffset += (Ny / outstpy) * (Nz / outstpz) * sizeof(double);
   }

   //MPI_File_write_at_all(file, fileoffset, **outxyz, (localNx / outstpx) * (Ny / outstpy) * (Nz / outstpz), MPI_DOUBLE, MPI_STATUS_IGNORE);

   return;
}
