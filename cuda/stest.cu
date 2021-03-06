// -*-c++-*-
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <list>
#include <iostream>

#include "Time.hh"
#include "Matriplex.h"
using namespace Matriplex;

#include "SMatrix.h"

#include <tbb/tbb.h>
#include <tbb/task_scheduler_init.h>

typedef int idx_t ;

using ROOT::Math::SMatrix;

#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, 
		      bool abort=true) 
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, 
	      line);
      if (abort) exit(code);
   }
}

template<typename T, idx_t DIM1, idx_t DIM2, idx_t DIM3>
__global__ void smatrixtest(const T * __restrict__ d1, 
			    const T * __restrict__ d2, 
			    T * __restrict__ d3,
			    int N)
{
  const int gti = blockIdx.x * blockDim.x + threadIdx.x;
  const int gStride = blockDim.x * gridDim.x;

  for ( idx_t gi = gti; gi < N; gi += gStride ) {  
    SMatrix<T, DIM1, DIM2> mat1;
    int pos = gi*DIM1*DIM2;
#pragma unroll 
    for ( int i = 0; i < DIM1; ++ i ) {
      for ( int j = 0; i < DIM2; ++ j ) {
      mat1[i][j] = d1[pos++];
      }
    }
    SMatrix<T, DIM2, DIM3> mat2;
    pos = gi*DIM2*DIM3;
#pragma unroll 
    for ( int i = 0; i < DIM2; ++ i ) {
      for ( int j = 0; i < DIM3; ++ j ) {
      mat2[i][j] = d2[pos++];
      }
    }

    SMatrix<T, DIM1, DIM3> mat3;

    mat3 = mat1 * mat2;
    pos = gi*DIM1*DIM3;
    for ( int i = 0; i < DIM2; ++ i ) {
      for ( int j = 0; i < DIM3; ++ j ) {
	d3[pos++] = mat3[i][j];
      }
    }
    
  }
  return;
}


template<typename T, idx_t DIM1, idx_t DIM2, idx_t DIM3>
__global__ void smallMatrix(const T * __restrict__ d1, 
			    const T * __restrict__ d2,
			    //			    T * __restrict__ s, // scratch
			    T * __restrict__ d3,
			    int N) 
{
  // data in remote is stored in matrix-order, so the data is just 
  // stored with an overall offset
  const int gti = blockIdx.x * blockDim.x + threadIdx.x;
  const int gStride = blockDim.x * gridDim.x;

  for ( idx_t gi = gti; gi < N; gi += gStride ) {  
    // copy data to local scratch area
    T mat1[DIM1*DIM2];
    T mat2[DIM2*DIM3];
  
    int pos = gi*DIM1*DIM2;
#pragma unroll 
    for ( int i = 0; i < DIM1*DIM2; ++i ) {
      mat1[i] = d1[pos++];
      //pos += 32;
    }
    pos = gi*DIM2*DIM3;
#pragma unroll 
    for ( int i = 0; i < DIM2*DIM3; ++i ) {
      mat2[i] = d2[pos++];
      //pos += 32;
    }

    // do multiplication
    T mat3[DIM1*DIM3];
    for (idx_t i = 0; i < DIM1; ++i) {
      for (idx_t j = 0; j < DIM3; ++j) {
	const idx_t ijo = (i * DIM3 + j);

	mat3[ijo] = 0.f;
#pragma unroll
	for (idx_t k = 0; k < DIM2; ++k) {
	  const idx_t iko = (i * DIM2 + k);
	  const idx_t kjo = (k * DIM3 + j);

	  mat3[ijo ] += mat1[iko ] * mat2[kjo];
	  //mat3[ijo ] += 100*i+j + 10000*gti;
	}
      }
    }

    // // do shuffle to get data out - warp cooperation
    // T tmp3[DIM1*DIM3];
    // // printf("thread %i (laneId %i, warp %d, block %d): calling shuffle_scatter, gi=%i\n", gti,
    // // 	   laneId(), threadIdx.x/32, blockIdx.x, gi);
    // shuffle_scatter(mat3, tmp3, DIM1*DIM3, laneId());

    // // copy result out again
    //pos = gi;//*DIM1*DIM3;
    //for ( int i = 0; i < DIM1*DIM3; ++i ) {
    //   printf("thread %i (laneId %i, warp %d): copy %f to %d (i=%d)\n", gti, laneId(),
    // 	     threadIdx.x/32, tmp3[i], pos,i);
    // d3[pos] = mat3[i];
       //   d3[pos] = tmp3[i];
       //pos += gStride; pos = pos%(N*DIM1*DIM3);
       //}
    // copy result out again
    pos = gi*DIM1*DIM3;
    for ( int i = 0; i < DIM1*DIM3; ++i, ++pos ) {
      d3[pos] = mat3[i];
    }
  }
}




int main(int argc, char **argv)
{
  int N = 25600;

  if ( argc >= 2 ) {
    N = atoi(argv[1]);
  }

  // Choose the best CUDA device 
  int num_devices, device;
  CUDA_SAFE_CALL(cudaGetDeviceCount(&num_devices));
  printf("This many devices: %d\n", num_devices);
  int max_multiprocessors = -1, max_device = -1;
  cudaDeviceProp best_prop;
  for ( device = 0; device < num_devices; ++device ) {
    cudaDeviceProp properties;
    CUDA_SAFE_CALL(cudaGetDeviceProperties(&properties, device));
    if ( max_multiprocessors < properties.multiProcessorCount ) {
      max_multiprocessors = properties.multiProcessorCount;
      max_device = device;
      best_prop = properties;
    }
  }
  if ( max_device >=0 )
    cudaSetDevice(max_device);
  else  {
    printf("problem finding a good device! aborting.\n");
    return 1;
  }
  printf("# Running on device %d (name %s)\n", max_device, best_prop.name);

  // set the memory limits on the device

  const int DIM1 = 6;
  const int DIM2 = 6;
  const int DIM3 = 6;
  //const int N = 103-6;
  const int nmatrix1 = DIM1*DIM2*N;
  const int nmatrix2 = DIM2*DIM3*N;
  const int nmatrixres = DIM1*DIM3*N;
  printf("Size of memory required: %5.1f kB\n",
	 sizeof(float)*(nmatrix1+nmatrix2+nmatrixres)/1024.);

  // Configure the GPU 
  // get the heap size
  size_t curSize = 0;
  cudaDeviceGetLimit(&curSize, cudaLimitMallocHeapSize);
  size_t setSize = sizeof(float)*(nmatrix1+nmatrix2+nmatrixres);
  if ( curSize < setSize ) {
    cudaError_t err = cudaDeviceSetLimit(cudaLimitMallocHeapSize, setSize);
    if ( err != cudaSuccess ) {
      printf("failed to set heap size to %d\n", curSize);
      return 1;
    }
    err = cudaDeviceGetLimit(&curSize, cudaLimitMallocHeapSize);
    if ( err != cudaSuccess ) {
      printf("failed to get heap size.\n");
      return 1;
    }
  } // if the size needs increasing
  printf("Current size: %5.0f kB\n", curSize/1024.);

  // Determine block and Grid Sizes
  // Launch heuristics API
  int blockSize;   // The launch configurator returned block size 
  int minGridSize; // The minimum grid size needed to achieve the 
                   // maximum occupancy for a full device launch 
  int gridSize;    // The actual grid size needed, based on input size 

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize, 
                                      (smatrixtest<float, DIM1, DIM2, DIM3>),
				       0, 0); 
  // Round up according to array size 
  gridSize = (N + blockSize - 1) / blockSize; 
  //gridSize = minGridSize;
  printf("blockSize = %d, gridSize = %d, minGridSize = %d\n", blockSize, gridSize,
	 minGridSize);
  printf("N=%d NBLOCKS= %d  NTHREADS= %d mat per thread=%5.2f\n", N, gridSize, 
	 blockSize, 1.0*N/(gridSize*blockSize));
  // 


  // Generate the data needed
  // fill matrices with random data
  float *mres = new float[nmatrixres];
  float *mres_gpu = new float[nmatrixres];
  memset(mres, 0,nmatrixres*sizeof(float));
  memset(mres_gpu, 0,nmatrixres*sizeof(float));

  // these vectors hold the pre-matriplex matrices

  float *h_f1 = 0;
  float *h_f2 = 0;
  cudaMallocHost(&h_f1, (nmatrix1+nmatrix2)*sizeof(float));
  h_f2 = h_f1 + nmatrix1;



  // space on gpu for the inputs and outputs
  float *d_f1 = 0;
  float *d_f2 = 0;
  float *d_fres = 0;
  cudaMalloc(&d_f1, (nmatrix1+nmatrix2+nmatrixres)*sizeof(float));
  d_f2 = d_f1 + nmatrix1;
  d_fres = d_f2 + nmatrix2;				      

  srand(123213UL);

  for ( int i = 0; i < nmatrix1; ++i ) {
    h_f1[i] = rand()*20./RAND_MAX;
  }
  for ( int i = 0; i < nmatrix2; ++i ) {
    h_f2[i] = rand()*20./RAND_MAX;
  }


  // copy to GPU
  printf("copying to GPU .... \n");
  // batched single copy
  cudaMemcpyAsync(d_f1, h_f1, sizeof(float)*(nmatrix1+nmatrix2), cudaMemcpyHostToDevice);

  // GPU
  // result is now in d_fres
  for ( int i = 0; i < 10; ++i ) 
    smatrixtest<float, DIM1, DIM2, DIM3><<<gridSize,blockSize>>>(d_f1,d_f2, d_fres,N );

#define NOTDEF
#ifdef NOTDEF
  const int CPU_MATRIPLEX_SIZE = 8; // sizeof vector register/sizeof(float)
  // Matriplex::MPlex<float, DIM1, DIM2, CPU_MATRIPLEX_SIZE> h_matrices1;
  // Matriplex::MPlex<float, DIM2, DIM3, CPU_MATRIPLEX_SIZE> h_matrices2;
  // memset(h_matrices1.fArray, 0, h_matrices1.kTotSize*sizeof(float));
  // memset(h_matrices2.fArray, 0, h_matrices2.kTotSize*sizeof(float));
  // Matriplex::Matriplex<float, DIM1, DIM3, CPU_MATRIPLEX_SIZE> h_result;
  // memset(h_result.fArray, 0, h_result.kTotSize*sizeof(float));

  // initialize tbb - do this before the timing
  int n = tbb::task_scheduler_init::default_num_threads();
  // loop over 
  size_t niter = (N/CPU_MATRIPLEX_SIZE) + ((N%CPU_MATRIPLEX_SIZE)?1:0);
  printf("niter = %d %d %d\n", niter, N, CPU_MATRIPLEX_SIZE);
  // TIME START -- CPU
  timepoint t0(now());
  tbb::parallel_for(size_t(0), niter, [=](size_t ii)  {
  //for ( auto ii = 0; ii < niter; ++ii ) {
      // these are constructed within the loop due to problems with the capture of the lambda 
      // function.
      Matriplex::MPlex<float, DIM1, DIM2, CPU_MATRIPLEX_SIZE> h_matrices1;
      Matriplex::MPlex<float, DIM2, DIM3, CPU_MATRIPLEX_SIZE> h_matrices2;
      Matriplex::Matriplex<float, DIM1, DIM3, CPU_MATRIPLEX_SIZE> h_result;
      //for ( int ii = 0; ii < niter; ++ii ) {
    // convert random data to matriplex
    int pos = CPU_MATRIPLEX_SIZE*ii;
    for ( idx_t i = 0; i < CPU_MATRIPLEX_SIZE ; ++i, ++pos ) {
      if ( pos == N ) break;
      h_matrices1.CopyIn(i, h_f1+pos*h_matrices1.kSize);
      h_matrices2.CopyIn(i, h_f2+pos*h_matrices2.kSize);
    }


    MultiplyGeneral(h_matrices1, h_matrices2, h_result);

    // convert resulting data from matriplex
    pos = CPU_MATRIPLEX_SIZE*ii; // reset
    for ( idx_t i = 0; i < CPU_MATRIPLEX_SIZE; ++i, ++pos ) {
      if ( pos == N ) break;
      h_result.CopyOut(i, mres+pos*(h_result.kSize));
    } 
    }); // loop over all matrices, CPU
  tick t1 = delta(t0);
  std::cout << "CPU delta t = " 
	    << std::chrono::duration_cast<std::chrono::microseconds>(t1).count() 
	    << " us"
	    << std::endl;
  // TIME END- CPU
#endif // NOTDEF
  //cudaThreadSynchronize();
  // check for error. this catches a kernel launch error
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
     // print the CUDA error message and exit
     printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__,cudaGetErrorString(error));
     exit(-1);
  }
  // copy result back
  CUDA_SAFE_CALL(cudaMemcpyAsync(mres_gpu,d_fres,sizeof(float)*nmatrixres, cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize();



  int mismatches = 0;
  printf("i:cpu\tgpu\n");
  for (int i = 0;i<nmatrixres; ++i ) {
    if  (fabs(mres[i]-mres_gpu[i])>1e-3 ) {
      ++mismatches;
      printf("%d: (%d) %8.3f\t%8.3f %s\n", i, int(i/DIM1*DIM3),mres[i], 
       	     mres_gpu[i], (fabs(mres[i]-mres_gpu[i])<1.0e-3)?"":"<<<");
    }
  }
  if ( mismatches)
    printf("This many mismatches: %d\n", mismatches);


  delete [] mres;
  delete [] mres_gpu;

  cudaFree(h_f1);
  cudaFree(d_f1);
  
  return mismatches;
}
