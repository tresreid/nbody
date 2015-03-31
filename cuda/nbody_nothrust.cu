// -*-c++-*-

#include <stdbool.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>
#include <cutil.h>
#include <cutil_math.h>



#ifdef __APPLE__
#include <mach/mach_time.h>
#else
#include <sys/time.h>
#endif

#define NBLOCKS 16
#define NTHREADS 256
#define OUTPUT_PATH "/var/tmp/wittich"

__global__ void nbody_kern(const float dt1, const float eps,
			 float4* pos_old,
			 float4* pos_new,
			 float4* vel ) 
{
  const float4 dt = make_float4(dt1,dt1,dt1,0.0f);

  const int nt = NTHREADS;
  const int nb = NBLOCKS;

  int gti = blockIdx.x * NTHREADS + threadIdx.x;


  float4 p = pos_old[gti];
  float4 v = vel[gti];
  float4 a = make_float4(0.0f,0.0f,0.0f,0.0f);
  //#define DUMB // - no cachinga
#ifndef DUMB
  int ti = threadIdx.x;
  __shared__ float4 pblock[NTHREADS];

  for(int jb=0; jb < nb; jb++) { //foreach block

    pblock[ti] = pos_old[jb*nt+ti]; /* Cache ONE particle position */
    __syncthreads(); // make sure the local cache is updated and visible to all threads

#pragma unroll 1 // turn off unrolling
//#pragma unroll  // turn on unrolling
    for(int j=0; j<nt; ) { /* For ALL cached particle positions ... */
      float4 p2 = pblock[j]; /* Read a cached particle position */
      float4 d = p2 - p;
      float invr = rsqrt(d.x*d.x + d.y*d.y + d.z*d.z + eps );
      float f = p2.w*invr*invr*invr; // this is actually f/(r^3 m_1), assume G = 1
      // d is direction btw two but not a unit vector
      // extra powers of invr above take care of length
      a += f*d; /* Accumulate acceleration */
      ++j;
    }

    __syncthreads(); // sync again before updating cache to next block
  }
#else
  for(int j=0; j < nb*nt; ++j) { //foreach particle
    float4 p2 = pos_old[j]; // from global memory
    float4 d = p2 - p;
    float invr = 1.0f/sqrt(d.x*d.x + d.y*d.y + d.z*d.z + eps );
    float f = p2.w*invr*invr*invr; // this is actually f/(r^3 m_1), assume G = 1
    // d is direction btw two but not a unit vector
    // extra powers of invr above take care of length
    a += f*d; /* Accumulate acceleration */
  }

#endif // DUMB
  p += dt*v + 0.5f*dt*dt*a ;
  v += dt*a ;

  pos_new[gti] = p;
  vel[gti] = v;

}



#include <signal.h>
static int saw_sigint = 0;
void siginthandler(int signal)
{
  saw_sigint = 1;
  return;
}


void writeDat(const char* fname, void* data, int ndata);

void savestate(const char* fname, float4* pos, float4* vel, int nparticle, 
	       int nsteps);
void readstate(const char* fname, float4 **pos, float4 **vel, int * nparticle,
	       int *nsteps);



/////////////////////////////////////////////////////////////////////////////
const int width  = 512;
const int height = 512;

int 
main(int argc, char **argv)
{

  signal(SIGINT, siginthandler);


  // steering inputs

  //int step,burst;

  int nparticle = NTHREADS*NBLOCKS; 
  //int nparticle = 1024; // nice power of 2 for simplicity
  const size_t nstep = 2*1250;
  //const size_t nstep = 50;
  int nburst = 128; // sub-steps
  //int nburst = 1; // sub-steps

  float dt = 0.001;
  float eps = 0.001;

  // create arrays -- host
  float4 *pos1 = 0;
  float4 *pos2 = 0;
  float4 *vel  = 0 ;
  // create arrays -- device
  float4 *pos1_d = 0;
  float4 *pos2_d = 0;
  float4 *vel_d  = 0 ;
  int nstep_start = 0;
  if ( argc == 1 ) {
    pos1 = (float4*)malloc(sizeof(float4)*nparticle);
    pos2 = (float4*)malloc(sizeof(float4)*nparticle);
    vel  = (float4*)malloc(sizeof(float4)*nparticle);

    srand(42137377L);

    // two black holes
    float offset = +220.;
    pos1[0].x = 1.5*width ;
    pos1[0].y = 1.5*height;
    pos1[0].z = 0.;
    pos1[0].w = 1;//5000.; // mass
    pos1[1].x = 3.*width ;
    pos1[1].y = 0.5*height;
    pos1[1].z = 0.;
    pos1[1].w = 1;//40000.; // mass
    pos1[2].x = 3.*width ;
    pos1[2].y = 0.75*height;
    pos1[2].z = 1.;
    pos1[2].w = 2.;//40000.; // mass
    pos1[3].x = 3.*width ;
    pos1[3].y = 0.75*height;
    pos1[3].z = -10.;
    pos1[3].w = 10.;//40000.; // mass
  

    for ( int i = 4; i < nparticle; ++i ){
      bool done = false;
      float mult = 0.5;
      const float el_a_sq= 60*40;
      const float el_b_sq = 80*60;
      const float cos_q = cos(3.14/3);
      const float sin_q = sin(3.14/3);
      while (! done ) {
	pos1[i].x = 1.0*rand()/RAND_MAX * 2*width - width/2. ;
	pos1[i].y = 1.0*rand()/RAND_MAX * 2*height - height/2. ;
	pos1[i].z = 1.0*rand()/RAND_MAX * 50. -  25.;
	pos1[i].w = 1.0*rand()/RAND_MAX * 5.; // mass
	//pos1[i].w = 2;
	float rsq = (pos1[i].x*pos1[i].x)/el_a_sq+(pos1[i].y*pos1[i].y)/el_b_sq+pos1[i].z*pos1[i].z;
	//if ( (rsq < (40*40)) && (rsq > (35*35))) 
	if ((rsq < 1) || (i<(nparticle *.5) )) {
	  done = true;
	  if ( i > (nparticle*.75)) {
	    mult = 2.0;
	    pos1[i].x = cos_q*pos1[i].x - sin_q * pos1[i].y;
	    pos1[i].y = sin_q*pos1[i].x + cos_q * pos1[i].y;
	  }
	  pos1[i].x += mult*offset;
	  pos1[i].y += mult*offset;

	}
      }
      pos2[i] = make_float4(0.f,0.f,0.f,0.f);
      vel[i]  = make_float4(0.f,0.f,0.f,0.f);
    }
  }
  else { // read from input file 
    fprintf(stderr, "reading from file %s\n", argv[1]);
    readstate(argv[1], &pos1, &vel, &nparticle, &nstep_start);
    fprintf(stderr, "Found %d particles\n", nparticle);
    pos2 = (float4*)malloc(sizeof(float4)*nparticle);

  }
  writeDat("start.bmp", (void*)pos1, nparticle);

  int which = 8;
  printf("Start: particle %d x=%f, y=%f, z=%f, m=%f\n",
	 which, pos1[which].x, pos1[which].y, pos1[which].z, pos1[which].w);
  float4 startpos = pos1[which];

  // for ( int j =120; j < 130; ++j ) {
  //   printf("Start1:   particle %d x=%f, y=%f, z=%f, m=%f\n",
  // 	   j, pos1[j].x, pos1[j].y, pos1[j].z, pos1[j].w);
  // }
  // for ( int j =120; j < 130; ++j ) {
  //   printf("Start2:   particle %d x=%f, y=%f, z=%f, m=%f\n",
  // 	   j, pos2[j].x, pos2[j].y, pos2[j].z, pos2[j].w);
  // }
  // for ( int j =120; j < 130; ++j ) {
  //   printf("Start3:   particle %d x=%f, y=%f, z=%f, m=%f\n",
  // 	   j, vel[j].x, vel[j].y, vel[j].z, vel[j].w);
  // }


  CUDA_SAFE_CALL(cudaMalloc((void **) &pos1_d, sizeof(float4)*nparticle));   // Allocate array on device
  CUDA_SAFE_CALL(cudaMalloc((void **) &pos2_d, sizeof(float4)*nparticle));   // Allocate array on device
  CUDA_SAFE_CALL(cudaMalloc((void **) &vel_d,  sizeof(float4)*nparticle));   // Allocate array on device
  

  //Fill the input array with the host allocated starting point data
  //
  CUDA_SAFE_CALL(cudaMemcpy(pos1_d, pos1, sizeof(float4)*nparticle, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(pos2_d, pos2, sizeof(float4)*nparticle, cudaMemcpyHostToDevice));
  CUDA_SAFE_CALL(cudaMemcpy(vel_d,  vel,  sizeof(float4)*nparticle, cudaMemcpyHostToDevice));

  // just to be sure I see what's going on
  bzero(pos1,sizeof(float4)*nparticle);



  // Start the timing loop and execute the kernel over several iterations
  //
  printf("starting ... \n");

  double tavg_0 = 0, tsqu_0 = 0;

  int iter = 0;
  for ( ; iter < nstep;++iter) {
    if ( saw_sigint != 0 ) {
      printf("Saw SIGINT, aborting on loop entry  %d.\n", iter);
      break;
    }

    printf("iter=%d\n", iter);
    int             k;

#ifdef __APPLE__
    uint64_t        t0, t1, t2;
    t0 = t1 = mach_absolute_time();
#else // linux
    struct timeval t0r, t1r;
    double t0, t1;
    gettimeofday(&t0r, NULL);
    t0 =  t0r.tv_sec*1000000 + (t0r.tv_usec);
#endif // !APPLE
    for (k = 0; k < nburst; k++) {
      //printf("k=%d\n", k);
      // I think this is a blocking call
      if ( k%2==0 ) {
	nbody_kern<<<NBLOCKS,NTHREADS>>>(dt, eps, pos1_d,pos2_d,vel_d);
      } else {
	nbody_kern<<<NBLOCKS,NTHREADS>>>(dt, eps, pos2_d,pos1_d,vel_d);
      }


    }
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    CUT_CHECK_ERROR("kernel invocation");
    //clFinish(queue);
#ifdef __APPLE__
    t2 = mach_absolute_time();
    struct mach_timebase_info info;
    mach_timebase_info(&info);
    double          t = 1e-9 * (t2 - t1) * info.numer / info.denom;
#else 
    gettimeofday(&t1r, NULL);
    t1 =  t1r.tv_sec*1000000 + (t1r.tv_usec);
    double t = (t1-t0)/1e6;
#endif // 
    //printf("done.\n");
    //Calculate the total bandwidth that was obtained on the device for all  memory transfers
    //
    printf("Time spent = %g\n", t/(1.*nburst));

    tavg_0 += t/nburst;
    tsqu_0 += t*t/(nburst*nburst);

    //Read back the results that were computed on the device
    //
    CUDA_SAFE_CALL(cudaMemcpy(pos1, pos1_d, sizeof(float4)*nparticle, cudaMemcpyDeviceToHost));
    CUDA_SAFE_CALL(cudaMemcpy(pos2, pos2_d, sizeof(float4)*nparticle, cudaMemcpyDeviceToHost));

    if ( iter%25==0 ) {    
      // get velocities
      CUDA_SAFE_CALL(cudaMemcpy(vel, vel_d, sizeof(float4)*nparticle, cudaMemcpyDeviceToHost));
      printf("End:   vel %d x=%f, y=%f, z=%f, m=%f\n",
	     which, vel[which].x, vel[which].y, vel[which].z, vel[which].w);

      // calculate total momentum
      float4 ptotv = make_float4(0.,0.,0.,0.);
      for ( int i = 0; i < nparticle; ++i ) {
	ptotv.x += pos1[i].w*vel[i].x;
	ptotv.y += pos1[i].w*vel[i].y;
	ptotv.z += pos1[i].w*vel[i].z;
      }
      float ptot = sqrt(ptotv.x*ptotv.x+ptotv.y*ptotv.y+ptotv.z*ptotv.z);
      printf("\t\tptot = %5.3f\n", ptot);
    }

    printf("End:   particle %d x=%f, y=%f, z=%f, m=%f\n",
	   which, pos1[which].x, pos1[which].y, pos1[which].z, pos1[which].w);
    printf("End:   particle %d x=%f, y=%f, z=%f, m=%f\n",
	   which, pos2[which].x, pos2[which].y, pos2[which].z, pos2[which].w);

    // for ( int j =120; j < 130; ++j ) {
    //   printf("End2:   particle %d x=%f, y=%f, z=%f, m=%f\n",
    // 	   j, pos1[j].x, pos1[j].y, pos1[j].z, pos1[j].w);
    // }

    // for ( int j =120; j < 130; ++j ) {
    //   printf("End2:   particle %d x=%f, y=%f, z=%f, m=%f\n",
    // 	   j, pos2[j].x, pos2[j].y, pos2[j].z, pos2[j].w);
    // }

    // for ( int j =120; j < 130; ++j ) {
    //   printf("End3:   particle %d x=%f, y=%f, z=%f, m=%f\n",
    // 	   j, vel[j].x, vel[j].y, vel[j].z, vel[j].w);
    // }

    float4 endpos = pos1[which];
  
    float4 sep = make_float4(endpos.x-startpos.x,endpos.y-startpos.y,endpos.z-startpos.z,0);
    float distance = sqrt(sep.x*sep.x + sep.y*sep.y + sep.z*sep.z);
    //printf("Distance travelled = %g\n", distance);

    char fname[256];
    sprintf(fname,  "%s/test_%d.bmp", OUTPUT_PATH , iter+nstep_start);
    writeDat(fname, (void*)pos1, nparticle);
  }

  double tavg = tavg_0/(iter);
  double trms = sqrt((tsqu_0- tavg_0*tavg_0/(1.*iter))/(iter-1.0));
  printf("Average time spent per single step  = %g pm %g\n", tavg, trms);

  savestate("current.dat", pos1, vel, nparticle, iter+nstep_start);

  free(pos1);
  free(pos2); 
  free(vel);
  cudaFree(pos1_d);
  cudaFree(pos2_d); 
  cudaFree(vel_d);


  return 0;
}




struct rgb_t {
  //unsigned char r,g,b, padding; // 8 bit depth, padded to 32 bits
  unsigned char b, g, r;
} ;

void writeDat(const char* fname, void* data, int ndata)
{
  //fprintf(stdout,"writing to %s\n", fname);
  FILE* fout = fopen(fname, "w");
  if ( fout == (FILE*)NULL ) {
    fprintf(stderr, "writeBMP: warning, couldn't open output file.\n");
    return ;
  }


  const int header_size = 14;
  unsigned char header[header_size];
  bzero(header, header_size);
  header[0] = 'B';
  header[1] = 'M';
  uint *ha = (uint*)(header+2); // 2 byte offset
  ha[2] = 54; // offset; 40+14; constant (two headers)

  const int dib_header_size = 40;
  unsigned char dib_header[dib_header_size];
  bzero(dib_header, dib_header_size);

  uint *a = (uint*)(dib_header);
  unsigned short int *b = (unsigned short int*)(dib_header);
  a[0] = dib_header_size;
  a[1] = width; // width
  a[2] = height; // height
  b[6] = 1; // must be 1
  b[7] = 24; // color depth
  a[4] = 0; // compression method - 0 is default 
  a[5] = 0; // image size - can be zero
  a[6] = 100; // vertical resolution, pixels/m, signed (?)
  a[7] = 100; // horizontal resolution, pixels/m, signed (?)
  a[8] = 0; // colormap entries
  a[9] = 0; // important colors. ignored.
  //int padding = (sizeof(struct rgb_t)*width)%4;

  struct rgb_t vals[height][width]; // I hate c
  bzero(vals, width*height*sizeof(struct rgb_t));
  //printf("size = %lu\n", width*height*sizeof(struct rgb_t)+header_size+dib_header_size);
  //printf("data size = %lu\n", width*height*sizeof(struct rgb_t));
  //printf("rgb_t size = %lu\n", sizeof(struct rgb_t));
  ha[0] = width*height*sizeof(struct rgb_t)+header_size+dib_header_size; // size of the file
  //printf("size=%d\n", ha[0]);
  // for ( int i = 0; i < header_size; ++i ) {
  //   printf("c[%d] = %d\n", i, header[i]);
  // }


  for ( int i = 0; i < width; ++i ) {
    for ( int j = 0; j < height; ++j ) {
      vals[j][i].r = 0;
      vals[j][i].g = 0;//4*j;
      vals[j][i].b = 0;
    }
  }

  // int pos = height/2;
  // printf("pos = %d\n", pos);
  // for ( int i = 0; i < width; ++i ) {
  //   for (int j = -10; j < 10; ++j ) {
  //     vals[pos+j][i].r = 255;
  //     vals[pos+j][i].g = 255;
  //     vals[pos+j][i].b = 255;
  //   }
  //   //printf("%d,%d,%d,%d\n", i, vals[i][pos].r, vals[i][pos].g, vals[i][pos].b);
  // }

  


  float4 *cdata = (float4*)data;
  int cnt = 0;
  for ( int i = 0; i < ndata; ++i ) {
    int x = int(cdata[i].x+0.5); //x += width/2; //x = x%width;
    int y = int(cdata[i].y+0.5); //y += height/2; //y = y%height;
    //printf("i=%d,x,y = %d, %d\n", i,x, y);
    if ( x < 0 || y < 0 ) continue;
    if ( (x >= width) || (y >= height) ) continue;
    ++cnt;
    if ( vals[y][x].r == 0 ) {
      vals[y][x].r = 255;
      vals[y][x].g = 255;
      vals[y][x].b = 255;
    }
    else {
      vals[y][x].r -= (vals[y][x].r>100?100:0);
    }
  }
  printf("File written (%d particles).\n", cnt);

  // for ( int i = 0; i < ndata; ++i ) 
  //   fprintf(fout, "%f %f %f\n", cdata[i].x,cdata[i].y,cdata[i].z);
  
  fwrite(header, sizeof(char), header_size, fout);
  fwrite(dib_header, sizeof(char), dib_header_size, fout);
  fwrite(vals, sizeof(struct rgb_t), width*height, fout);
  //fwrite(cdata, sizeof(float4), ndata, fout);

  fclose(fout);
  return;

}

void savestate(const char* fname, float4* pos, float4* vel, int nparticle,
	       int nsteps)
{
  FILE* fout = fopen(fname, "w");
  if ( fout == (FILE*)NULL ) {
    perror("savestate: warning, couldn't open output file.\n");
    return ;
  }
  //
  fprintf(fout, "%d %d\n", nparticle, nsteps); // number of lines
  for ( int i = 0; i < nparticle; ++i ) 
    fprintf(fout, "%d %f %f %f  %f %f %f %f\n",
	    i, pos[i].x, pos[i].y, pos[i].z, pos[i].w,
	    vel[i].x, vel[i].y, vel[i].z);

  fclose(fout);
  return;
}

void readstate(const char* fname, float4 **pos, float4 **vel, int * nparticle,
	       int *nsteps)
{
  FILE* fin = fopen(fname, "r");
  if ( fin == (FILE*)NULL ) {
    perror("readstate: warning, couldn't open output file.\n");
    return ;
  }
  fscanf(fin, "%d", nparticle);
  fscanf(fin, "%d", nsteps);
  int n = *nparticle; // just easier to type
  printf("This many particles in %s: %d, %d steps\n", fname, n, *nsteps);
  *pos = (float4*)malloc(n*sizeof(float4));
  *vel = (float4*)malloc(n*sizeof(float4));
  //
  int i = 0;
  while ( !feof(fin) && (i < n) ) {
    int j;
    float x,y,z,m, vx, vy, vz;
    int ret = fscanf(fin, "%d %f %f %f  %f %f %f %f",
		     &j, &x,&y,&z,&m,&vx,&vy,&vz);
    if ( ret !=8 ) {	 	
      fprintf(stderr, "readstate Error: fscanf failed (%d, i = %d)\n", ret,i);
      exit(1);
    }
    (*pos)[i].x = x;
    (*pos)[i].y = y;
    (*pos)[i].z = z;
    (*pos)[i].w = m;
    (*vel)[i].x = vx;
    (*vel)[i].y = vy;
    (*vel)[i].z = vz;
    (*vel)[i].w = 0.;
    // fprintf(stderr, "ret=%d\n", ret);
    // fprintf(stderr, "%d %f %f %f  %f %f %f %f\n",
    // 	    j, (*pos)[i].x, (*pos)[i].y, (*pos)[i].z, (*pos)[i].w,
    // 	    (*vel)[i].x, (*vel)[i].y, (*vel)[i].z);
    ++i;
  }
  fclose(fin);
  return;
}