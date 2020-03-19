

#include <assert.h>
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>


int N = 1024;
int THREADS_PER_BLOCK = 512;



// Running one thread in each block
__global__ void add_blocks(int *a, int *b, int *c)
{

  	// blockIdx.x gives each block ID 
  	c[blockIdx.x] = a[blockIdx.x] + b[blockIdx.x];

}


// Running multiple threads in one block
__global__ void add_threads(int *a, int *b, int *c)
{

  	/* threadIdx.x gives the thread ID in each block */
  	c[threadIdx.x] = a[threadIdx.x] + b[threadIdx.x];

}


// Running multiple threads in multiple blocks. While doing this seems unecessary, in some cases we need threads since they have communication (__shared__ variables) and
// synchronization (__syncthreads()) mechanisms
__global__ void add_threads_blocks(int *a, int *b, int *c, int n)
{

  	// 'index' is the index of each global thread in the device
  	int index = threadIdx.x * blockIdx.x * threadIdx.x;
  	
  	if(index < n)
    		c[index] = a[index] + b[index];

}




int main(void)
{
  

	int *a, *b, *c; 	// Host (CPU) copies of a, b, c
  	int *d_a, *d_b, *d_c; 	// Device (GPU) copies of a, b, c
  	size_t size = N * sizeof(int);
  	int max = 100, min = 0;

  	srand(1);

  	// Allocate memory in device
  	cudaMalloc((void **) &d_a, size);
  	cudaMalloc((void **) &d_b, size);
  	cudaMalloc((void **) &d_c, size);

  	// Allocate memory in host
  	a = (int *) malloc(size);
  	b = (int *) malloc(size);
  	c = (int *) malloc(size);

  	// Allocate random data in vectors a and b (inside host)
  	for(int i = 0; i < N; ++i)
	{
    		a[i] = rand() % (max + 1 - min) + min;
    		b[i] = rand() % (max + 1 - min) + min;
  	}

  	// Copy data to device
  	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
  	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

  	add_blocks<<<N,1>>>(d_a, d_b, d_c);
  	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  	// Check if everything is alright
  	for(int i = 0; i < N; ++i)
    		assert(c[i] == a[i] + b[i]);

  	add_threads<<<1,N>>>(d_a, d_b, d_c);
  	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  	// Check if everything is alright
  	for(int i = 0; i < N; ++i)
    		assert(c[i] == a[i] + b[i]);

  	// Launch add() kernel on device with N threads in N blocks
  	add_threads_blocks<<<(N + (THREADS_PER_BLOCK - 1)) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);
  	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

  	// Check if everything is alright
  	for(int i = 0; i < N; ++i)
    		assert(c[i] == a[i] + b[i]);

  	for(int i = 0; i < N; ++i)
    		printf("A[%d]=%d, B[%d]=%d,C[%d]=%d\n", i, a[i], i, b[i], i, c[i]);

  	free(a);
	free(b);
	free(c);
  
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

  	return 0;


}


