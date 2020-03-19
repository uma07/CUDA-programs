
# include <stdio.h>
# include <stdlib.h>
# include <cuda.h>

# define N (2048)
# define THREADS_PER_BLOCK 512



__global__ void add(int *a, int *b, int *c)
{

	int index = threadIdx.x + blockIdx.x * blockDim.x;

	c[index] = a[index] + b[index];

}




int main(void)
{


	int *a, *b, *c; // host copies of a, b, c
	int *d_a, *d_b, *d_c; // device copies of a, b, c
	int size = N*sizeof(int); // we need space for N integers

	// allocate device copies of a, b, c
	cudaMalloc((void**)&d_a, size);
	cudaMalloc((void**)&d_b, size);
	cudaMalloc((void**)&d_c, size);

	a = (int*)malloc(size);
	b = (int*)malloc(size);
	c = (int*)malloc(size);

	for(int i = 0; i < N; i++)
	{
		a[i] = i;
		b[i] = i+1;
	}

	// copy inputs to device
	cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);

	// launch add() kernel with blocks and threads
	add<<< N/THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>(d_a, d_b, d_c);

	// copy device result back to host copy of c
	cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);

	for(int i = 0; i < N; i++)
        	printf("%5d + %5d = %5d\n", a[i], b[i], c[i]);

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;


}
