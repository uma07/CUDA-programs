

# include <stdio.h>
# include <stdlib.h>
# include <cuda.h>
# include <sys/time.h>
# include <unistd.h>


# define BLOCK_SIZE (32)
//# define n 128
//# define n 256
//# define n 512
//# define n 1024
//# define n 2048
//# define n 4096
# define n 8192

# define threshold 1e-8




double rtclock(void)
{

  	struct timezone Tzp;
  	struct timeval Tp;
  	int stat;
  
	stat = gettimeofday (&Tp, &Tzp);
  
	if (stat != 0)
		printf("Error return from gettimeofday: %d",stat);
  
	return(Tp.tv_sec + Tp.tv_usec*1.0e-6);

}




void compare(int N, double *wref, double *w)
{
	double maxdiff,this_diff;
	int numdiffs;
	int i,j;
  	numdiffs = 0;
  	maxdiff = 0;
  
	for(i=0;i<N;i++)
   		for(j=0;j<N;j++)
    		{
     			this_diff = wref[i*N+j]-w[i*N+j];
     
			if(this_diff < 0)
				this_diff = -1.0*this_diff;

     			if(this_diff>threshold)
      			{
				numdiffs++;

				if(this_diff > maxdiff)
					maxdiff=this_diff;
      			}
    		}

   	if(numdiffs > 0)
      		printf("%d Diffs found over threshold %f; Max Diff = %f\n", numdiffs, threshold, maxdiff);

   	else
      		printf("No differences found between reference and test versions\n");
}






int *mat_mul_ord(int *A, int *B, int *C)
{

	for(int i = 0; i < n; i++)
		for(int j = 0; j < n; j++)
		{
			int sum = 0;

			for(int k = 0; k < n; k++)
				sum += A[i*n+k] * B[k*n+j];

			C[i*n+j] = sum;
		}

	return C;

}




__global__ void mat_mul_dev(int *A, int *B, int *C)
{

	int x=threadIdx.y+blockIdx.y*blockDim.y;
	int y=threadIdx.x+blockIdx.x*blockDim.x;
	int sum=0;
	
	if((x<n)&&(y<n))
		for (int k=0;k<n;k++)
    			 sum += A[x*n+k]*B[y*n+k];

	C[x*n+y]=sum;

}





__global__ void matrixMul(int *A, int *B, int *C)
{

	// Declaration of the shared memory arrays As and Bs used to store the sub-matrices of A and B respectively
	__shared__ int As[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ int Bs[BLOCK_SIZE][BLOCK_SIZE];

	int w = BLOCK_SIZE;

	// Block Index
	int bx = blockIdx.x;
        int by = blockIdx.y;

	// Thread Index
   	int tx = threadIdx.x;
        int ty = threadIdx.y;

	// Row 'row' and Column 'col' of matrix A or B
   	int col = bx*w + tx;
        int row = by*w + ty;

    	// Cv is used to store the element of the block sub-matrix that is computed by the thread
   	int Cv = 0;
   
	// Loop over all the sub-matrices of A and B required to compute the block sub-matrix
	for(int k = 0; k < n/w; k++)
   	{
		// Load the matrices from device memory to shared memory; each thread loads one element of each matrix
      		As[ty][tx] = A[row*n + (k*w + tx)];
      		Bs[ty][tx] = B[(k*w + ty)*n + col];

		// Synchronize to make sure the matrices are loaded
      		__syncthreads();

		// Multiply the two matrices together; each thread computes one element of the block sub-matrix
      		for(int l = 0; l < w; l++)
         		Cv += As[ty][l] * Bs[l][tx];
	}
	
	// Write the block sub-matrix to device memory; each thread writes one element
      	C[row*n + col] = Cv;

}




int main()
{


	int *A, *B, *C, *Cref1, *Cref2;
        int *A_d, *B_d, *C_d;
        int i, j;
	double clkbegin, clkend, t;

	A = (int *) malloc(n*n*sizeof(int*));
	B = (int *) malloc(n*n*sizeof(int*));
	C = (int *) malloc(n*n*sizeof(int*));
	Cref1 = (int *) malloc(n*n*sizeof(int*));
	Cref2 = (int *) malloc(n*n*sizeof(int*));

        int size = n*n*sizeof(int);

	// Initialise the input data on the CPU
        for(i = 0; i < n; i++)
                for(j = 0; j < n; j++)
                {
                        A[i*n+j] = 2;//i+j;
                        B[i*n+j] = 1;//2+i+j;
                }

	clkbegin = rtclock();
	C = mat_mul_ord(A, B, C);
	clkend = rtclock();

	t = clkend-clkbegin;

	printf("GPU: Approx GFLOPS: %.1f ; Time = %f sec ; c[n/2][n/2-1] = %d\n", 2.0*n*n*n/t/1e9, t, C[((n/2)*n)+n/2-1]);

	// Create corresponding int arrays on the GPU
        cudaMalloc((void**)&A_d, size);
        cudaMalloc((void**)&B_d, size);
        cudaMalloc((void**)&C_d, size);
        
	// Copy input data to array on GPU
        cudaMemcpy(A_d, A, size, cudaMemcpyHostToDevice);
        cudaMemcpy(B_d, B, size, cudaMemcpyHostToDevice);

	// Set the grid and block sizes to launch kernel
        dim3 block(BLOCK_SIZE, BLOCK_SIZE);
        dim3 grid(n/BLOCK_SIZE, n/BLOCK_SIZE);

	clkbegin = rtclock();
        mat_mul_dev<<<grid, block>>>(A_d, B_d, C_d);
	clkend = rtclock();

	t = clkend-clkbegin;

	cudaMemcpy(Cref1, C_d, size, cudaMemcpyDeviceToHost);
	printf("GPU: Approx GFLOPS: %.1f ; Time = %f sec ; c[n/2][n/2-1] = %d\n", 2.0*n*n*n/t/1e9, t, Cref1[((n/2)*n)+n/2-1]);

	clkbegin = rtclock();
        matrixMul<<<grid, block>>>(A_d, B_d, C_d);
	clkend = rtclock();

	t = clkend-clkbegin;

	// Copy output array from GPU back to CPU
        cudaMemcpy(Cref2, C_d, size, cudaMemcpyDeviceToHost);

	// Free up the arrays on the GPU
        cudaFree(A_d);
        cudaFree(B_d);
        cudaFree(C_d);
/*
	for(i = 0; i < n; i++)
	{
		for(j = 0; j < n; j++)
			printf("%d  ", C[i*n+j]);

		printf("\n");
	}
*/
	printf("GPU: Approx GFLOPS: %.1f ; Time = %f sec ; C[n/2][n/2-1] = %d\n", 2.0*n*n*n/t/1e9, t, Cref2[((n/2)*n)+n/2-1]);
//	printf("GPU: Approx GFLOPS: %.1f ; Time = %f sec ; c[n/2][n/2-1] = %d\n", 2.0*n*n*n/t/1e9, t, Cref[((n/2)*n)+n/2-1]);

	compare(n, (double *) C,(double *) Cref1);
	compare(n, (double *) C,(double *) Cref2);

	return 0;


}




