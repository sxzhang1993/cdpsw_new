#include "../headers/book.h"
#include "stdlib.h"
#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <ios>
#include <iostream>
#include <stdio.h>
#include <string>
#include <vector>
#define delta(X,Y) ((X == Y) ? 1 : 0)
#define ceilDiv(X, Y) (((X) + (Y) - 1) / (Y))
#define max2(A,B) ((A) > (B) ? (A) : (B))
#define max4(A,B,C,D) (max2((A) , max2( (B) , max2( (C) , (D)))))
#include "unixtimer.h"

__device__ int g_uids = 0;

char * readFragment(FILE * file, size_t size){

	char * str;
	int c;
	size_t length = 0;

	str = (char *) realloc(NULL, sizeof(char) * size);
	if(!str)return str;
	while( (c = fgetc(file)) != EOF && c != '\n'){
		str[length++] = c;
		if(length == size){
			str = (char*) realloc(str, sizeof(char) * (size+=size));
			if(!str)return str;
		}
	}
	str[length++] = '\0';

	return (char *)realloc(str, sizeof(char) * length);


}



typedef struct kernelData{
	int start;
	int end;
	int windowLength;
	int xlength;
	int * matrix;
}kData;

typedef struct resultData{
	char * cigar;
	int score;
	int location;

}result;


__device__ char * my_strcpy(char *dest, const char *src){
	int i = 0;
	do {
		dest[i] = src[i];}
	while (src[i++] != 0);
	return dest;
}


__device__ char * my_strcat(char *dest, const char *src){
	int i = 0;
	while (dest[i] != 0) i++;
	my_strcpy(dest+i, src);
	return dest;
}

char * compressCigar(char * uncompressedCigar){

	int length = strlen(uncompressedCigar);
	char * compressedCigar = (char*) calloc(sizeof(char), length);
	int start = length-1;

	while(start > 0){

		if(uncompressedCigar[start] == uncompressedCigar[start-1]){
			int count = 1;

			while(uncompressedCigar[start] == uncompressedCigar[start-1]){

				count++;
				start--;

			}

			char buf[15];
			sprintf(buf, "%d%c", count, uncompressedCigar[start]);
			strcat(compressedCigar, buf);

		}else{
			char * buf = (char*) calloc(sizeof(char), 1);
			*buf = uncompressedCigar[start];
			strcat(compressedCigar, buf);
			free(buf);	
		}


		start--;
	}


	return compressedCigar;
}

/*__global__ void cdprun(int const iSize,int iDepth, char * x, char * y, kData* data, result * results) {
    
//   int tid = threadIdx.x；

	char * c_x = x;
	char * c_y = y;
//	 kData* host_data = (kData*) malloc(sizeof(kData) * nWindows);
        kData* device_data = data;
	 result * device_results = results;


	     
   // printf("Recursion=%d: Hello World from thread %d block %d\n",iDepth,threadIdx.x,blockIdx.x);
    // condition to stop recursive execution
    if (iSize == 1) return;
    // reduce block size to half
    int nthreads = iSize>>1;
    // thread 0 launches child grid recursively
    if(threadIdx.x == 0 && nthreads > 0) {
        //nestedHelloWorld<<<1, nthreads>>>(nthreads,++iDepth);
	alignKernel<<<1, nthreads>>>(c_x, c_y, device_data, device_results);
       // printf("-------> nested execution depth: %d\n",iDepth);
    }
}*/

/*__global__ void cdp_kernel(int max_depth, int depth, int thread,
                           int parent_uid) {
  // We create a unique ID per block. Thread 0 does that and shares the value
  // with the other threads.
  __shared__ int s_uid;

  if (threadIdx.x == 0) {
    s_uid = atomicAdd(&g_uids, 1);
    //atomicAdd() reads a word at some address in global or shared memory, adds a number to it, and writes the result back to the same address. Atomic functions can only be used in device functions.
  }

  __syncthreads();

  // We print the ID of the block and information about its parent.
 // print_info(depth, thread, s_uid, parent_uid);

  // We launch new blocks if we haven't reached the max_depth yet.
  if (++depth >= max_depth) {
    return;
  }

  cdp_kernel<<<gridDim.x, blockDim.x>>>(max_depth, depth, threadIdx.x, s_uid);
  //number of blocks in a grid
  //number of threads in a block
  //A group of threads is called a CUDA block. CUDA blocks are grouped into a grid. A kernel is executed as a grid of blocks of threads
}*/
__global__ void alignKernel(char * x, char * y, kData* data, result * results){



	int id = blockIdx.x;
	int start = data[id].start;
	int end = data[id].end;

	int length = end-start;
	int n = data[id].xlength;
	int * device_matrix = data[id].matrix;	
	int max = 0, innerX, innerY;

	for(int i = 0; i <= n; i++){
		device_matrix[i * length + 0] = 0; 
	}
	for(int j = 0; j <= length; j++){
		device_matrix[0 * length + j] = 0;
	}

	for(int i = 1; i <= n; i++){
		for(int j = 1; j <= length; j++){

			int val = max4(0, device_matrix[(i-1)*length+j] -1, device_matrix[i*length+ (j-1)] -1, device_matrix[((i-1)*length+(j-1))] + delta(x[i-1], y[start+j-1]));
			device_matrix[i*length+j] = val; 
			if(val > max){
				max = val;
				innerX = i;
				innerY = j;
			}
		}
	}
	int xCord = innerX, yCord = innerY;
	
	result myResults = results[id];
	while(device_matrix[xCord* length + yCord] > 0 && (yCord > 0 && yCord > 0)){
		if(device_matrix[xCord* length +yCord] == device_matrix[(xCord-1)* length +(yCord-1)] + delta(x[xCord-1], y[start + yCord-1])){

			my_strcat(myResults.cigar, "M");
			xCord--;
			yCord--;

		}else{
			if(device_matrix[xCord* length + yCord] == device_matrix[(xCord-1) * length + yCord] - 1){
				my_strcat(myResults.cigar, "I");
				xCord--;
			}else if(device_matrix[xCord * length + yCord] == device_matrix[xCord * length +(yCord-1)] -1){
				my_strcat(myResults.cigar, "D");
				yCord--;
			}	
		}
	}
	myResults.location = yCord + start + 1;
	myResults.score = max;
	results[id] = myResults;
	return;
}

__global__ void cdprun(int const iSize,int iDepth, char * x, char * y, kData* data, result * results) {

//   int tid = threadIdx.x；

        char * c_x = x;
        char * c_y = y;
//       kData* host_data = (kData*) malloc(sizeof(kData) * nWindows);
        kData* device_data = data;
         result * device_results = results;



   // printf("Recursion=%d: Hello World from thread %d block %d\n",iDepth,threadIdx.x,blockIdx.x);
    // condition to stop recursive execution
    if (iSize == 1) return;
    // reduce block size to half
    int nthreads = iSize>>1;
    // thread 0 launches child grid recursively
    if(threadIdx.x == 0 && nthreads > 0) {
        //nestedHelloWorld<<<1, nthreads>>>(nthreads,++iDepth);
        alignKernel<<<1, nthreads>>>(c_x, c_y, device_data, device_results);
       // printf("-------> nested execution depth: %d\n",iDepth);
    }
}


void print_usage(char * cmd){


	fprintf(stderr, "Usage: %s ", cmd);
	fprintf(stderr, "[-threads] ");
	fprintf(stderr, "[-overlap] ");
	fprintf(stderr, "[-largefile] ");
	fprintf(stderr, "[-smallfile] ");
	fprintf(stderr, "[-windowsize] \n");

}

int main(int argc, char * argv[]){

	FILE * xFile = stdin, * yFile = stdin;

	int numThreads = 16, windowSize = 0, overlap = 0;

	for(int i = 1; i < argc; i++){

		if(!strncmp(argv[i], "-t", strlen("-t"))){
			int userInput = atoi(argv[++i]);
			if(userInput < 16){
				printf("Invalid thread size entered. Using default thread number: %d\n", numThreads);

			}else{

				numThreads = userInput;	

			}

		}else if(!strncmp(argv[i], "-o", strlen("-o"))){

			overlap = atoi(argv[++i]);

		}else if(!strncmp(argv[i], "-w", strlen("-w"))){

			windowSize = atoi(argv[++i]);

		}else if(!strncmp(argv[i], "-s", strlen("-s"))){

			xFile =fopen(argv[++i], "r+");

		}else if(!strncmp(argv[i], "-l", strlen("-l"))){

			yFile = fopen(argv[++i], "r++");

		}else{
			print_usage(argv[0]);	
		}
	}

	if(xFile == stdin)
		printf("Please enter the smaller fragment: ");

	char * xFragment;
	xFragment = readFragment(xFile, 256);

	if(yFile == stdin)
		printf("Please enter the larger fragment: ");
	char * yFragment;
	yFragment = readFragment(yFile, 2048);

	int lenX = strlen(xFragment), lenY = strlen(yFragment);

	if(overlap == 0)	
		overlap = lenX;

	if(windowSize == 0)
		windowSize = lenX * 3;

	int nWindows = ceilDiv(lenY, windowSize);

	char * x, *y;
	HANDLE_ERROR(cudaMalloc((void**) &x, sizeof(char) * lenX));
	HANDLE_ERROR(cudaMalloc((void**) &y, sizeof(char) * lenY));
	HANDLE_ERROR(cudaMemcpy(x, xFragment, sizeof(char) * lenX, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(y, yFragment, sizeof(char) * lenY, cudaMemcpyHostToDevice));

	kData* host_data = (kData*) malloc(sizeof(kData) * nWindows);
	kData* device_data;
	HANDLE_ERROR(cudaMalloc((void**) &device_data, sizeof(kData) * nWindows));
	//We now have our initialized data;

	result * host_results = (result *) malloc(sizeof(result) * nWindows);
	result * device_results;
	HANDLE_ERROR(cudaMalloc((void**) &device_results, sizeof(result) * nWindows));
	//Initialized result structs

	char * cigs[nWindows];

	for(int i = 0; i < nWindows; i++){
		int start = 0;
		if(i == 0)
			start = 0;
		else
			start = host_data[i-1].start - overlap + windowSize;

		host_data[i].start = start;
		int end = start + windowSize;
		end = (end > lenY ? lenY : end);
		host_data[i].end = end;	
		host_data[i].xlength = lenX;
		host_data[i].windowLength = windowSize;
		cigs[i] = (char *) malloc(sizeof(char) * lenX * 2);
		HANDLE_ERROR(cudaMalloc(&(host_results[i].cigar), sizeof(char) * lenX * 2));
		HANDLE_ERROR(cudaMalloc(&(host_data[i].matrix), sizeof(int) * (lenX +1) * (windowSize + 1)));

	}
	HANDLE_ERROR(cudaMemcpy(device_data, host_data, sizeof(kData) * nWindows, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(device_results, host_results, sizeof(result) * nWindows, cudaMemcpyHostToDevice));


	int NITER = 10;
	int blocksize = 8;
	 dim3 block (blocksize, 1);
	//start_timer();
	for(int i = 0; i < NITER; i++){
	//	alignKernel<<<nWindows, 1>>>(x, y, device_data, device_results);		
		cdprun<<<nWindows, 1>>>(block.x, 0, x, y, device_data, device_results);
		HANDLE_ERROR(cudaDeviceSynchronize());
	}
	
      //	fprintf(stderr, "Average kernel time for %d iterations: %lf\n", NITER, NITER/cpu_seconds());


	HANDLE_ERROR(cudaMemcpy(host_results, device_results, sizeof(result) * nWindows, cudaMemcpyDeviceToHost));

	for(int i = 0; i < nWindows; i++){
		HANDLE_ERROR(cudaMemcpy(cigs[i], host_results[i].cigar, sizeof(char) * lenX * 2, cudaMemcpyDeviceToHost));
	}

	int overallMax = 0, location = 0, index = 0;
	for(int i = 0; i < nWindows; i++){
		if(overallMax < host_results[i].score){
			overallMax = host_results[i].score;
			location = host_results[i].location;
			index = i;
		}	
	}	
	//char * compressed = compressCigar(cigs[index]);
	printf("Best alignment found at %d :\n", location);
	
	for(int i = 0; i < nWindows; i++){
		free(cigs[i]);
		HANDLE_ERROR(cudaFree(host_results[i].cigar));
		HANDLE_ERROR(cudaFree(host_data[i].matrix));
	}

	free(host_results);
	free(host_data);
	HANDLE_ERROR(cudaFree(x));
	HANDLE_ERROR(cudaFree(y));
	HANDLE_ERROR(cudaFree(device_data));
	HANDLE_ERROR(cudaFree(device_results));
	return 0;
}
