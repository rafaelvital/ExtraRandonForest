#include "NodeCu.cuh"
#include <time.h>
#include <iostream>
#include <cstdio>
#include <ctime>
#include "defineVariable.h"
#include "errorCheck.cu"

#include "cub/device/device_partition.cuh"

#define FLT_MAX  3.40282347E+38F

#define THREAD THREAD_V
#define BLOCK BLOCK_V
#define BLOCK_SAMPLE BLOCK_SAMPLE_V

#define D_INPUT D_INPUT_V
#define D_OUTPUT D_OUTPUT_V
#define D_STRIDE D_STRIDE_V

//using namespace PoliFitted;

namespace PoliFitted
{

unsigned NodeCu::mNMin = 0;
unsigned NodeCu::mNumSplits = 0;
float NodeCu::mScoreThreshold = 0;

//This kernel construct a initial map for the first node of the tree
//This map is a the sequence i*stride
__global__ void sequence( unsigned * map, const  unsigned size, const unsigned stride){
	unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
	const unsigned gridSize = blockDim.x * gridDim.x;

	while(i < size){
		map[i] = i*stride;
		i+=gridSize;
	}
}

NodeCu::NodeCu(const DataCu* ds_c):ds(ds_c), h_size(ds_c->totalSize), map(NULL){
	//reserve and specify the initial map
	CudaSafeCall(cudaMalloc((void**)&map,sizeof(unsigned)*h_size));
	sequence<<<BLOCK,THREAD>>>(map,h_size, ds->mSize);  //map include stride
	CudaCheckError();
}

NodeCu::NodeCu(const DataCu* ds_c,  unsigned* map_c, const  unsigned h_size_c):ds(ds_c), map(map_c), h_size(h_size_c){}

NodeCu::~NodeCu (){
	CudaSafeCall(cudaFree(map));
}

Dataset* NodeCu::getDataset(){

	//copy the map from device to host
	unsigned* h_map = NULL;
	unsigned  sizeAlloc = sizeof(unsigned)*h_size;
	h_map = ( unsigned*)malloc(sizeAlloc);
	CudaSafeCall(cudaMemcpy(h_map,map,sizeAlloc,cudaMemcpyDeviceToHost));

	//create a empty reduced dataset
    Dataset* newDs = new Dataset(ds->mInputSize, ds->mOutputSize);

    //fill the reduced data set with the information of the Global dataset and the map
    newDs->resize(h_size);
    for(unsigned i = 0;i < h_size ; i++){
    	newDs->at(i)  = ds->ds->at(h_map[i]/ds->mSize);    //map include stride
    }

    free(h_map);

    return newDs;
}

unsigned* NodeCu::getMap(){

	//copy the map from device to host
	unsigned* h_map = NULL;
	unsigned sizeAlloc = sizeof(unsigned)*h_size;
	h_map = (unsigned*)malloc(sizeAlloc);
	CudaSafeCall(cudaMemcpy(h_map,map,sizeAlloc,cudaMemcpyDeviceToHost));

	//map include stride, but h_map don't
	for(unsigned i = 0;i < h_size ; i++){
		h_map[i] /= ds->mSize;
	}

    return h_map;
}

//declared to get a clean code
#define sharedInput shared[0]
#define sharedOutput shared[1]

//This is a template kernel that says when all the input are constant and when the output is constant
//We use different types of kernel if the number of the attribute is greater than the number of thread per block
template <unsigned int blockSize, unsigned block_per_sample, unsigned mInput, unsigned mOutput, unsigned stride >
__global__ void verifyConst(const float* data, const  unsigned* map, bool* d_ans, const  unsigned size){
	//case where we have more attribute than tread per block
	if(block_per_sample > 1){

		//the answer is shared on the block, because if we discovery that one attribute is different we can stop the search
		__shared__ bool shared[2];

		//Variable to control which attribute each thread will read
		const unsigned mod = threadIdx.x+blockIdx.x % block_per_sample * blockSize;
		const unsigned stepIdx = gridDim.x/block_per_sample;
		unsigned step = (blockIdx.x /block_per_sample);

		//All samples attributes will compare there attributes with the attributes of the first sample
		float compare = data[map[0]+mod];

		//declare that the inputs are constant
		sharedInput = true;

		//In a case where a block don't have the output attribute, the sharedOutput is declared false and don't interfere on the work
		//But if the block have the output attribute sharedOutput is declared true and interfere on the work
		sharedOutput = false;
		__syncthreads();

		//In a case where a block don't have the output attribute, the sharedOutput is declared false and don't interfere on the work
		//But if the block have the output attribute sharedOutput is declared true and interfere on the work
		if(mod == mInput){
			sharedOutput = true;
		}
		__syncthreads();

		//only the valid thread work
		if(mod < stride){
			//search for a value different of compare, if find one ends the search
			//note we do two different search, one for the input and another for the output
			while(step < size  && (sharedInput || sharedOutput)){
				if(fabs((data[map[step]+mod]-compare)) > 1e-6f){
					if(mod < mInput){
						sharedInput = false;
					}else{
						sharedOutput = false;
					}
				}
				step+=stepIdx;
			}
		}

		__syncthreads();

		//Assign the result to the global memory
		if(threadIdx.x == 0){
			d_ans[0+2*blockIdx.x] = sharedInput;
			d_ans[1+2*blockIdx.x] = sharedOutput;
		}

	}else{
		//case where  we have more tread per block  than attribute

		//the aswer is shared on the block, because if we discovery that one atribute is different we can stop of search
		__shared__ bool shared[2];

		//Variable to control which attribute each thread will read
		const unsigned readOnly = (blockSize/stride) * stride;
		const unsigned mod = threadIdx.x % stride;
		const unsigned stepIdx = readOnly * gridDim.x / stride;
		unsigned step = (blockIdx.x*readOnly+threadIdx.x) / stride;

		//the kernel is very similar to other case, the only difference is that the sharedOutput always interfere
		float compara = data[map[0]+mod];
		sharedInput = true;
		sharedOutput = true;

		__syncthreads();

		if(threadIdx.x < readOnly){
			while(step < size  && (sharedInput || sharedOutput)){
				if(fabs((data[map[step]+mod]-compara)) > 1e-6f){
					if(mod < mInput){
						sharedInput = false;
					}else{
						sharedOutput = false;
					}
				}
				step+=stepIdx;
			}
		}

		__syncthreads();

		if(threadIdx.x == 0){
			d_ans[0+2*blockIdx.x] = sharedInput;
			d_ans[1+2*blockIdx.x] = sharedOutput;
		}
	}
}

#undef sharedInput
#undef sharedOutput

//This is a template kernel response what is the maximun and minimun value of each attibute
//We use different types of kernel if the number of the attribute is greater than the number of thread per block
template <unsigned int blockSize, unsigned block_per_sample, unsigned mInput, unsigned mOutput, unsigned stride >
__global__ void searchMaxMin(const float* data,const  unsigned* map, float* max, float* min, const  unsigned size){
	//case where we have more attribute than tread per block
	if(block_per_sample > 1){

		//Variable to control which attribute each thread will read
		const unsigned mod = threadIdx.x+blockIdx.x % block_per_sample * blockSize;
		const unsigned stepIdx = gridDim.x / block_per_sample;
		unsigned step = (blockIdx.x /block_per_sample);

		//define initial value
		float max_local = -FLT_MAX;
		float min_local = +FLT_MAX;
		float read;

		//each valid thread read a value and analysis if it is a maximum or a minimum
		if(mod<stride){
			while(step < size){
				read = data[map[step]+mod];
				if(read > max_local) max_local = read;
				if(read < min_local) min_local = read;
				step+=stepIdx;
			}
		}
		__syncthreads();

		//Assign the result to the global memory
		if(mod<stride){
			max[mod+stride*(blockIdx.x/block_per_sample)] = max_local;
			min[mod+stride*(blockIdx.x/block_per_sample)] = min_local;
		}

	}else{
		//case where we have more tread per block than attribute

		//shared variable to do the block reduce
		__shared__ float maxShare[blockSize];
		__shared__ float minShare[blockSize];

		//Variable to control which attribute each thread will read
		const unsigned tid = threadIdx.x;
		const unsigned readOnly = (blockSize/stride) * stride;
		const unsigned mod = threadIdx.x % stride;
		const unsigned stepIdx = readOnly * gridDim.x / stride;
		unsigned step = (blockIdx.x*readOnly+threadIdx.x) / stride;

		//define initial value
		float read;
		float maxLocal = -FLT_MAX;
		float minLocal = +FLT_MAX;

		//each valid thread read a value and analysis if it is a maximum or a minimum
		if(tid<readOnly){
			while(step < size){
				read = data[map[step]+mod];
				if(read > maxLocal) maxLocal = read;
				if(read < minLocal) minLocal = read;
				step+=stepIdx;
			}
		}
		__syncthreads();

		//block reduce
		unsigned lenght = size < readOnly/stride ? size : readOnly/stride;
		if(lenght > 1){
			//copy the variable from the local to the shared memory
			maxShare[tid] = maxLocal;
			minShare[tid] = minLocal;
			unsigned s;
			unsigned base;
			//start reduce
			while(lenght > 1){
				base = lenght >> 1;
				if(lenght & 1 == 1 ) base++;
				s = tid+base*stride;
				if(s < lenght * stride){
					if(maxShare[s] > maxShare[tid]) maxShare[tid] = maxShare[s];
					if(minShare[s] < minShare[tid]) minShare[tid] = minShare[s];
				}
				lenght = base;
				__syncthreads();
			}
			//Assign the result to the global memory after the block reduce
			if(tid<stride){
				max[tid+stride*blockIdx.x] = maxShare[tid];
				min[tid+stride*blockIdx.x] = minShare[tid];
			}
		}else{
			//Assign the result to the global memory if we don't need to do the block reduce
			if(tid<stride){
				max[tid+stride*blockIdx.x] = maxLocal;
				min[tid+stride*blockIdx.x] = minLocal;
			}
		}
	}
}

//MultiNode Kernel
//This is a template kernel response what is the maximun and minimun value of each attibute
//We use different types of kernel if the number of the attribute is greater than the number of thread per block
template <unsigned int blockSize, unsigned block_per_sample, unsigned mInput, unsigned mOutput, unsigned stride >
__global__ void searchMaxMin(const float* data, unsigned** map, float* max, float* min, const unsigned* size, const unsigned nNode){

	const unsigned node = blockIdx.x * nNode / gridDim.x; //the exact expression is blockIdx.x / (gridDim.x / nNode), but this operation is more expensive
	const unsigned newblockIdx = blockIdx.x % (gridDim.x / nNode);
	const unsigned newgridDim = (gridDim.x / nNode);
	const unsigned d_size = size[node];

	//case where we have more attribute than tread per block
	if(block_per_sample > 1){

		//Variable to control which attribute each thread will read
		const unsigned mod = threadIdx.x+newblockIdx % block_per_sample * blockSize;
		const unsigned stepIdx = newgridDim / block_per_sample;
		unsigned step = (newblockIdx /block_per_sample);

		//define initial value
		float max_local = -FLT_MAX;
		float min_local = +FLT_MAX;
		float read;

		//each valid thread read a value and analysis if it is a maximum or a minimum
		if(mod<stride){
			while(step < d_size){
				read = data[map[node][step]+mod];
				if(read > max_local) max_local = read;
				if(read < min_local) min_local = read;
				step+=stepIdx;
			}
		}
		__syncthreads();

		//Assign the result to the global memory
		if(mod<stride){
			max[mod+stride*(blockIdx.x/block_per_sample)] = max_local;
			min[mod+stride*(blockIdx.x/block_per_sample)] = min_local;
		}

	}else{
		//case where we have more tread per block than attribute

		//shared variable to do the block reduce
		__shared__ float maxShare[blockSize];
		__shared__ float minShare[blockSize];

		//Variable to control which attribute each thread will read
		const unsigned tid = threadIdx.x;
		const unsigned readOnly = (blockSize/stride) * stride;
		const unsigned mod = threadIdx.x % stride;
		const unsigned stepIdx = readOnly * newgridDim / stride;
		unsigned step = (newblockIdx*readOnly+threadIdx.x) / stride;

		//define initial value
		float read;
		float maxLocal = -FLT_MAX;
		float minLocal = +FLT_MAX;

		//each valid thread read a value and analysis if it is a maximum or a minimum
		if(tid<readOnly){
			while(step < d_size){
				read = data[map[node][step]+mod];
				if(read > maxLocal) maxLocal = read;
				if(read < minLocal) minLocal = read;
				step+=stepIdx;
			}
		}
		__syncthreads();

		//block reduce
		unsigned lenght = d_size < readOnly/stride ? d_size : readOnly/stride;
		if(lenght > 1){
			//copy the variable from the local to the shared memory
			maxShare[tid] = maxLocal;
			minShare[tid] = minLocal;
			unsigned s;
			unsigned base;
			//start reduce
			while(lenght > 1){
				base = lenght >> 1;
				if(lenght & 1 == 1 ) base++;
				s = tid+base*stride;
				if(s < lenght * stride){
					if(maxShare[s] > maxShare[tid]) maxShare[tid] = maxShare[s];
					if(minShare[s] < minShare[tid]) minShare[tid] = minShare[s];
				}
				lenght = base;
				__syncthreads();
			}
			//Assign the result to the global memory after the block reduce
			if(tid<stride){
				max[tid+stride*blockIdx.x] = maxShare[tid];
				min[tid+stride*blockIdx.x] = minShare[tid];
			}
		}else{
			//Assign the result to the global memory if we don't need to do the block reduce
			if(tid<stride){
				max[tid+stride*blockIdx.x] = maxLocal;
				min[tid+stride*blockIdx.x] = minLocal;
			}
		}
	}
}


//MultiNode Kernel
//This is a template kernel that response what is the sum, sum square and the number of elements of each split
//The sum ...  are taken with respect the output value, the split data is with respect to the input
//we calculate the sum ... of the part of data that is smaller than the split value
//the part that is greater is obtained with a subtract operation

//We use different types of kernel if the number of the attribute is greater than the number of thread per block
template <unsigned int blockSize, unsigned block_per_sample, unsigned mInput, unsigned mOutput, unsigned stride >
__global__ void reduceVariance(const float* data,  unsigned** map, const float* rSplit, float* reduceSum, float* reduceSumSquare,  unsigned* reduceMinorSize,const unsigned* size, const unsigned nNode){

	const unsigned node = blockIdx.x * nNode / gridDim.x; //the exact expression is blockIdx.x / (gridDim.x / nNode), but this operation is more expensive
	const unsigned newblockIdx = blockIdx.x % (gridDim.x / nNode);
	const unsigned newgridDim = (gridDim.x / nNode);
	const unsigned d_size = size[node];

	//case where we have more attribute than tread per block
	if(block_per_sample > 1){

		//Variable to control which attribute each thread will read
		const unsigned mod = threadIdx.x+newblockIdx % block_per_sample * blockSize; //the last will be greater than stride
		const unsigned stepIdx = newgridDim/block_per_sample;
		unsigned step = (newblockIdx /block_per_sample);

		float split;
		float read;
		float sum;
		float sumSquare;
		unsigned minorSize;

		//define initial value
		if(mod < stride){
			split = rSplit[mod+stride*node];
			sum = 0.0f;
			sumSquare = 0.0f;
			minorSize = 0;

			//each valid thread read a input value,if this value is smaller than the split, the kernel will read the
			//output and add it to the sum, sumSquare and number of elements(minorSize)
			while(step < d_size){
				if(data[map[node][step]+mod] < split){
					read = data[map[node][step]+mInput]; //data of output[0]
					sum += read;
					sumSquare += read*read;
					minorSize += 1;
				}
				step += stepIdx;
			}
		}
		__syncthreads();

		//Assign the result to the global memory
		if(mod < stride){
			reduceSum[mod+stride*(blockIdx.x/block_per_sample)] = sum;
			reduceSumSquare[mod+stride*(blockIdx.x/block_per_sample)] = sumSquare;
			reduceMinorSize[mod+stride*(blockIdx.x/block_per_sample)] = minorSize;
		}
	}else{
		//case where we have more tread per block than attribute

		//shared variable to do the block reduce
		__shared__ float sumShared[blockSize];
		__shared__ float sumSquareShared[blockSize];
		__shared__ unsigned minorSizeShared[blockSize];

		//Variable to control which attribute each thread will read
		const unsigned tid = threadIdx.x;
		const unsigned readOnly = (blockSize/stride) * stride;
		const unsigned mod = threadIdx.x % stride;
		const unsigned stepIdx = readOnly * newgridDim / stride;
		unsigned step = (newblockIdx*readOnly+threadIdx.x) / stride;

		//define initial value
		float read;
		float split = rSplit[mod+stride*node];
		float sum = 0.0f;
		float sumSquare = 0.0f;
		float minorSize = 0;

		//each valid thread read a input value,if this value is smaller than the split, the kernel will read the
		//output and add it to the sum, sumSquare and number of elements(minorSize)
		if(tid< readOnly){
			while(step < d_size){
				if(data[map[node][step]+mod] < split){
					read = data[map[node][step]+mInput]; //data of output[0]
					sum += read;
					sumSquare += read*read;
					minorSize += 1;
				}
				step += stepIdx;
			}
		}
		__syncthreads();

		//block reduce
		unsigned lenght = d_size <readOnly/stride ? d_size : readOnly /stride;
		if(lenght > 1){
			//copy the variable from the local to the shared memory
			sumShared[tid] = sum;
			sumSquareShared[tid] = sumSquare;
			minorSizeShared[tid] = minorSize;
			unsigned s;
			unsigned base;
			//start reduce
			while(lenght > 1){
				base = lenght >> 1;
				if(lenght & 1 == 1 ) base++;
				s = tid+base*stride;
				if(s < lenght *stride ){
					sumShared[tid] += sumShared[s];
					sumSquareShared[tid] += sumSquareShared[s];
					minorSizeShared[tid] += minorSizeShared[s];
				}
				lenght = base;
				__syncthreads();
			}
			//Assign the result to the global memory after the block reduce
			if(tid < stride){
				reduceSum[tid+stride*blockIdx.x] = sumShared[tid];
				reduceSumSquare[tid+stride*blockIdx.x] = sumSquareShared[tid];
				reduceMinorSize[tid+stride*blockIdx.x] = minorSizeShared[tid];
			}

		}else{
			//Assign the result to the global memory if we don't need to do the block reduce
			if(tid < stride){
				reduceSum[tid+stride*blockIdx.x] = sum;
				reduceSumSquare[tid+stride*blockIdx.x] = sumSquare;
				reduceMinorSize[tid+stride*blockIdx.x] = minorSize;
			}
		}
	}
}

//We use different types of kernel if the number of the attribute is greater than the number of thread per block
template <unsigned int blockSize, unsigned block_per_sample, unsigned mInput, unsigned mOutput, unsigned stride >
__global__ void reduceVariance(const float* data, const  unsigned* map, const float* rSplit, float* reduceSum, float* reduceSumSquare,  unsigned* reduceMinorSize,const unsigned size){
	//case where we have more attribute than tread per block
	if(block_per_sample > 1){

		//Variable to control which attribute each thread will read
		const unsigned mod = threadIdx.x+blockIdx.x % block_per_sample * blockSize; //the last will be greater than stride
		const unsigned stepIdx = gridDim.x/block_per_sample;
		unsigned step = (blockIdx.x /block_per_sample);

		float split;
		float read;
		float sum;
		float sumSquare;
		unsigned minorSize;

		//define initial value
		if(mod < stride){
			split = rSplit[mod];
			sum = 0.0f;
			sumSquare = 0.0f;
			minorSize = 0;

			//each valid thread read a input value,if this value is smaller than the split, the kernel will read the
			//output and add it to the sum, sumSquare and number of elements(minorSize)
			while(step < size){
				if(data[map[step]+mod] < split){
					read = data[map[step]+mInput]; //data of output[0]
					sum += read;
					sumSquare += read*read;
					minorSize += 1;
				}
				step += stepIdx;
			}
		}
		__syncthreads();

		//Assign the result to the global memory
		if(mod < stride){
			reduceSum[mod+stride*(blockIdx.x/block_per_sample)] = sum;
			reduceSumSquare[mod+stride*(blockIdx.x/block_per_sample)] = sumSquare;
			reduceMinorSize[mod+stride*(blockIdx.x/block_per_sample)] = minorSize;
		}
	}else{
		//case where we have more tread per block than attribute

		//shared variable to do the block reduce
		__shared__ float sumShared[blockSize];
		__shared__ float sumSquareShared[blockSize];
		__shared__ unsigned minorSizeShared[blockSize];

		//Variable to control which attribute each thread will read
		const unsigned tid = threadIdx.x;
		const unsigned readOnly = (blockSize/stride) * stride;
		const unsigned mod = threadIdx.x % stride;
		const unsigned stepIdx = readOnly * gridDim.x / stride;
		unsigned step = (blockIdx.x*readOnly+threadIdx.x) / stride;

		//define initial value
		float read;
		float split = rSplit[mod];
		float sum = 0.0f;
		float sumSquare = 0.0f;
		float minorSize = 0;

		//each valid thread read a input value,if this value is smaller than the split, the kernel will read the
		//output and add it to the sum, sumSquare and number of elements(minorSize)
		if(tid< readOnly){
			while(step < size){
				if(data[map[step]+mod] < split){
					read = data[map[step]+mInput]; //data of output[0]
					sum += read;
					sumSquare += read*read;
					minorSize += 1;
				}
				step += stepIdx;
			}
		}
		__syncthreads();

		//block reduce
		unsigned lenght = size <readOnly/stride ? size : readOnly /stride;
		if(lenght > 1){
			//copy the variable from the local to the shared memory
			sumShared[tid] = sum;
			sumSquareShared[tid] = sumSquare;
			minorSizeShared[tid] = minorSize;
			unsigned s;
			unsigned base;
			//start reduce
			while(lenght > 1){
				base = lenght >> 1;
				if(lenght & 1 == 1 ) base++;
				s = tid+base*stride;
				if(s < lenght *stride ){
					sumShared[tid] += sumShared[s];
					sumSquareShared[tid] += sumSquareShared[s];
					minorSizeShared[tid] += minorSizeShared[s];
				}
				lenght = base;
				__syncthreads();
			}
			//Assign the result to the global memory after the block reduce
			if(tid < stride){
				reduceSum[tid+stride*blockIdx.x] = sumShared[tid];
				reduceSumSquare[tid+stride*blockIdx.x] = sumSquareShared[tid];
				reduceMinorSize[tid+stride*blockIdx.x] = minorSizeShared[tid];
			}

		}else{
			//Assign the result to the global memory if we don't need to do the block reduce
			if(tid < stride){
				reduceSum[tid+stride*blockIdx.x] = sum;
				reduceSumSquare[tid+stride*blockIdx.x] = sumSquare;
				reduceMinorSize[tid+stride*blockIdx.x] = minorSize;
			}
		}
	}
}

//create a bool vector with one if the select input is smaller than the split value
__global__ void createStencil(const float* data, const  unsigned* map, bool* stencilA,  const unsigned axis, const float split, const  unsigned size){
	unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
	const  unsigned gridSize = gridDim.x*blockDim.x;

	while(i < size){
		stencilA[i] = data[map[i]+axis] < split;
		i+= gridSize;
	}
}

//Divided the newMap in two other vector  mapRight and mapLeft according to minorSize
__global__ void copyMap( unsigned* newMap,  unsigned* mapLeft, unsigned* mapRight, unsigned size,  unsigned minorSize){
	 unsigned i = blockIdx.x*blockDim.x+threadIdx.x;
	const  unsigned gridSize = gridDim.x*blockDim.x;

	while(i<minorSize){
			mapLeft[i] = newMap[i];
			i+=gridSize;
	}

	while(i<size){
		mapRight[i-minorSize] = newMap[i];
		i+=gridSize;
	}
}

void  NodeCu::processNode(bool& isLeaf,  unsigned& axis, float& split, NodeCu*& right, NodeCu*& left){


	//We don't use anymore the end conditions by constant input/output
	//The end condition of size, is verified by the the function trainList of ExtraTreeCuda
	//The end codition of equality is verified by the score function
	//when the input/output is constant, the score function must be the smaller value

     /*************** part 1 - END CONDITIONS ********************/
	/*
	//END SIZE
    if(h_size < mNMin){
         isLeaf = true;
         right = this;
         std::cout << " m min" << std::endl;
         return;
    }

    //Declare variable and copy
    bool* d_ans = NULL;
    bool* h_ans = NULL;
    unsigned sizeAlloc = sizeof(bool)*2*BLOCK;
    CudaSafeCall(cudaMalloc((void**)&d_ans,sizeAlloc));
	h_ans = (bool*)malloc(sizeAlloc);

	bool inputIsConst = true;
	bool outputIsConst = true;

	//kernel launch
	//REMARK: The mean of each  kernel's response is different, so the joined part is a little different
	verifyConst<THREAD, BLOCK_SAMPLE, D_INPUT, D_OUTPUT,D_STRIDE><<<BLOCK,THREAD>>>(ds->data, map, d_ans, h_size);
	CudaCheckError();

	//copy and join the response
	CudaSafeCall(cudaMemcpy(h_ans,d_ans,sizeAlloc,cudaMemcpyDeviceToHost));
	if(BLOCK_SAMPLE == 1){
		for(unsigned i = 0; i < BLOCK ; i++){
			inputIsConst = inputIsConst && h_ans[2*i];
			outputIsConst = outputIsConst && h_ans[2*i+1];
		}
	}else{
		for(unsigned i = 0; i < BLOCK ; i++){
			inputIsConst = inputIsConst && h_ans[2*i];
		}
		for(unsigned i = BLOCK_SAMPLE-1; i < BLOCK ; i+=BLOCK_SAMPLE){
			outputIsConst = outputIsConst && h_ans[2*i+1];
		}
	}

	//Return for constant Input or Output
    if( inputIsConst || outputIsConst == true){
    	isLeaf = true;
    	right = this;
    	std::cout << " is constant " <<  inputIsConst  << "  "  <<  outputIsConst << std::endl;
    	CudaSafeCall(cudaFree(d_ans));
    	free(h_ans);
    	return;
    }

    CudaSafeCall(cudaFree(d_ans));
    free(h_ans);

 */


    /************** part 2 - SEARCHE MAXIMUN AND MINIMUN *******************/

    //declare variable and copy
    float* d_max = NULL;
    float* d_min = NULL;
    float* h_max = NULL;
    float* h_min = NULL;
    unsigned sizeAlloc = sizeof(float)*(ds->mSize)*BLOCK/BLOCK_SAMPLE;
    CudaSafeCall(cudaMalloc((void**)&d_max,sizeAlloc));
    CudaSafeCall(cudaMalloc((void**)&d_min,sizeAlloc));
    h_max = (float*)malloc(sizeAlloc);
    h_min = (float*)malloc(sizeAlloc);

    //kernel launch
    searchMaxMin<THREAD,BLOCK_SAMPLE,D_INPUT, D_OUTPUT,D_STRIDE><<<BLOCK,THREAD>>>(ds->data, map, d_max, d_min, h_size);
    CudaCheckError();

    //copy and join the response
    CudaSafeCall(cudaMemcpy(h_max,d_max,sizeAlloc,cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(h_min,d_min,sizeAlloc,cudaMemcpyDeviceToHost));
    for(unsigned i = 1; i < BLOCK / BLOCK_SAMPLE ; i++){
    	for(unsigned j=0;j < ds->mSize; j++){
    		if(!(h_max[j] > h_max[j+ds->mSize*i])) h_max[j] = h_max[j+ds->mSize*i];
    		if(!(h_min[j] <= h_max[j+ds->mSize*i])) h_min[j] = h_min[j+ds->mSize*i];
    	}
    }

    //The CPU calculate the random split with respect to to the maximum and minimum
    float* d_rSplit = NULL;
    float* h_rSplit = NULL;
    sizeAlloc = sizeof(float)*(ds->mSize);
    CudaSafeCall(cudaMalloc((void**)&d_rSplit,sizeAlloc));
    h_rSplit = (float*)malloc(sizeAlloc);

    for( unsigned i = 0; i < ds->mInputSize; ++i){
    	h_rSplit[i] = h_min[i] + (h_max[i] - h_min[i]) * (float)((rand() % 99) + 1) / 100.0;
    }

    for( unsigned i = ds->mInputSize; i < ds->mSize; ++i){
    	h_rSplit[i] = FLT_MAX; //the output split is meanless, we put the maximum value for do not split the output during the calulation of variance
    							//So when we calc the variance for the "output split" we calc the variance for the entire data
    }

    //the split information is copied to the device
    CudaSafeCall(cudaMemcpy(d_rSplit,h_rSplit,sizeAlloc,cudaMemcpyHostToDevice));

    //free resource
    CudaSafeCall(cudaFree(d_max));
    CudaSafeCall(cudaFree(d_min));
    free(h_max);
    free(h_min);


    /************** part 3 -CALCULATE THE SCORE *******************/

    //Declare and copy variable
    float* d_sum = NULL;
    float* d_sumSquare = NULL;
    float* h_sum = NULL;
    float* h_sumSquare = NULL;
    sizeAlloc = sizeof(float)*(ds->mSize)*BLOCK / BLOCK_SAMPLE;
    CudaSafeCall(cudaMalloc((void**)&d_sum,sizeAlloc));
    CudaSafeCall(cudaMalloc((void**)&d_sumSquare,sizeAlloc));
    h_sum = (float*)malloc(sizeAlloc);
    h_sumSquare = (float*)malloc(sizeAlloc);

    unsigned* d_minorSize = NULL;
    unsigned* h_minorSize = NULL;
    sizeAlloc = sizeof( unsigned)*(ds->mSize)*BLOCK/ BLOCK_SAMPLE;
    CudaSafeCall(cudaMalloc((void**)&d_minorSize,sizeAlloc));
    h_minorSize = ( unsigned*)malloc(sizeAlloc);

    //Launch kernel to calculate the sum and sumSquare for each split
    reduceVariance<THREAD, BLOCK_SAMPLE, D_INPUT, D_OUTPUT,D_STRIDE><<<BLOCK,THREAD>>>(ds->data, map, d_rSplit, d_sum, d_sumSquare, d_minorSize, h_size);
    CudaCheckError();

    //copy and join the response
    CudaSafeCall(cudaMemcpy(h_minorSize,d_minorSize,sizeAlloc,cudaMemcpyDeviceToHost));
    sizeAlloc = sizeof(float)*(ds->mSize)*BLOCK /BLOCK_SAMPLE;
    CudaSafeCall(cudaMemcpy(h_sum,d_sum,sizeAlloc,cudaMemcpyDeviceToHost));
    CudaSafeCall(cudaMemcpy(h_sumSquare,d_sumSquare,sizeAlloc,cudaMemcpyDeviceToHost));

    for(unsigned i = 1; i < BLOCK / BLOCK_SAMPLE ; i++){
    	for(unsigned j=0;j < ds->mSize; j++){
    		h_sum[j] += h_sum[j+(ds->mSize)*i];
    		h_sumSquare[j] += h_sumSquare[j+(ds->mSize)*i];
    		h_minorSize[j] += h_minorSize[j+(ds->mSize)*i];
    	}
    }

    //score variable
    std::vector<float> score(ds->mInputSize);
    float bestScore = -1;
    float varMinor;
    float varMajor;
    float varTotal;
    unsigned minorSize;

    //calculate score
    //All the variance is calculate unless a constant value, due to the fact that this same value will be used to calculate the score
    //the split cause by the split of output term, represent the entire data
    varTotal =  h_sumSquare[ds->mInputSize] - (h_sum[ds->mInputSize] * h_sum[ds->mInputSize]) / h_size;
    for(unsigned i = 0; i< ds->mInputSize;++i){
    	if(h_minorSize[i]!= 0 && h_minorSize[i]!= h_size){
    		varMinor = h_sumSquare[i]-(h_sum[i] * h_sum[i]) / (float) h_minorSize[i];
    		varMajor = (h_sumSquare[ds->mInputSize] - h_sumSquare[i]) - ((h_sum[ds->mInputSize] - h_sum[i])*(h_sum[ds->mInputSize] - h_sum[i])) / (float)(h_size-h_minorSize[i]);
    		score[i] = 1 - (varMinor + varMajor) / varTotal;
    	}else{
    		score[i] = 0;
    	}
    }

    //choose the best score and set the best axis and split
    for(unsigned i = 0; i < ds->mInputSize; ++i){
    	if(score[i] > bestScore) {
    		bestScore = score[i];
    		axis = i;
    		split = h_rSplit[i];
    		minorSize = h_minorSize[i];
    		//		std::cout << "axis: "<< axis << "  score " <<  bestScore << std::endl;
    	}
    }

    //free resource
    CudaSafeCall(cudaFree(d_rSplit));
    CudaSafeCall(cudaFree(d_sum));
    CudaSafeCall(cudaFree(d_sumSquare));
    CudaSafeCall(cudaFree(d_minorSize));
    free(h_rSplit);
    free(h_sum);
    free(h_sumSquare);
    free(h_minorSize);

    //Leaf condition for the score
    if (bestScore <= mScoreThreshold) {
    	isLeaf = true;
    	right = this;
    	//std::cout << " score basso" << std::endl;
    	return;
    }

    //Create a stencil with respect to the axis and split choosed
    bool* stencil = NULL;
    sizeAlloc = sizeof(bool)*h_size;
    CudaSafeCall(cudaMalloc((void**)&stencil,sizeAlloc));
    createStencil<<<BLOCK, THREAD>>>(ds->data, map, stencil, axis, split , h_size);
    CudaCheckError();


    //Partitionate the current map using the stencil
    //This is done using the function DevicePartition::Flagged of the library CUB
    unsigned* newMap = NULL;
    CudaSafeCall(cudaMalloc((void**)&newMap, sizeof( unsigned)*h_size));
    unsigned* d_num_selected_out = NULL;
    CudaSafeCall(cudaMalloc((void**)&d_num_selected_out, sizeof( unsigned)));

    void	*d_temp_storage = NULL;
    size_t	temp_storage_bytes = 0;
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, map, stencil, newMap, d_num_selected_out, h_size);
    CudaSafeCall(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, map, stencil, newMap, d_num_selected_out, h_size);
    CudaSafeCall(cudaFree(d_temp_storage));

    //Copy the partitioned map in two diferent maps, one for each child
    unsigned* mapLeft = NULL;
    unsigned* mapRight = NULL;
    CudaSafeCall(cudaMalloc((void**)&mapLeft, sizeof( unsigned)*minorSize));
    CudaSafeCall(cudaMalloc((void**)&mapRight,sizeof( unsigned)*(h_size-minorSize)));
    copyMap<<<BLOCK, THREAD>>>(newMap, mapLeft,mapRight, h_size, minorSize);
    CudaCheckError();

    //free resource
    CudaSafeCall(cudaFree(newMap));
    CudaSafeCall(cudaFree(stencil));
    CudaSafeCall(cudaFree(d_num_selected_out));

    //Create the child
    isLeaf = false;
    left = new NodeCu(ds, mapLeft, minorSize);
    right = new NodeCu(ds, mapRight, h_size-minorSize);


   // std::cout << "h_size = " << h_size << std::endl;


}


void processVectorNode(std::vector<NodeCu*>& nodeCuVector, std::vector<bool>& isLeaf,  std::vector<unsigned>& axis, std::vector<float>& split, std::vector<NodeCu*>& right, std::vector<NodeCu*>& left){


	/*************** part 0 - CONTROL VARIABLE AND NODE SETUP ********************/

	//number of node to analysis/process
	unsigned nNode = nodeCuVector.size();
	unsigned mSize = nodeCuVector[0]->ds->mSize;
	unsigned mInput = nodeCuVector[0]->ds->mInputSize;
	unsigned mOutput = nodeCuVector[0]->ds->mOutputSize;
	unsigned nodeId;


	//define on the device the size of each node
	unsigned* d_sizeVec = NULL;
	unsigned* h_sizeVec = NULL;
	unsigned sizeAlloc = sizeof(unsigned)*nNode;
	CudaSafeCall(cudaMalloc((void**)&d_sizeVec,sizeAlloc));
	h_sizeVec = (unsigned*)malloc(sizeAlloc);
	for(unsigned i=0; i< nNode; ++i){
		h_sizeVec[i] = nodeCuVector[i]->h_size;
	}
	CudaSafeCall(cudaMemcpy(d_sizeVec,h_sizeVec,sizeAlloc,cudaMemcpyHostToDevice));

	//define vector of map of each node
	unsigned** d_mapVec = NULL;
	unsigned** h_mapVec = NULL;
	sizeAlloc = sizeof(unsigned*)*nNode;
	CudaSafeCall(cudaMalloc((void**)&d_mapVec,sizeAlloc));
	h_mapVec = (unsigned**)malloc(sizeAlloc);
	for(unsigned i=0; i< nNode; ++i){
		h_mapVec[i] = nodeCuVector[i]->map;
	}
	CudaSafeCall(cudaMemcpy(d_mapVec, h_mapVec,sizeAlloc,cudaMemcpyHostToDevice));

	/*************** part 1 - END CONDITIONS ********************/
	//We don't use anymore the end conditions by constant input/output
	//The end condition of size, is verified by the the function trainList of ExtraTreeCuda
	//The end codition of equality is verified by the score function
	//when the input/output is constant, the score function must be the smaller value


	/************** part 2 - SEARCHE MAXIMUN AND MINIMUN *******************/

	//the answer of max and min is only one vector, this vector contains the max and min of all nodes processed
	//declare variable and copy
	float* d_max = NULL;
	float* d_min = NULL;
	float* h_max = NULL;
	float* h_min = NULL;
	sizeAlloc = sizeof(float)* mSize * BLOCK/BLOCK_SAMPLE;
	CudaSafeCall(cudaMalloc((void**)&d_max,sizeAlloc));
	CudaSafeCall(cudaMalloc((void**)&d_min,sizeAlloc));
	h_max = (float*)malloc(sizeAlloc);
	h_min = (float*)malloc(sizeAlloc);

	//kernel launch
	searchMaxMin<THREAD,BLOCK_SAMPLE,D_INPUT, D_OUTPUT,D_STRIDE><<<BLOCK,THREAD>>>(nodeCuVector[0]->ds->data, d_mapVec, d_max, d_min, d_sizeVec, nNode);
	CudaCheckError();

	//copy and join the response
	CudaSafeCall(cudaMemcpy(h_max,d_max,sizeAlloc,cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(h_min,d_min,sizeAlloc,cudaMemcpyDeviceToHost));
	for(unsigned n = 0; n < nNode; n++){
		nodeId = n * BLOCK / BLOCK_SAMPLE / nNode * mSize;
		for(unsigned i = 1; i < BLOCK / BLOCK_SAMPLE / nNode; i++){
			for(unsigned j=0;j < mSize; j++){
				//reduction between blocks of same nodeId
				if(!(h_max[j+nodeId] > h_max[j+mSize*i+nodeId]))
					h_max[j+nodeId] = h_max[j+mSize*i+nodeId];
				if(!(h_min[j + nodeId] <= h_min[j+mSize*i +nodeId]))
					h_min[j + nodeId] = h_min[j+mSize*i + nodeId];
			}
		}
	}

	//The CPU calculate the random split with respect to to the maximum and minimum
	float* d_rSplit = NULL;
	float* h_rSplit = NULL;
	sizeAlloc = sizeof(float) * mSize * nNode;
	CudaSafeCall(cudaMalloc((void**)&d_rSplit, sizeAlloc));
	h_rSplit = (float*)malloc(sizeAlloc);

	for(unsigned n = 0; n < nNode; ++n){
		nodeId = n * BLOCK / BLOCK_SAMPLE / nNode * mSize;
		for( unsigned i = 0; i < mInput; ++i){
			h_rSplit[i+ n * mSize] = h_min[i+nodeId] + (h_max[i+ nodeId] - h_min[i+ nodeId]) * (float)((rand() % 99) + 1) / 100.0;
		}
		for( unsigned i = mInput; i < mSize; ++i){
			h_rSplit[i + n * mSize] = FLT_MAX; //the output split is meanless, we put the maximum value for do not split the output during the calulation of variance
												//So when we calc the variance for the "output split" we calc the variance for the entire data
		}
	}

	//the split information is copied to the device
	CudaSafeCall(cudaMemcpy(d_rSplit,h_rSplit,sizeAlloc,cudaMemcpyHostToDevice));

	//free resource
	CudaSafeCall(cudaFree(d_max));
	CudaSafeCall(cudaFree(d_min));
	free(h_max);
	free(h_min);

	/************** part 3 -CALCULATE THE SCORE *******************/

	//Declare and copy variable
	float* d_sum = NULL;
	float* d_sumSquare = NULL;
	float* h_sum = NULL;
	float* h_sumSquare = NULL;
	sizeAlloc = sizeof(float)* mSize *BLOCK / BLOCK_SAMPLE;
	CudaSafeCall(cudaMalloc((void**)&d_sum,sizeAlloc));
	CudaSafeCall(cudaMalloc((void**)&d_sumSquare,sizeAlloc));
	h_sum = (float*)malloc(sizeAlloc);
	h_sumSquare = (float*)malloc(sizeAlloc);

	unsigned* d_minorSize = NULL;
	unsigned* h_minorSize = NULL;
	sizeAlloc = sizeof( unsigned)* mSize *BLOCK/ BLOCK_SAMPLE;
	CudaSafeCall(cudaMalloc((void**)&d_minorSize,sizeAlloc));
	h_minorSize = ( unsigned*)malloc(sizeAlloc);

	//Launch kernel to calculate the sum and sumSquare for each split
	reduceVariance<THREAD, BLOCK_SAMPLE, D_INPUT, D_OUTPUT,D_STRIDE><<<BLOCK,THREAD>>>(nodeCuVector[0]->ds->data, d_mapVec, d_rSplit, d_sum, d_sumSquare, d_minorSize, d_sizeVec, nNode);
	CudaCheckError();

	//copy and join the response
	CudaSafeCall(cudaMemcpy(h_minorSize,d_minorSize,sizeAlloc,cudaMemcpyDeviceToHost));
	sizeAlloc = sizeof(float)* mSize *BLOCK /BLOCK_SAMPLE;
	CudaSafeCall(cudaMemcpy(h_sum,d_sum,sizeAlloc,cudaMemcpyDeviceToHost));
	CudaSafeCall(cudaMemcpy(h_sumSquare,d_sumSquare,sizeAlloc,cudaMemcpyDeviceToHost));

	std::vector<unsigned> minorSize(nNode);
	float mScoreThreshold = 0;
	for(unsigned n = 0;n < nNode; n ++){
		nodeId = n * BLOCK / BLOCK_SAMPLE / nNode * mSize;
		for(unsigned i = 1; i < BLOCK / BLOCK_SAMPLE /nNode ; i++){
			for(unsigned j=0;j < mSize; j++){
				h_sum[j + nodeId ] += h_sum[j+mSize*i + nodeId];
				h_sumSquare[j + nodeId] += h_sumSquare[j+mSize*i+ nodeId];
				h_minorSize[j + nodeId] += h_minorSize[j+mSize*i + nodeId];
			}
		}
		//}

		//score variable
		float score;
		float bestScore = 0;
		float varMinor;
		float varMajor;
		float varTotal;

		//calculate score
		//All the variance is calculate unless a constant value, due to the fact that this same value will be used to calculate the score
		//the split cause by the split of output term, represent the entire data
		varTotal =  h_sumSquare[mInput + nodeId] - (h_sum[mInput + nodeId] * h_sum[mInput + nodeId]) / h_sizeVec[n];
		for(unsigned i = 0; i< mInput;++i){
			if(h_minorSize[i + nodeId]!= 0 && h_minorSize[i + nodeId]!= h_sizeVec[n]){
				varMinor = h_sumSquare[i + nodeId]-(h_sum[i + nodeId] * h_sum[i + nodeId]) / (float) h_minorSize[i +nodeId];
				varMajor = (h_sumSquare[mInput + nodeId] - h_sumSquare[i +nodeId]) - ((h_sum[mInput + nodeId] - h_sum[i + nodeId])*(h_sum[mInput + nodeId] - h_sum[i + nodeId])) / (float)(h_sizeVec[n]-h_minorSize[i + nodeId]);
				score = 1 - (varMinor + varMajor) / varTotal;
			}else{
				score = 0;
			}

			//choose the best score and set the best axis and split
			if(score > bestScore) {
				bestScore = score;
				axis[n] = i;
				split[n] = h_rSplit[i + n * mSize];
				minorSize[n] = h_minorSize[i +  nodeId];
				//		std::cout << "axis: "<< axis << "  score " <<  bestScore << std::endl;
			}
		}

		//Leaf condition for the score
		if (bestScore <= mScoreThreshold) {
			isLeaf[n] = true;
			right[n] = nodeCuVector[n];
			//std::cout << " score basso" << std::endl;
		}else{

			//Create a stencil with respect to the axis and split choosed
			bool* stencil = NULL;
			sizeAlloc = sizeof(bool)*h_sizeVec[n];
			CudaSafeCall(cudaMalloc((void**)&stencil,sizeAlloc));
			createStencil<<<BLOCK, THREAD>>>(nodeCuVector[n]->ds->data, nodeCuVector[n]->map, stencil, axis[n], split[n] , h_sizeVec[n]);
			CudaCheckError();


			//Partitionate the current map using the stencil
			//This is done using the function DevicePartition::Flagged of the library CUB
			unsigned* newMap = NULL;
			CudaSafeCall(cudaMalloc((void**)&newMap, sizeof( unsigned)*h_sizeVec[n]));
			unsigned* d_num_selected_out = NULL;
			CudaSafeCall(cudaMalloc((void**)&d_num_selected_out, sizeof( unsigned)));

			void	*d_temp_storage = NULL;
			size_t	temp_storage_bytes = 0;
			cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, nodeCuVector[n]->map, stencil, newMap, d_num_selected_out, h_sizeVec[n]);
			CudaSafeCall(cudaMalloc(&d_temp_storage, temp_storage_bytes));
			cub::DevicePartition::Flagged(d_temp_storage, temp_storage_bytes, nodeCuVector[n]->map, stencil, newMap, d_num_selected_out, h_sizeVec[n]);
			CudaSafeCall(cudaFree(d_temp_storage));

			/*
	       unsigned *h_l = ( unsigned*) malloc(sizeof( unsigned));
	       cudaMemcpy(h_l,d_num_selected_out,sizeof( unsigned),cudaMemcpyDeviceToHost);
	       if((*h_l)!=minorSize){

	           	std::cout << "Error in map partition  " << (*h_l)-minorSize << std::endl;

	       }
	       free(h_l);
			 */


			//Copy the partitioned map in two diferent maps, one for each child
			unsigned* mapLeft = NULL;
			unsigned* mapRight = NULL;
			CudaSafeCall(cudaMalloc((void**)&mapLeft, sizeof( unsigned)*minorSize[n]));
			CudaSafeCall(cudaMalloc((void**)&mapRight,sizeof( unsigned)*(h_sizeVec[n]-minorSize[n])));
			copyMap<<<BLOCK, THREAD>>>(newMap, mapLeft,mapRight, h_sizeVec[n], minorSize[n]);
			CudaCheckError();

			//free resource
			CudaSafeCall(cudaFree(newMap));
			CudaSafeCall(cudaFree(stencil));
			CudaSafeCall(cudaFree(d_num_selected_out));

			//Create the child
			isLeaf[n] = false;
			left[n] = new NodeCu( nodeCuVector[n]->ds, mapLeft, minorSize[n]);
			right[n] = new NodeCu(nodeCuVector[n]->ds, mapRight, h_sizeVec[n]-minorSize[n]);


			//std::cout << "h_size = " << h_sizeVec[n] << std::endl;
		}
	}

	//free resource
	CudaSafeCall(cudaFree(d_rSplit));
	CudaSafeCall(cudaFree(d_sum));
	CudaSafeCall(cudaFree(d_sumSquare));
	CudaSafeCall(cudaFree(d_minorSize));
	CudaSafeCall(cudaFree(d_sizeVec));
	CudaSafeCall(cudaFree(d_mapVec));

	free(h_mapVec);
	free(h_sizeVec);
	free(h_rSplit);
	free(h_sum);
	free(h_sumSquare);
	free(h_minorSize);
}

}

