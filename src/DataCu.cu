/*
 * DataCu.h
 *
 *  Created on: Jul 12, 2015
 *      Author: vital
 */

#ifndef DATACU_H_
#define DATACU_H_

#include "Dataset.h"
#include "errorCheck.cu"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>


namespace PoliFitted {

/*
 * This class is responsible to copy the Dataset from Host to Device
 * On the device the dataset is ordered by samples, i.e, we put in order all the input and the output of the first sample
 * after the input and output of the second samples, and so on.
 */

class DataCu{
public:

	//The constructor receives a pointer a Dataset*, extract a float* from it and copy it to the device
	DataCu(Dataset * ds_c):ds(ds_c), data(NULL), mInputSize(ds_c->GetInputSize()), mOutputSize(ds_c->GetOutputSize()) , totalSize(ds_c->size()), mSize(ds_c->GetInputSize()+ds_c->GetOutputSize()) {

		float * h_data = ds->GetMatrix();

		//copy the dataset to the device
		const unsigned sizeAlloc = sizeof(float)*totalSize*mSize;
		CudaSafeCall(cudaMalloc((void**)&data, sizeAlloc));
		CudaSafeCall(cudaMemcpy(data,h_data,sizeAlloc,cudaMemcpyHostToDevice));

		//More L1 cache is faster, the program don't use sharedMemory
		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

		free(h_data);
	};
	
	// The destructor Free the memory on the Device
	virtual ~DataCu(){
		CudaSafeCall(cudaFree(data));
	};	
	
	//the pointer to the device memory
	float* data;
	Dataset* ds;

	const unsigned mInputSize;
	const unsigned mOutputSize;
	const unsigned totalSize;
	const unsigned mSize;

};

} /* namespace PoliFitted */
#endif /* DATACU_H_ */
