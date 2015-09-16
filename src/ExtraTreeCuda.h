/*
 * ExtreeCuda.h
 *
 *  Created on: Jul 16, 2015
 *      Author: vital
 */

#ifndef EXTREECUDA_H_
#define EXTREECUDA_H_

#include "ExtraTree.h"
#include "DataCu.cu"
#include "OrderedLink.h"
#include <atomic>
#include "LinkCu.h"

namespace PoliFitted {

/**************************************************************************
*   This class implements a regression tree                               *
*  It is implemented to use a CUDA code	to train the samples     		  *
***************************************************************************/

class ExtraTreeCuda: public PoliFitted::ExtraTree {
public:
		 /**
          * Basic constructor
          * @param input_size , output_size of the train dataset
          * @param k number of selectable attributes to be randomly picked 
		     (The cuda implementation use always all the input, so k = input_size)
          * @param nmin minimum number of tuples in a leaf
		  * @param score_th minimum score to consider a split valid
          */
	ExtraTreeCuda(unsigned int input_size = 1,
            unsigned int output_size = 1,
            int k = 5,
            int nmin = 2,
            float score_th = 0.0,
            LeafType leaf = CONSTANT) : ExtraTree(input_size, output_size, k, nmin, score_th, leaf){};
			
		/**
         * Empty destructor
         */

	virtual ~ExtraTreeCuda();
	
	 /**
         * Builds a approximation model for the training set
         * @param  data The training set
         * @param   overwrite and normalize aren't implemented, there are only for compatibility reason 
         */
	virtual void Train(Dataset* data, bool overwrite = true, bool normalize = true);

	 /**
         * Builds a approximation model for the training set
         * @param  data The training set
         * @param   overwrite and normalize aren't implemented, there are only for compatibility reason 
         */
	void Train(DataCu* dataCu, bool overwrite = true, bool normalize = true);


};

/**
Free function that says the order of each note LinkCu is computed by the GPU
The atomic variable threadOpen exist because in some situations the code create treads in CPU
to help the GPU to train the data, so the functions need control if all threads are done
**/
void trainList(OrderedLink& lista, std::atomic<unsigned>& threadOpen);

} /* namespace PoliFitted */
#endif /* EXTREECUDA_H_ */
