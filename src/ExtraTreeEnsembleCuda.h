/*
 * ExtraTreeEnsembleCuda.h
 *
 *  Created on: Jul 20, 2015
 *      Author: vital
 */

#ifndef EXTRATREEENSEMBLECUDA_H_
#define EXTRATREEENSEMBLECUDA_H_

#include <iostream>
#include <vector>
#include "Regressor.h"
#include "ExtraTreeCuda.h"

namespace PoliFitted {

class ExtraTreeEnsembleCuda: public PoliFitted::Regressor {
public:

       /**
         * The basic constructor
         * @param input_size , output_size of the train dataset
         * @param m number of trees in the ensemble
         * @param k number of selectable attributes to be randomly picked 
		     (The cuda implementation use always all the input, so k = input_size)
          * @param nmin minimum number of tuples in a leaf
		  * @param score_th minimum score to consider a split valid
          */
		ExtraTreeEnsembleCuda(unsigned int input_size = 1,
                         unsigned int output_size = 1,
                         int m = 50,
                         int k = 5,
                         int nmin = 2,
                         float score_th = 0.0,
                         LeafType leaf = CONSTANT);

       /**
         * Empty destructor
         */
       virtual ~ExtraTreeEnsembleCuda();

       /**
        * Initialize the ExtraTreeEnsemble by clearing the internal structures
        */
       virtual void Initialize();

       /**
        * Set nmin
        * @param nmin the minimum number of inputs for splitting
        */
       void SetNMin(int nm);

       /**
        * Builds an approximation model for the training set with a GPU device
        * @param  data The training set
        * @param   overwrite and normalize aren't implemented, there are only for compatibility reason 
         */
       virtual void Train(Dataset* data, bool overwrite = true, bool normalize = true);

	   /**
        * Builds an approximation model for the training set with a GPU device
        * @param  data The training set
        * @param   overwrite and normalize aren't implemented, there are only for compatibility reason 
         */
       void Train(DataCu* dataCu, bool overwrite = true, bool normalize = true);

       /**
        * @return Tuple
        * @param  input The input data on which the model is evaluated
        */
       virtual void Evaluate (Tuple* input, Tuple& output);

       /**
        * @return Value
        * @param  input The input data on which the model is evaluated
        */
       virtual void Evaluate (Tuple* input, float& output);

       /**
        *
        */
       virtual Regressor* GetNewRegressor ();

       /**
        *
        */
       virtual void WriteOnStream (ofstream& out);

       /**
        *
        */
       virtual void ReadFromStream (ifstream& in);



   private:

       unsigned int mNumTrees; //number of trees in the ensemble
       unsigned int mNumSplits; //number of selectable attributes to be randomly picked
       unsigned int mNMin; //minimum number of tuples for splitting
       float mScoreThreshold;
       vector<ExtraTreeCuda*> mEnsemble; 
       LeafType mLeafType;
       float mSum;

};

} /* namespace PoliFitted */
#endif /* EXTRATREEENSEMBLECUDA_H_ */
