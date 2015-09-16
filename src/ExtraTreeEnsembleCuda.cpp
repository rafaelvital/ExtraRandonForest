/*
 * ExtraTreeEnsembleCuda.cpp
 *
 *  Created on: Jul 20, 2015
 *      Author: vital
 */

#include "ExtraTreeEnsembleCuda.h"
#include "OrderedLink.h"
#include <atomic>
#include "NodeCu.cuh"
#include "LinkCu.h"

using namespace PoliFitted;

ExtraTreeEnsembleCuda::ExtraTreeEnsembleCuda(unsigned int input_size,
                                     	 unsigned int output_size,
                                     	 int m,
                                     	 int k,
                                     	 int nmin,
                                     	 float score_th,
                                     LeafType leaf) :
    Regressor("ExtraTreeCuda", input_size, output_size)
{
    mNumTrees = m;
    mNumSplits = k;
    mNMin = nmin;
    mScoreThreshold = score_th;
    mLeafType = leaf;
    mSum = 0;
    for (unsigned int i = 0; i < mNumTrees; i++) {
        mEnsemble.push_back(new ExtraTreeCuda(mInputSize, mOutputSize, mNumSplits, mNMin, mScoreThreshold, mLeafType));
    }
    mParameters << "M" << m << "K" << k << "Nmin" << nmin << "Sth" << mScoreThreshold;
}

ExtraTreeEnsembleCuda::~ExtraTreeEnsembleCuda()
{
    for (unsigned int i = 0; i < mEnsemble.size(); i++) {
        delete mEnsemble.at(i);
    }
}

void ExtraTreeEnsembleCuda::Initialize()
{
    for (unsigned int i = 0; i < mEnsemble.size(); i++) {
        delete mEnsemble.at(i);
    }
    mEnsemble.clear();
    for (unsigned int i = 0; i < mNumTrees; i++) {
        mEnsemble.push_back(new ExtraTreeCuda(mInputSize, mOutputSize, mNumSplits, mNMin, mScoreThreshold, mLeafType));
    }
}

void ExtraTreeEnsembleCuda::Evaluate(Tuple* input, Tuple& output)
{
	if (mEnsemble.size() != mNumTrees) {
		cout << mEnsemble.size() << " not equal to " << mNumTrees << endl;
		return;
	}
	mSum = 0.0;
	float out;
	for (unsigned i = 0; i <  mEnsemble.size(); ++i) {
		mEnsemble[i]->Evaluate(input, out);
		mSum += out;
	}
	output[0] = mSum / (float)mNumTrees;
}

void ExtraTreeEnsembleCuda::Evaluate(Tuple* input, float& output)
{
    if (mEnsemble.size() != mNumTrees) {
        cout << mEnsemble.size() << " not equal to " << mNumTrees << endl;
        return;
    }
    mSum = 0.0;
    float out;
    for (unsigned i = 0; i <  mEnsemble.size(); ++i) {
    	mEnsemble[i]->Evaluate(input, out);
    	mSum += out;
    }
    output = mSum / (float)mNumTrees;
}

void ExtraTreeEnsembleCuda::Train(Dataset* data, bool overwrite, bool normalize)
{
	DataCu* dataCu = new DataCu(data);
	Train(dataCu, overwrite, normalize);
	delete dataCu;
}

void ExtraTreeEnsembleCuda::Train(DataCu* dataCu, bool overwrite, bool normalize){
	if (!overwrite)
	    {
	        cerr << "Not implemented!" << endl;
	    }

		//Pass the parameter mNIN and mScoreThreshold to the NodeCu
	    NodeCu::mNMin = mNMin;
	    NodeCu::mNumSplits = mNumSplits;
	    NodeCu::mScoreThreshold = mScoreThreshold;

		//create the atomic variable to control how much thread are open
	    std::atomic<unsigned> threadOpen;
	    threadOpen = 0;
		
		/*create and add to the list the first LinkCu node for each tree in the forest
		The first linkCu is the link between the entire train dataset(already loaded in GPU)
		and the root of the each tree
		However, the Root of the tree can't be expressed explicit, it need be a child of another node,
		so was created the fake node in which the right child is the Root
		*/
	    std::vector<NodeCu*> rootCu(mNumTrees);
	    std::vector<rtINode*> fake(mNumTrees);
	    std::vector<LinkCu*>  linkRoot(mNumTrees);
	    OrderedLink lista;
	  
	 
	    for(unsigned i = 0; i < mNumTrees;i++){
	    	rootCu[i] = new NodeCu(dataCu);
	    	fake[i] = new rtINode(0, 0.f, nullptr, nullptr);
	    	linkRoot[i] = new LinkCu(rootCu[i], fake[i], true);
	    	lista.insert(linkRoot[i]);
	    }

		//Process the list
	    trainList(lista, threadOpen);

		//retrieve the real root of the tree from the fake node and delete it
	    for(unsigned i = 0; i < mNumTrees;i++){
	    	mEnsemble[i]->setRoot(fake[i]->getRight());
	    	fake[i]->setRight(nullptr);
	    	delete fake[i];
	    }

	    //rootCu is deleted on the process
	    //LinkCu is deleted on the process
}


Regressor* ExtraTreeEnsembleCuda::GetNewRegressor()
{
    return new ExtraTreeEnsembleCuda(mInputSize, mOutputSize, mNumTrees, mNumSplits, mNMin, mScoreThreshold, mLeafType);
}

void ExtraTreeEnsembleCuda::WriteOnStream(ofstream& out)
{
  out << mNumTrees << " " << mNumSplits << " " << mNMin << " " << mInputSize
      << " " << mOutputSize << " " << mScoreThreshold << " " << mLeafType << endl;
  for (unsigned int i = 0; i < mNumTrees; i++)
  {
    out << *mEnsemble[i] << endl;
  }
}

void ExtraTreeEnsembleCuda::ReadFromStream(ifstream& in)
{
  string type;
  int leaf_type;
  in >> mNumTrees >> mNumSplits >> mNMin >> mInputSize >> mOutputSize >> mScoreThreshold >> leaf_type;
//  in >> mNumTrees >> mNumSplits >> mNMin >> mInputSize >> mOutputSize >> leaf_type;
  mLeafType = (LeafType)leaf_type;
  //   in >> type;
  //   ExtraTree* tree = new ExtraTree(mInputSize,mOutputSize,mNumSplits,mNMin);
  for (unsigned int i = 0; i < mEnsemble.size(); i++)
  {
    delete mEnsemble.at(i);
  }
  mEnsemble.clear();
  for (unsigned int i = 0; i < mNumTrees; i++)
  {
    mEnsemble.push_back(new ExtraTreeCuda(mInputSize, mOutputSize, mNumSplits, mNMin));
    in >> *mEnsemble[i];
    //     mEnsemble.push_back(tree);
  }
}
