/*
 * ExtreeCuda.cpp
 *
 *  Created on: Jul 16, 2015
 *      Author: vital
 */

#include "ExtraTreeCuda.h"
#include "NodeCu.cuh"
#include "defineVariable.h"

#define TUNING_LIST TUNING_LIST_V

using namespace PoliFitted;

ExtraTreeCuda::~ExtraTreeCuda() {
}

void ExtraTreeCuda::Train(Dataset* data, bool overwrite, bool normalize)
{
	DataCu* dataCu = new DataCu(data);
	Train(dataCu, overwrite, normalize);
	delete dataCu;
}

void ExtraTreeCuda::Train(DataCu* dataCu, bool overwrite, bool normalize)
{
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

	/*create the first LinkCu node
	 the first linkCu is the link between the entire train dataset(already loaded in GPU)
	 and the root of the Tree
	However, the Root of the tree can't be expressed explicit, it need be a child of another node,
	so was created the fake node in which the right child is the Root
	*/
    NodeCu* rootCu = new NodeCu(dataCu);
    rtINode* fake = new rtINode(0, 0.f, nullptr, nullptr);
    LinkCu* linkRoot = new LinkCu(rootCu, fake, true);
	
	//Add the linkRott to the list
    OrderedLink lista;
    lista.insert(linkRoot);

	//Process the list
    trainList(lista, threadOpen);

	//retrieve the real root of the tree from the fake node and delete it
    root = fake->getRight();
    fake->setRight(NULL);
    delete fake;
	
	//rootCu is deleted on the process
	//LinkCu is deleted on the process
}

void PoliFitted::trainList(OrderedLink& lista, std::atomic<unsigned>& threadOpen){
	
	//Process the first LinkCu and add the child in the list
	std::vector<LinkCu*> toProcess;
	std::vector<LinkCu*> linkR;
	std::vector<LinkCu*> linkL;

	unsigned nNode;
	LinkCu* A = nullptr;
	LinkCu* B = nullptr;
	while(!lista.empty()){

		//verify how much node will be processed
		for(unsigned i = 1; i <= BLOCK_V / BLOCK_SAMPLE_V ; i++){
			if( (BLOCK_V / BLOCK_SAMPLE_V) % i == 0){
				if(lista.size()>=i){
					nNode = i;
					//Empirical expression with good results
					if((*lista.begin())->getSize() > TUNING_LIST * (BLOCK_V / i) * (THREAD_V)) break;
				}else{
					break;
				}
			}
		}

		//create vector with the nodes that will be processed now and their children
		//One time that the node go to the vector toProcess, the node is deleted of the orderedList
		toProcess.resize(nNode);
		for(unsigned i = 0; i < nNode; i++){
			toProcess[i] = *lista.begin();
			lista.erase(lista.begin());
		}
		linkR.resize(nNode, nullptr);
		linkL.resize(nNode, nullptr);
		processVectorLink(toProcess, linkR,linkL,threadOpen);

		//verify each child, if it's a leaf beacause of low number of samples, build the leaf
		//Otherwise, put it on the list
		for(unsigned i = 0; i < nNode; i++){
			if(linkR[i] != nullptr){
				if(linkR[i]->getSize() > NodeCu::mNMin){
					lista.insert(linkR[i]);
				}else{
					linkR[i]->buildLeaf(threadOpen);
				}
			}
			if(linkL[i]!=nullptr){
				if(linkL[i]->getSize() > NodeCu::mNMin){
					lista.insert(linkL[i]);
				}else{
					 linkL[i]->buildLeaf(threadOpen);
				}
			}
		}

		//clear the toProcess node vector and their children
		//Because these vector are vector of pointer, clear then don't destroy the pointer
		//The toProcess pointer and their children destroy themselves.
		toProcess.clear();
		linkR.clear();
		linkL.clear();
	}
	
	//Wait all thread finish their job
	while(threadOpen>0){
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}
