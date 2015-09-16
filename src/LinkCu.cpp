#include "LinkCu.h"
#include "defineVariable.h"

#define CPU_PROCESS CPU_PROCESS_V

using namespace PoliFitted;

LinkCu::LinkCu(NodeCu* nodeC, rtINode* parentN, bool rightS)
              : nodeCu(nodeC), parentNode(parentN), rightSide(rightS) {}

void LinkCu::processLink(LinkCu*& linkR, LinkCu*& linkL, std::atomic<unsigned>& threadOpen){
    bool isLeaf;
    unsigned axis;
    float split;
    NodeCu* nodeCuR = nullptr;
    NodeCu* nodeCuL = nullptr;

	//specify the nodeCu size to process or with CPU or GPU
    if(nodeCu->getSize() > CPU_PROCESS){
    	
		//GPU process the node
		nodeCu->processNode(isLeaf, axis, split, nodeCuR, nodeCuL);
		
		if(isLeaf){
			//if the node is a leaf, the CPU will finish the job calling the function cpuLeafAndFit with a thread in order to continue to build the tree with GPU simultaneously
			unsigned* map = nodeCu->getMap();
			threadOpen++; //indicate that a thread was launched
			std::thread t(&LinkCu::cpuLeafAndFit, this, map, std::ref(threadOpen));
			t.detach();
		}else{
			//if the node is a internal node, we add the node to the tree, and its children go to the list 
			rtINode* intern = new rtINode(axis, split, nullptr, nullptr);
			if(rightSide){
				//delete parent->getRight();
				parentNode->setRight(intern);
			}else{
				//delete parent->getRight();
				parentNode->setLeft(intern);
			}
			linkR = new LinkCu(nodeCuR, intern, true);
			linkL = new LinkCu(nodeCuL, intern, false);
			delete this;
		}
    }else{
		
    	//CPU process, the CPU will finish the job calling the function cpuTrain with a thread in order to continue to build the tree  with GPU simultaneously
    	unsigned* map = nodeCu->getMap();
    	threadOpen++;//indicate that a thread was launched
    	std::thread t(&LinkCu::cpuTrain, this, map, std::ref(threadOpen));
    	t.detach();
    }
}

void LinkCu::buildLeaf(std::atomic<unsigned>& threadOpen){
	unsigned* map = nodeCu->getMap();
	threadOpen++; //indicate that a thread was launched
	std::thread t(&LinkCu::cpuLeafAndFit, this, map, std::ref(threadOpen));
	t.detach();
}

void LinkCu::cpuTrain(unsigned* map, std::atomic<unsigned>& threadOpen){
	
	//create a reduced dataset from the map received
	Dataset* dsPartial = nodeCu->getAllDataset()->GetReducedDataset(nodeCu->getSize(),map);
	
	//create and train a tree starting from the reduced Dataset
	ExtraTree* treePartial = new ExtraTree(dsPartial->GetInputSize(), dsPartial->GetOutputSize(), nodeCu->mNumSplits, nodeCu->mNMin, nodeCu->mScoreThreshold);
	treePartial->Train(dsPartial);
	
	//Join the small tree to the bigger one 	
	if(rightSide){
		//delete parent->getRight();
		parentNode->setRight(treePartial->GetRoot());
	}else{
		//delete parent->getRight();
		parentNode->setLeft(treePartial->GetRoot());
	}	
	delete dsPartial;
	delete map;
	threadOpen--; //the thread finished its job
	delete this;
}

void LinkCu::cpuLeafAndFit(unsigned* map, std::atomic<unsigned>& threadOpen){
	
	//Create a leaf and fit it with the samples indicated in map
	rtLeaf* leaf = new rtLeaf();
	Dataset* dsPartial = nodeCu->getAllDataset()->GetReducedDataset(nodeCu->getSize(),map);
	leaf->Fit(dsPartial);
	
	//Add the leaf to the tree
	if(rightSide){
		//delete parent->getRight();
		parentNode->setRight(leaf);
	}else{
		//delete parent->getRight();
		parentNode->setLeft(leaf);
	}	
	delete dsPartial;
	delete map;
	threadOpen--; //the thread finished its job
	delete this;
}

LinkCu::~LinkCu(){
    delete nodeCu;
}

namespace PoliFitted
{
void processVectorLink(std::vector<LinkCu*>& toProcess, std::vector<LinkCu*>& linkR, std::vector<LinkCu*>& linkL, std::atomic<unsigned>& threadOpen){

	unsigned nNode = toProcess.size();
	if(nNode>1){
		std::vector<bool> isLeaf(nNode);
		std::vector<unsigned> axis(nNode);
		std::vector<float> split(nNode);
		std::vector<NodeCu*> nodeCuR(nNode,nullptr);
		std::vector<NodeCu*> nodeCuL(nNode,nullptr);

		//GPU process the node
		std::vector<NodeCu*> nodeCuVector(toProcess.size());
		for(unsigned i = 0;i < nNode; i++){
			nodeCuVector[i] = toProcess[i]->nodeCu;
		}

		processVectorNode(nodeCuVector, isLeaf, axis, split, nodeCuR, nodeCuL);
		for(unsigned i = 0;i < toProcess.size(); i++){
			if(isLeaf[i]){
				//if the node is a leaf, the CPU will finish the job calling the function cpuLeafAndFit with a thread in order to continue to build the tree with GPU simultaneously
				unsigned* map = toProcess[i]->nodeCu->getMap();
				threadOpen++; //indicate that a thread was launched
				std::thread t(&LinkCu::cpuLeafAndFit, toProcess[i], map, std::ref(threadOpen));
				t.detach();
			}else{
				//if the node is a internal node, we add the node to the tree, and its children go to the list
				rtINode* intern = new rtINode(axis[i], split[i], nullptr, nullptr);
				if(toProcess[i]->rightSide){
					//delete parent->getRight();
					toProcess[i]->parentNode->setRight(intern);
				}else{
					//delete parent->getRight();
					toProcess[i]->parentNode->setLeft(intern);
				}
				linkR[i] = new LinkCu(nodeCuR[i], intern, true);
				linkL[i] = new LinkCu(nodeCuL[i], intern, false);
				delete toProcess[i];
			}
		}
	}else{
		toProcess[0]->processLink(linkR[0],linkL[0],threadOpen);
	}
}
}




