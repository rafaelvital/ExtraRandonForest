#ifndef LINKCU_H
#define LINKCU_H

#include "rtnode.h"
#include "NodeCu.cuh"
#include "Dataset.h"
#include "ExtraTree.h"
#include <thread>
#include <atomic>

namespace PoliFitted
{
	
/**  This class do the Link between the tree node (rtAnode) and the samples of the Dataset that 
this node represents.
It is required because rtAnode Classes don't track the correspondent Dataset.
The tree node is represented by a pointer to the parent node
and a variable that says if this node is the right or the left child.

**/

class LinkCu
{
    public:
		/**
			Basic Contructor
		**/
        LinkCu(NodeCu* nodeC, rtINode* parentN, bool rightS);
		
		/**
			Destroyer only the NodeCu
		 **/
        virtual ~LinkCu (); 

        /**
         * Split the the current LinkCu in two new node, constructing the Tree
         * Who decide how split the data, is NodeCU, that uses the GPU for it
         * @param  linkR: new LinkCu right child, need be NULL
         * @param  linkL: new LinkCu left child, need be NULL
         * @param  threadOpen, parameter to control the thread job
         */
        void processLink(LinkCu*& linkR, LinkCu*& linkL, std::atomic<unsigned>& threadOpen);

        /**
         * Create a leaf and fit the data in general is a task low cost task, so it is done with the CPU
         * This functions create a leaf and fit its data.
         * To use this function this object must be a leaf, the function don't verify it
         * @param  threadOpen, parameter to control the thread job
        */
        void buildLeaf(std::atomic<unsigned>& threadOpen);


        /**
         * Split the the a vector of LinkCu* in two new node, constructing the Tree
         * Who decide how split the data, is NodeCU, that uses the GPU for it
         * @param  toProcess: a vector of LinkCu* node
         * @param  linkR: a vector of LinkCu* right child, need be a vector of NULL
         * @param  linkL: a vector of LinkCu* left child, need be a vector of NULL
         * @param  threadOpen, parameter to control the thread job
        */
        friend void processVectorLink(std::vector<LinkCu*>& toProcess, std::vector<LinkCu*>& linkR, std::vector<LinkCu*>& linkL, std::atomic<unsigned>& threadOpen);

        unsigned getSize() const{
        	return nodeCu->getSize();
        }

    private:
        NodeCu* nodeCu;
        rtINode* parentNode;
        bool rightSide;

        /**
         * When the number of samples is low, is faster do the computation with the CPU rather than the GPU
         * This functions create a tree starting from the reduced Dataset and add small tree to the tree of the caller
         * @param  map: A vector that indicates the samples to create the reduced Dataset
         * @param  threadOpen, parameter to control the thread job
         */		 
        void cpuTrain(unsigned* map, std::atomic<unsigned>& threadOpen);

        /**
         * Create a leaf and fit the data in general is a task low cost task, so it is done with the CPU
         * This functions create a leaf and fit its data.
         * @param  map: A vector that indicates the samples to fit the data
         * @param  threadOpen, parameter to control the thread job
         */	
        void cpuLeafAndFit(unsigned* map, std::atomic<unsigned>& threadOpen);



};


}
#endif // LINKCU_H
