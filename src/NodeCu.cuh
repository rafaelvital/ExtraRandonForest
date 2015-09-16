#ifndef NODECU_H
#define NODECU_H

#include "DataCu.cu"
#include <vector>

namespace PoliFitted
{
/*
 * This class specify a dataset on the device memory
 * The most important method is the processNode where we decide
 * if this node is a internal or leaf node of the tree.
 *  If is a internal node we find the best split *
 */

class NodeCu
{
    public:
		//Construct the node from the global dataset already loaded in device memory
        NodeCu(const DataCu* ds_c); //

        /*Construct the node from the a reduced dataset
         @param ds_c, Global dataset loaded on the device
         @param map_c, Indicate the index of the dataset where the samples are.
         	 	 	   Is important highlight that this map include the stride (number input + umber output)
         	 	 	   So, the map do not show the index i, but the index stride * i
         @param h_size_c, size of the current dataset     */
        NodeCu(const DataCu* ds_c,  unsigned * map_c, const unsigned h_size_c); //

        //free the map only
        virtual ~NodeCu ();

        /*Compute the best split for the node, creating two new children
        	@param isLeaf, Response if this node is a leaf or not
        	@param axis, If this node are a internal node, it response what is the best input  to split
   	   	 	@param split, If this node are a internal node, it response what is the best split value
       	    @param right, If this node are a internal node, it response the right child
        	@param left,  If this node are a internal node, it response the left child     */
        void processNode(bool& isLeaf,  unsigned& axis, float& split, NodeCu*& right, NodeCu*& left);



        //Return the current dataset on the host memory
        Dataset* getDataset();

        //Return the Global dataset on the host memory
        Dataset* getAllDataset(){return ds->ds;}

        //Return what are the index i of the samples that correspond to the reduced Dataset
        unsigned* getMap();

        unsigned getSize(){return h_size;}

        //parameter to build the tree
        //Another possibility is put this parameter how template parameter
        static  unsigned mNMin;
        static  unsigned mNumSplits;
        static float mScoreThreshold;

        friend void processVectorNode(std::vector<NodeCu*>& nodeCuVector, std::vector<bool>& isLeaf,  std::vector<unsigned>& axis, std::vector<float>& split, std::vector<NodeCu*>& right, std::vector<NodeCu*>& left);


    private:
        const DataCu* ds; //pointer to the global dataset (on the device)
        const unsigned h_size; //Size of the current dataset
        unsigned* map; //map of the samples of the current dataset (on the device)
        			   //remember, this map store the index stride * i

};

}
#endif // NODECU_H
