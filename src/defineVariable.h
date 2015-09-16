/*
 * defineVariable.h
 *
 *  Created on: Jul 21, 2015
 *      Author: vital
 */

//define all the define here

#define THREAD_V  512      //Thread per block
#define BLOCK_V  6        //Number of blocks
#define BLOCK_SAMPLE_V  1  //How much blocks is needed to the number of thread be greater than the number of input+output
						  //set to 1 if we have more thread than input

#define D_INPUT_V  442  //number of input
#define D_OUTPUT_V 1    //number of output
#define D_STRIDE_V 443  //number of input + output

#define TUNING_LIST_V 2 //Tuning parameter that decide when process more than one node of the list

#define CPU_PROCESS_V 0  //how much small need be the dataset to process it with the CPU



