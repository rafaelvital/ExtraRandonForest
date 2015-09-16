/*
 * main3.cpp
 *
 *  Created on: Jul 12, 2015
 *      Author: vital
 */

/*
 * main.cpp
 *
 *  Created on: Jul 8, 2015
 *      Author: vital
 */

#include "Dataset.h"
#include "NodeCu.cuh"
#include "ExtraTreeCuda.h"
#include "ExtraTreeEnsembleCuda.h"
#include "ExtraTreeEnsemble.h"
#include <iostream>
#include <map>
#include <fstream>
#include <cstdio>
#include <ctime>


using namespace PoliFitted;

int main(){


	//read data set
	Dataset* ds = new Dataset();
	ifstream fileIn;
	//fileIn.open("/home/vital/dataset/image-seg/Dataset2.data");
	fileIn.open("/home/vital/dataset/ufficiale/dataset.dat");
	fileIn >> *ds;
	fileIn.close();

	std::map<string, float> metrics;
	double durationCu, durationCP;
	std::clock_t start;
	ofstream fileOut;

	/************************************************************************************************/
	//Extra tree

/*
	start = std::clock();
	ExtraTreeCuda* treeCu = new ExtraTreeCuda(ds->GetInputSize(), ds->GetOutputSize(), ds->GetInputSize(), 1, 0.0f);
	treeCu->Train(ds);
	durationCu = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout << "tempo Cuda: " << durationCu <<std::endl;

	//fileOut.open("/home/vital/dataset/ufficiale/resTre.dat");
	fileOut.open("/home/vital/dataset/image-seg/resGPU.data");
	treeCu->WriteOnStream(fileOut);
	fileOut.close();


	//metrics = treeCu->CrossValidate(ds,5,true);
	metrics = treeCu->ComputePerfMetrics(ds);
	std::cout << "R2 " << metrics.at("R2") <<std::endl;
	std::cout << "RAAE " << metrics.at("RAAE") <<std::endl;
	std::cout << "RMAE " << metrics.at("RMAE") <<std::endl;
	std::cout << "RMSE " << metrics.at("RMSE") <<std::endl;
	std::cout << "PEP " << metrics.at("PEP") <<std::endl;
*/
	/************************************************************************************************/
	//Esemble Extra tree

	start = std::clock();
	ExtraTreeEnsembleCuda* treeEnsembleCuda = new ExtraTreeEnsembleCuda(ds->GetInputSize(), ds->GetOutputSize(), 1, ds->GetInputSize(), 2, 0.0f);
	treeEnsembleCuda->Train(ds);
	durationCu = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout << "tempo Cuda: " << durationCu <<std::endl;

	fileOut.open("/home/vital/dataset/ufficiale/resEseTre.dat");
	treeEnsembleCuda->WriteOnStream(fileOut);
	fileOut.close();

//	metrics = treeEnsembleCuda->CrossValidate(ds,5);
	metrics = treeEnsembleCuda->ComputePerfMetrics(ds);

	std::cout << "R2 " << metrics.at("R2") <<std::endl;
	std::cout << "RAAE " << metrics.at("RAAE") <<std::endl;
	std::cout << "RMAE " << metrics.at("RMAE") <<std::endl;
	std::cout << "RMSE " << metrics.at("RMSE") <<std::endl;
	std::cout << "PEP " << metrics.at("PEP") <<std::endl;


	/************************************************************************************************/
	//CPU
/*
	start = std::clock();
	ExtraTreeEnsemble* treeCP = new ExtraTreeEnsemble(ds->GetInputSize(), ds->GetOutputSize(),1, ds->GetInputSize(), 1, 0.0f);
	//ExtraTree* treeCP = new ExtraTree(ds->GetInputSize(), ds->GetOutputSize(), ds->GetInputSize(), 1, 0.0f);
	treeCP->Train(ds);
	durationCP = ( std::clock() - start ) / (double) CLOCKS_PER_SEC;
	std::cout << "tempo CPU: " << durationCP <<std::endl;


	fileOut.open("/home/vital/dataset/image-seg/resCPU.data");
	treeCP->WriteOnStream(fileOut);
	fileOut.close();

	//metrics = treeCP->ComputePerfMetrics(ds);
	metrics = treeCP->CrossValidate(ds,5);

	std::cout << "R2 " << metrics.at("R2") <<std::endl;
	std::cout << "RAAE " << metrics.at("RAAE") <<std::endl;
	std::cout << "RMAE " << metrics.at("RMAE") <<std::endl;
	std::cout << "RMSE " << metrics.at("RMSE") <<std::endl;
	std::cout << "PEP " << metrics.at("PEP") <<std::endl;

*/
	delete ds;


	return 0;
}
