################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/Dataset.cpp \
../src/ExtraTree.cpp \
../src/ExtraTreeCuda.cpp \
../src/ExtraTreeEnsemble.cpp \
../src/ExtraTreeEnsembleCuda.cpp \
../src/LinkCu.cpp \
../src/Regressor.cpp \
../src/Sample.cpp \
../src/Tree.cpp \
../src/Tuple.cpp \
../src/main.cpp \
../src/rtLeaf.cpp \
../src/rtLeafLinearInterp.cpp 

CU_SRCS += \
../src/DataCu.cu \
../src/NodeCu.cu \
../src/errorCheck.cu 

CU_DEPS += \
./src/DataCu.d \
./src/NodeCu.d \
./src/errorCheck.d 

OBJS += \
./src/DataCu.o \
./src/Dataset.o \
./src/ExtraTree.o \
./src/ExtraTreeCuda.o \
./src/ExtraTreeEnsemble.o \
./src/ExtraTreeEnsembleCuda.o \
./src/LinkCu.o \
./src/NodeCu.o \
./src/Regressor.o \
./src/Sample.o \
./src/Tree.o \
./src/Tuple.o \
./src/errorCheck.o \
./src/main.o \
./src/rtLeaf.o \
./src/rtLeafLinearInterp.o 

CPP_DEPS += \
./src/Dataset.d \
./src/ExtraTree.d \
./src/ExtraTreeCuda.d \
./src/ExtraTreeEnsemble.d \
./src/ExtraTreeEnsembleCuda.d \
./src/LinkCu.d \
./src/Regressor.d \
./src/Sample.d \
./src/Tree.d \
./src/Tuple.d \
./src/main.d \
./src/rtLeaf.d \
./src/rtLeafLinearInterp.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O3 -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O3 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_20,code=compute_20 -gencode arch=compute_20,code=sm_21  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.0/bin/nvcc -G -g -O3 -std=c++11 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_20,code=sm_21  -odir "src" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.0/bin/nvcc -G -g -O3 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


