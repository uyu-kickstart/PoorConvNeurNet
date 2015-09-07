//
//  ConvolutionNeuralNetwork.h
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/03.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#ifndef __PoorConvNeurNet__ConvolutionNeuralNetwork__
#define __PoorConvNeurNet__ConvolutionNeuralNetwork__

#include "Layer.h"
#include "ActivationFunction.h"
#include "ErrorFunction.h"

class ConvolutionalNeuralNetwork {
public:
    std::vector<MatrixD> Teachers;
    std::vector<MatrixD> Inputs;
    std::vector<Layer*> network;
    ErrorFunction* errorFunc;
    MatrixD ZL;
    ConvolutionalNeuralNetwork(std::vector<Layer*> network, std::vector<MatrixD> Teachers, std::vector<MatrixD> Inputs, ErrorFunction* errorFunction):
    ZL(Teachers.at(0).rows,Teachers.at(0).cols){
        this->Teachers = Teachers;
        this->Inputs = Inputs;
        this->network = network;
        this->errorFunc = errorFunction;
    }
    MatrixD FP(MatrixD X);
    void FP(int index);
    void BP(int index);
};

#endif /* defined(__PoorConvNeurNet__ConvolutionNeuralNetwork__) */
