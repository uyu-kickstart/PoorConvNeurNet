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
    std::vector<MatrixF> Teachers;
    std::vector<MatrixF> Inputs;
    std::vector<Layer*> network;
    ErrorFunction* errorFunc;
    MatrixF ZL;
    ConvolutionalNeuralNetwork(std::vector<Layer*> network, std::vector<MatrixF> Teachers, std::vector<MatrixF> Inputs, ErrorFunction* errorFunction){
        this->Teachers = Teachers;
        this->Inputs = Inputs;
        this->network = network;
        this->errorFunc = errorFunction;
    }
    MatrixF FP(MatrixF X);
    void FP(int index);
    void BP(int index);
};

#endif /* defined(__PoorConvNeurNet__ConvolutionNeuralNetwork__) */
