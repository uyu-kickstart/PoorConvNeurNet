//
//  ConvolutionNeuralNetwork.cpp
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/03.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#include "ConvolutionNeuralNetwork.h"

MatrixF ConvolutionalNeuralNetwork::FP(MatrixF X){
    MatrixF Z;
    X.copyTo(Z);
    for(int i = 0; i < network.size(); i++){
        network.at(i)->ForwardPropargation(Z).copyTo(Z);
    }
    return Z;
}

void ConvolutionalNeuralNetwork::FP(int index){
    MatrixF Z = Inputs.at(index).reshape(1,1).t();
    for(int i = 0; i < network.size(); i++){
        (network.at(i)->ForwardPropargation(Z)).copyTo(Z);
    }
    Z.copyTo(ZL);
}

void ConvolutionalNeuralNetwork::BP(int index){
    MatrixF Delta;
    network.at((int)network.size()-1)->BackPropargation_Last(Teachers.at(index), network.at((int)network.size()-2)->Z, errorFunc).copyTo(Delta);
    for(int i = (int)network.size() - 2; i > 0; i--){
        network.at(i)->BackPropargation(Delta, network.at(i+1)->W,network.at(i-1)->Z).copyTo(Delta);
    }
    network.at(0)->BackPropargation(Delta, network.at(1)->W, Inputs.at(index).reshape(1,1).t()).copyTo(Delta);
}