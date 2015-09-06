//
//  XORNeurNetTest.cpp
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/06.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

//  XORを3層のNeuralNetworkで学習させるテスト
//  初期化時の乱数は一様乱数なのでときどき失敗する。

/*
#include <iostream>
#include "ConvolutionNeuralNetwork.h"

#define LEARNING_RATE 0.1
#define LEARNING_REPEAT 1000

int main(int argc, char* argv[]){
    FullyConnectedLayer layer1(2, 2, Logistic::getInstance(), LEARNING_RATE);
    FullyConnectedLayer layer2(1, 2, Logistic::getInstance(), LEARNING_RATE);
    std::vector<Layer*> network(2);
    network.at(0) = &layer1;
    network.at(1) = &layer2;
    MatrixF Input1 = (MatrixF(2,1)<<0,0); MatrixF Teacher1 = (MatrixF(1,1)<<0);
    MatrixF Input2 = (MatrixF(2,1)<<0,1); MatrixF Teacher2 = (MatrixF(1,1)<<1);
    MatrixF Input3 = (MatrixF(2,1)<<1,0); MatrixF Teacher3 = (MatrixF(1,1)<<1);
    MatrixF Input4 = (MatrixF(2,1)<<1,1); MatrixF Teacher4 = (MatrixF(1,1)<<0);
    std::vector<MatrixF> Teachers(4);
    Teachers.at(0) = Teacher1;
    Teachers.at(1) = Teacher2;
    Teachers.at(2) = Teacher3;
    Teachers.at(3) = Teacher4;
    std::vector<MatrixF> Inputs(4);
    Inputs.at(0) = Input1;
    Inputs.at(1) = Input2;
    Inputs.at(2) = Input3;
    Inputs.at(3) = Input4;
    ConvolutionalNeuralNetwork model(network,Teachers ,Inputs , SquareError::getInstance());
    
    
    for(int j = 0; j < LEARNING_REPEAT;++j){
        for (int i = 0; i < 4; ++i) {
            model.FP(i);
            model.BP(i);
        }
    }
    
    for (int i = 0; i < 4; ++i) {
        std::cout<<Inputs.at(i).t()<<" : "<<Teachers.at(i)<<std::endl;
        std::cout<<model.FP(Inputs.at(i))<<std::endl;
    }
    return 0;
}
*/