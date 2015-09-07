//
//  main.cpp
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/02.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#include <iostream>
#include "ConvolutionNeuralNetwork.h"
#include "makeArray.h"
#include <opencv2/highgui.hpp>

#define INPUT_WIDTH 16
#define LAYERS 4
#define LEARNING_RATE 0.9
#define LEARNING_REPEAT 2
#define TESTCLASSES 3

int main(int argc, const char * argv[]) {
    std::cout<<"Loading Test Datas..."<<std::endl;
    
    std::vector<MatrixD> Inputs(CLASSES);
    std::vector<MatrixD> Teachers(CLASSES);
    for(int n = 0; n < CLASSES; n++){
        Inputs.at(n) = MatrixD::zeros(INPUT_WIDTH, INPUT_WIDTH);
        Teachers.at(n) = MatrixD::zeros(CLASSES, 1);
        Teachers.at(n).at<double_t>(n,0) = 1;
    }
    GetDataMat(INPUT_WIDTH, "Hanazono", Inputs);
    /*
    std::vector<MatrixD> TestInputs(TESTCLASSES);
    std::vector<MatrixD> TestTeachers(TESTCLASSES);
    for (int n = 0; n < TESTCLASSES; ++n) {
        TestInputs.at(n) = Inputs.at(n);
        TestTeachers.at(n) = MatrixD::zeros(TESTCLASSES, 1);
        TestTeachers.at(n).at<double_t>(n,0) = 1;
    }*/
    
    std::cout<<"Succeeded in Loading Datas"<<std::endl;
    
    std::cout<<"Building ConvNetwork..."<<std::endl;
    
    //ConvolutionLayer(int filterWidth, int beforeWidth, ActivationFunction* function, double_t alpha = 0.1)
    //PoolingLayer(int poolingWidth, int beforeWidth, ActivationFunction* function, int stride, double_t alpha = 0.1)
    //FullyConnectedLayer(int units, int belowUnits, ActivationFunction* function, double_t alpha = 0.1)
    std::vector<Layer*> network(LAYERS);
    ConvolutionLayer convlayer(3, INPUT_WIDTH, Logistic::getInstance(), LEARNING_RATE);
    network.at(0) = &convlayer;
    PoolingLayer poolinglayer(3, ConvolutionLayer::UnitWidth(3, INPUT_WIDTH) , Logistic::getInstance(), 1, LEARNING_RATE);
    network.at(1) = &poolinglayer;
    FullyConnectedLayer fullayer0(CLASSES + 120,network.at(1)->U.rows,Logistic::getInstance(),LEARNING_RATE);
    network.at(2) = &fullayer0;
    FullyConnectedLayer fullayer1(CLASSES, network.at(2)->U.rows, Softmax::getInstance(),LEARNING_RATE);
    network.at(3) = &fullayer1;
    
    ConvolutionalNeuralNetwork convNet(network, Teachers, Inputs, CrossEntropy_MultiClass::getInstance());
    
    std::cout<<"Succeeded in Building ConvNetwork "<<std::endl;
    std::cout<<"ConvNetwork Learning..."<<std::endl;
    
    //時間計測スタート
    double f = 1/cv::getTickFrequency();
    int64 time = cv::getTickCount();
    
    double_t belowE = 0;
    for(int index = 0; index < CLASSES; ++index){
        convNet.FP(index);
        belowE += convNet.errorFunc->f(convNet.Teachers.at(index), convNet.ZL);
        convNet.BP(index);
    }
    for(int i = 0; i < LEARNING_REPEAT - 1; ++i){
        double_t E = 0;
        for(int index = 0; index < CLASSES; ++index){
            convNet.FP(index);
            E += convNet.errorFunc->f(convNet.Teachers.at(index), convNet.ZL);
            convNet.BP(index);
        }
        std::cout<<"E:"<<E<<std::endl;
        if(belowE < E)
            break;
        belowE = E;
    }
    
    std::cout<<"filter\n"<< convlayer.H.reshape(1,3) <<std::endl;
    std::cout<<"Complete Learning"<<std::endl;
    std::cout<<(cv::getTickCount() - time)*f<<" [s]"<<std::endl;
    //ここまで時間計測
    
    
    std::cout<<"Recognition Test"<<std::endl;
    
    double_t macthCount = 0;
    for (int i = 0; i < CLASSES; ++i) {
        MatrixD OutPutMat = convNet.FP(Inputs.at(i));
        int index_max = -1;
        double_t maxVal = 0;
        for(int j = 0; j < OutPutMat.rows; ++j){
            if(maxVal < OutPutMat.at<double_t>(j,0)){
                index_max = j;
                maxVal = OutPutMat.at<double_t>(j,0);
            }
        }
        if(index_max == i){
            macthCount++;
            std::cout<<index_max<<std::endl;
        }
    }
    std::cout<<"Recognize Count:"<<macthCount<<std::endl;
    return 0;
}