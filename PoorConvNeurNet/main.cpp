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
#define LEARNING_RATE 0.325
#define MOMENTAM_RATE 0.635
#define LEARNING_REPEAT 1000
#define TESTCLASSES 5

int main(int argc, const char * argv[]) {
    std::cout<<"Loading Test Datas..."<<std::endl;
    
    std::vector<MatrixD> Inputs(CLASSES);
    std::vector<MatrixD> Tests(CLASSES);
    std::vector<MatrixD> Teachers(CLASSES);
    for(int n = 0; n < CLASSES; n++){
        Inputs.at(n) = MatrixD::zeros(INPUT_WIDTH, INPUT_WIDTH);
        Tests.at(n) = MatrixD::zeros(INPUT_WIDTH, INPUT_WIDTH);
        Teachers.at(n) = MatrixD::zeros(CLASSES, 1);
        Teachers.at(n).at<double_t>(n,0) = 1;
    }
    GetDataMat(INPUT_WIDTH, "Hanazono", Inputs);
    GetDataMat(INPUT_WIDTH, "NotoSans", Tests);
    
    std::vector<MatrixD> TestInputs(TESTCLASSES);
    std::vector<MatrixD> TestTests(TESTCLASSES);
    std::vector<MatrixD> TestTeachers(TESTCLASSES);
    for (int n = 0; n < TESTCLASSES; ++n) {
        TestInputs.at(n) = Inputs.at(n+1);
        TestTests.at(n) = Tests.at(n+1);
        TestTeachers.at(n) = MatrixD::zeros(TESTCLASSES, 1);
        TestTeachers.at(n).at<double_t>(n+1,0) = 1;
    }
    
    std::cout<<"Succeeded in Loading Datas"<<std::endl;
    
    std::cout<<"Building ConvNetwork..."<<std::endl;
    
    //ConvolutionLayer(int filterWidth, int beforeWidth, ActivationFunction* function, double_t alpha, double_t momentam)
    //PoolingLayer(int poolingWidth, int beforeWidth, ActivationFunction* function, int stride, double_t alpha, double_t momentam)
    //FullyConnectedLayer(int units, int belowUnits, ActivationFunction* function, double_t alpha, double_t momentam)
    std::vector<Layer*> network(LAYERS);
    ConvolutionLayer convlayer(3, INPUT_WIDTH, Logistic::getInstance(), LEARNING_RATE, MOMENTAM_RATE);
    network.at(0) = &convlayer;
    PoolingLayer poolinglayer(3, ConvolutionLayer::UnitWidth(3, INPUT_WIDTH) , Logistic::getInstance(), 2, LEARNING_RATE, MOMENTAM_RATE);
    network.at(1) = &poolinglayer;
    FullyConnectedLayer fullayer0(TESTCLASSES + 2,network.at(1)->U.rows,Logistic::getInstance(),LEARNING_RATE, MOMENTAM_RATE);
    network.at(2) = &fullayer0;
    FullyConnectedLayer fullayer1(TESTCLASSES, network.at(2)->U.rows, Softmax::getInstance(),LEARNING_RATE, MOMENTAM_RATE);
    network.at(3) = &fullayer1;
    
    ConvolutionalNeuralNetwork convNet(network, TestTeachers, TestInputs, CrossEntropy_MultiClass::getInstance());
    
    std::cout<<"Succeeded in Building ConvNetwork "<<std::endl;
    std::cout<<"ConvNetwork Learning..."<<std::endl;
    
    //時間計測スタート
    double f = 1/cv::getTickFrequency();
    int64 time = cv::getTickCount();
    
    double_t belowE = 0;
    for(int index = 0; index < TESTCLASSES; ++index){
        convNet.FP(index);
        convNet.BP(index);
        belowE += convNet.errorFunc->f(convNet.Teachers.at(index), convNet.FP(TestTests.at(index)));
    }
    std::cout<<belowE<<std::endl;
    for(int i = 0; i < LEARNING_REPEAT - 1; ++i){
        double_t E = 0;
        for(int index = 0; index < TESTCLASSES; ++index){
            convNet.FP(index);
            convNet.BP(index);
            E += convNet.errorFunc->f(convNet.Teachers.at(index), convNet.FP(TestTests.at(index)));
        }
        std::cout<<"E:"<<E<<std::endl;
        belowE = E;
    }
    
    std::cout<<"filter\n"<< convlayer.H.reshape(1,3) <<std::endl;
    std::cout<<"Complete Learning"<<std::endl;
    std::cout<<(cv::getTickCount() - time)*f<<" [s]"<<std::endl;
    //ここまで時間計測
    
    
    std::cout<<"Recognition Test"<<std::endl;
    
    double_t macthCount = 0;
    for (int i = 0; i < TESTCLASSES; ++i) {
        MatrixD OutPutMat = convNet.FP(TestInputs.at(i));
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