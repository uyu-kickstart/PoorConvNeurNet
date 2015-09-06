//
//  ActivationFunction.cpp
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/02.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#include "ActivationFunction.h"

void Identity::f(MatrixF U, MatrixF Z){
    U.copyTo(Z);
    Z.forEach([](float& uij, const void*)->void{
        if(uij<0)
            uij = 0;
        else if(uij>1)
            uij = 1;
    });
}

void Identity::df(MatrixF Z, MatrixF Out){
    Out = 1;
}

void Rectifer::f(MatrixF U, MatrixF Z){
    U.copyTo(Z);
    Z.forEach([](float& uij, const void*)->void{
        uij = fmax(uij, 0);
    });
}

void Rectifer::df(MatrixF Z, MatrixF Out){
    Z.copyTo(Out);
    Out.forEach([](float& zij, const void*)->void{
        if(zij>0)
            zij = 1;
        else
            zij = 0;
    });
}

void Logistic::f(MatrixF U, MatrixF Z){
    U.copyTo(Z);
    Z.forEach([](float& uij, const void*)->void{
        uij = 1 / (1 + exp(-uij));
    });
}

void Logistic::df(MatrixF Z, MatrixF Out){
    Z.copyTo(Out);
    Out.forEach([](float& zij, const void*)->void{
        zij = zij * (1 - zij);
    });
}

void Softmax::f(MatrixF U, MatrixF Z){
    //オーバーフローを防ぐためにUの最大値を見つけて各値から引く
    MatrixF SumU = U.clone(), MaxU = U.clone();
    cv::reduce(MaxU, MaxU, 0, CV_REDUCE_MAX);
    cv::reduce(MaxU, MaxU, 1, CV_REDUCE_MAX);
    float max = MaxU.at<float>(0,0);
    SumU.forEach([&](float& t, const void*)-> void{
        t = exp(t - max);
    });
    //std::cout<<SumU<<std::endl;
    cv::reduce(SumU, SumU, 0, CV_REDUCE_SUM);
    cv::reduce(SumU, SumU, 1, CV_REDUCE_SUM);
    float sum = SumU.at<float>(0,0);
    //std::cout<<"\n"<<sum<<"\n"<<std::endl;
    U.copyTo(Z);
    Z.forEach([&](float& zij, const void*)->void{
        zij = exp(zij - max) / sum;
    });
    //std::cout<<Z<<std::endl;
    /*
    MatrixF TestMat = Z.clone();
    cv::reduce(TestMat, TestMat, 0, CV_REDUCE_SUM);
    cv::reduce(TestMat, TestMat, 1, CV_REDUCE_SUM);
    std::cout<<TestMat<<std::endl;
    */
}

void Softmax::df(MatrixF Z, MatrixF Out){
    Z.copyTo(Out);
    Out.forEach([](float& zij, const void*)->void{
        zij = zij * (1 - zij);
    });
}