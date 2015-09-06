//
//  ErrorFunction.cpp
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/03.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#include "ErrorFunction.h"

float CrossEntropy_MultiClass::f(MatrixF Teacher, MatrixF ZL){
    //未実装
    return 1;
}

void CrossEntropy_MultiClass::df(MatrixF Teacher, MatrixF ZL, MatrixF DeltaL){
    DeltaL = ZL - Teacher;
}

float Likelihood_BinaryClass::f(MatrixF Teacher, MatrixF ZL){
    MatrixF Y = ZL.clone();
    MatrixF _1minusY = 1 - ZL;
    cv::log(Y, Y);
    cv::log(_1minusY, _1minusY);
    MatrixF E = Teacher.mul(Y) + _1minusY.mul(1 - Teacher);
    cv::reduce(E, E, 0, CV_REDUCE_SUM);
    cv::reduce(E, E, 1, CV_REDUCE_SUM);
    return E.at<float>(0,0);
}

void Likelihood_BinaryClass::df(MatrixF Teacher, MatrixF ZL, MatrixF DeltaL){
    DeltaL = Teacher - ZL;
}

float SquareError::f(MatrixF Teacher, MatrixF ZL){
    MatrixF Difference = ZL - Teacher;
    return cv::norm(Difference)/2;
}

void SquareError::df(MatrixF Teacher, MatrixF ZL, MatrixF DeltaL){
    DeltaL =  ZL - Teacher;
}
