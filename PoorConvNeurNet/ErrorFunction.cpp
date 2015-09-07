//
//  ErrorFunction.cpp
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/03.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#include "ErrorFunction.h"

double_t CrossEntropy_MultiClass::f(MatrixD Teacher, MatrixD ZL){
    MatrixD logZL;
    cv::log(ZL, logZL);
    logZL = logZL.mul(Teacher);
    cv::reduce(logZL, logZL, 0, CV_REDUCE_SUM);
    cv::reduce(logZL, logZL, 1, CV_REDUCE_SUM);
    //未実装
    return -1 * logZL.at<double_t>(0,0);
}

void CrossEntropy_MultiClass::df(MatrixD Teacher, MatrixD ZL, MatrixD DeltaL){
    DeltaL = ZL - Teacher;
}

double_t Likelihood_BinaryClass::f(MatrixD Teacher, MatrixD ZL){
    MatrixD Y = ZL.clone();
    MatrixD _1minusY = 1 - ZL;
    cv::log(Y, Y);
    cv::log(_1minusY, _1minusY);
    MatrixD E = Teacher.mul(Y) + _1minusY.mul(1 - Teacher);
    cv::reduce(E, E, 0, CV_REDUCE_SUM);
    cv::reduce(E, E, 1, CV_REDUCE_SUM);
    return E.at<double_t>(0,0);
}

void Likelihood_BinaryClass::df(MatrixD Teacher, MatrixD ZL, MatrixD DeltaL){
    DeltaL = Teacher - ZL;
}

double_t SquareError::f(MatrixD Teacher, MatrixD ZL){
    MatrixD Difference = ZL - Teacher;
    return cv::norm(Difference)/2;
}

void SquareError::df(MatrixD Teacher, MatrixD ZL, MatrixD DeltaL){
    DeltaL =  ZL - Teacher;
}
