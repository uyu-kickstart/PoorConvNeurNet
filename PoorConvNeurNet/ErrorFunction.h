//
//  ErrorFunction.h
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/03.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#ifndef __PoorConvNeurNet__ErrorFunction__
#define __PoorConvNeurNet__ErrorFunction__

#include "Definitions.h"
#include "ActivationFunction.h"

class ErrorFunction {
public:
    virtual double_t f(MatrixD Teacher, MatrixD ZL)=0;
    virtual void df(MatrixD Teacher, MatrixD ZL, MatrixD DeltaL)=0;
};

class CrossEntropy_MultiClass : public ErrorFunction {
private:
    CrossEntropy_MultiClass(){}
    CrossEntropy_MultiClass(const CrossEntropy_MultiClass& r){}
    CrossEntropy_MultiClass& operator=(const CrossEntropy_MultiClass& r){return *this;}
public:
    static CrossEntropy_MultiClass* getInstance(){
        static CrossEntropy_MultiClass ce_mc;
        return &ce_mc;
    }
    double_t f(MatrixD Teacher, MatrixD ZL) override;
    void df(MatrixD Teacher, MatrixD ZL, MatrixD DeltaL) override;
};

class Likelihood_BinaryClass : public ErrorFunction {
private:
    Likelihood_BinaryClass(){}
    Likelihood_BinaryClass(const Likelihood_BinaryClass& r){}
    Likelihood_BinaryClass& operator=(const Likelihood_BinaryClass& r){return *this;}
public:
    static Likelihood_BinaryClass* getInstance(){
        static Likelihood_BinaryClass l_bc;
        return &l_bc;
    }
    double_t f(MatrixD Teacher, MatrixD ZL) override;
    void df(MatrixD Teacher, MatrixD ZL, MatrixD DeltaL) override;
};

class SquareError : public ErrorFunction {
private:
    SquareError(){}
    SquareError(const SquareError& r){}
    SquareError& operator=(const SquareError& r){return *this;}
public:
    static SquareError* getInstance(){
        static SquareError se;
        return &se;
    }
    double_t f(MatrixD Teacher, MatrixD ZL) override;
    void df(MatrixD Teacher, MatrixD ZL, MatrixD DeltaL) override;
};


#endif /* defined(__PoorConvNeurNet__ErrorFunction__) */
