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
    virtual float f(MatrixF Teacher, MatrixF ZL)=0;
    virtual void df(MatrixF Teacher, MatrixF ZL, MatrixF DeltaL)=0;
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
    float f(MatrixF Teacher, MatrixF ZL) override;
    void df(MatrixF Teacher, MatrixF ZL, MatrixF DeltaL) override;
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
    float f(MatrixF Teacher, MatrixF ZL) override;
    void df(MatrixF Teacher, MatrixF ZL, MatrixF DeltaL) override;
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
    float f(MatrixF Teacher, MatrixF ZL) override;
    void df(MatrixF Teacher, MatrixF ZL, MatrixF DeltaL) override;
};


#endif /* defined(__PoorConvNeurNet__ErrorFunction__) */
