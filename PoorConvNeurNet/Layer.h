//
//  Layer.h
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/02.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#ifndef __PoorConvNeurNet__Layer__
#define __PoorConvNeurNet__Layer__

#include "Definitions.h"
#include "ActivationFunction.h"
#include "ErrorFunction.h"

class Layer{
public:
    MatrixD W;
    MatrixD U;
    MatrixD Z;
    MatrixD Delta;
    MatrixD dW;
    ActivationFunction* func;
    double_t alpha;
    double_t momentum;
    
    Layer(int units, int belowUnits, ActivationFunction* function, double_t alpha, double_t momentum = 0.5);
    virtual ~Layer(){}
    //参照を返す
    virtual MatrixD ForwardPropargation(MatrixD X) = 0;
    virtual MatrixD BackPropargation(MatrixD Delta_Higher, MatrixD W_Higher, MatrixD Z_lower) = 0;
    virtual MatrixD BackPropargation_Last(MatrixD Teacher, MatrixD Z_lower, ErrorFunction* erFunc) = 0;
};

class FullyConnectedLayer : public Layer{
public:
    MatrixD b;
    MatrixD db;
    
    FullyConnectedLayer(int units, int belowUnits, ActivationFunction* function, double_t alpha, double_t momentum = 0.5);
    ~FullyConnectedLayer(){}
    
    MatrixD ForwardPropargation(MatrixD X)override;
    MatrixD BackPropargation(MatrixD Delta_Higher, MatrixD W_Higher, MatrixD Z_lower)override;
    MatrixD BackPropargation_Last(MatrixD Teacher, MatrixD Z_lower, ErrorFunction* erFunc)override;
};

class ConvolutionLayer : public Layer {
public:
    MatrixD H;
    MatrixD b;
    MatrixD db;
    MatrixD dH;
    std::vector<MatrixD> T;
    
    static int UnitWidth(int filterWidth, int beforeWidth);
    void Initialize();
    void makeW();
    
    ConvolutionLayer(int filterWidth, int beforeWidth, ActivationFunction* function, double_t alpha, double_t momentum = 0.5);
    ~ConvolutionLayer(){}
    
    MatrixD ForwardPropargation(MatrixD X)override;
    MatrixD BackPropargation(MatrixD Delta_Higher, MatrixD W_Higher, MatrixD Z_lower)override;
    MatrixD BackPropargation_Last(MatrixD Teacher, MatrixD Z_lower, ErrorFunction* erFunc)override;
};

class PoolingLayer : public Layer {
public:
    int stride;
    int poolingWidth;
    
    static int UnitWidth(int beforeWidth, int stride);
    
    PoolingLayer(int poolingWidth, int beforeWidth, ActivationFunction* function, int stride, double_t alpha, double_t momentum = 0.5);
    ~PoolingLayer(){}
    
    MatrixD ForwardPropargation(MatrixD X)override;
    MatrixD BackPropargation(MatrixD Delta_Higher, MatrixD W_Higher, MatrixD Z_lower)override;
    MatrixD BackPropargation_Last(MatrixD Teacher, MatrixD Z_lower, ErrorFunction* erFunc)override;
};

#endif /* defined(__PoorConvNeurNet__Layer__) */
