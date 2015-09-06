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
    MatrixF W;
    MatrixF U;
    MatrixF Z;
    MatrixF Delta;
    ActivationFunction* func;
    float alpha;
    
    Layer(int units, int belowUnits, ActivationFunction* function, float alpha = 0.1);
    virtual ~Layer(){}
    //参照を返す
    virtual MatrixF ForwardPropargation(MatrixF X) = 0;
    virtual MatrixF BackPropargation(MatrixF Delta_Higher, MatrixF W_Higher, MatrixF Z_lower) = 0;
    virtual MatrixF BackPropargation_Last(MatrixF Teacher, MatrixF Z_lower, ErrorFunction* erFunc) = 0;
};

class FullyConnectedLayer : public Layer{
public:
    MatrixF b;
    
    FullyConnectedLayer(int units, int belowUnits, ActivationFunction* function, float alpha = 0.1);
    ~FullyConnectedLayer(){}
    
    MatrixF ForwardPropargation(MatrixF X)override;
    MatrixF BackPropargation(MatrixF Delta_Higher, MatrixF W_Higher, MatrixF Z_lower)override;
    MatrixF BackPropargation_Last(MatrixF Teacher, MatrixF Z_lower, ErrorFunction* erFunc)override;
};

class ConvolutionLayer : public Layer {
public:
    MatrixF H;
    MatrixF b;
    std::vector<MatrixF> T;
    
    static int UnitWidth(int filterWidth, int beforeWidth);
    void Initialize();
    void makeW();
    
    ConvolutionLayer(int filterWidth, int beforeWidth, ActivationFunction* function, float alpha = 0.1);
    ~ConvolutionLayer(){}
    
    MatrixF ForwardPropargation(MatrixF X)override;
    MatrixF BackPropargation(MatrixF Delta_Higher, MatrixF W_Higher, MatrixF Z_lower)override;
    MatrixF BackPropargation_Last(MatrixF Teacher, MatrixF Z_lower, ErrorFunction* erFunc)override;
};

class PoolingLayer : public Layer {
public:
    int stride;
    int poolingWidth;
    
    static int UnitWidth(int beforeWidth, int stride);
    
    PoolingLayer(int poolingWidth, int beforeWidth, ActivationFunction* function, int stride, float alpha = 0.1);
    ~PoolingLayer(){}
    
    MatrixF ForwardPropargation(MatrixF X)override;
    MatrixF BackPropargation(MatrixF Delta_Higher, MatrixF W_Higher, MatrixF Z_lower)override;
    MatrixF BackPropargation_Last(MatrixF Teacher, MatrixF Z_lower, ErrorFunction* erFunc)override;
};

#endif /* defined(__PoorConvNeurNet__Layer__) */
