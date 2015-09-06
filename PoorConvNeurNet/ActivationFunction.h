//
//  ActivationFunction.h
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/02.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#ifndef __PoorConvNeurNet__ActivationFunction__
#define __PoorConvNeurNet__ActivationFunction__

#include "Definitions.h"
class ActivationFunction {
public:
    virtual void f(MatrixF U, MatrixF Z) = 0;
    virtual void df(MatrixF Z, MatrixF Out) = 0;
};

class Identity : public ActivationFunction {
private:
    Identity(){}
    Identity(const Identity& r){}
    Identity& operator=(const Identity& r){return *this;}
public:
    static Identity* getInstance(){
        static Identity identity;
        return &identity;
    }
    void f(MatrixF U, MatrixF Z) override;
    void df(MatrixF Z, MatrixF Out) override;
};

class Rectifer : public ActivationFunction{
private:
    Rectifer(){}
    Rectifer(const Rectifer& r){}
    Rectifer& operator=(const Rectifer& r){return *this;}
public:
    static Rectifer* getInstance(){
        static Rectifer rectifer;
        return &rectifer;
    }
    void f(MatrixF U, MatrixF Z) override;
    void df(MatrixF Z, MatrixF Out) override;
};

class Logistic : public ActivationFunction {
private:
    Logistic(){}
    Logistic(const Logistic& r){}
    Logistic& operator=(const Logistic& r){return *this;}
public:
    static Logistic* getInstance(){
        static Logistic logistic;
        return &logistic;
    }
    void f(MatrixF U, MatrixF Z) override;
    void df(MatrixF Z, MatrixF Out) override;
};

class Softmax : public ActivationFunction {
private:
    Softmax(){}
    Softmax(const Softmax& r){}
    Softmax& operator=(const Softmax& r){return *this;}
public:
    static Softmax* getInstance(){
        static Softmax softmax;
        return &softmax;
    }
    void f(MatrixF U, MatrixF Z) override;
    void df(MatrixF Z, MatrixF Out) override;
};

#endif /* defined(__PoorConvNeurNet__ActivationFunction__) */
