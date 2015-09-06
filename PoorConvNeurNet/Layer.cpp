//
//  Layer.cpp
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/02.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#include "Layer.h"

Layer::Layer(int units, int belowUnits, ActivationFunction* function, float alpha) :
    W(units,belowUnits), U(units,1), Z(units,1), Delta(units,1){
    func = function;
    this->alpha = alpha;
}

FullyConnectedLayer::FullyConnectedLayer(int units, int belowUnits, ActivationFunction* function, float learning_rate) :
    Layer(units,belowUnits,function,learning_rate),
    b(units,1){
    alpha = learning_rate;
    cv::RNG gen(cv::getTickCount());
    gen.fill(W, cv::RNG::UNIFORM, cv::Scalar(-1.0), cv::Scalar(1.0));
    b = 0;
}

MatrixF FullyConnectedLayer::ForwardPropargation(MatrixF X){
    MatrixF X_ = X.reshape(1,1).t();
    U = W * X_ + b;
    //std::cout<<W * X_<<std::endl;
    func->f(U, Z);
    return Z;
}

MatrixF FullyConnectedLayer::BackPropargation(MatrixF Delta_Higher, MatrixF W_Higher, MatrixF Z_lower){
    func->df(Z, Delta);
    Delta = Delta.mul(W_Higher.t() * Delta_Higher);
    MatrixF dW = Delta * Z_lower.t();
    MatrixF db = Delta;
    //モメンタム項なし
    W -= alpha * dW;
    b -= alpha * db;
    return Delta;
}
MatrixF FullyConnectedLayer::BackPropargation_Last(MatrixF Teacher, MatrixF Z_lower,ErrorFunction* erFunc){
    erFunc->df(Teacher, Z, Delta);
    MatrixF dW = Delta * Z_lower.t();
    MatrixF db = Delta;
    //モメンタム項なし
    W -= alpha * dW;
    b -= alpha * db;
    return Delta;
}


int ConvolutionLayer::UnitWidth(int filterWidth, int beforeWidth){
    return beforeWidth - 2 * (int)floor(filterWidth/2);
}

void ConvolutionLayer::Initialize(){
    cv::RNG gen(cv::getTickCount());
    gen.fill(H, cv::RNG::UNIFORM, cv::Scalar(0.0), cv::Scalar(1.0));
    b = 0;      //全成分0
    makeW();
}

//見直す必要あり
void ConvolutionLayer::makeW(){
    for(int j = 0; j < W.rows; ++j){
        for(int i = 0; i < W.cols; ++i){
            MatrixF t_ji = MatrixF::zeros((int)T.size(),1);
            for(int r = 0; r < T.size(); r++){
                t_ji(r,0) += T.at(r).at<float>(j,i);
            }
            W.at<float>(j,i) = cv::norm(t_ji.t() * H);
        }
    }
}

ConvolutionLayer::ConvolutionLayer(int filterWidth, int beforeWidth, ActivationFunction* function, float learning_rate) :
    Layer((int)pow(UnitWidth(filterWidth, beforeWidth), 2), beforeWidth*beforeWidth, function, learning_rate),
    H(filterWidth*filterWidth,1),
    b((int)pow(UnitWidth(filterWidth, beforeWidth), 2), 1){
    alpha = learning_rate;
    for(int i = 0; i < H.rows; i++){
        T.push_back(MatrixF::zeros(W.rows, W.cols));
    }
    for(int j = 0; j < T[0].rows; j++){
        for(int i = 0; i < filterWidth; ++i){
            for(int k = 0; k < filterWidth; ++k){
                T.at(i * filterWidth+k).at<float>(j,i+j +k*(beforeWidth - filterWidth)) = 1.0;
            }
        }
    }
    ConvolutionLayer::Initialize();
}

MatrixF ConvolutionLayer::ForwardPropargation(MatrixF X){
    MatrixF X_ = X.reshape(1,1).t();
    U = W * X_ + b;
    func->f(U, Z);
    return Z;
}

MatrixF ConvolutionLayer::BackPropargation(MatrixF Delta_Higher, MatrixF W_Higher, MatrixF Z_lower){
    func->df(Z, Delta);
    Delta = Delta.mul(W_Higher.t() * Delta_Higher);
    MatrixF dW = Delta * Z_lower.t();
    MatrixF db = Delta;
    //モメンタム項なし
    W -= alpha * dW;
    b -= alpha * db;
    
    std::vector<MatrixF> dHr(H.rows);
    MatrixF dH(H.rows,1);
    for(int r = 0; r < dHr.size(); r++){
        dHr.at(r) = T.at(r).mul(dW);
        reduce(dHr.at(r), dHr.at(r), 0, CV_REDUCE_SUM);
        reduce(dHr.at(r), dHr.at(r), 1, CV_REDUCE_SUM);
        dH(r,0) = dHr.at(r)(0,0);
    }
    //改善すべき点：AdaGrad等未実装
    H -= dH * alpha;
    makeW();
    return Delta;
}

MatrixF ConvolutionLayer::BackPropargation_Last(MatrixF Teacher, MatrixF Z_lower, ErrorFunction* erFunc){
    erFunc->df(Teacher, Z, Delta);
    MatrixF dW = Delta * Z_lower.t();
    MatrixF db = Delta;
    //モメンタム項なし
    W -= alpha * dW;
    b -= alpha * db;
    std::vector<MatrixF> dHr(H.rows);
    MatrixF dH(H.rows,1);
    for(int r = 0; r < dHr.size(); r++){
        dHr.at(r) = T.at(r).mul(dW);
        reduce(dHr.at(r), dHr.at(r), 0, CV_REDUCE_SUM);
        reduce(dHr.at(r), dHr.at(r), 1, CV_REDUCE_SUM);
        dH(r,0) = dHr.at(r)(0,0);
    }
    //改善すべき点：AdaGrad等未実装
    H -= dH * alpha;
    makeW();

    return Delta;
}


int PoolingLayer::UnitWidth(int beforeWidth, int stride){
    return (int)floor((beforeWidth - 1)/stride) + 1;
}

PoolingLayer::PoolingLayer(int poolingWidth, int beforeWidth, ActivationFunction* function, int stride, float learning_rate):
    Layer((int)pow(UnitWidth(beforeWidth,stride), 2),beforeWidth*beforeWidth, function, learning_rate){
    PoolingLayer::poolingWidth = poolingWidth;
    PoolingLayer::stride = stride;
    int halfPoolingWidth = (int)floor(poolingWidth/2);
    int unitwidth = UnitWidth(beforeWidth,stride);
    float weight4pooling = 1.0/(poolingWidth*poolingWidth);
    W = 0;
    //改善すべき点：ここのforループは時間がかかるので、どうにかして行列演算等で高速化したい
    for(int j = 0; j < W.rows; j++){
        MatrixF PaddedW_i(beforeWidth + halfPoolingWidth*2, beforeWidth+halfPoolingWidth*2);
        PaddedW_i = 0;
        for(int b = 0; b < poolingWidth; b++){
            for(int a = 0; a < poolingWidth; a++){
                PaddedW_i.at<float>((int)floor(j*stride/unitwidth)+b, (j*stride)%unitwidth+a) = weight4pooling;
            }
        }
        for(int y = 0; y < beforeWidth; y++){
            for(int x = 0; x < beforeWidth; x++){
                W.at<float>(j,y*beforeWidth + x) = PaddedW_i.at<float>(y+halfPoolingWidth,x+halfPoolingWidth);
            }
        }
    }
}

MatrixF PoolingLayer::ForwardPropargation(MatrixF X){
    MatrixF X_ = X.reshape(1,1).t();
    //改善するべき点:用意した重み行列によるpoolingは精度が悪い
    U = W * X_;
    U.copyTo(Z);
    /*
    int padWidth = (int)floor(PoolingLayer::poolingWidth/2);
    MatrixF paddedX = X.reshape(1,(int)sqrt(X.rows*X.cols));
    std::cout<<paddedX<<std::endl;
    copyMakeBorder(paddedX, paddedX, padWidth, padWidth, padWidth, padWidth, cv::BORDER_CONSTANT,cv::Scalar(0));
    std::cout<<paddedX<<std::endl;
    int index_i = 0, index_j;
    for(int i = 0; i + padWidth < Z.cols - padWidth; i += stride){
        index_j = 0;
        for(int j = 0; j + padWidth < Z.rows - padWidth; j += stride){
            MatrixF subMat;
            X(cv::Rect(j, i, poolingWidth, poolingWidth)).copyTo(subMat);
            reduce(subMat, subMat, 0, CV_REDUCE_AVG);
            reduce(subMat, subMat, 1, CV_REDUCE_AVG);
            Z.at<float>(index_j, index_i) = subMat.at<float>(0,0);
            index_j++;
        }
        index_i++;
    }*/
    return Z;
}

MatrixF PoolingLayer::BackPropargation(MatrixF Delta_Higher, MatrixF W_Higher, MatrixF Z_lower){
    func->df(Z, Delta);
    Delta = Delta.mul(W_Higher.t() * Delta_Higher);
    //Wの更新は行わない
    return Delta;
}

MatrixF PoolingLayer::BackPropargation_Last(MatrixF Teacher, MatrixF Z_lower, ErrorFunction *erFunc){
    erFunc->df(Teacher, Z, Delta);
    return Delta;
}