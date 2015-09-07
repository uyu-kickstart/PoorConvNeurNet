//
//  Layer.cpp
//  PoorConvNeurNet
//
//  Created by IGUCHI Yusuke on 2015/09/02.
//  Copyright (c) 2015年 uyu-kickstart. All rights reserved.
//

#include "Layer.h"

Layer::Layer(int units, int belowUnits, ActivationFunction* function, double_t alpha, double_t momentum) :
    W(units,belowUnits), U(units,1), Z(units,1), Delta(units,1), dW(units,belowUnits){
        dW = 0;
    func = function;
    this->alpha = alpha;
    this->momentum = momentum;
}

FullyConnectedLayer::FullyConnectedLayer(int units, int belowUnits, ActivationFunction* function, double_t learning_rate, double_t momentum) :
    Layer(units,belowUnits,function,learning_rate, momentum),
    b(units,1), db(units,1){
    alpha = learning_rate;
    cv::RNG gen(cv::getTickCount());
    gen.fill(W, cv::RNG::UNIFORM, cv::Scalar(-1.0), cv::Scalar(1.0));
    b = 0;
    db = 0;
}

MatrixD FullyConnectedLayer::ForwardPropargation(MatrixD X){
    MatrixD X_ = X.reshape(1,1).t();
    U = W * X_ + b;
    //std::cout<<W * X_<<std::endl;
    func->f(U, Z);
    return Z;
}

MatrixD FullyConnectedLayer::BackPropargation(MatrixD Delta_Higher, MatrixD W_Higher, MatrixD Z_lower){
    func->df(Z, Delta);
    Delta = Delta.mul(W_Higher.t() * Delta_Higher);
    dW = momentum * dW - alpha * Delta * Z_lower.t();
    db = momentum * db - alpha * Delta;
    W += dW;
    b += db;
    return Delta;
}
MatrixD FullyConnectedLayer::BackPropargation_Last(MatrixD Teacher, MatrixD Z_lower,ErrorFunction* erFunc){
    erFunc->df(Teacher, Z, Delta);
    dW = momentum * dW - alpha * Delta * Z_lower.t();
    db = momentum * db - alpha * Delta;
    W += dW;
    b += db;
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
            MatrixD t_ji = MatrixD::zeros((int)T.size(),1);
            for(int r = 0; r < T.size(); r++){
                t_ji(r,0) += T.at(r).at<double_t>(j,i);
            }
            W.at<double_t>(j,i) = cv::norm(t_ji.t() * H);
        }
    }
}

ConvolutionLayer::ConvolutionLayer(int filterWidth, int beforeWidth, ActivationFunction* function, double_t learning_rate, double_t momentum) :
    Layer((int)pow(UnitWidth(filterWidth, beforeWidth), 2), beforeWidth*beforeWidth, function, learning_rate, momentum),
    H(filterWidth*filterWidth,1),
    b((int)pow(UnitWidth(filterWidth, beforeWidth), 2), 1), db(b.rows, 1), dH(H.rows,1){
    alpha = learning_rate;
    for(int i = 0; i < H.rows; i++){
        T.push_back(MatrixD::zeros(W.rows, W.cols));
    }
    for(int j = 0; j < T[0].rows; j++){
        for(int i = 0; i < filterWidth; ++i){
            for(int k = 0; k < filterWidth; ++k){
                T.at(i * filterWidth+k).at<double_t>(j,i+j +k*(beforeWidth - filterWidth)) = 1.0;
            }
        }
    }
    db = 0;
    dH = 0;
    ConvolutionLayer::Initialize();
}

MatrixD ConvolutionLayer::ForwardPropargation(MatrixD X){
    MatrixD X_ = X.reshape(1,1).t();
    U = W * X_ + b;
    func->f(U, Z);
    return Z;
}

MatrixD ConvolutionLayer::BackPropargation(MatrixD Delta_Higher, MatrixD W_Higher, MatrixD Z_lower){
    func->df(Z, Delta);
    Delta = Delta.mul(W_Higher.t() * Delta_Higher);
    dW = Delta * Z_lower.t();
    db = momentum * db - alpha * Delta;
    W += dW;
    b += db;
    
    std::vector<MatrixD> dHr(H.rows);
    MatrixD dH_(H.rows,1);
    dH_ = 0;
    for(int r = 0; r < dHr.size(); r++){
        dHr.at(r) = T.at(r).mul(dW);
        reduce(dHr.at(r), dHr.at(r), 0, CV_REDUCE_SUM);
        reduce(dHr.at(r), dHr.at(r), 1, CV_REDUCE_SUM);
        dH_(r,0) = dHr.at(r)(0,0);
    }
    //改善すべき点：AdaGrad等未実装
    dH = momentum * dH - alpha * dH_;
    H += dH;
    makeW();
    return Delta;
}

MatrixD ConvolutionLayer::BackPropargation_Last(MatrixD Teacher, MatrixD Z_lower, ErrorFunction* erFunc){
    erFunc->df(Teacher, Z, Delta);
    dW = Delta * Z_lower.t();
    db = momentum * db - alpha * Delta;
    W += dW;
    b += db;
    
    std::vector<MatrixD> dHr(H.rows);
    MatrixD dH_(H.rows,1);
    dH_ = 0;
    for(int r = 0; r < dHr.size(); r++){
        dHr.at(r) = T.at(r).mul(dW);
        reduce(dHr.at(r), dHr.at(r), 0, CV_REDUCE_SUM);
        reduce(dHr.at(r), dHr.at(r), 1, CV_REDUCE_SUM);
        dH_(r,0) = dHr.at(r)(0,0);
    }
    //改善すべき点：AdaGrad等未実装
    dH = momentum * dH - alpha * dH_;
    H += dH;
    makeW();
    return Delta;
}


int PoolingLayer::UnitWidth(int beforeWidth, int stride){
    return (int)floor((beforeWidth - 1)/stride) + 1;
}

PoolingLayer::PoolingLayer(int poolingWidth, int beforeWidth, ActivationFunction* function, int stride, double_t learning_rate, double_t momentum):
    Layer((int)pow(UnitWidth(beforeWidth,stride), 2),beforeWidth*beforeWidth, function, learning_rate, momentum){
    PoolingLayer::poolingWidth = poolingWidth;
    PoolingLayer::stride = stride;
    int halfPoolingWidth = (int)floor(poolingWidth/2);
    int unitwidth = UnitWidth(beforeWidth,stride);
    double_t weight4pooling = 1.0/(poolingWidth*poolingWidth);
    W = 0;
    //改善すべき点：ここのforループは時間がかかるので、どうにかして行列演算等で高速化したい
    for(int j = 0; j < W.rows; j++){
        MatrixD PaddedW_i(beforeWidth + halfPoolingWidth*2, beforeWidth+halfPoolingWidth*2);
        PaddedW_i = 0;
        for(int b = 0; b < poolingWidth; b++){
            for(int a = 0; a < poolingWidth; a++){
                PaddedW_i.at<double_t>((int)floor(j*stride/unitwidth)+b, (j*stride)%unitwidth+a) = weight4pooling;
            }
        }
        for(int y = 0; y < beforeWidth; y++){
            for(int x = 0; x < beforeWidth; x++){
                W.at<double_t>(j,y*beforeWidth + x) = PaddedW_i.at<double_t>(y+halfPoolingWidth,x+halfPoolingWidth);
            }
        }
    }
}

MatrixD PoolingLayer::ForwardPropargation(MatrixD X){
    MatrixD X_ = X.reshape(1,1).t();
    //改善するべき点:用意した重み行列によるpoolingは精度が悪い
    //正規化を行っていないので値が発散しやすい?
    U = W * X_;
    U.copyTo(Z);
    /*
    int padWidth = (int)floor(PoolingLayer::poolingWidth/2);
    MatrixD paddedX = X.reshape(1,(int)sqrt(X.rows*X.cols));
    std::cout<<paddedX<<std::endl;
    copyMakeBorder(paddedX, paddedX, padWidth, padWidth, padWidth, padWidth, cv::BORDER_CONSTANT,cv::Scalar(0));
    std::cout<<paddedX<<std::endl;
    int index_i = 0, index_j;
    for(int i = 0; i + padWidth < Z.cols - padWidth; i += stride){
        index_j = 0;
        for(int j = 0; j + padWidth < Z.rows - padWidth; j += stride){
            MatrixD subMat;
            X(cv::Rect(j, i, poolingWidth, poolingWidth)).copyTo(subMat);
            reduce(subMat, subMat, 0, CV_REDUCE_AVG);
            reduce(subMat, subMat, 1, CV_REDUCE_AVG);
            Z.at<double_t>(index_j, index_i) = subMat.at<double_t>(0,0);
            index_j++;
        }
        index_i++;
    }*/
    return Z;
}

MatrixD PoolingLayer::BackPropargation(MatrixD Delta_Higher, MatrixD W_Higher, MatrixD Z_lower){
    func->df(Z, Delta);
    Delta = Delta.mul(W_Higher.t() * Delta_Higher);
    //Wの更新は行わない
    return Delta;
}

MatrixD PoolingLayer::BackPropargation_Last(MatrixD Teacher, MatrixD Z_lower, ErrorFunction *erFunc){
    erFunc->df(Teacher, Z, Delta);
    return Delta;
}