#pragma once

#include "NvInferRuntimeCommon.h"
#include <iostream>
#include <memory>

struct InferDeleter{
    template <typename T>
    void operator()(T* obj) const{
        if(obj){
            obj->destroy();
        }
    }
};

template <typename T>
using InferUniquePtr = std::unique_ptr<T, InferDeleter>;

class Logger:public nvinfer1::ILogger{
    public:
        void log(Severity severity,const char* msg) noexcept override{
            if(severity==Severity::kERROR)
                std::cout<<"ERROR: "<<msg<<std::endl;
        }
    nvinfer1::ILogger &getTRTLogger() noexcept {return *this;}
};