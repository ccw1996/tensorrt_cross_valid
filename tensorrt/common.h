#pragma once

#include "NvInferRuntimeCommon.h"
#include <iostream>
#include <memory>
#include "params.h"

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

// struct Params{
//     std::string onnxFileName;
//     bool int8Enabled;
//     bool fp16Enabled;
//     int32_t batch{1};
//     std::string dataDirs;
//     int inputTensorNums;
//     int outputTensorNums;
//     std::string inputTensorNames;
//     std::string outputTensorNames;
//     std::string engineFileName;
//     std::string calibFile;
//     int calibType;
// };