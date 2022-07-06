#pragma once

#include "common.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "NvOnnxParser.h"
#include "NvOnnxConfig.h"

#include <vector>
#include <string>

using namespace nvinfer1;

class TensorRT_Inference{
public:
    TensorRT_Inference(const char* model_path, const char* engine_path);
    ~TensorRT_Inference();
    bool build();
    bool infer();
    bool loadEngine();
    void run(float* input, float* output);
private:
    bool parse();
    bool prepareImage(const char* image_path);
    bool outputResult(std::string output_name);
    std::vector<Dims> getInputDims(std::vector<std::string> input_name);
    std::vector<Dims> getOutputDims(std::vector<std::string> output_name);
    void storeEngine();
    bool constructNetwork(InferUniquePtr<nvinfer1::IBuilder>& builder,
    InferUniquePtr<nvinfer1::INetworkDefinition>& network, 
    InferUniquePtr<nvinfer1::IBuilderConfig>& config,
    InferUniquePtr<nvonnxparser::IParser>& parser);

    nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
    nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.
    std::shared_ptr<nvinfer1::ICudaEngine> mEngine;
    Logger mLogger;

    std::string modelPath;
    std::string enginePath;



    void** dev_buffers_;
    cudaStream_t stream_;
    std::vector<std::string> input_name_;
    std::vector<std::string> output_name_;
    int input_index_ = 0;
};