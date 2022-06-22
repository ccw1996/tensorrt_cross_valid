#include "tensorrt.h"
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>


cv::Mat preprocess_img(cv::Mat& img,int input_w,int input_h){
    int w,h,x,y;
    float r_w=input_w/(img.cols*1.0);
    float r_h=input_h/(img.rows*1.0);
    if(r_h>r_w){
        w=input_w;
        h=r_w*img.rows;
        x=0;
        y=(input_h-h)/2;
    }else{
        w= r_h * img.cols;
        h=input_h;
        x=(input_w-w)/2;
        y=0;
    }
    cv::Mat img_resize(h,w,CV_8UC3);
    cv::resize(img,img_resize,img_resize.size(),0,0,cv::INTER_LINEAR);
    cv::Mat out(input_h,input_w,CV_8UC3);
    img_resize.copyTo(out(cv::Rect(x,y,img_resize.cols,img_resize.rows)));
    return out;
}

//FIXME: check glooger
TensorRT_Inference::TensorRT_Inference(const char* model_path, const char* engine_path){
    modelPath=model_path;
    enginePath=engine_path;
    mEngine=nullptr;
    input_name_.push_back("input");
    output_name_.push_back("output");
    dev_buffers_=static_cast<void**>(malloc(sizeof(void*)*(output_name_.size()+input_name_.size())));
    //FIXME: size and buffer need change
    //engine_->getBindingIndex(input_name_[0].c_str(),&input_index_);
    //for(int i=0;i<output_name_.size();i++){
        cudaMalloc(&dev_buffers_[1],1000*sizeof(float));
    //}
    for(int i=0;i<input_name_.size();i++){
        cudaMalloc(&dev_buffers_[i], 224 * 224* 3 * sizeof(float));
    }
}

TensorRT_Inference::~TensorRT_Inference(){

}

bool TensorRT_Inference::build(){
    auto builder=InferUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(mLogger));
    if(!builder){
        return false;
    }

    //dynamic range
    const auto explicitBatch=1U<<static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    auto network=InferUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if(!network){
        return false;
    }

    auto config=InferUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if(!config){
        return false;
    }

    auto parser=InferUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, mLogger));
    if(!parser){
        return false;
    }
    auto constructed =constructNetwork(builder, network, config, parser);
    if(!constructed){
        return false;
    }
    //FIXME: CUDA stream used for profiling by the builder.

    InferUniquePtr<IHostMemory> engine_data{builder->buildSerializedNetwork(*network, *config)};
    if(!engine_data){
        return false;
    }
    
    //FIXME: glogger
    InferUniquePtr<IRuntime> runtime{nvinfer1::createInferRuntime(mLogger)};
    if(!runtime){
        return false;
    }

    mEngine=std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engine_data->data(), engine_data->size()));
    if(!mEngine){
        return false;
    }
    storeEngine();
    mInputDims=network->getInput(0)->getDimensions();
    mOutputDims=network->getOutput(0)->getDimensions();
    return true;
}

bool TensorRT_Inference::constructNetwork(InferUniquePtr<nvinfer1::IBuilder>& builder,
    InferUniquePtr<nvinfer1::INetworkDefinition>& network, 
    InferUniquePtr<nvinfer1::IBuilderConfig>& config,
    InferUniquePtr<nvonnxparser::IParser>& parser){
    
    //FIXME: glogger
    auto parsed=parser->parseFromFile(modelPath.c_str(), 0);
    if(!parsed){
        return false;
    }
    // 1mb
    config->setMaxWorkspaceSize(1<<20);
    
    // config->setFlag(BuilderFlag::kFP16);
    // config->setFlag(BuilderFlag::kINT8);
    // setAllDynamicRanges
    return true;
}

bool TensorRT_Inference::infer(){

    auto context=InferUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    if(!context){
        return false;
    }

    prepareImage("D:/code/trt_test/image02.png");
    //HACK: change euqueue to execute
    bool status=context->executeV2(dev_buffers_);
    if(!status){
        return false;
    }
    outputResult("output");
}

void TensorRT_Inference::run(float* input, float* output){
    return;
}

bool TensorRT_Inference::prepareImage(const char* image_path){
    const int inputC=mInputDims.d[1];
    const int inputH=mInputDims.d[2];
    const int inputW=mInputDims.d[3];
    cv::Mat image=cv::imread(image_path);
    cv::Mat pr_img=preprocess_img(image,inputW,inputH);

    std::vector<cv::Mat> split_mat;
    cv::split(pr_img,split_mat);
    uint8_t* b_data=split_mat[0].data;
    uint8_t* g_data=split_mat[1].data;
    uint8_t* r_data=split_mat[2].data;

    float* host_data=new float[inputC * inputH * inputW];
    for(int i=0;i<inputH * inputW;i++){
        host_data[i]=b_data[i]/255.0;
        host_data[i+inputH*inputW]=g_data[i]/255.0;
        host_data[i+2*inputH*inputW]=r_data[i]/255.0;
    }
    cudaMemcpy(static_cast<float*>(dev_buffers_[input_index_]),host_data, inputC * inputH * inputW *sizeof(float),cudaMemcpyHostToDevice);
    delete[] host_data;
    return true;
}
bool TensorRT_Inference::outputResult(std::string output_name) {
    std::vector<float*> host_output(1);
    host_output[0]=new float[1000];
    cudaMemcpy(host_output[0],dev_buffers_[1],1000*sizeof(float),cudaMemcpyDeviceToHost);
    float max_value=-1;
    int max_index=-1;
    for(int i=0;i<1000;i++){
        if(host_output[0][i]>max_value){
            max_value=host_output[0][i];
            max_index=i;
        }
    }
    std::cout<<"max_value:"<<max_value<<" max_index:"<<max_index<<std::endl;
    delete[] host_output[0];
    return true;
}
std::vector<Dims> TensorRT_Inference::getInputDims(std::vector<std::string> input_name) {
    std::vector<Dims> result;
    return result;
}
std::vector<Dims> TensorRT_Inference::getOutputDims(std::vector<std::string> output_name) {
    std::vector<Dims> result;
    return result;
}

void TensorRT_Inference::storeEngine() {
    std::ofstream engineFile(enginePath, std::ios::binary);
    if(!engineFile){
        std::cout<<"open engine file failed"<<std::endl;
        return;
    }
    IHostMemory* engine_data=mEngine->serialize();
    if(!engine_data){
        std::cout<<"serialize engine failed"<<std::endl;
        return;
    }
    engineFile.write(static_cast<char*>(engine_data->data()),engine_data->size());
    if(!engineFile.fail()){
        std::cout<<"store engine success"<<std::endl;
        return;
    }
    return;
}
bool TensorRT_Inference::loadEngine() {
    std::ifstream engineFile(enginePath, std::ios::binary);
    if(!engineFile.good()){
        std::cout<<"engine file not good"<<std::endl;
        return false;
    }
    engineFile.seekg(0, std::ifstream::end);
    int64_t fsize = engineFile.tellg();
    engineFile.seekg(0, std::ifstream::beg);
    
    std::vector<char> engineData(fsize);
    engineFile.read(engineData.data(), fsize);
    if(!engineFile.good()){
        std::cout<<"engine file not good"<<std::endl;
        return false;
    }
    InferUniquePtr<IRuntime> runtime{nvinfer1::createInferRuntime(mLogger)};
    mEngine=std::shared_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(engineData.data(), engineData.size()));
    mInputDims=mEngine->getBindingDimensions(0);
    mOutputDims=mEngine->getBindingDimensions(1);
    return mEngine!=nullptr;
}