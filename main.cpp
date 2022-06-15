#include <iostream>
#include "tensorrt/tensorrt.h"

using namespace std;

int main(){
    TensorRT_Inference trt_test("D:\\code\\trt_test\\end2end.onnx","D:\\code\\trt_test\\test.engine");
    trt_test.build();
    trt_test.infer();
    return 0;
}