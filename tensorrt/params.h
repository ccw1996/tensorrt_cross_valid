#include <iostream>
#include <vector>

class Params{
public:
    int readParam(const char*filename);
    
private:
    std::string onnxFileName;
    bool int8Enabled;
    bool fp16Enabled;
    int32_t batch{1};
    std::string dataDirs;
    int inputTensorNums;
    int outputTensorNums;
    std::vector<std::string> inputTensorNames;
    std::vector<std::string> outputTensorNames;
    std::string engineFileName;
    std::string calibFile;
    int calibType;
};