#include "params.h"
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept>
#include <iostream>

int Params::readParam(const char*filename){
    std::stringstream buffer;
    std::string line;
    std::string paramName;
    int paramValue=0;
    std::string paramValuestr;

    std::ifstream fin(filename);
    if(!fin.good()){
        std::string msg("file read failed");
        throw std::runtime_error(msg);
    }
    while(fin.good()){
        getline(fin,line);
        if(line[0]!='#'){
            buffer.clear();
            buffer<<line;
            buffer>>paramName;
            if(paramName.compare("onnxFile:")==0){
                buffer>>paramValuestr;
                onnxFileName=paramValuestr;
            }else if(paramName.compare("int8:")==0){
                buffer>>paramValuestr;
                if(paramValuestr=="false"||paramValuestr=="0"){
                    int8Enabled=0;
                }else if (paramValuestr == "true" || paramValuestr == "1") {
                    int8Enabled=1;
                }
                else {
                    throw std::runtime_error("must be true or false");
                }
            }
            else if (paramName.compare("fp16:") == 0) {
                buffer >> paramValuestr;
                if (paramValuestr == "false" || paramValuestr == "0") {
                    fp16Enabled = 0;
                }
                else if (paramValuestr == "true" || paramValuestr == "1") {
                    fp16Enabled = 1;
                }
                else {
                    throw std::runtime_error("must be true or false");
                }
            }
            else if (paramName.compare("batch:") == 0) {
                buffer >> paramValuestr;
                batch = stoi(paramValuestr);
            }
            else if (paramName.compare("dataDirs:") == 0) {
                buffer >> paramValuestr;
                dataDirs = paramValuestr;
            }
            else if (paramName.compare("calibFile:") == 0) {
                buffer >> paramValuestr;
                calibFile = paramValuestr;
            }
            else if (paramName.compare("engineFileName:") == 0) {
                buffer >> paramValuestr;
                engineFileName = paramValuestr;
            }
            else if (paramName.compare("inputTensorNames:") == 0) {
                buffer >> paramValuestr;
                std::stringstream ss(paramValuestr);
                std::string str;
                while (getline(ss, str, ',')) {
                    inputTensorNames.push_back(str);
                }
            }
            else if (paramName.compare("outputTensorNames:") == 0) {
                buffer >> paramValuestr;
                std::stringstream ss(paramValuestr);
                std::string str;
                while (getline(ss, str, ',')) {
                    outputTensorNames.push_back(str);
                }
            }
            else {
                throw std::runtime_error("the format must be params: xxx, such as \"onnxFile: path \"");
            }
        }
    }
    fin.close();
    return 1;
}

// int main(){
//     Params params;
//     params.readParam("D:\\code\\cpp_learning\\cpp_feature\\test.txt");
//     return 0;
// }