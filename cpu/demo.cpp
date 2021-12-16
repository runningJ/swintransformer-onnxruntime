#include <iostream>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>
#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <numeric>
#include <chrono>

using namespace std;
using namespace cv;
using namespace Ort;


template <typename T>
T vectorProduct(const std::vector<T>& v)
{
    return accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
};

int main(int argc,char**argv)
{
    if (argc != 3)
    {
        cerr<<"usage "<< argv[0] <<" image_path model_path"<<endl;
        return 0;
    }
    cv::Mat image = imread(argv[1]);
    if(image.empty())
    {
        cerr <<"input image has problem "<< argv[1]<<endl;
        return 0;
    }

    string model_path = argv[2];

    Env env;
    SessionOptions options{nullptr};
    Session session(env, model_path.c_str(),options);

    size_t numInputNodes = session.GetInputCount();
    size_t numOutputNodes = session.GetOutputCount();

    std::cout << "Number of Input Nodes: " << numInputNodes << std::endl;
    std::cout << "Number of Output Nodes: " << numOutputNodes << std::endl;

    AllocatorWithDefaultOptions allocator;
    const char* inputName = session.GetInputName(0, allocator);
    std::cout << "Input Name: " << inputName << std::endl;
    TypeInfo inputTypeInfo = session.GetInputTypeInfo(0);
    auto inputTensorInfo = inputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType inputType = inputTensorInfo.GetElementType();
    std::vector<int64_t> inputDims = inputTensorInfo.GetShape();
    std::cout << "Input Dimensions: ";
    for(int i = 0; i < inputDims.size(); ++i)
    {
        cout<< inputDims[i]<<" ";
    }
    cout <<endl;
    cout <<"-----------------------------------------"<<endl;
    const char* outputName = session.GetOutputName(0, allocator);
    cout << "Output Name: " << outputName << std::endl;
    TypeInfo outputTypeInfo = session.GetOutputTypeInfo(0);
    auto outputTensorInfo = outputTypeInfo.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType outputType = outputTensorInfo.GetElementType();
    std::vector<int64_t> outputDims = outputTensorInfo.GetShape();
    std::cout << "Output Dimensions: ";
    for(int i = 0; i < outputDims.size(); ++i)
    {
        cout<< outputDims[i]<<" ";
    }
    cout <<endl;

    //data preprocess
    cv::Mat resizedImageBGR, resizedImageRGB, resizedImage, preprocessedImage;
    cv::resize(image, resizedImageBGR,cv::Size(inputDims.at(2), inputDims.at(3)));
    resizedImageRGB = resizedImageBGR;
    //cv::cvtColor(resizedImageBGR, resizedImageRGB,cv::ColorConversionCodes::COLOR_BGR2RGB);
    resizedImageRGB.convertTo(resizedImage, CV_32F, 1.0 / 255);
    cv::Mat channels[3];
    cv::split(resizedImage, channels);
    channels[0] = (channels[0] - 0.485) / 0.229;
    channels[1] = (channels[1] - 0.456) / 0.224;
    channels[2] = (channels[2] - 0.406) / 0.225;
    cv::merge(channels, 3, resizedImage);
    cv::dnn::blobFromImage(resizedImage, preprocessedImage);

    size_t inputTensorSize = vectorProduct(inputDims);
    std::vector<float> inputTensorValues(inputTensorSize);
    inputTensorValues.assign(preprocessedImage.begin<float>(),
                             preprocessedImage.end<float>());

    size_t outputTensorSize = vectorProduct(outputDims);
    std::vector<float> outputTensorValues(outputTensorSize);

    std::vector<const char*> inputNames{inputName};
    std::vector<const char*> outputNames{outputName};
    
    std::vector<Value> inputTensors;
    std::vector<Value> outputTensors;

    MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(
        OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);

    inputTensors.push_back(Value::CreateTensor<float>(
        memoryInfo, inputTensorValues.data(), inputTensorSize, inputDims.data(),
        inputDims.size()));

    outputTensors.push_back(Value::CreateTensor<float>(
        memoryInfo, outputTensorValues.data(), outputTensorSize,
        outputDims.data(), outputDims.size()));
    for(int i = 0; i < 100; ++i)
    {
        auto s_t=std::chrono::steady_clock::now();
        session.Run(Ort::RunOptions{nullptr}, inputNames.data(),
                    inputTensors.data(), 1, outputNames.data(),
                    outputTensors.data(), 1);
        auto e_t=std::chrono::steady_clock::now();
        double dr_s=std::chrono::duration<double,std::milli>(e_t-s_t).count();
        cout <<"runing inference cost time "<< dr_s <<"ms"<<endl;
    }

     for(int j = 0; j < 10; ++j)
     {
         cout << outputTensorValues.at(j)<<endl;
    }
    return 0;
}