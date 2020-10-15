#include <fstream>
#include <vector>
#include <memory>
#include <string>
#include <map>
#include <condition_variable>
#include <mutex>
#include <inference_engine.hpp>
#include <format_reader_ptr.h>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
#include <samples/classification_results.h>
#include <sys/stat.h>
#include <opencv2/opencv.hpp>
#include <iostream>
using namespace cv;
using namespace InferenceEngine;


void main()
{
	Core ie;
	CNNNetwork network = ie.ReadNetwork("crnn.xml","crnn.bin");
	network.setBatchSize(1);
	InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
	std::string input_name = network.getInputsInfo().begin()->first;

	input_info->getPreProcess().setResizeAlgorithm(RESIZE_BILINEAR);
	input_info->setLayout(Layout::NCHW);
	input_info->setPrecision(Precision::FP32);
	DataPtr output_info = network.getOutputsInfo().begin()->second;
	std::string out_name = network.getOutputsInfo().begin()->first;
	output_info->setPrecision(Precision::FP32);

	ExecutableNetwork executable_network = ie.LoadNetwork(network,"CPU");
	InferRequest infer_request = executable_network.CreateInferRequest();
	cv::Mat image = imread("TestImg.jpg");
	cv::Mat dst;
	


	auto input = infer_request.GetBlob(input_name);
	size_t num_channels = input->getTensorDesc().getDims()[1];
	size_t h = input->getTensorDesc().getDims()[2];
	size_t w = input->getTensorDesc().getDims()[3];
	size_t image_size = h * w;
	cv::resize(image, dst, cv::Size(w, w));
	float* data = static_cast<float*>(input->buffer());
	for (size_t row = 0; row < h; row++) {
		for (size_t col = 0; col < w; col++) {
			for (size_t ch = 0; ch < num_channels; ch++) {
				data[image_size*ch + row * w + col] = ((float)dst.at<Vec3b>(row, col)[ch])/255;
			}
		}
	}
	infer_request.Infer();

	auto output = infer_request.GetBlob(out_name);
	float* buff = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output->buffer());
	const int c = output->getTensorDesc().getDims()[0];
	const int l = output->getTensorDesc().getDims()[1];
	std::string result = "";
	std::string alph = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-:";
	for (int i = 0; i < c; i++)
	{
		int index = 0;
		double max = 0;
		for (int j = 0; j < l; j++)
		{
			double v = buff[i*l + j];
			std::cout << v << "--";
			if (v > max)
			{
				max = v;
				index = j;
			}
		}
		std::string vs = alph.substr(index+8888, 1);
		result += alph.substr(index, 1);
	}
	


	std::cout << "done";
}
