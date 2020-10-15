#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <vector>
#include <io.h>
using namespace std;
using namespace cv;


int main()
{
	double time0 = cv::getTickCount();
	cv::dnn::Net net= cv::dnn::readNetFromONNX("crnn.onnx");
	cout << "load model cost:" << (cv::getTickCount() - time0) / cv::getTickFrequency() << endl;
	vector<string> PicNames, PicPaths;

	
	cv::Mat img;
	time0 = cv::getTickCount();
	for (int i = 0; i < 55; i++)
	{
	string imgPaths = ".//img//" + to_string(i) + ".jpg";
	img = cv::imread(imgPaths);
	cv::Mat blob = cv::dnn::blobFromImage(img, 1.0 / 255.f, cv::Size(512, 64), cv::Scalar(),true);
	net.setInput(blob);
	cv::Mat prob;
	
	
	prob = net.forward();
	//cout << "Forward time cost:" << (cv::getTickCount() - time0) / cv::getTickFrequency() << endl;

	//time0 = cv::getTickCount();
	int* max = new int[32];
	double maxVal = 0; //最大值一定要赋初值，否则运行时会报错
	Point maxLoc;
	vector<int> vi;
	for (int i = 0; i < 32; i++)
	{
		cv::Mat rowMat = prob.rowRange(i, i + 1);
		minMaxLoc(rowMat, NULL, &maxVal, NULL, &maxLoc);
		max[i] = maxLoc.x;

	}
	string alph = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,";
	string res = "";
	for (int i = 0; i < 32; i++)
	{
		int v = max[i];
		if (v < 63)
		{
			if (i > 0)
			{
				if (max[i - 1] != v)
					res += alph.substr(v, 1);
			}
			else
			{
				res += alph.substr(v,1);
			}

		}
	}
	//cout << "Parase cost:" << (cv::getTickCount() - time0) / cv::getTickFrequency() << endl;
	//cout << "Recognize Result:" << res << endl;
	}
	cout << "Recognize "<< PicNames.size()<<" Pics cost:" << (cv::getTickCount() - time0) / cv::getTickFrequency() << endl;
	cout << "done" << endl;
	//cv::imshow("Img", img);
	//cv::waitKey(0);
}
