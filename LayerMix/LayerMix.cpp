// LayerMix.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include <iostream>
#include <opencv2/opencv.hpp>  
#include <cmath>

#define UNKNOWN_FLOW_THRESH 1e9
#define PI 3.14159265359

using namespace std;
using namespace cv;

void LayerMix(InputArray _grayImage, OutputArray _mixImage,int blurSize);

int main()
{
	std::cout << "Please enter image path : ";
	string infile;
	std::cin >> infile;

	std::cout << "Please enter blur square size : ";
	int blurSize = 0;
	std::cin >> blurSize;

	if (blurSize % 2 == 0)
	{
		--blurSize;
	}


	/*設定輸出文件名*/

	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));							//檔案路徑
	string infilename(infile.substr(pos1 + 1, pos2 - pos1 - 1));		//檔案名稱

	/*載入原圖*/

	Mat srcImage = imread(infile);	//原始圖
	if (!srcImage.data) { printf("Oh，no，讀取srcImage錯誤~！ \n"); return false; }

	/*將原圖像轉換為灰度圖像*/

	Mat grayImage;
	if (srcImage.type() != CV_8UC1)
	{
		cvtColor(srcImage, grayImage, CV_BGR2GRAY);

		string grayoutfile = filepath + "\\" + infilename + "_0_GRAY.png";		//灰階影像
		imwrite(grayoutfile, grayImage);
	}
	else
		grayImage = srcImage;

	/*將灰度圖像中值模糊*/

	Mat mixImage;
	LayerMix(grayImage, mixImage, blurSize);
	string lmOutfile = filepath + "\\" + infilename + "_LM.png";		//補集(疊合)
	imwrite(lmOutfile, mixImage);

    return 0;
}


void LayerMix(InputArray _grayImage, OutputArray _mixImage, int blurSize)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	_mixImage.create(grayImage.size(), CV_8UC1);
	Mat mixImage = _mixImage.getMat();

	Mat blurImage;
	medianBlur(grayImage, blurImage, blurSize);

	double divide = 0;
	for (int i = 0; i < grayImage.rows; ++i)
	{
		for (int j = 0; j < grayImage.cols; ++j)
		{
			divide = (double)grayImage.at<uchar>(i, j) / (double)blurImage.at<uchar>(i, j) > 1 ? 255 : ((double)grayImage.at<uchar>(i, j) / (double)blurImage.at<uchar>(i, j)) * 255.0;
			mixImage.at<uchar>(i, j) = divide + (double)blurImage.at<uchar>(i, j) < 255.0 ? 0 : 255;
		}
	}
}
