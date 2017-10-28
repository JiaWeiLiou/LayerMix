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

void LayerMix(InputArray _grayImage, InputArray _blurImage, OutputArray _mixImage);
void Differential(InputArray _grayImage, OutputArray _grad_x, OutputArray _grad_y, OutputArray _grad_mag);
void DrawAbsGraySystem(InputArray _field, OutputArray _grayField);

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
	Mat blurImage;
	GaussianBlur(grayImage, blurImage, Size(blurSize, blurSize), 0, 0);

	string bluroutfile = filepath + "\\" + infilename + "_1_BLUR.png";		//模糊影像
	imwrite(bluroutfile, blurImage);

	/*基於面的圖層混合模式*/

	Mat mixImageArea;
	LayerMix(grayImage, blurImage, mixImageArea);
	string lmaOutfile = filepath + "\\" + infilename + "_LMA.png";			//基於面的圖層混合模式
	imwrite(lmaOutfile, mixImageArea);

	/*計算影像梯度*/
	Mat gradx,grady,gradm;
	Differential(grayImage, gradx, grady, gradm);
	DrawAbsGraySystem(gradx, gradx);
	DrawAbsGraySystem(grady, grady);
	DrawAbsGraySystem(gradm, gradm);

	string gxOutfile = filepath + "\\" + infilename + "_Gx.png";			//影像梯度(水平)
	imwrite(gxOutfile, gradx);
	string gyOutfile = filepath + "\\" + infilename + "_Gy.png";			//影像梯度(垂直)
	imwrite(gyOutfile, grady);
	string gmOutfile = filepath + "\\" + infilename + "_Gm.png";			//影像梯度(幅值)
	imwrite(gmOutfile, gradm);

	/*梯度分割混合*/

    return 0;
}


void LayerMix(InputArray _grayImage, InputArray _blurImage, OutputArray _mixImage)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	Mat blurImage = _blurImage.getMat();
	CV_Assert(blurImage.type() == CV_8UC1);

	_mixImage.create(grayImage.size(), CV_8UC1);
	Mat mixImage = _mixImage.getMat();

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

void Differential(InputArray _grayImage, OutputArray _grad_x, OutputArray _grad_y, OutputArray _grad_mag) {

	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	_grad_x.create(grayImage.size(), CV_8UC1);
	Mat grad_x = _grad_x.getMat();

	_grad_y.create(grayImage.size(), CV_8UC1);
	Mat grad_y = _grad_y.getMat();

	_grad_mag.create(grayImage.size(), CV_8UC1);
	Mat grad_mag = _grad_mag.getMat();

	Mat grayImageRef;
	copyMakeBorder(grayImage, grayImageRef, 1, 1, 1, 1, BORDER_REPLICATE);
	float gradx_temp = 0;
	float grady_temp = 0;
	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j) {
			gradx_temp = ((float)grayImageRef.at<uchar>(i + 1, j) - (float)grayImageRef.at<uchar>(i + 1, j + 2))*0.5;
			grady_temp = ((float)grayImageRef.at<uchar>(i, j + 1) - (float)grayImageRef.at<uchar>(i + 2, j + 1))*0.5;

			grad_x.at<uchar>(i, j) = abs(gradx_temp);
			grad_y.at<uchar>(i, j) = abs(grady_temp);
			grad_mag.at<uchar>(i, j) = sqrt(pow(gradx_temp, 2) + pow(grady_temp, 2));
		}
}

void DrawAbsGraySystem(InputArray _field, OutputArray _grayField)
{
	Mat field = _field.getMat();

	_grayField.create(field.size(), CV_8UC1);
	Mat grayField = _grayField.getMat();

	// determine motion range:  
	double maxvalue = 0;

	// Find max flow to normalize fx and fy  
	for (int i = 0; i < field.rows; ++i)
		for (int j = 0; j < field.cols; ++j)
			maxvalue = maxvalue > field.at<uchar>(i, j) ? maxvalue : field.at<uchar>(i, j);

	for (int i = 0; i < field.rows; ++i)
		for (int j = 0; j < field.cols; ++j) {
			grayField.at<uchar>(i, j) = ((double)field.at<uchar>(i, j) / maxvalue) * 255;
		}
}