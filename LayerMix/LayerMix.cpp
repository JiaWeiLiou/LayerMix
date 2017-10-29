// LayerMix.cpp : 定義主控台應用程式的進入點。
//

#include "stdafx.h"
#include "basic_processing.h"
#include <iostream>
#include <opencv2/opencv.hpp>  
#include <cmath>

#define UNKNOWN_FLOW_THRESH 1e9
#define PI 3.14159265359

using namespace std;
using namespace cv;

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

	Mat divideArea;
	Divide(grayImage, blurImage, divideArea);

	string divideaOutfile = filepath + "\\" + infilename + "_DIVIDEA.png";			//基於面的圖層混合模式
	imwrite(divideaOutfile, divideArea);

	Mat hardmixArea;
	HardMix(grayImage, divideArea, hardmixArea);

	string hardmixaOutfile = filepath + "\\" + infilename + "_HARDMIXA.png";			//基於面的圖層混合模式
	imwrite(hardmixaOutfile, hardmixArea);

	/*去除影像雜訊*/

	Mat clearWiteArea,clearBlackArea;
	ClearNoise(hardmixArea, clearWiteArea, 20, 4, 1);

	string clearwOutfile = filepath + "\\" + infilename + "_CLEARW.png";			//基於面的圖層混合模式
	imwrite(clearwOutfile, clearWiteArea);

	ClearNoise(clearWiteArea, clearBlackArea, 20, 4, 0);

	string clearbOutfile = filepath + "\\" + infilename + "_CLEARB.png";			//基於面的圖層混合模式
	imwrite(clearbOutfile, clearBlackArea);

	/*計算影像梯度*/

	Mat gradx,grady;		//16SC1
	Differential(grayImage, gradx, grady);

	Mat gradField;			//16SC2
	GradientField(gradx, grady, gradField);			//結合梯度場

	Mat gradm, gradd;		//32FC1
	CalculateGradient(gradField, gradm, gradd);		//計算梯度幅值及方向

	Mat gradx_out, grady_out, gradm_out, gradd_out, gradf_out;		//8UC1
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradField, gradf_out);

	string gxOutfile = filepath + "\\" + infilename + "_GX.png";			//影像梯度(水平)
	imwrite(gxOutfile, gradx_out);
	string gyOutfile = filepath + "\\" + infilename + "_GY.png";			//影像梯度(垂直)
	imwrite(gyOutfile, grady_out);
	string gmOutfile = filepath + "\\" + infilename + "_GM.png";			//影像梯度(幅值)
	imwrite(gmOutfile, gradm_out);
	string gdOutfile = filepath + "\\" + infilename + "_GD.png";			//影像梯度(方向)
	imwrite(gdOutfile, gradd_out);
	string gfOutfile = filepath + "\\" + infilename + "_GF.png";			//影像梯度(場)
	imwrite(gfOutfile, gradf_out);

	/*非最大值抑制*/

	Mat gradNMS;			//16SC2
	NonMaximumSuppression(gradField, gradNMS);

	Mat gradmNMS_out, gradfNMS_out;		//8UC1
	DrawAbsGraySystem(gradNMS, gradmNMS_out);
	DrawColorSystem(gradNMS, gradfNMS_out);

	string mNMSOutfile = filepath + "\\" + infilename + "_M_NMS.png";			//非最大值抑制(幅值)
	imwrite(mNMSOutfile, gradmNMS_out);
	string fNMSOutfile = filepath + "\\" + infilename + "_F_NMS.png";			//非最大值抑制(場)
	imwrite(fNMSOutfile, gradfNMS_out);

	/*差分混合模式*/

	Mat gradDivide;			//8UC1
	Divide(gradmNMS_out, grayImage, gradDivide);		//差分混合模式	

	string divideOutfile = filepath + "\\" + infilename + "_Divide.png";			//差分混合模式
	imwrite(divideOutfile, gradDivide);

	/*滯後閥值*/
	Mat edgeHT;		//8UC1
	HysteresisThreshold(gradDivide, edgeHT, 150, 100);
	string edgehtOutfile = filepath + "\\" + infilename + "_HT.png";			//滯後閥值
	imwrite(edgehtOutfile, edgeHT);

    return 0;
}


