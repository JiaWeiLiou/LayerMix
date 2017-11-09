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

	std::cout << "Please enter blur square size for Area : ";
	int blurAreaSize = 0;
	std::cin >> blurAreaSize;

	if (blurAreaSize % 2 == 0)
	{
		--blurAreaSize;
	}

	std::cout << "Please enter blur square size for Line : ";
	int blurLineSize = 0;
	std::cin >> blurLineSize;

	if (blurLineSize % 2 == 0)
	{
		--blurLineSize;
	}

	/*設定輸出文件名*/

	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));							//檔案路徑
	string infilename(infile.substr(pos1 + 1, pos2 - pos1 - 1));		//檔案名稱

	/*載入原圖*/

	Mat srcImage = imread(infile);	//原始影像(8UC1 || 8UC3 )
	if (!srcImage.data) { printf("Oh，no，讀取srcImage錯誤~！ \n"); return false; }

	/*將原圖像轉換為灰度圖像*/

	Mat grayImage;		//灰階影像(8UC1)
	if (srcImage.type() != CV_8UC1)
	{
		cvtColor(srcImage, grayImage, CV_BGR2GRAY);

		string grayoutfile = filepath + "\\" + infilename + "_0_GRAY.png";		//灰階影像
		imwrite(grayoutfile, grayImage);
	}
	else
		grayImage = srcImage;


	/****基於面的影像萃取****/

	/*將灰度圖像高斯模糊*/

	Mat blurAImage;	//模糊影像(8UC1)	
	GaussianBlur(grayImage, blurAImage, Size(blurAreaSize, blurAreaSize), 0, 0);

	string bluraoutfile = filepath + "\\" + infilename + "_1_BLURA.png";		//模糊影像
	imwrite(bluraoutfile, blurAImage);

	/*圖層混合模式*/

	Mat divideArea;		//分割混合模式(8UC1)
	Divide(grayImage, blurAImage, divideArea);

	string divideaOutfile = filepath + "\\" + infilename + "_2.1_DIVIDEA.png";			//分割混合模式
	imwrite(divideaOutfile, divideArea);

	Mat hardmixArea;	//實色印疊合混合模式(8UC1 and 二值影像)
	HardMix(grayImage, divideArea, hardmixArea);			

	string hardmixaOutfile = filepath + "\\" + infilename + "_2.2_HARDMIXA.png";			//實色印疊合混合模式
	imwrite(hardmixaOutfile, hardmixArea);

	/*去除影像雜訊*/

	Mat clearWiteArea;		//去除白色雜訊(8UC1 and 二值影像)
	ClearNoise(hardmixArea, clearWiteArea, 20, 4, 1);

	string clearwOutfile = filepath + "\\" + infilename + "_3.1_CLEARW.png";			//去除白色雜訊
	imwrite(clearwOutfile, clearWiteArea);

	Mat clearBlackArea;		//去除黑色雜訊(8UC1 and 二值影像)
	ClearNoise(clearWiteArea, clearBlackArea, 20, 4, 0);

	string clearbOutfile = filepath + "\\" + infilename + "_3.2_CLEARB.png";			//去除黑色雜訊
	imwrite(clearbOutfile, clearBlackArea);


	/****基於線的影像萃取****/

	/*將灰度圖像中值模糊*/

	Mat blurLImage;	//模糊影像(8UC1)	
	medianBlur(grayImage, blurLImage, blurLineSize);

	string blurloutfile = filepath + "\\" + infilename + "_4_BLURL.png";		//模糊影像
	imwrite(blurloutfile, blurLImage);

	/*計算影像梯度*/

	Mat gradx,grady;		//水平及垂直梯度(16SC1)
	Differential(blurLImage, gradx, grady);

	Mat gradf;			//梯度場(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;		//梯度幅值及梯度方向(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_out, grady_out, gradm_out, gradd_out, gradf_out;		//輸出用(8UC1 or 8UC3)
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradf, gradf_out);

	string gxOutfile = filepath + "\\" + infilename + "_5.1_GX.png";			//輸出用影像梯度(水平)
	imwrite(gxOutfile, gradx_out);
	string gyOutfile = filepath + "\\" + infilename + "_5.2_GY.png";			//輸出用影像梯度(垂直)
	imwrite(gyOutfile, grady_out);
	string gmOutfile = filepath + "\\" + infilename + "_5.3_GM.png";			//輸出用影像梯度(幅值)
	imwrite(gmOutfile, gradm_out);
	string gdOutfile = filepath + "\\" + infilename + "_5.4_GD.png";			//輸出用影像梯度(方向)
	imwrite(gdOutfile, gradd_out);
	string gfOutfile = filepath + "\\" + infilename + "_5.5_GF.png";			//輸出用影像梯度(場)
	imwrite(gfOutfile, gradf_out);

	/*分割混合模式*/

	Mat divideLine;										//分割混合模式(8UC1)
	Divide(gradm, blurLImage, divideLine);

	Mat gradmDivide_out, gradfDivide_out;				//輸出用(8UC1 or 8UC3)
	DrawAbsGraySystem(divideLine, gradmDivide_out);
	DrawColorSystem(divideLine, gradd, gradfDivide_out);

	string gradmDivideOutfile = filepath + "\\" + infilename + "_6.1_DIVIDEM.png";			//輸出用分割混合模式(幅值)
	imwrite(gradmDivideOutfile, gradmDivide_out);
	string gradfDivideOutfile = filepath + "\\" + infilename + "_6.2_DIVIDEF.png";			//輸出用分割混合模式(場)
	imwrite(gradfDivideOutfile, gradfDivide_out);

	/*非極大值抑制*/
	
	Mat gradmNMS, graddNMS;			//非最大值抑制(8UC1、32FC1)
	NonMaximumSuppression(divideLine, gradd, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//輸出用(8UC1 or 8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMSOutfile = filepath + "\\" + infilename + "_7.1_NMSM.png";			//非最大值抑制(幅值)
	imwrite(gradmNMSOutfile, gradmNMS_out);
	string graddNMSOutfile = filepath + "\\" + infilename + "_7.2_NMSD.png";			//非最大值抑制(方向)
	imwrite(graddNMSOutfile, graddNMS_out);
	string gradFNMSOutfile = filepath + "\\" + infilename + "_7.3_NMSF.png";			//非最大值抑制(場)
	imwrite(gradFNMSOutfile, gradfNMS_out);

	/*清除異方向點*/
	
	Mat gradmCDD, graddCDD;			//清除異方向點(8UC1、32FC1)
	ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//輸出用(8UC1 or 8UC3)
	DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	DrawColorSystem(graddCDD, graddCDD_out);
	DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	string gradmCDDOutfile = filepath + "\\" + infilename + "_8.1_CDDM.png";			//清除異方向點(幅值)
	imwrite(gradmCDDOutfile, gradmCDD_out);
	string graddCDDOutfile = filepath + "\\" + infilename + "_8.2_CDDD.png";			//清除異方向點(方向)
	imwrite(graddCDDOutfile, graddCDD_out);
	string gradFCDDOutfile = filepath + "\\" + infilename + "_8.3_CDDF.png";			//清除異方向點(場)
	imwrite(gradFCDDOutfile, gradfCDD_out);

	/*初步斷線連通*/

	Mat gradmWCBL, graddWCBL;			//斷線連通(8UC1、32FC1)
	ConnectBreakLine(gradmCDD, graddCDD, gradmWCBL, graddWCBL, 2, 3, 60, 1);

	Mat gradmWCBL_out, graddWCBL_out, gradfWCBL_out;		//輸出用(8UC1 or 8UC3)
	DrawAbsGraySystem(gradmWCBL, gradmWCBL_out);
	DrawColorSystem(graddWCBL, graddWCBL_out);
	DrawColorSystem(gradmWCBL, graddWCBL, gradfWCBL_out);

	string gradmWCBLOutfile = filepath + "\\" + infilename + "_9.1_WCBLM.png";			//初步斷線連通(幅值)
	imwrite(gradmWCBLOutfile, gradmWCBL_out);
	string graddWCBLOutfile = filepath + "\\" + infilename + "_9.2_WCBLD.png";			//初步斷線連通(方向)
	imwrite(graddWCBLOutfile, graddWCBL_out);
	string gradWFCBLOutfile = filepath + "\\" + infilename + "_9.3_WCBLF.png";			//初步斷線連通(場)
	imwrite(gradWFCBLOutfile, gradfWCBL_out);

	/*滯後閥值*/

	Mat edgeHT;		//滯後閥值(8UC1 and 二值影像)
	HysteresisThreshold(gradmWCBL, edgeHT, 150, 50);
	string edgehtOutfile = filepath + "\\" + infilename + "_10_HT.png";			//滯後閥值
	imwrite(edgehtOutfile, edgeHT);

	/*強制斷線連通*/

	Mat edgeFCBL;		//強制斷線連通(8UC1)
	BWConnectBreakLine(gradmWCBL, graddWCBL, edgeHT, edgeFCBL, 2, 5, 90, 0);
	string edgefcblOutfile = filepath + "\\" + infilename + "_11_FCBL.png";			//強制斷線連通
	imwrite(edgefcblOutfile, edgeFCBL);

    return 0;
}


