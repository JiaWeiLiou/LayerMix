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

	if (blurAreaSize % 2 == 0) { --blurAreaSize; }

	std::cout << "Please enter blur square size for Line : ";
	int blurLineSize = 0;
	std::cin >> blurLineSize;

	if (blurLineSize % 2 == 0) { --blurLineSize; }

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

		string gray_outfile = filepath + "\\" + infilename + "_0_GRAY.png";		//灰階影像
		imwrite(gray_outfile, grayImage);
	}
	else
		grayImage = srcImage;


	/****基於面的影像萃取****/

	/*將灰度圖像高斯模糊*/

	Mat blurImage;	//模糊影像(8UC1)	
	GaussianBlur(grayImage, blurImage, Size(blurAreaSize, blurAreaSize), 0, 0);

	string blurA_outfile = filepath + "\\" + infilename + "_1_BLURA.png";		//模糊灰階影像
	imwrite(blurA_outfile, blurImage);

	/*圖層混合模式*/

	Mat divideArea;		//分割混合模式(8UC1)
	DivideArea(grayImage, blurImage, divideArea);

	string divideA_outfile = filepath + "\\" + infilename + "_2.1_DIVIDEA.png";			//面分割混合模式
	imwrite(divideA_outfile, divideArea);

	Mat hardmixArea;	//實色印疊合混合模式(8UC1(BW))
	HardMix(grayImage, divideArea, hardmixArea);

	string hardmixA_outfile = filepath + "\\" + infilename + "_2.2_HARDMIXA.png";		//實色印疊合混合模式
	imwrite(hardmixA_outfile, hardmixArea);

	/*去除影像雜訊*/

	Mat clearWiteArea;		//去除白色雜訊(8UC1(BW))
	ClearNoise(hardmixArea, clearWiteArea, 20, 4, 1);

	string clearWA_outfile = filepath + "\\" + infilename + "_3.1_CLEARW.png";			//去除白色雜訊
	imwrite(clearWA_outfile, clearWiteArea);

	Mat clearBlackArea;		//去除黑色雜訊(8UC1(BW))
	ClearNoise(clearWiteArea, clearBlackArea, 20, 4, 0);

	string clearBA_outfile = filepath + "\\" + infilename + "_3.2_CLEARB.png";			//去除黑色雜訊
	imwrite(clearBA_outfile, clearBlackArea);


	/****基於線的影像萃取****/

	/*計算影像梯度*/

	Mat gradx, grady;		//水平及垂直梯度(16SC1)
	Differential(grayImage, gradx, grady);

	Mat gradf;			//梯度場(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;		//梯度幅值及梯度方向(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_out, grady_out, gradm_out, gradd_out, gradf_out;		//輸出用(8UC1、8UC1、8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradf, gradf_out);

	string gradx_outfile = filepath + "\\" + infilename + "_4.1_GX.png";			//輸出用影像梯度(水平)
	imwrite(gradx_outfile, gradx_out);
	string grady_outfile = filepath + "\\" + infilename + "_4.2_GY.png";			//輸出用影像梯度(垂直)
	imwrite(grady_outfile, grady_out);
	string gradm_outfile = filepath + "\\" + infilename + "_4.3_GM.png";			//輸出用影像梯度(幅值)
	imwrite(gradm_outfile, gradm_out);
	string gradd_outfile = filepath + "\\" + infilename + "_4.4_GD.png";			//輸出用影像梯度(方向)
	imwrite(gradd_outfile, gradd_out);
	string gradf_outfile = filepath + "\\" + infilename + "_4.5_GF.png";			//輸出用影像梯度(場)
	imwrite(gradf_outfile, gradf_out);

	/*方向模糊*/

	Mat graddBlur;	//模糊方向(8UC1)	
	BlurDirection(gradd, graddBlur, 11);

	Mat graddBlur_out, gradfBlur_out;		//輸出用(8UC3、8UC3)
	DrawColorSystem(graddBlur, graddBlur_out);
	DrawColorSystem(gradm, graddBlur, gradfBlur_out);

	string graddBlur_outfile = filepath + "\\" + infilename + "_5.1_BLURD.png";			//模糊方向(方向)
	imwrite(graddBlur_outfile, graddBlur_out);
	string gradfBlur_outfile = filepath + "\\" + infilename + "_5.2_BLURF.png";			//模糊方向(場)
	imwrite(gradfBlur_outfile, gradfBlur_out);

	/*將幅值高斯模糊*/

	Mat gradmBlur;	//模糊幅值(8UC1)	
	GaussianBlur(gradm, gradmBlur, Size(blurLineSize, blurLineSize), 0, 0);

	string blurM_outfile = filepath + "\\" + infilename + "_6_BLURM.png";			//模糊幅值
	imwrite(blurM_outfile, gradmBlur);

	/*線分割混合模式*/

	Mat gradmDivide;										//線分割混合模式(8UC1)
	DivideLine(gradm, gradmBlur, gradmDivide);

	Mat gradmDivide_out, gradfDivide_out;		//輸出用(8UC1、8UC3)
	DrawAbsGraySystem(gradmDivide, gradmDivide_out);
	DrawColorSystem(gradmDivide, graddBlur, gradfDivide_out);

	string gradmDivide_outfile = filepath + "\\" + infilename + "_7.1_DIVIDEM.png";			//線分割混合模式(幅值)
	imwrite(gradmDivide_outfile, gradmDivide_out);
	string gradfDivide_outfile = filepath + "\\" + infilename + "_7.2_DIVIDEF.png";			//線分割混合模式(場)
	imwrite(gradfDivide_outfile, gradfDivide_out);

	/*非極大值抑制*/

	Mat gradmNMS, graddNMS;			//非最大值抑制(8UC1、32FC1)
	NonMaximumSuppression(gradmDivide, graddBlur, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMS_outfile = filepath + "\\" + infilename + "_8.1_NMSM.png";			//非最大值抑制(幅值)
	imwrite(gradmNMS_outfile, gradmNMS_out);
	string graddNMS_outfile = filepath + "\\" + infilename + "_8.2_NMSD.png";			//非最大值抑制(方向)
	imwrite(graddNMS_outfile, graddNMS_out);
	string gradfNMS_outfile = filepath + "\\" + infilename + "_8.3_NMSF.png";			//非最大值抑制(場)
	imwrite(gradfNMS_outfile, gradfNMS_out);

	/*清除異方向點*/

	Mat gradmCDD, graddCDD;			//清除異方向點(8UC1、32FC1)
	ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	DrawColorSystem(graddCDD, graddCDD_out);
	DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	string gradmCDD_outfile = filepath + "\\" + infilename + "_9.1_CDDM.png";			//清除異方向點(幅值)
	imwrite(gradmCDD_outfile, gradmCDD_out);
	string graddCDD_outfile = filepath + "\\" + infilename + "_9.2_CDDD.png";			//清除異方向點(方向)
	imwrite(graddCDD_outfile, graddCDD_out);
	string gradfCDD_outfile = filepath + "\\" + infilename + "_9.3_CDDF.png";			//清除異方向點(場)
	imwrite(gradfCDD_outfile, gradfCDD_out);

	/*斷線連通*/

	Mat gradmCBL, graddCBL;			//斷線連通(8UC1、32FC1)
	ConnectBreakLine(gradmCDD, graddCDD, gradmCBL, graddCBL, 2, 5, 60, 0, 0);

	Mat gradmCBL_out, graddCBL_out, gradfCBL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmCBL, gradmCBL_out);
	DrawColorSystem(graddCBL, graddCBL_out);
	DrawColorSystem(gradmCBL, graddCBL, gradfCBL_out);

	string gradmCBL_outfile = filepath + "\\" + infilename + "_10.1_CBLM.png";			//斷線連通(幅值)
	imwrite(gradmCBL_outfile, gradmCBL_out);
	string graddCBL_outfile = filepath + "\\" + infilename + "_10.2_CBLD.png";			//斷線連通(方向)
	imwrite(graddCBL_outfile, graddCBL_out);
	string gradfCBL_outfile = filepath + "\\" + infilename + "_10.3_CBLF.png";			//斷線連通(場)
	imwrite(gradfCBL_outfile, gradfCBL_out);

	/*滯後閥值*/

	Mat lineHT;		//滯後閥值(8UC1(BW))
	HysteresisThreshold(gradmCBL, lineHT, 200, 2);
	string LHT_outfile = filepath + "\\" + infilename + "_11_HT.png";			//滯後閥值
	imwrite(LHT_outfile, lineHT);

	/*對稱端點連通*/

	Mat gradmSCBL, graddSCBL, lineSCBL;			//短對稱端點連通(8UC1、32FC1、8UC1(BW))
	BWConnectBreakLine(gradmCBL, graddCBL, lineHT, gradmSCBL, graddSCBL, lineSCBL, 2, 20, 90, 0, 0);

	Mat gradmSCBL_out, graddSCBL_out, gradfSCBL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmSCBL, gradmSCBL_out);
	DrawColorSystem(graddSCBL, graddSCBL_out);
	DrawColorSystem(gradmSCBL, graddSCBL, gradfSCBL_out);

	string LSCBL_outfile = filepath + "\\" + infilename + "_12.0_SCBL.png";				//對稱端點連通(二值)
	imwrite(LSCBL_outfile, lineSCBL);
	string gradmSCBL_outfile = filepath + "\\" + infilename + "_12.1_SCBLM.png";			//對稱端點連通(幅值)
	imwrite(gradmSCBL_outfile, gradmSCBL_out);
	string graddSCBL_outfile = filepath + "\\" + infilename + "_12.2_SCBLD.png";			//對稱端點連通(方向)
	imwrite(graddSCBL_outfile, graddSCBL_out);
	string gradfSCBL_outfile = filepath + "\\" + infilename + "_12.3_SCBLF.png";			//對稱端點連通(場)
	imwrite(gradfSCBL_outfile, gradfSCBL_out);

	/*去除孤立點*/

	Mat lineCIP;	//去除孤立點(8UC1(BW))
	ClearSpecialPoint(lineSCBL, lineCIP, 0, 1, 0);
	string LCIP_outfile = filepath + "\\" + infilename + "_13_CIP.png";			//去除孤立點
	imwrite(LCIP_outfile, lineCIP);

	/*強制端點連通*/

	Mat gradmACBL, graddACBL, lineACBL;			//短對稱端點連通(8UC1、32FC1、8UC1(BW))
	BWConnectBreakLine(gradmSCBL, graddSCBL, lineCIP, gradmACBL, graddACBL, lineACBL, 2, 5, 90, 1, 1);

	Mat gradmACBL_out, graddACBL_out, gradfACBL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmACBL, gradmACBL_out);
	DrawColorSystem(graddACBL, graddACBL_out);
	DrawColorSystem(gradmACBL, graddACBL, gradfACBL_out);

	string LACBL_outfile = filepath + "\\" + infilename + "_14.0_ACBL.png";				//強制端點連通(二值)
	imwrite(LACBL_outfile, lineACBL);
	string gradmACBL_outfile = filepath + "\\" + infilename + "_14.1_ACBLM.png";			//強制端點連通(幅值)
	imwrite(gradmACBL_outfile, gradmACBL_out);
	string graddACBL_outfile = filepath + "\\" + infilename + "_14.2_ACBLD.png";			//強制端點連通(方向)
	imwrite(graddACBL_outfile, graddACBL_out);
	string gradfACBL_outfile = filepath + "\\" + infilename + "_14.3_ACBLF.png";			//強制端點連通(場)
	imwrite(gradfACBL_outfile, gradfACBL_out);

	/*去除雜線*/

	Mat lineCNL;	//去除雜線(8UC1(BW))
	//ClearSpecialPoint(lineACBL, lineCNL, blurLineSize, 5, 1);
	ClearNoise(lineACBL, lineCNL, 20, 8, 1);
	string LCNL_outfile = filepath + "\\" + infilename + "_15_CNL.png";			//去除雜線
	imwrite(LCNL_outfile, lineCNL);

	///*分水嶺演算法切割*/

	return 0;
}


