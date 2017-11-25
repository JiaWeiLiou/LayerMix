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

		string gray_outfile = filepath + "\\" + infilename + "_0.0_GRAY.png";		//灰階影像(灰階)
		imwrite(gray_outfile, grayImage);
	}
	else { grayImage = srcImage; }

	Mat grayColor_out;		//輸出用(8UC3)
	DrawColorbar(grayImage, grayColor_out);

	string grayColor_outfile = filepath + "\\" + infilename + "_0.1_GRAY.png";		//灰階影像(彩色)
	imwrite(grayColor_outfile, grayColor_out);

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

	Mat gradx_out, grady_out, gradm_out, gradc_out,gradd_out, gradf_out;		//輸出用(8UC1、8UC1、8UC1、8UC3、8UC3、8UC3)
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorBar(gradm, gradc_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradf, gradf_out);

	string gradx_outfile = filepath + "\\" + infilename + "_4.1_GX.png";			//輸出用影像梯度(水平)
	imwrite(gradx_outfile, gradx_out);
	string grady_outfile = filepath + "\\" + infilename + "_4.2_GY.png";			//輸出用影像梯度(垂直)
	imwrite(grady_outfile, grady_out);
	string gradm_outfile = filepath + "\\" + infilename + "_4.3_GM.png";			//輸出用影像梯度(幅值)
	imwrite(gradm_outfile, gradm_out);
	string gradc_outfile = filepath + "\\" + infilename + "_4.4_GC.png";			//輸出用影像梯度(彩色)
	imwrite(gradc_outfile, gradc_out);
	string gradd_outfile = filepath + "\\" + infilename + "_4.5_GD.png";			//輸出用影像梯度(方向)
	imwrite(gradd_outfile, gradd_out);
	string gradf_outfile = filepath + "\\" + infilename + "_4.6_GF.png";			//輸出用影像梯度(場)
	imwrite(gradf_outfile, gradf_out);

	/*方向模糊*/

	Mat graddBlur;	//模糊方向(8UC1)	
	BlurDirection(gradd, graddBlur, 5);

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

	string blurM_outfile = filepath + "\\" + infilename + "_6_BLURM.png";			//模糊幅值(幅值)
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

	/*滯後切割*/

	Mat gradmHC, graddHC;		//滯後切割(8UC1、32FC1)
	HysteresisCut(gradmDivide, graddBlur, clearBlackArea, gradmHC, graddHC);

	Mat gradmHC_out, graddHC_out, gradfHC_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmHC, gradmHC_out);
	DrawColorSystem(graddHC, graddHC_out);
	DrawColorSystem(gradmHC, graddHC, gradfHC_out);

	string gradmHC_outfile = filepath + "\\" + infilename + "_8.1_HCM.png";			//滯後切割(幅值)
	imwrite(gradmHC_outfile, gradmHC_out);
	string graddHC_outfile = filepath + "\\" + infilename + "_8.2_HCD.png";			//滯後切割(方向)
	imwrite(graddHC_outfile, graddHC_out);
	string gradfHC_outfile = filepath + "\\" + infilename + "_8.3_HCF.png";			//滯後切割(場)
	imwrite(gradfHC_outfile, gradfHC_out);

	/*非極大值抑制*/

	Mat gradmNMS, graddNMS;			//非最大值抑制(8UC1、32FC1)
	NonMaximumSuppression(gradmHC, graddHC, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMS_outfile = filepath + "\\" + infilename + "_9.1_NMSM.png";			//非最大值抑制(幅值)
	imwrite(gradmNMS_outfile, gradmNMS_out);
	string graddNMS_outfile = filepath + "\\" + infilename + "_9.2_NMSD.png";			//非最大值抑制(方向)
	imwrite(graddNMS_outfile, graddNMS_out);
	string gradfNMS_outfile = filepath + "\\" + infilename + "_9.3_NMSF.png";			//非最大值抑制(場)
	imwrite(gradfNMS_outfile, gradfNMS_out);

	/*清除異方向點*/

	Mat gradmCDD, graddCDD;			//清除異方向點(8UC1、32FC1)
	ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	DrawColorSystem(graddCDD, graddCDD_out);
	DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	string gradmCDD_outfile = filepath + "\\" + infilename + "_10.1_CDDM.png";			//清除異方向點(幅值)
	imwrite(gradmCDD_outfile, gradmCDD_out);
	string graddCDD_outfile = filepath + "\\" + infilename + "_10.2_CDDD.png";			//清除異方向點(方向)
	imwrite(graddCDD_outfile, graddCDD_out);
	string gradfCDD_outfile = filepath + "\\" + infilename + "_10.3_CDDF.png";			//清除異方向點(場)
	imwrite(gradfCDD_outfile, gradfCDD_out);

	/*二值化*/

	Mat lineHT;		//二值化(8UC1(BW))
	threshold(gradmCDD, lineHT, 1, 255, THRESH_BINARY);

	string LHT_outfile = filepath + "\\" + infilename + "_11_BW.png";			//二值化(二值)
	imwrite(LHT_outfile, lineHT);

	/*對稱端點連通*/

	Mat gradmSCL, graddSCL, lineSCL;			//短對稱端點連通(8UC1、32FC1、8UC1(BW))
	BWConnectLine(gradmCDD, graddCDD, lineHT, gradmSCL, graddSCL, lineSCL, 2, 5, 60, 0, 0);

	Mat gradmSCL_out, graddSCL_out, gradfSCL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmSCL, gradmSCL_out);
	DrawColorSystem(graddSCL, graddSCL_out);
	DrawColorSystem(gradmSCL, graddSCL, gradfSCL_out);

	string LSCL_outfile = filepath + "\\" + infilename + "_12.0_SCL.png";				//對稱端點連通(二值)
	imwrite(LSCL_outfile, lineSCL);
	string gradmSCL_outfile = filepath + "\\" + infilename + "_12.1_SCLM.png";			//對稱端點連通(幅值)
	imwrite(gradmSCL_outfile, gradmSCL_out);
	string graddSCL_outfile = filepath + "\\" + infilename + "_12.2_SCLD.png";			//對稱端點連通(方向)
	imwrite(graddSCL_outfile, graddSCL_out);
	string gradfSCL_outfile = filepath + "\\" + infilename + "_12.3_SCLF.png";			//對稱端點連通(場)
	imwrite(gradfSCL_outfile, gradfSCL_out);

	/*去除孤立點*/

	Mat lineCP;	//去除孤立點(8UC1(BW))
	ClearPoint(lineSCL, lineCP, 0, 1, 0);

	string LCP_outfile = filepath + "\\" + infilename + "_13_CP.png";			//去除孤立點(二值)
	imwrite(LCP_outfile, lineCP);

	/*強制端點連通*/

	Mat gradmACL, graddACL, lineACL;			//短對稱端點連通(8UC1、32FC1、8UC1(BW))
	BWConnectLine(gradmSCL, graddSCL, lineCP, gradmACL, graddACL, lineACL, 2, 5, 90, 1, 1);

	Mat gradmACL_out, graddACL_out, gradfACL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmACL, gradmACL_out);
	DrawColorSystem(graddACL, graddACL_out);
	DrawColorSystem(gradmACL, graddACL, gradfACL_out);

	string LACL_outfile = filepath + "\\" + infilename + "_14.0_ACL.png";				//強制端點連通(二值)
	imwrite(LACL_outfile, lineACL);
	string gradmACL_outfile = filepath + "\\" + infilename + "_14.1_ACLM.png";			//強制端點連通(幅值)
	imwrite(gradmACL_outfile, gradmACL_out);
	string graddACL_outfile = filepath + "\\" + infilename + "_14.2_ACLD.png";			//強制端點連通(方向)
	imwrite(graddACL_outfile, graddACL_out);
	string gradfACL_outfile = filepath + "\\" + infilename + "_14.3_ACLF.png";			//強制端點連通(場)
	imwrite(gradfACL_outfile, gradfACL_out);

	/*去除雜線*/

	Mat lineCNL;	//去除雜線(8UC1(BW))
	//ClearSpecialPoint(lineACBL, lineCNL, blurLineSize, 5, 1);
	ClearNoise(lineACL, lineCNL, 30, 8, 1);

	string LCNL_outfile = filepath + "\\" + infilename + "_15_CNL.png";			//去除雜線(二值)
	imwrite(LCNL_outfile, lineCNL);


	/****結合面與線的萃取結果****/

	/*結合面與線*/

	Mat combine;	//結合面與線(8UC1(BW))
	BWCombine(clearBlackArea, lineCNL, combine);

	Mat combineLabel_out, combineImage_out;		//輸出用(8UC3、8UC3)
	DrawLabel(combine, combineLabel_out);
	DrawEdge(combine, srcImage, combineImage_out);

	string combineBW_outfile = filepath + "\\" + infilename + "_16.0_COMBINE.png";				//結合面與線(二值)
	imwrite(combineBW_outfile, combine);
	string combineLabel_outfile = filepath + "\\" + infilename + "_16.1_COMBINE.png";			//結合面與線(標籤)
	imwrite(combineLabel_outfile, combineLabel_out);
	string combineImage_outfile = filepath + "\\" + infilename + "_16.2_COMBINE.png";			//結合面與線(疊圖)
	imwrite(combineImage_outfile, combineImage_out);

	///*分水嶺演算法切割*/

	return 0;
}


