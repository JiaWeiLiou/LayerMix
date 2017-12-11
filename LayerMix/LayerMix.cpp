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

	Mat gray;		//灰階影像(8UC1)
	if (srcImage.type() != CV_8UC1)
	{
		cvtColor(srcImage, gray, CV_BGR2GRAY);

		string gray_file = filepath + "\\" + infilename + "_0.0_GRAY.png";				//灰階影像(灰階)
		imwrite(gray_file, gray);
	}
	else { gray = srcImage; }

	Mat gray_BR;		//輸出用(8UC3)
	DrawColorBar(gray, gray_BR);

	string gray_BR_file = filepath + "\\" + infilename + "_0.1_GRAY(BR).png";			//灰階影像(藍紅)
	imwrite(gray_BR_file, gray_BR);

	/****基於面的影像萃取****/

	/*面模糊灰階影像*/

	Mat blurA;	//模糊影像(8UC1)	
	GaussianBlur(gray, blurA, Size(blurAreaSize, blurAreaSize), 0, 0);

	string blurA_file = filepath + "\\" + infilename + "_1.0_BLURA.png";				//面模糊灰階影像(灰階)
	imwrite(blurA_file, blurA);

	/*面消除區域亮度*/

	Mat divideA;		//面消除區域亮度(8UC1)
	DivideArea(gray, blurA, divideA);

	Mat divideA_BR;		//輸出用(8UC3)
	DrawColorBar(divideA, divideA_BR);

	string divideA_file = filepath + "\\" + infilename + "_2.0_DIVIDEA.png";			//面消除區域亮度(灰階)
	imwrite(divideA_file, divideA);
	string divideA_BR_file = filepath + "\\" + infilename + "_2.1_DIVIDEA(BR).png";		//面消除區域亮度(藍紅)
	imwrite(divideA_BR_file, divideA_BR);

	/*面二值化*/

	Mat bwA;		//面積二值化(8UC1(BW))
	threshold(divideA, bwA, 127, 255, THRESH_BINARY);

	Mat bwA_L_out, bwA_I_out;		//輸出用(8UC3、8UC3)
	DrawLabel(bwA, bwA_L_out);
	DrawEdge(bwA, srcImage, bwA_I_out);

	string bwA_file = filepath + "\\" + infilename + "_3.0_AREA(BW).png";			//面二值化(二值)
	imwrite(bwA_file, bwA);
	string bwA_L_file = filepath + "\\" + infilename + "_3.1_AREA(L).png";			//面二值化(標籤)
	imwrite(bwA_L_file, bwA_L_out);
	string bwA_I_file = filepath + "\\" + infilename + "_3.2_AREA(I).png";			//面二值化(疊圖)
	imwrite(bwA_I_file, bwA_I_out);

	/*去除白色雜訊*/

	Mat cwA;		//去除白色雜訊(8UC1(BW))
	ClearNoise(bwA, cwA, 5, 4, 1);

	Mat cwA_L_out, cwA_I_out;		//輸出用(8UC3、8UC3)
	DrawLabel(cwA, cwA_L_out);
	DrawEdge(cwA, srcImage, cwA_I_out);

	string cwA_file = filepath + "\\" + infilename + "_4.0_CW(BW).png";				//去除白色雜訊(二值)
	imwrite(cwA_file, cwA);
	string cwA_L_file = filepath + "\\" + infilename + "_4.1_CW(L).png";			//去除白色雜訊(標籤)
	imwrite(cwA_L_file, cwA_L_out);
	string cwA_I_file = filepath + "\\" + infilename + "_4.2_CW(I).png";			//去除白色雜訊(疊圖)
	imwrite(cwA_I_file, cwA_I_out);

	/*去除黑色雜訊*/

	Mat cbA;		//去除黑色雜訊(8UC1(BW))
	ClearNoise(cwA, cbA, 5, 4, 0);

	Mat cbA_L_out, cbA_I_out;		//輸出用(8UC3、8UC3)
	DrawLabel(cbA, cbA_L_out);
	DrawEdge(cbA, srcImage, cbA_I_out);

	string cbA_file = filepath + "\\" + infilename + "_5.0_CB(BW).png";				//去除黑色雜訊(二值)
	imwrite(cbA_file, cbA);
	string cbA_L_file = filepath + "\\" + infilename + "_5.1_CB(L).png";			//去除白色雜訊(標籤)
	imwrite(cbA_L_file, cbA_L_out);
	string cbA_I_file = filepath + "\\" + infilename + "_5.2_CB(I).png";			//去除白色雜訊(疊圖)
	imwrite(cbA_I_file, cbA_I_out);

	/****基於線的影像萃取****/

	///*模糊灰階影像*/

	//Mat blurImageL;	//模糊影像(8UC1)	
	//GaussianBlur(grayImage, blurImageL, Size(blurLineSize, blurLineSize), 0, 0);

	//string blurL_file = filepath + "\\" + infilename + "_4_BLURL.png";			//模糊灰階影像(灰階)
	//imwrite(blurL_file, blurImageL);

	/*計算影像梯度*/

	Mat gradx, grady;		//水平及垂直梯度(16SC1)
	Differential(divideA, gradx, grady);

	Mat gradf;				//梯度場(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;		//梯度幅值及梯度方向(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_out, grady_out, gradm_out,gradd_out, gradf_out;		//輸出用(8UC1、8UC1、8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradf, gradf_out);

	string gradx_file = filepath + "\\" + infilename + "_6.1_GX.png";			//輸出用影像梯度(水平)
	imwrite(gradx_file, gradx_out);
	string grady_file = filepath + "\\" + infilename + "_6.2_GY.png";			//輸出用影像梯度(垂直)
	imwrite(grady_file, grady_out);
	string gradm_file = filepath + "\\" + infilename + "_6.3_GM.png";			//輸出用影像梯度(幅值)
	imwrite(gradm_file, gradm_out);
	string gradd_file = filepath + "\\" + infilename + "_6.4_GD.png";			//輸出用影像梯度(方向)
	imwrite(gradd_file, gradd_out);
	string gradf_file = filepath + "\\" + infilename + "_6.5_GF.png";			//輸出用影像梯度(場)
	imwrite(gradf_file, gradf_out);

	/*方向模糊*/

	Mat graddBlur;	//模糊方向(8UC1)	
	BlurDirection(gradd, graddBlur, 5);

	Mat graddBlur_out, gradfBlur_out;		//輸出用(8UC3、8UC3)
	DrawColorSystem(graddBlur, graddBlur_out);
	DrawColorSystem(gradm, graddBlur, gradfBlur_out);

	string graddBlur_file = filepath + "\\" + infilename + "_7.0_BLUR(D).png";			//模糊方向(方向)
	imwrite(graddBlur_file, graddBlur_out);
	string gradfBlur_file = filepath + "\\" + infilename + "_7.1_BLUR(F).png";			//模糊方向(場)
	imwrite(gradfBlur_file, gradfBlur_out);

	///*模糊幅值*/

	//Mat gradmBlur;	//模糊幅值(8UC1)	
	//GaussianBlur(gradm, gradmBlur, Size(blurLineSize, blurLineSize), 0, 0);

	//string blurM_file = filepath + "\\" + infilename + "_7.2_BLUR(M).png";				//模糊幅值(幅值)
	//imwrite(blurM_file, gradmBlur);

	///*線消除區域亮度*/

	//Mat divideL;										//線消除區域亮度(8UC1)
	//DivideLine(gradm, gradmBlur, divideL);

	//Mat divideL_M_out, divideL_F_out;		//輸出用(8UC1、8UC3)
	//DrawAbsGraySystem(divideL, divideL_M_out);
	//DrawColorSystem(divideL, graddBlur, divideL_F_out);

	//string divideL_M_file = filepath + "\\" + infilename + "_8.1_DIVIDE(M).png";			//線消除區域亮度(幅值)
	//imwrite(divideL_M_file, divideL_M_out);
	//string divideL_F_file = filepath + "\\" + infilename + "_8.2_DIVIDE(F).png";		//線消除區域亮度(場)
	//imwrite(divideL_F_file, divideL_F_out);

	/*非極大值抑制*/

	Mat gradmNMS, graddNMS;			//非最大值抑制(8UC1、32FC1)
	NonMaximumSuppression(gradm, graddBlur, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMS_file = filepath + "\\" + infilename + "_8.1_NMS(M).png";			//非最大值抑制(幅值)
	imwrite(gradmNMS_file, gradmNMS_out);
	string graddNMS_file = filepath + "\\" + infilename + "_8.2_NMS(D).png";			//非最大值抑制(方向)
	imwrite(graddNMS_file, graddNMS_out);
	string gradfNMS_file = filepath + "\\" + infilename + "_8.3_NMS(F).png";			//非最大值抑制(場)
	imwrite(gradfNMS_file, gradfNMS_out);

	///*清除分岔點*/

	//Mat gradmCBP, graddCBP;			//清除異方向點(8UC1、32FC1)
	//ClearBifPoint(gradmNMS, graddNMS, gradmCBP, graddCBP);

	//Mat gradmCBP_out, graddCBP_out, gradfCBP_out;		//輸出用(8UC1、8UC3、8UC3)
	//DrawAbsGraySystem(gradmCBP, gradmCBP_out);
	//DrawColorSystem(graddCBP, graddCBP_out);
	//DrawColorSystem(gradmCBP, graddCBP, gradfCBP_out);

	//string gradmCBP_file = filepath + "\\" + infilename + "_7.1_CBPM.png";			//清除分岔點(幅值)
	//imwrite(gradmCBP_file, gradmCBP_out);
	//string graddCBP_file = filepath + "\\" + infilename + "_7.2_CBPD.png";			//清除分岔點(方向)
	//imwrite(graddCBP_file, graddCBP_out);
	//string gradfCBP_file = filepath + "\\" + infilename + "_7.3_CBPF.png";			//清除分岔點(場)
	//imwrite(gradfCBP_file, gradfCBP_out);

	/*清除異方向點*/

	Mat gradmCDD, graddCDD;			//清除異方向點(8UC1、32FC1)
	ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	DrawColorSystem(graddCDD, graddCDD_out);
	DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	string gradmCDD_file = filepath + "\\" + infilename + "_9.1_CDD(M).png";			//清除異方向點(幅值)
	imwrite(gradmCDD_file, gradmCDD_out);
	string graddCDD_file = filepath + "\\" + infilename + "_9.2_CDD(D).png";			//清除異方向點(方向)
	imwrite(graddCDD_file, graddCDD_out);
	string gradfCDD_file = filepath + "\\" + infilename + "_9.3_CDD(F).png";			//清除異方向點(場)
	imwrite(gradfCDD_file, gradfCDD_out);

	/*對稱端點連通*/

	Mat gradmSCL, graddSCL;			//短對稱端點連通(8UC1、32FC1)
	ConnectLine(gradmCDD, graddCDD, gradmSCL, graddSCL, 2, 2, 60, 0, 1);

	Mat gradmSCL_out, graddSCL_out, gradfSCL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmSCL, gradmSCL_out);
	DrawColorSystem(graddSCL, graddSCL_out);
	DrawColorSystem(gradmSCL, graddSCL, gradfSCL_out);

	string gradmSCL_file = filepath + "\\" + infilename + "_10.1_SCL(M).png";			//對稱端點連通(幅值)
	imwrite(gradmSCL_file, gradmSCL_out);
	string graddSCL_file = filepath + "\\" + infilename + "_10.2_SCL(D).png";			//對稱端點連通(方向)
	imwrite(graddSCL_file, graddSCL_out);
	string gradfSCL_file = filepath + "\\" + infilename + "_10.3_SCL(F).png";			//對稱端點連通(場)
	imwrite(gradfSCL_file, gradfSCL_out);

	///*滯後切割*/

	//Mat gradmHC, graddHC;		//滯後切割(8UC1、32FC1)
	//HysteresisCut(gradmSCL, graddSCL, cbA, gradmHC, graddHC);

	//Mat gradmHC_out, graddHC_out, gradfHC_out;		//輸出用(8UC1、8UC3、8UC3)
	//DrawAbsGraySystem(gradmHC, gradmHC_out);
	//DrawColorSystem(graddHC, graddHC_out);
	//DrawColorSystem(gradmHC, graddHC, gradfHC_out);

	//string gradmHC_file = filepath + "\\" + infilename + "_12.1_HC(M).png";				//滯後切割(幅值)
	//imwrite(gradmHC_file, gradmHC_out);
	//string graddHC_file = filepath + "\\" + infilename + "_12.2_HC(D).png";				//滯後切割(方向)
	//imwrite(graddHC_file, graddHC_out);
	//string gradfHC_file = filepath + "\\" + infilename + "_12.3_HC(F).png";				//滯後切割(場)
	//imwrite(gradfHC_file, gradfHC_out);

	/*去除孤立點*/

	Mat gradmCIP, graddCIP;			//去除孤立點(8UC1、32FC1)
	ClearIsoPoint(gradmSCL, graddSCL, gradmCIP, graddCIP);

	Mat gradmCIP_out, graddCIP_out, gradfCIP_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmCIP, gradmCIP_out);
	DrawColorSystem(graddCIP, graddCIP_out);
	DrawColorSystem(gradmCIP, graddCIP, gradfCIP_out);

	string gradmCIP_file = filepath + "\\" + infilename + "_11.1_CIP(M).png";			//去除孤立點(幅值)
	imwrite(gradmCIP_file, gradmCIP_out);
	string graddCIP_file = filepath + "\\" + infilename + "_11.2_CIP(D).png";			//去除孤立點(方向)
	imwrite(graddCIP_file, graddCIP_out);
	string gradfCIP_file = filepath + "\\" + infilename + "_11.3_CIP(F).png";			//去除孤立點(場)
	imwrite(gradfCIP_file, gradfCIP_out);

	/*強制端點連通*/

	Mat gradmACL, graddACL;			//短對稱端點連通(8UC1、32FC1)
	ConnectLine(gradmCIP, graddCIP, gradmACL, graddACL, 2, 3, 120, 1, 1);

	Mat gradmACL_out, graddACL_out, gradfACL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmACL, gradmACL_out);
	DrawColorSystem(graddACL, graddACL_out);
	DrawColorSystem(gradmACL, graddACL, gradfACL_out);

	string gradmACL_file = filepath + "\\" + infilename + "_12.1_ACL(M).png";			//強制端點連通(幅值)
	imwrite(gradmACL_file, gradmACL_out);
	string graddACL_file = filepath + "\\" + infilename + "_12.2_ACL(D).png";			//強制端點連通(方向)
	imwrite(graddACL_file, graddACL_out);
	string gradfACL_file = filepath + "\\" + infilename + "_12.3_ACL(F).png";			//強制端點連通(場)
	imwrite(gradfACL_file, gradfACL_out);

	/*滯後閥值*/

	Mat bwL;		//二值化(8UC1(BW))
	HysteresisThreshold(gradmCIP, bwL, 75, 10);

	string bwL_file = filepath + "\\" + infilename + "_13_LINE(BW).png";				//滯後閥值(二值)
	imwrite(bwL_file, bwL);

	///*線二值化*/

	//Mat bwL;		//二值化(8UC1(BW))
	//threshold(gradmCIP, bwL, 1, 255, THRESH_BINARY);

	//string bwL_file = filepath + "\\" + infilename + "_14_LINE(BW).png";				//線二值化(二值)
	//imwrite(bwL_file, bwL);

	///*強制端點連通*/

	//Mat gradmACL, graddACL, lineACL;			//短對稱端點連通(8UC1、32FC1、8UC1(BW))
	//BWConnectLine(gradmCIP, graddCIP, bwL, gradmACL, graddACL, lineACL, 2, 5, 90, 1, 1);

	//Mat gradmACL_out, graddACL_out, gradfACL_out;		//輸出用(8UC1、8UC3、8UC3)
	//DrawAbsGraySystem(gradmACL, gradmACL_out);
	//DrawColorSystem(graddACL, graddACL_out);
	//DrawColorSystem(gradmACL, graddACL, gradfACL_out);

	//string lineACL_file = filepath + "\\" + infilename + "_15.0_ACL.png";			//強制端點連通(二值)
	//imwrite(lineACL_file, lineACL);
	//string gradmACL_file = filepath + "\\" + infilename + "_15.1_ACLM.png";			//強制端點連通(幅值)
	//imwrite(gradmACL_file, gradmACL_out);
	//string graddACL_file = filepath + "\\" + infilename + "_15.2_ACLD.png";			//強制端點連通(方向)
	//imwrite(graddACL_file, graddACL_out);
	//string gradfACL_file = filepath + "\\" + infilename + "_15.3_ACLF.png";			//強制端點連通(場)
	//imwrite(gradfACL_file, gradfACL_out);

	///*去除雜線*/

	//Mat lineCNL;	//去除雜線(8UC1(BW))
	//ClearNoise(lineACL, lineCNL, 20, 8, 1);

	//string lineCNL_file = filepath + "\\" + infilename + "_16_CNL.png";				//去除雜線(二值)
	//imwrite(lineCNL_file, lineCNL);


	/****結合面與線的萃取結果****/

	/*結合面與線*/

	Mat combineBW;	//結合面與線(8UC1(BW))
	BWCombine(cbA, bwL, combineBW);

	Mat combine_L_out, combine_I_out;		//輸出用(8UC3、8UC3)
	DrawLabel(combineBW, combine_L_out);
	DrawEdge(combineBW, srcImage, combine_I_out);

	string combine_file = filepath + "\\" + infilename + "_14.0_COMBINE(BW).png";			//結合面與線(二值)
	imwrite(combine_file, combineBW);
	string combine_L_file = filepath + "\\" + infilename + "_14.1_COMBINE(L).png";			//結合面與線(標籤)
	imwrite(combine_L_file, combine_L_out);
	string combine_I_file = filepath + "\\" + infilename + "_14.2_COMBINE(I).png";			//結合面與線(疊圖)
	imwrite(combine_I_file, combine_I_out);

	/*填補不足面積*/

	Mat fillBW;		//填補不足面積(8UC1(BW))
	ClearNoise(combineBW, fillBW, 50, 4, 1);

	Mat fill_L_out, fill_I_out;		//輸出用(8UC3、8UC3)
	DrawLabel(fillBW, fill_L_out);
	DrawEdge(fillBW, srcImage, fill_I_out);

	string fill_file = filepath + "\\" + infilename + "_15.0_FILL(BW).png";			//填補不足面積(二值)
	imwrite(fill_file, fillBW);
	string fill_L_file = filepath + "\\" + infilename + "_15.1_FILL(L).png";		//填補不足面積(標籤)
	imwrite(fill_L_file, fill_L_out);
	string fill_I_file = filepath + "\\" + infilename + "_15.2_FILL(I).png";		//填補不足面積(疊圖)
	imwrite(fill_I_file, fill_I_out);

	/*距離轉換*/

	Mat dtBW;		//距離轉換(32FC1(BW))
	distanceTransform(combineBW, dtBW, CV_DIST_L2, 3);

	Mat dt_out,dt_BR_out;		//輸出用(8UC1)
	DrawAbsGraySystem(dtBW, dt_out);
	DrawColorBar(dt_out, dt_BR_out);

	string dt_file = filepath + "\\" + infilename + "_16.0_DT.png";					//距離轉換(灰階)
	imwrite(dt_file, dt_out);
	string dt_BR_file = filepath + "\\" + infilename + "_16.1_DT(BR).png";					//距離轉換(藍紅)
	imwrite(dt_BR_file, dt_BR_out);

	/*分水嶺演算法切割*/

	Mat wsBW;		//分水嶺演算法切割(32SC1(BW))
	BWWatershed(srcImage, combineBW, combineBW, wsBW);

	Mat ws_L_out, ws_I_out;		//輸出用(8UC3、8UC3)
	DrawLabel(wsBW, ws_L_out);
	DrawEdge(wsBW, srcImage, ws_I_out);

	string ws_file = filepath + "\\" + infilename + "_17.0_WS(BW).png";				//分水嶺演算法切割(二值)
	imwrite(ws_file, wsBW);
	string ws_L_file = filepath + "\\" + infilename + "_17.1_WS(L).png";			//分水嶺演算法切割(標籤)
	imwrite(ws_L_file, ws_L_out);
	string ws_I_file = filepath + "\\" + infilename + "_17.2_WS(I).png";			//分水嶺演算法切割(疊圖)
	imwrite(ws_I_file, ws_I_out);

	return 0;
}


