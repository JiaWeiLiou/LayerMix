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

	if (blurAreaSize % 2 == 0)	{ --blurAreaSize; }

	std::cout << "Please enter blur square size for Line : ";
	int blurLineSize = 0;
	std::cin >> blurLineSize;

	if (blurLineSize % 2 == 0)	{ --blurLineSize; }

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

	Mat blurAImage;	//模糊影像(8UC1)	
	GaussianBlur(grayImage, blurAImage, Size(blurAreaSize, blurAreaSize), 0, 0);

	string blurA_outfile = filepath + "\\" + infilename + "_1_BLURA.png";		//模糊影像
	imwrite(blurA_outfile, blurAImage);

	/*圖層混合模式*/

	Mat divideArea;		//分割混合模式(8UC1)
	Divide(grayImage, blurAImage, divideArea);

	string divideA_outfile = filepath + "\\" + infilename + "_2.1_DIVIDEA.png";			//分割混合模式
	imwrite(divideA_outfile, divideArea);

	Mat hardmixArea;	//實色印疊合混合模式(8UC1(BW))
	HardMix(grayImage, divideArea, hardmixArea);			

	string hardmixA_outfile = filepath + "\\" + infilename + "_2.2_HARDMIXA.png";			//實色印疊合混合模式
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

	/*將灰度圖像中值模糊*/

	Mat blurLImage;	//模糊影像(8UC1)	
	medianBlur(grayImage, blurLImage, blurLineSize);

	string blurL_outfile = filepath + "\\" + infilename + "_4_BLURL.png";		//模糊影像
	imwrite(blurL_outfile, blurLImage);

	/*計算影像梯度*/

	Mat gradx,grady;		//水平及垂直梯度(16SC1)
	Differential(blurLImage, gradx, grady);

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

	string gradx_outfile = filepath + "\\" + infilename + "_5.1_GX.png";			//輸出用影像梯度(水平)
	imwrite(gradx_outfile, gradx_out);
	string grady_outfile = filepath + "\\" + infilename + "_5.2_GY.png";			//輸出用影像梯度(垂直)
	imwrite(grady_outfile, grady_out);
	string gradm_outfile = filepath + "\\" + infilename + "_5.3_GM.png";			//輸出用影像梯度(幅值)
	imwrite(gradm_outfile, gradm_out);
	string gradd_outfile = filepath + "\\" + infilename + "_5.4_GD.png";			//輸出用影像梯度(方向)
	imwrite(gradd_outfile, gradd_out);
	string gradf_outfile = filepath + "\\" + infilename + "_5.5_GF.png";			//輸出用影像梯度(場)
	imwrite(gradf_outfile, gradf_out);

	/*分割混合模式*/

	Mat divideLine;										//分割混合模式(8UC1)
	Divide(gradm, blurLImage, divideLine);

	Mat gradmDivide_out, gradfDivide_out;				//輸出用(8UC1、8UC3)
	DrawAbsGraySystem(divideLine, gradmDivide_out);
	DrawColorSystem(divideLine, gradd, gradfDivide_out);

	string divideM_outfile = filepath + "\\" + infilename + "_6.1_DIVIDEM.png";			//輸出用分割混合模式(幅值)
	imwrite(divideM_outfile, gradmDivide_out);
	string divideF_outfile = filepath + "\\" + infilename + "_6.2_DIVIDEF.png";			//輸出用分割混合模式(場)
	imwrite(divideF_outfile, gradfDivide_out);

	/*非極大值抑制*/
	
	Mat gradmNMS, graddNMS;			//非最大值抑制(8UC1、32FC1)
	NonMaximumSuppression(divideLine, gradd, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMS_outfile = filepath + "\\" + infilename + "_7.1_NMSM.png";			//非最大值抑制(幅值)
	imwrite(gradmNMS_outfile, gradmNMS_out);
	string graddNMS_outfile = filepath + "\\" + infilename + "_7.2_NMSD.png";			//非最大值抑制(方向)
	imwrite(graddNMS_outfile, graddNMS_out);
	string gradfNMS_outfile = filepath + "\\" + infilename + "_7.3_NMSF.png";			//非最大值抑制(場)
	imwrite(gradfNMS_outfile, gradfNMS_out);

	/*清除異方向點*/
	
	Mat gradmCDD, graddCDD;			//清除異方向點(8UC1、32FC1)
	ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	DrawColorSystem(graddCDD, graddCDD_out);
	DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	string gradmCDD_outfile = filepath + "\\" + infilename + "_8.1_CDDM.png";			//清除異方向點(幅值)
	imwrite(gradmCDD_outfile, gradmCDD_out);
	string graddCDD_outfile = filepath + "\\" + infilename + "_8.2_CDDD.png";			//清除異方向點(方向)
	imwrite(graddCDD_outfile, graddCDD_out);
	string gradfCDD_outfile = filepath + "\\" + infilename + "_8.3_CDDF.png";			//清除異方向點(場)
	imwrite(gradfCDD_outfile, gradfCDD_out);

	/*斷線連通*/

	Mat gradmCBL, graddCBL;			//斷線連通(8UC1、32FC1)
	ConnectBreakLine(gradmCDD, graddCDD, gradmCBL, graddCBL, 2, 3, 60, 0, 0);

	Mat gradmCBL_out, graddCBL_out, gradfCBL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmCBL, gradmCBL_out);
	DrawColorSystem(graddCBL, graddCBL_out);
	DrawColorSystem(gradmCBL, graddCBL, gradfCBL_out);

	string gradmCBL_outfile = filepath + "\\" + infilename + "_9.1_CBLM.png";			//斷線連通(幅值)
	imwrite(gradmCBL_outfile, gradmCBL_out);
	string graddCBL_outfile = filepath + "\\" + infilename + "_9.2_CBLD.png";			//斷線連通(方向)
	imwrite(graddCBL_outfile, graddCBL_out);
	string gradfCBL_outfile = filepath + "\\" + infilename + "_9.3_CBLF.png";			//斷線連通(場)
	imwrite(gradfCBL_outfile, gradfCBL_out);

	/*滯後閥值*/

	Mat lineHT;		//滯後閥值(8UC1(BW))
	HysteresisThreshold(gradmCBL, lineHT, 150, 50);
	string LHT_outfile = filepath + "\\" + infilename + "_10_HT.png";			//滯後閥值
	imwrite(LHT_outfile, lineHT);

	/*去除孤立點*/

	Mat lineCIP;	//去除孤立點(8UC1(BW))
	ClearSpecialPoint(lineHT, lineCIP, 0, 1, 0);
	string LCIP_outfile = filepath + "\\" + infilename + "_11_CIP.png";			//去除孤立點
	imwrite(LCIP_outfile, lineCIP);

	/*短對稱端點連通*/

	Mat gradmSSCBL, graddSSCBL, lineSSCBL;			//短對稱端點連通(8UC1、32FC1、8UC1(BW))
	BWConnectBreakLine(gradmCBL, graddCBL, lineCIP, gradmSSCBL, graddSSCBL, lineSSCBL, 2, 10, 90, 0, 0);

	Mat gradmSSCBL_out, graddSSCBL_out, gradfSSCBL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmSSCBL, gradmSSCBL_out);
	DrawColorSystem(graddSSCBL, graddSSCBL_out);
	DrawColorSystem(gradmSSCBL, graddSSCBL, gradfSSCBL_out);

	string LSSCBL_outfile = filepath + "\\" + infilename + "_12.1.0_SSCBL.png";			//短對稱端點連通(二值)
	imwrite(LSSCBL_outfile, lineSSCBL);
	string gradmSSCBL_outfile = filepath + "\\" + infilename + "_12.1.1_SSCBLM.png";			//短對稱端點連通(幅值)
	imwrite(gradmSSCBL_outfile, gradmSSCBL_out);
	string graddSSCBL_outfile = filepath + "\\" + infilename + "_12.1.2_SSCBLD.png";			//短對稱端點連通(方向)
	imwrite(graddSSCBL_outfile, graddSSCBL_out);
	string gradfSSCBL_outfile = filepath + "\\" + infilename + "_12.1.3_SSCBLF.png";			//短對稱端點連通(場)
	imwrite(gradfSSCBL_outfile, gradfSSCBL_out);

	/*短斷線強制連通*/

	Mat gradmSACBL, graddSACBL, lineSACBL;			//短斷線強制連通(8UC1、32FC1、8UC1(BW))
	BWConnectBreakLine(gradmSSCBL, graddSSCBL, lineSSCBL, gradmSACBL, graddSACBL, lineSACBL, 2, 5, 180, 1, 1);

	Mat gradmSACBL_out, graddSACBL_out, gradfSACBL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmSACBL, gradmSACBL_out);
	DrawColorSystem(graddSACBL, graddSACBL_out);
	DrawColorSystem(gradmSACBL, graddSACBL, gradfSACBL_out);

	string LSACBL_outfile = filepath + "\\" + infilename + "_12.2.0_SACBL.png";					//短斷線強制連通(二值)
	imwrite(LSACBL_outfile, lineSACBL);
	string gradmSACBL_outfile = filepath + "\\" + infilename + "_12.2.1_SACBLM.png";			//短斷線強制連通(幅值)
	imwrite(gradmSACBL_outfile, gradmSACBL_out);
	string graddSACBL_outfile = filepath + "\\" + infilename + "_12.2.2_SACBLD.png";			//短斷線強制連通(方向)
	imwrite(graddSACBL_outfile, graddSACBL_out);
	string gradfSACBL_outfile = filepath + "\\" + infilename + "_12.2.3_SACBLF.png";			//短斷線強制連通(場)
	imwrite(gradfSACBL_outfile, gradfSACBL_out);

	/*去除雜訊端點*/

	Mat lineCNP;	//去除孤立點(8UC1(BW))
	ClearSpecialPoint(lineSACBL, lineCNP, blurLineSize, 3, 1);
	string LCNP_outfile = filepath + "\\" + infilename + "_13_CIP.png";			//去除孤立點
	imwrite(LCNP_outfile, lineCNP);

	/*長對稱端點連通*/

	Mat gradmLSCBL, graddLSCBL, lineLSCBL;			//長對稱端點連通(8UC1、32FC1、8UC1(BW))
	BWConnectBreakLine(gradmSACBL, graddSACBL, lineCNP, gradmLSCBL, graddLSCBL, lineLSCBL, 2, 100, 90, 0, 0);

	Mat gradmLSCBL_out, graddLSCBL_out, gradfLSCBL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmLSCBL, gradmLSCBL_out);
	DrawColorSystem(graddLSCBL, graddLSCBL_out);
	DrawColorSystem(gradmLSCBL, graddLSCBL, gradfLSCBL_out);

	string LLSCBL_outfile = filepath + "\\" + infilename + "_14.1.0_LSCBL.png";					//長對稱端點連通(二值)
	imwrite(LLSCBL_outfile, lineLSCBL);
	string gradmLSCBL_outfile = filepath + "\\" + infilename + "_14.1.1_LSCBLM.png";			//長對稱端點連通(幅值)
	imwrite(gradmLSCBL_outfile, gradmLSCBL_out);
	string graddLSCBL_outfile = filepath + "\\" + infilename + "_14.1.2_LSCBLD.png";			//長對稱端點連通(方向)
	imwrite(graddLSCBL_outfile, graddLSCBL_out);
	string gradfLSCBL_outfile = filepath + "\\" + infilename + "_14.1.3_LSCBLF.png";			//長對稱端點連通(場)
	imwrite(gradfLSCBL_outfile, gradfLSCBL_out);

	/*長斷線強制連通*/
	
	Mat gradmLACBL, graddLACBL, lineLACBL;			//長斷線強制連通(8UC1、32FC1、8UC1(BW))
	BWConnectBreakLine(gradmLSCBL, graddLSCBL, lineLSCBL, gradmLACBL, graddLACBL, lineLACBL, 2, 20, 90, 1, 0);

	Mat gradmLACBL_out, graddLACBL_out, gradfLACBL_out;		//輸出用(8UC1、8UC3、8UC3)
	DrawAbsGraySystem(gradmLACBL, gradmLACBL_out);
	DrawColorSystem(graddLACBL, graddLACBL_out);
	DrawColorSystem(gradmLACBL, graddLACBL, gradfLACBL_out);

	string LLACBL_outfile = filepath + "\\" + infilename + "_14.2.0_LACBL.png";					//長斷線強制連通(二值)
	imwrite(LLACBL_outfile, lineLACBL);
	string gradmLACBL_outfile = filepath + "\\" + infilename + "_14.2.1_LACBLM.png";			//長斷線強制連通(幅值)
	imwrite(gradmLACBL_outfile, gradmLACBL_out);
	string graddLACBL_outfile = filepath + "\\" + infilename + "_14.2.2_LACBLD.png";			//長斷線強制連通(方向)
	imwrite(graddLACBL_outfile, graddLACBL_out);
	string gradfLACBL_outfile = filepath + "\\" + infilename + "_14.2.3_LACBLF.png";			//長斷線強制連通(場)
	imwrite(gradfLACBL_outfile, gradfLACBL_out);

	/*去除突出雜線及孤立線*/

	/*分水嶺演算法切割*/

    return 0;
}


