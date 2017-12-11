// LayerMix.cpp : �w�q�D���x���ε{�����i�J�I�C
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

	/*�]�w��X���W*/

	int pos1 = infile.find_last_of('/\\');
	int pos2 = infile.find_last_of('.');
	string filepath(infile.substr(0, pos1));							//�ɮ׸��|
	string infilename(infile.substr(pos1 + 1, pos2 - pos1 - 1));		//�ɮצW��

	/*���J���*/

	Mat srcImage = imread(infile);	//��l�v��(8UC1 || 8UC3 )
	if (!srcImage.data) { printf("Oh�Ano�AŪ��srcImage���~~�I \n"); return false; }

	/*�N��Ϲ��ഫ���ǫ׹Ϲ�*/

	Mat gray;		//�Ƕ��v��(8UC1)
	if (srcImage.type() != CV_8UC1)
	{
		cvtColor(srcImage, gray, CV_BGR2GRAY);

		string gray_file = filepath + "\\" + infilename + "_0.0_GRAY.png";				//�Ƕ��v��(�Ƕ�)
		imwrite(gray_file, gray);
	}
	else { gray = srcImage; }

	Mat gray_BR;		//��X��(8UC3)
	DrawColorBar(gray, gray_BR);

	string gray_BR_file = filepath + "\\" + infilename + "_0.1_GRAY(BR).png";			//�Ƕ��v��(�Ŭ�)
	imwrite(gray_BR_file, gray_BR);

	/****��󭱪��v���Ѩ�****/

	/*���ҽk�Ƕ��v��*/

	Mat blurA;	//�ҽk�v��(8UC1)	
	GaussianBlur(gray, blurA, Size(blurAreaSize, blurAreaSize), 0, 0);

	string blurA_file = filepath + "\\" + infilename + "_1.0_BLURA.png";				//���ҽk�Ƕ��v��(�Ƕ�)
	imwrite(blurA_file, blurA);

	/*�������ϰ�G��*/

	Mat divideA;		//�������ϰ�G��(8UC1)
	DivideArea(gray, blurA, divideA);

	Mat divideA_BR;		//��X��(8UC3)
	DrawColorBar(divideA, divideA_BR);

	string divideA_file = filepath + "\\" + infilename + "_2.0_DIVIDEA.png";			//�������ϰ�G��(�Ƕ�)
	imwrite(divideA_file, divideA);
	string divideA_BR_file = filepath + "\\" + infilename + "_2.1_DIVIDEA(BR).png";		//�������ϰ�G��(�Ŭ�)
	imwrite(divideA_BR_file, divideA_BR);

	/*���G�Ȥ�*/

	Mat bwA;		//���n�G�Ȥ�(8UC1(BW))
	threshold(divideA, bwA, 127, 255, THRESH_BINARY);

	Mat bwA_L_out, bwA_I_out;		//��X��(8UC3�B8UC3)
	DrawLabel(bwA, bwA_L_out);
	DrawEdge(bwA, srcImage, bwA_I_out);

	string bwA_file = filepath + "\\" + infilename + "_3.0_AREA(BW).png";			//���G�Ȥ�(�G��)
	imwrite(bwA_file, bwA);
	string bwA_L_file = filepath + "\\" + infilename + "_3.1_AREA(L).png";			//���G�Ȥ�(����)
	imwrite(bwA_L_file, bwA_L_out);
	string bwA_I_file = filepath + "\\" + infilename + "_3.2_AREA(I).png";			//���G�Ȥ�(�|��)
	imwrite(bwA_I_file, bwA_I_out);

	/*�h���զ����T*/

	Mat cwA;		//�h���զ����T(8UC1(BW))
	ClearNoise(bwA, cwA, 5, 4, 1);

	Mat cwA_L_out, cwA_I_out;		//��X��(8UC3�B8UC3)
	DrawLabel(cwA, cwA_L_out);
	DrawEdge(cwA, srcImage, cwA_I_out);

	string cwA_file = filepath + "\\" + infilename + "_4.0_CW(BW).png";				//�h���զ����T(�G��)
	imwrite(cwA_file, cwA);
	string cwA_L_file = filepath + "\\" + infilename + "_4.1_CW(L).png";			//�h���զ����T(����)
	imwrite(cwA_L_file, cwA_L_out);
	string cwA_I_file = filepath + "\\" + infilename + "_4.2_CW(I).png";			//�h���զ����T(�|��)
	imwrite(cwA_I_file, cwA_I_out);

	/*�h���¦����T*/

	Mat cbA;		//�h���¦����T(8UC1(BW))
	ClearNoise(cwA, cbA, 5, 4, 0);

	Mat cbA_L_out, cbA_I_out;		//��X��(8UC3�B8UC3)
	DrawLabel(cbA, cbA_L_out);
	DrawEdge(cbA, srcImage, cbA_I_out);

	string cbA_file = filepath + "\\" + infilename + "_5.0_CB(BW).png";				//�h���¦����T(�G��)
	imwrite(cbA_file, cbA);
	string cbA_L_file = filepath + "\\" + infilename + "_5.1_CB(L).png";			//�h���զ����T(����)
	imwrite(cbA_L_file, cbA_L_out);
	string cbA_I_file = filepath + "\\" + infilename + "_5.2_CB(I).png";			//�h���զ����T(�|��)
	imwrite(cbA_I_file, cbA_I_out);

	/****���u���v���Ѩ�****/

	///*�ҽk�Ƕ��v��*/

	//Mat blurImageL;	//�ҽk�v��(8UC1)	
	//GaussianBlur(grayImage, blurImageL, Size(blurLineSize, blurLineSize), 0, 0);

	//string blurL_file = filepath + "\\" + infilename + "_4_BLURL.png";			//�ҽk�Ƕ��v��(�Ƕ�)
	//imwrite(blurL_file, blurImageL);

	/*�p��v�����*/

	Mat gradx, grady;		//�����Ϋ������(16SC1)
	Differential(divideA, gradx, grady);

	Mat gradf;				//��׳�(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;		//��״T�Ȥα�פ�V(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_out, grady_out, gradm_out,gradd_out, gradf_out;		//��X��(8UC1�B8UC1�B8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradf, gradf_out);

	string gradx_file = filepath + "\\" + infilename + "_6.1_GX.png";			//��X�μv�����(����)
	imwrite(gradx_file, gradx_out);
	string grady_file = filepath + "\\" + infilename + "_6.2_GY.png";			//��X�μv�����(����)
	imwrite(grady_file, grady_out);
	string gradm_file = filepath + "\\" + infilename + "_6.3_GM.png";			//��X�μv�����(�T��)
	imwrite(gradm_file, gradm_out);
	string gradd_file = filepath + "\\" + infilename + "_6.4_GD.png";			//��X�μv�����(��V)
	imwrite(gradd_file, gradd_out);
	string gradf_file = filepath + "\\" + infilename + "_6.5_GF.png";			//��X�μv�����(��)
	imwrite(gradf_file, gradf_out);

	/*��V�ҽk*/

	Mat graddBlur;	//�ҽk��V(8UC1)	
	BlurDirection(gradd, graddBlur, 5);

	Mat graddBlur_out, gradfBlur_out;		//��X��(8UC3�B8UC3)
	DrawColorSystem(graddBlur, graddBlur_out);
	DrawColorSystem(gradm, graddBlur, gradfBlur_out);

	string graddBlur_file = filepath + "\\" + infilename + "_7.0_BLUR(D).png";			//�ҽk��V(��V)
	imwrite(graddBlur_file, graddBlur_out);
	string gradfBlur_file = filepath + "\\" + infilename + "_7.1_BLUR(F).png";			//�ҽk��V(��)
	imwrite(gradfBlur_file, gradfBlur_out);

	///*�ҽk�T��*/

	//Mat gradmBlur;	//�ҽk�T��(8UC1)	
	//GaussianBlur(gradm, gradmBlur, Size(blurLineSize, blurLineSize), 0, 0);

	//string blurM_file = filepath + "\\" + infilename + "_7.2_BLUR(M).png";				//�ҽk�T��(�T��)
	//imwrite(blurM_file, gradmBlur);

	///*�u�����ϰ�G��*/

	//Mat divideL;										//�u�����ϰ�G��(8UC1)
	//DivideLine(gradm, gradmBlur, divideL);

	//Mat divideL_M_out, divideL_F_out;		//��X��(8UC1�B8UC3)
	//DrawAbsGraySystem(divideL, divideL_M_out);
	//DrawColorSystem(divideL, graddBlur, divideL_F_out);

	//string divideL_M_file = filepath + "\\" + infilename + "_8.1_DIVIDE(M).png";			//�u�����ϰ�G��(�T��)
	//imwrite(divideL_M_file, divideL_M_out);
	//string divideL_F_file = filepath + "\\" + infilename + "_8.2_DIVIDE(F).png";		//�u�����ϰ�G��(��)
	//imwrite(divideL_F_file, divideL_F_out);

	/*�D���j�ȧ��*/

	Mat gradmNMS, graddNMS;			//�D�̤j�ȧ��(8UC1�B32FC1)
	NonMaximumSuppression(gradm, graddBlur, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMS_file = filepath + "\\" + infilename + "_8.1_NMS(M).png";			//�D�̤j�ȧ��(�T��)
	imwrite(gradmNMS_file, gradmNMS_out);
	string graddNMS_file = filepath + "\\" + infilename + "_8.2_NMS(D).png";			//�D�̤j�ȧ��(��V)
	imwrite(graddNMS_file, graddNMS_out);
	string gradfNMS_file = filepath + "\\" + infilename + "_8.3_NMS(F).png";			//�D�̤j�ȧ��(��)
	imwrite(gradfNMS_file, gradfNMS_out);

	///*�M�������I*/

	//Mat gradmCBP, graddCBP;			//�M������V�I(8UC1�B32FC1)
	//ClearBifPoint(gradmNMS, graddNMS, gradmCBP, graddCBP);

	//Mat gradmCBP_out, graddCBP_out, gradfCBP_out;		//��X��(8UC1�B8UC3�B8UC3)
	//DrawAbsGraySystem(gradmCBP, gradmCBP_out);
	//DrawColorSystem(graddCBP, graddCBP_out);
	//DrawColorSystem(gradmCBP, graddCBP, gradfCBP_out);

	//string gradmCBP_file = filepath + "\\" + infilename + "_7.1_CBPM.png";			//�M�������I(�T��)
	//imwrite(gradmCBP_file, gradmCBP_out);
	//string graddCBP_file = filepath + "\\" + infilename + "_7.2_CBPD.png";			//�M�������I(��V)
	//imwrite(graddCBP_file, graddCBP_out);
	//string gradfCBP_file = filepath + "\\" + infilename + "_7.3_CBPF.png";			//�M�������I(��)
	//imwrite(gradfCBP_file, gradfCBP_out);

	/*�M������V�I*/

	Mat gradmCDD, graddCDD;			//�M������V�I(8UC1�B32FC1)
	ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	DrawColorSystem(graddCDD, graddCDD_out);
	DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	string gradmCDD_file = filepath + "\\" + infilename + "_9.1_CDD(M).png";			//�M������V�I(�T��)
	imwrite(gradmCDD_file, gradmCDD_out);
	string graddCDD_file = filepath + "\\" + infilename + "_9.2_CDD(D).png";			//�M������V�I(��V)
	imwrite(graddCDD_file, graddCDD_out);
	string gradfCDD_file = filepath + "\\" + infilename + "_9.3_CDD(F).png";			//�M������V�I(��)
	imwrite(gradfCDD_file, gradfCDD_out);

	/*��ٺ��I�s�q*/

	Mat gradmSCL, graddSCL;			//�u��ٺ��I�s�q(8UC1�B32FC1)
	ConnectLine(gradmCDD, graddCDD, gradmSCL, graddSCL, 2, 2, 60, 0, 1);

	Mat gradmSCL_out, graddSCL_out, gradfSCL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmSCL, gradmSCL_out);
	DrawColorSystem(graddSCL, graddSCL_out);
	DrawColorSystem(gradmSCL, graddSCL, gradfSCL_out);

	string gradmSCL_file = filepath + "\\" + infilename + "_10.1_SCL(M).png";			//��ٺ��I�s�q(�T��)
	imwrite(gradmSCL_file, gradmSCL_out);
	string graddSCL_file = filepath + "\\" + infilename + "_10.2_SCL(D).png";			//��ٺ��I�s�q(��V)
	imwrite(graddSCL_file, graddSCL_out);
	string gradfSCL_file = filepath + "\\" + infilename + "_10.3_SCL(F).png";			//��ٺ��I�s�q(��)
	imwrite(gradfSCL_file, gradfSCL_out);

	///*�������*/

	//Mat gradmHC, graddHC;		//�������(8UC1�B32FC1)
	//HysteresisCut(gradmSCL, graddSCL, cbA, gradmHC, graddHC);

	//Mat gradmHC_out, graddHC_out, gradfHC_out;		//��X��(8UC1�B8UC3�B8UC3)
	//DrawAbsGraySystem(gradmHC, gradmHC_out);
	//DrawColorSystem(graddHC, graddHC_out);
	//DrawColorSystem(gradmHC, graddHC, gradfHC_out);

	//string gradmHC_file = filepath + "\\" + infilename + "_12.1_HC(M).png";				//�������(�T��)
	//imwrite(gradmHC_file, gradmHC_out);
	//string graddHC_file = filepath + "\\" + infilename + "_12.2_HC(D).png";				//�������(��V)
	//imwrite(graddHC_file, graddHC_out);
	//string gradfHC_file = filepath + "\\" + infilename + "_12.3_HC(F).png";				//�������(��)
	//imwrite(gradfHC_file, gradfHC_out);

	/*�h���t���I*/

	Mat gradmCIP, graddCIP;			//�h���t���I(8UC1�B32FC1)
	ClearIsoPoint(gradmSCL, graddSCL, gradmCIP, graddCIP);

	Mat gradmCIP_out, graddCIP_out, gradfCIP_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmCIP, gradmCIP_out);
	DrawColorSystem(graddCIP, graddCIP_out);
	DrawColorSystem(gradmCIP, graddCIP, gradfCIP_out);

	string gradmCIP_file = filepath + "\\" + infilename + "_11.1_CIP(M).png";			//�h���t���I(�T��)
	imwrite(gradmCIP_file, gradmCIP_out);
	string graddCIP_file = filepath + "\\" + infilename + "_11.2_CIP(D).png";			//�h���t���I(��V)
	imwrite(graddCIP_file, graddCIP_out);
	string gradfCIP_file = filepath + "\\" + infilename + "_11.3_CIP(F).png";			//�h���t���I(��)
	imwrite(gradfCIP_file, gradfCIP_out);

	/*�j����I�s�q*/

	Mat gradmACL, graddACL;			//�u��ٺ��I�s�q(8UC1�B32FC1)
	ConnectLine(gradmCIP, graddCIP, gradmACL, graddACL, 2, 3, 120, 1, 1);

	Mat gradmACL_out, graddACL_out, gradfACL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmACL, gradmACL_out);
	DrawColorSystem(graddACL, graddACL_out);
	DrawColorSystem(gradmACL, graddACL, gradfACL_out);

	string gradmACL_file = filepath + "\\" + infilename + "_12.1_ACL(M).png";			//�j����I�s�q(�T��)
	imwrite(gradmACL_file, gradmACL_out);
	string graddACL_file = filepath + "\\" + infilename + "_12.2_ACL(D).png";			//�j����I�s�q(��V)
	imwrite(graddACL_file, graddACL_out);
	string gradfACL_file = filepath + "\\" + infilename + "_12.3_ACL(F).png";			//�j����I�s�q(��)
	imwrite(gradfACL_file, gradfACL_out);

	/*����֭�*/

	Mat bwL;		//�G�Ȥ�(8UC1(BW))
	HysteresisThreshold(gradmCIP, bwL, 75, 10);

	string bwL_file = filepath + "\\" + infilename + "_13_LINE(BW).png";				//����֭�(�G��)
	imwrite(bwL_file, bwL);

	///*�u�G�Ȥ�*/

	//Mat bwL;		//�G�Ȥ�(8UC1(BW))
	//threshold(gradmCIP, bwL, 1, 255, THRESH_BINARY);

	//string bwL_file = filepath + "\\" + infilename + "_14_LINE(BW).png";				//�u�G�Ȥ�(�G��)
	//imwrite(bwL_file, bwL);

	///*�j����I�s�q*/

	//Mat gradmACL, graddACL, lineACL;			//�u��ٺ��I�s�q(8UC1�B32FC1�B8UC1(BW))
	//BWConnectLine(gradmCIP, graddCIP, bwL, gradmACL, graddACL, lineACL, 2, 5, 90, 1, 1);

	//Mat gradmACL_out, graddACL_out, gradfACL_out;		//��X��(8UC1�B8UC3�B8UC3)
	//DrawAbsGraySystem(gradmACL, gradmACL_out);
	//DrawColorSystem(graddACL, graddACL_out);
	//DrawColorSystem(gradmACL, graddACL, gradfACL_out);

	//string lineACL_file = filepath + "\\" + infilename + "_15.0_ACL.png";			//�j����I�s�q(�G��)
	//imwrite(lineACL_file, lineACL);
	//string gradmACL_file = filepath + "\\" + infilename + "_15.1_ACLM.png";			//�j����I�s�q(�T��)
	//imwrite(gradmACL_file, gradmACL_out);
	//string graddACL_file = filepath + "\\" + infilename + "_15.2_ACLD.png";			//�j����I�s�q(��V)
	//imwrite(graddACL_file, graddACL_out);
	//string gradfACL_file = filepath + "\\" + infilename + "_15.3_ACLF.png";			//�j����I�s�q(��)
	//imwrite(gradfACL_file, gradfACL_out);

	///*�h�����u*/

	//Mat lineCNL;	//�h�����u(8UC1(BW))
	//ClearNoise(lineACL, lineCNL, 20, 8, 1);

	//string lineCNL_file = filepath + "\\" + infilename + "_16_CNL.png";				//�h�����u(�G��)
	//imwrite(lineCNL_file, lineCNL);


	/****���X���P�u���Ѩ����G****/

	/*���X���P�u*/

	Mat combineBW;	//���X���P�u(8UC1(BW))
	BWCombine(cbA, bwL, combineBW);

	Mat combine_L_out, combine_I_out;		//��X��(8UC3�B8UC3)
	DrawLabel(combineBW, combine_L_out);
	DrawEdge(combineBW, srcImage, combine_I_out);

	string combine_file = filepath + "\\" + infilename + "_14.0_COMBINE(BW).png";			//���X���P�u(�G��)
	imwrite(combine_file, combineBW);
	string combine_L_file = filepath + "\\" + infilename + "_14.1_COMBINE(L).png";			//���X���P�u(����)
	imwrite(combine_L_file, combine_L_out);
	string combine_I_file = filepath + "\\" + infilename + "_14.2_COMBINE(I).png";			//���X���P�u(�|��)
	imwrite(combine_I_file, combine_I_out);

	/*��ɤ������n*/

	Mat fillBW;		//��ɤ������n(8UC1(BW))
	ClearNoise(combineBW, fillBW, 50, 4, 1);

	Mat fill_L_out, fill_I_out;		//��X��(8UC3�B8UC3)
	DrawLabel(fillBW, fill_L_out);
	DrawEdge(fillBW, srcImage, fill_I_out);

	string fill_file = filepath + "\\" + infilename + "_15.0_FILL(BW).png";			//��ɤ������n(�G��)
	imwrite(fill_file, fillBW);
	string fill_L_file = filepath + "\\" + infilename + "_15.1_FILL(L).png";		//��ɤ������n(����)
	imwrite(fill_L_file, fill_L_out);
	string fill_I_file = filepath + "\\" + infilename + "_15.2_FILL(I).png";		//��ɤ������n(�|��)
	imwrite(fill_I_file, fill_I_out);

	/*�Z���ഫ*/

	Mat dtBW;		//�Z���ഫ(32FC1(BW))
	distanceTransform(combineBW, dtBW, CV_DIST_L2, 3);

	Mat dt_out,dt_BR_out;		//��X��(8UC1)
	DrawAbsGraySystem(dtBW, dt_out);
	DrawColorBar(dt_out, dt_BR_out);

	string dt_file = filepath + "\\" + infilename + "_16.0_DT.png";					//�Z���ഫ(�Ƕ�)
	imwrite(dt_file, dt_out);
	string dt_BR_file = filepath + "\\" + infilename + "_16.1_DT(BR).png";					//�Z���ഫ(�Ŭ�)
	imwrite(dt_BR_file, dt_BR_out);

	/*�������t��k����*/

	Mat wsBW;		//�������t��k����(32SC1(BW))
	BWWatershed(srcImage, combineBW, combineBW, wsBW);

	Mat ws_L_out, ws_I_out;		//��X��(8UC3�B8UC3)
	DrawLabel(wsBW, ws_L_out);
	DrawEdge(wsBW, srcImage, ws_I_out);

	string ws_file = filepath + "\\" + infilename + "_17.0_WS(BW).png";				//�������t��k����(�G��)
	imwrite(ws_file, wsBW);
	string ws_L_file = filepath + "\\" + infilename + "_17.1_WS(L).png";			//�������t��k����(����)
	imwrite(ws_L_file, ws_L_out);
	string ws_I_file = filepath + "\\" + infilename + "_17.2_WS(I).png";			//�������t��k����(�|��)
	imwrite(ws_I_file, ws_I_out);

	return 0;
}


