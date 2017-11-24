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

	Mat grayImage;		//�Ƕ��v��(8UC1)
	if (srcImage.type() != CV_8UC1)
	{
		cvtColor(srcImage, grayImage, CV_BGR2GRAY);

		string gray_outfile = filepath + "\\" + infilename + "_0_GRAY.png";		//�Ƕ��v��
		imwrite(gray_outfile, grayImage);
	}
	else
		grayImage = srcImage;


	/****��󭱪��v���Ѩ�****/

	/*�N�ǫ׹Ϲ������ҽk*/

	Mat blurImage;	//�ҽk�v��(8UC1)	
	GaussianBlur(grayImage, blurImage, Size(blurAreaSize, blurAreaSize), 0, 0);

	string blurA_outfile = filepath + "\\" + infilename + "_1_BLURA.png";		//�ҽk�Ƕ��v��
	imwrite(blurA_outfile, blurImage);

	/*�ϼh�V�X�Ҧ�*/

	Mat divideArea;		//���βV�X�Ҧ�(8UC1)
	DivideArea(grayImage, blurImage, divideArea);

	string divideA_outfile = filepath + "\\" + infilename + "_2.1_DIVIDEA.png";			//�����βV�X�Ҧ�
	imwrite(divideA_outfile, divideArea);

	Mat hardmixArea;	//���L�|�X�V�X�Ҧ�(8UC1(BW))
	HardMix(grayImage, divideArea, hardmixArea);

	string hardmixA_outfile = filepath + "\\" + infilename + "_2.2_HARDMIXA.png";		//���L�|�X�V�X�Ҧ�
	imwrite(hardmixA_outfile, hardmixArea);

	/*�h���v�����T*/

	Mat clearWiteArea;		//�h���զ����T(8UC1(BW))
	ClearNoise(hardmixArea, clearWiteArea, 20, 4, 1);

	string clearWA_outfile = filepath + "\\" + infilename + "_3.1_CLEARW.png";			//�h���զ����T
	imwrite(clearWA_outfile, clearWiteArea);

	Mat clearBlackArea;		//�h���¦����T(8UC1(BW))
	ClearNoise(clearWiteArea, clearBlackArea, 20, 4, 0);

	string clearBA_outfile = filepath + "\\" + infilename + "_3.2_CLEARB.png";			//�h���¦����T
	imwrite(clearBA_outfile, clearBlackArea);


	/****���u���v���Ѩ�****/

	/*�p��v�����*/

	Mat gradx, grady;		//�����Ϋ������(16SC1)
	Differential(grayImage, gradx, grady);

	Mat gradf;			//��׳�(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;		//��״T�Ȥα�פ�V(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_out, grady_out, gradm_out, gradd_out, gradf_out;		//��X��(8UC1�B8UC1�B8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradf, gradf_out);

	string gradx_outfile = filepath + "\\" + infilename + "_4.1_GX.png";			//��X�μv�����(����)
	imwrite(gradx_outfile, gradx_out);
	string grady_outfile = filepath + "\\" + infilename + "_4.2_GY.png";			//��X�μv�����(����)
	imwrite(grady_outfile, grady_out);
	string gradm_outfile = filepath + "\\" + infilename + "_4.3_GM.png";			//��X�μv�����(�T��)
	imwrite(gradm_outfile, gradm_out);
	string gradd_outfile = filepath + "\\" + infilename + "_4.4_GD.png";			//��X�μv�����(��V)
	imwrite(gradd_outfile, gradd_out);
	string gradf_outfile = filepath + "\\" + infilename + "_4.5_GF.png";			//��X�μv�����(��)
	imwrite(gradf_outfile, gradf_out);

	/*��V�ҽk*/

	Mat graddBlur;	//�ҽk��V(8UC1)	
	BlurDirection(gradd, graddBlur, 11);

	Mat graddBlur_out, gradfBlur_out;		//��X��(8UC3�B8UC3)
	DrawColorSystem(graddBlur, graddBlur_out);
	DrawColorSystem(gradm, graddBlur, gradfBlur_out);

	string graddBlur_outfile = filepath + "\\" + infilename + "_5.1_BLURD.png";			//�ҽk��V(��V)
	imwrite(graddBlur_outfile, graddBlur_out);
	string gradfBlur_outfile = filepath + "\\" + infilename + "_5.2_BLURF.png";			//�ҽk��V(��)
	imwrite(gradfBlur_outfile, gradfBlur_out);

	/*�N�T�Ȱ����ҽk*/

	Mat gradmBlur;	//�ҽk�T��(8UC1)	
	GaussianBlur(gradm, gradmBlur, Size(blurLineSize, blurLineSize), 0, 0);

	string blurM_outfile = filepath + "\\" + infilename + "_6_BLURM.png";			//�ҽk�T��
	imwrite(blurM_outfile, gradmBlur);

	/*�u���βV�X�Ҧ�*/

	Mat gradmDivide;										//�u���βV�X�Ҧ�(8UC1)
	DivideLine(gradm, gradmBlur, gradmDivide);

	Mat gradmDivide_out, gradfDivide_out;		//��X��(8UC1�B8UC3)
	DrawAbsGraySystem(gradmDivide, gradmDivide_out);
	DrawColorSystem(gradmDivide, graddBlur, gradfDivide_out);

	string gradmDivide_outfile = filepath + "\\" + infilename + "_7.1_DIVIDEM.png";			//�u���βV�X�Ҧ�(�T��)
	imwrite(gradmDivide_outfile, gradmDivide_out);
	string gradfDivide_outfile = filepath + "\\" + infilename + "_7.2_DIVIDEF.png";			//�u���βV�X�Ҧ�(��)
	imwrite(gradfDivide_outfile, gradfDivide_out);

	/*�D���j�ȧ��*/

	Mat gradmNMS, graddNMS;			//�D�̤j�ȧ��(8UC1�B32FC1)
	NonMaximumSuppression(gradmDivide, graddBlur, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMS_outfile = filepath + "\\" + infilename + "_8.1_NMSM.png";			//�D�̤j�ȧ��(�T��)
	imwrite(gradmNMS_outfile, gradmNMS_out);
	string graddNMS_outfile = filepath + "\\" + infilename + "_8.2_NMSD.png";			//�D�̤j�ȧ��(��V)
	imwrite(graddNMS_outfile, graddNMS_out);
	string gradfNMS_outfile = filepath + "\\" + infilename + "_8.3_NMSF.png";			//�D�̤j�ȧ��(��)
	imwrite(gradfNMS_outfile, gradfNMS_out);

	/*�M������V�I*/

	Mat gradmCDD, graddCDD;			//�M������V�I(8UC1�B32FC1)
	ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	DrawColorSystem(graddCDD, graddCDD_out);
	DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	string gradmCDD_outfile = filepath + "\\" + infilename + "_9.1_CDDM.png";			//�M������V�I(�T��)
	imwrite(gradmCDD_outfile, gradmCDD_out);
	string graddCDD_outfile = filepath + "\\" + infilename + "_9.2_CDDD.png";			//�M������V�I(��V)
	imwrite(graddCDD_outfile, graddCDD_out);
	string gradfCDD_outfile = filepath + "\\" + infilename + "_9.3_CDDF.png";			//�M������V�I(��)
	imwrite(gradfCDD_outfile, gradfCDD_out);

	/*�_�u�s�q*/

	Mat gradmCBL, graddCBL;			//�_�u�s�q(8UC1�B32FC1)
	ConnectBreakLine(gradmCDD, graddCDD, gradmCBL, graddCBL, 2, 5, 60, 0, 0);

	Mat gradmCBL_out, graddCBL_out, gradfCBL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmCBL, gradmCBL_out);
	DrawColorSystem(graddCBL, graddCBL_out);
	DrawColorSystem(gradmCBL, graddCBL, gradfCBL_out);

	string gradmCBL_outfile = filepath + "\\" + infilename + "_10.1_CBLM.png";			//�_�u�s�q(�T��)
	imwrite(gradmCBL_outfile, gradmCBL_out);
	string graddCBL_outfile = filepath + "\\" + infilename + "_10.2_CBLD.png";			//�_�u�s�q(��V)
	imwrite(graddCBL_outfile, graddCBL_out);
	string gradfCBL_outfile = filepath + "\\" + infilename + "_10.3_CBLF.png";			//�_�u�s�q(��)
	imwrite(gradfCBL_outfile, gradfCBL_out);

	/*����֭�*/

	Mat lineHT;		//����֭�(8UC1(BW))
	HysteresisThreshold(gradmCBL, lineHT, 200, 2);
	string LHT_outfile = filepath + "\\" + infilename + "_11_HT.png";			//����֭�
	imwrite(LHT_outfile, lineHT);

	/*��ٺ��I�s�q*/

	Mat gradmSCBL, graddSCBL, lineSCBL;			//�u��ٺ��I�s�q(8UC1�B32FC1�B8UC1(BW))
	BWConnectBreakLine(gradmCBL, graddCBL, lineHT, gradmSCBL, graddSCBL, lineSCBL, 2, 20, 90, 0, 0);

	Mat gradmSCBL_out, graddSCBL_out, gradfSCBL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmSCBL, gradmSCBL_out);
	DrawColorSystem(graddSCBL, graddSCBL_out);
	DrawColorSystem(gradmSCBL, graddSCBL, gradfSCBL_out);

	string LSCBL_outfile = filepath + "\\" + infilename + "_12.0_SCBL.png";				//��ٺ��I�s�q(�G��)
	imwrite(LSCBL_outfile, lineSCBL);
	string gradmSCBL_outfile = filepath + "\\" + infilename + "_12.1_SCBLM.png";			//��ٺ��I�s�q(�T��)
	imwrite(gradmSCBL_outfile, gradmSCBL_out);
	string graddSCBL_outfile = filepath + "\\" + infilename + "_12.2_SCBLD.png";			//��ٺ��I�s�q(��V)
	imwrite(graddSCBL_outfile, graddSCBL_out);
	string gradfSCBL_outfile = filepath + "\\" + infilename + "_12.3_SCBLF.png";			//��ٺ��I�s�q(��)
	imwrite(gradfSCBL_outfile, gradfSCBL_out);

	/*�h���t���I*/

	Mat lineCIP;	//�h���t���I(8UC1(BW))
	ClearSpecialPoint(lineSCBL, lineCIP, 0, 1, 0);
	string LCIP_outfile = filepath + "\\" + infilename + "_13_CIP.png";			//�h���t���I
	imwrite(LCIP_outfile, lineCIP);

	/*�j����I�s�q*/

	Mat gradmACBL, graddACBL, lineACBL;			//�u��ٺ��I�s�q(8UC1�B32FC1�B8UC1(BW))
	BWConnectBreakLine(gradmSCBL, graddSCBL, lineCIP, gradmACBL, graddACBL, lineACBL, 2, 5, 90, 1, 1);

	Mat gradmACBL_out, graddACBL_out, gradfACBL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmACBL, gradmACBL_out);
	DrawColorSystem(graddACBL, graddACBL_out);
	DrawColorSystem(gradmACBL, graddACBL, gradfACBL_out);

	string LACBL_outfile = filepath + "\\" + infilename + "_14.0_ACBL.png";				//�j����I�s�q(�G��)
	imwrite(LACBL_outfile, lineACBL);
	string gradmACBL_outfile = filepath + "\\" + infilename + "_14.1_ACBLM.png";			//�j����I�s�q(�T��)
	imwrite(gradmACBL_outfile, gradmACBL_out);
	string graddACBL_outfile = filepath + "\\" + infilename + "_14.2_ACBLD.png";			//�j����I�s�q(��V)
	imwrite(graddACBL_outfile, graddACBL_out);
	string gradfACBL_outfile = filepath + "\\" + infilename + "_14.3_ACBLF.png";			//�j����I�s�q(��)
	imwrite(gradfACBL_outfile, gradfACBL_out);

	/*�h�����u*/

	Mat lineCNL;	//�h�����u(8UC1(BW))
	//ClearSpecialPoint(lineACBL, lineCNL, blurLineSize, 5, 1);
	ClearNoise(lineACBL, lineCNL, 20, 8, 1);
	string LCNL_outfile = filepath + "\\" + infilename + "_15_CNL.png";			//�h�����u
	imwrite(LCNL_outfile, lineCNL);

	///*�������t��k����*/

	return 0;
}


