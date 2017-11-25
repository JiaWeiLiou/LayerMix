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

		string gray_outfile = filepath + "\\" + infilename + "_0.0_GRAY.png";		//�Ƕ��v��(�Ƕ�)
		imwrite(gray_outfile, grayImage);
	}
	else { grayImage = srcImage; }

	Mat grayColor_out;		//��X��(8UC3)
	DrawColorbar(grayImage, grayColor_out);

	string grayColor_outfile = filepath + "\\" + infilename + "_0.1_GRAY.png";		//�Ƕ��v��(�m��)
	imwrite(grayColor_outfile, grayColor_out);

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

	Mat gradx_out, grady_out, gradm_out, gradc_out,gradd_out, gradf_out;		//��X��(8UC1�B8UC1�B8UC1�B8UC3�B8UC3�B8UC3)
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorBar(gradm, gradc_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradf, gradf_out);

	string gradx_outfile = filepath + "\\" + infilename + "_4.1_GX.png";			//��X�μv�����(����)
	imwrite(gradx_outfile, gradx_out);
	string grady_outfile = filepath + "\\" + infilename + "_4.2_GY.png";			//��X�μv�����(����)
	imwrite(grady_outfile, grady_out);
	string gradm_outfile = filepath + "\\" + infilename + "_4.3_GM.png";			//��X�μv�����(�T��)
	imwrite(gradm_outfile, gradm_out);
	string gradc_outfile = filepath + "\\" + infilename + "_4.4_GC.png";			//��X�μv�����(�m��)
	imwrite(gradc_outfile, gradc_out);
	string gradd_outfile = filepath + "\\" + infilename + "_4.5_GD.png";			//��X�μv�����(��V)
	imwrite(gradd_outfile, gradd_out);
	string gradf_outfile = filepath + "\\" + infilename + "_4.6_GF.png";			//��X�μv�����(��)
	imwrite(gradf_outfile, gradf_out);

	/*��V�ҽk*/

	Mat graddBlur;	//�ҽk��V(8UC1)	
	BlurDirection(gradd, graddBlur, 5);

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

	string blurM_outfile = filepath + "\\" + infilename + "_6_BLURM.png";			//�ҽk�T��(�T��)
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

	/*�������*/

	Mat gradmHC, graddHC;		//�������(8UC1�B32FC1)
	HysteresisCut(gradmDivide, graddBlur, clearBlackArea, gradmHC, graddHC);

	Mat gradmHC_out, graddHC_out, gradfHC_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmHC, gradmHC_out);
	DrawColorSystem(graddHC, graddHC_out);
	DrawColorSystem(gradmHC, graddHC, gradfHC_out);

	string gradmHC_outfile = filepath + "\\" + infilename + "_8.1_HCM.png";			//�������(�T��)
	imwrite(gradmHC_outfile, gradmHC_out);
	string graddHC_outfile = filepath + "\\" + infilename + "_8.2_HCD.png";			//�������(��V)
	imwrite(graddHC_outfile, graddHC_out);
	string gradfHC_outfile = filepath + "\\" + infilename + "_8.3_HCF.png";			//�������(��)
	imwrite(gradfHC_outfile, gradfHC_out);

	/*�D���j�ȧ��*/

	Mat gradmNMS, graddNMS;			//�D�̤j�ȧ��(8UC1�B32FC1)
	NonMaximumSuppression(gradmHC, graddHC, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMS_outfile = filepath + "\\" + infilename + "_9.1_NMSM.png";			//�D�̤j�ȧ��(�T��)
	imwrite(gradmNMS_outfile, gradmNMS_out);
	string graddNMS_outfile = filepath + "\\" + infilename + "_9.2_NMSD.png";			//�D�̤j�ȧ��(��V)
	imwrite(graddNMS_outfile, graddNMS_out);
	string gradfNMS_outfile = filepath + "\\" + infilename + "_9.3_NMSF.png";			//�D�̤j�ȧ��(��)
	imwrite(gradfNMS_outfile, gradfNMS_out);

	/*�M������V�I*/

	Mat gradmCDD, graddCDD;			//�M������V�I(8UC1�B32FC1)
	ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	DrawColorSystem(graddCDD, graddCDD_out);
	DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	string gradmCDD_outfile = filepath + "\\" + infilename + "_10.1_CDDM.png";			//�M������V�I(�T��)
	imwrite(gradmCDD_outfile, gradmCDD_out);
	string graddCDD_outfile = filepath + "\\" + infilename + "_10.2_CDDD.png";			//�M������V�I(��V)
	imwrite(graddCDD_outfile, graddCDD_out);
	string gradfCDD_outfile = filepath + "\\" + infilename + "_10.3_CDDF.png";			//�M������V�I(��)
	imwrite(gradfCDD_outfile, gradfCDD_out);

	/*�G�Ȥ�*/

	Mat lineHT;		//�G�Ȥ�(8UC1(BW))
	threshold(gradmCDD, lineHT, 1, 255, THRESH_BINARY);

	string LHT_outfile = filepath + "\\" + infilename + "_11_BW.png";			//�G�Ȥ�(�G��)
	imwrite(LHT_outfile, lineHT);

	/*��ٺ��I�s�q*/

	Mat gradmSCL, graddSCL, lineSCL;			//�u��ٺ��I�s�q(8UC1�B32FC1�B8UC1(BW))
	BWConnectLine(gradmCDD, graddCDD, lineHT, gradmSCL, graddSCL, lineSCL, 2, 5, 60, 0, 0);

	Mat gradmSCL_out, graddSCL_out, gradfSCL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmSCL, gradmSCL_out);
	DrawColorSystem(graddSCL, graddSCL_out);
	DrawColorSystem(gradmSCL, graddSCL, gradfSCL_out);

	string LSCL_outfile = filepath + "\\" + infilename + "_12.0_SCL.png";				//��ٺ��I�s�q(�G��)
	imwrite(LSCL_outfile, lineSCL);
	string gradmSCL_outfile = filepath + "\\" + infilename + "_12.1_SCLM.png";			//��ٺ��I�s�q(�T��)
	imwrite(gradmSCL_outfile, gradmSCL_out);
	string graddSCL_outfile = filepath + "\\" + infilename + "_12.2_SCLD.png";			//��ٺ��I�s�q(��V)
	imwrite(graddSCL_outfile, graddSCL_out);
	string gradfSCL_outfile = filepath + "\\" + infilename + "_12.3_SCLF.png";			//��ٺ��I�s�q(��)
	imwrite(gradfSCL_outfile, gradfSCL_out);

	/*�h���t���I*/

	Mat lineCP;	//�h���t���I(8UC1(BW))
	ClearPoint(lineSCL, lineCP, 0, 1, 0);

	string LCP_outfile = filepath + "\\" + infilename + "_13_CP.png";			//�h���t���I(�G��)
	imwrite(LCP_outfile, lineCP);

	/*�j����I�s�q*/

	Mat gradmACL, graddACL, lineACL;			//�u��ٺ��I�s�q(8UC1�B32FC1�B8UC1(BW))
	BWConnectLine(gradmSCL, graddSCL, lineCP, gradmACL, graddACL, lineACL, 2, 5, 90, 1, 1);

	Mat gradmACL_out, graddACL_out, gradfACL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmACL, gradmACL_out);
	DrawColorSystem(graddACL, graddACL_out);
	DrawColorSystem(gradmACL, graddACL, gradfACL_out);

	string LACL_outfile = filepath + "\\" + infilename + "_14.0_ACL.png";				//�j����I�s�q(�G��)
	imwrite(LACL_outfile, lineACL);
	string gradmACL_outfile = filepath + "\\" + infilename + "_14.1_ACLM.png";			//�j����I�s�q(�T��)
	imwrite(gradmACL_outfile, gradmACL_out);
	string graddACL_outfile = filepath + "\\" + infilename + "_14.2_ACLD.png";			//�j����I�s�q(��V)
	imwrite(graddACL_outfile, graddACL_out);
	string gradfACL_outfile = filepath + "\\" + infilename + "_14.3_ACLF.png";			//�j����I�s�q(��)
	imwrite(gradfACL_outfile, gradfACL_out);

	/*�h�����u*/

	Mat lineCNL;	//�h�����u(8UC1(BW))
	//ClearSpecialPoint(lineACBL, lineCNL, blurLineSize, 5, 1);
	ClearNoise(lineACL, lineCNL, 30, 8, 1);

	string LCNL_outfile = filepath + "\\" + infilename + "_15_CNL.png";			//�h�����u(�G��)
	imwrite(LCNL_outfile, lineCNL);


	/****���X���P�u���Ѩ����G****/

	/*���X���P�u*/

	Mat combine;	//���X���P�u(8UC1(BW))
	BWCombine(clearBlackArea, lineCNL, combine);

	Mat combineLabel_out, combineImage_out;		//��X��(8UC3�B8UC3)
	DrawLabel(combine, combineLabel_out);
	DrawEdge(combine, srcImage, combineImage_out);

	string combineBW_outfile = filepath + "\\" + infilename + "_16.0_COMBINE.png";				//���X���P�u(�G��)
	imwrite(combineBW_outfile, combine);
	string combineLabel_outfile = filepath + "\\" + infilename + "_16.1_COMBINE.png";			//���X���P�u(����)
	imwrite(combineLabel_outfile, combineLabel_out);
	string combineImage_outfile = filepath + "\\" + infilename + "_16.2_COMBINE.png";			//���X���P�u(�|��)
	imwrite(combineImage_outfile, combineImage_out);

	///*�������t��k����*/

	return 0;
}


