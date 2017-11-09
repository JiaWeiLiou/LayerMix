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

		string grayoutfile = filepath + "\\" + infilename + "_0_GRAY.png";		//�Ƕ��v��
		imwrite(grayoutfile, grayImage);
	}
	else
		grayImage = srcImage;


	/****��󭱪��v���Ѩ�****/

	/*�N�ǫ׹Ϲ������ҽk*/

	Mat blurAImage;	//�ҽk�v��(8UC1)	
	GaussianBlur(grayImage, blurAImage, Size(blurAreaSize, blurAreaSize), 0, 0);

	string bluraoutfile = filepath + "\\" + infilename + "_1_BLURA.png";		//�ҽk�v��
	imwrite(bluraoutfile, blurAImage);

	/*�ϼh�V�X�Ҧ�*/

	Mat divideArea;		//���βV�X�Ҧ�(8UC1)
	Divide(grayImage, blurAImage, divideArea);

	string divideaOutfile = filepath + "\\" + infilename + "_2.1_DIVIDEA.png";			//���βV�X�Ҧ�
	imwrite(divideaOutfile, divideArea);

	Mat hardmixArea;	//���L�|�X�V�X�Ҧ�(8UC1 and �G�ȼv��)
	HardMix(grayImage, divideArea, hardmixArea);			

	string hardmixaOutfile = filepath + "\\" + infilename + "_2.2_HARDMIXA.png";			//���L�|�X�V�X�Ҧ�
	imwrite(hardmixaOutfile, hardmixArea);

	/*�h���v�����T*/

	Mat clearWiteArea;		//�h���զ����T(8UC1 and �G�ȼv��)
	ClearNoise(hardmixArea, clearWiteArea, 20, 4, 1);

	string clearwOutfile = filepath + "\\" + infilename + "_3.1_CLEARW.png";			//�h���զ����T
	imwrite(clearwOutfile, clearWiteArea);

	Mat clearBlackArea;		//�h���¦����T(8UC1 and �G�ȼv��)
	ClearNoise(clearWiteArea, clearBlackArea, 20, 4, 0);

	string clearbOutfile = filepath + "\\" + infilename + "_3.2_CLEARB.png";			//�h���¦����T
	imwrite(clearbOutfile, clearBlackArea);


	/****���u���v���Ѩ�****/

	/*�N�ǫ׹Ϲ����ȼҽk*/

	Mat blurLImage;	//�ҽk�v��(8UC1)	
	medianBlur(grayImage, blurLImage, blurLineSize);

	string blurloutfile = filepath + "\\" + infilename + "_4_BLURL.png";		//�ҽk�v��
	imwrite(blurloutfile, blurLImage);

	/*�p��v�����*/

	Mat gradx,grady;		//�����Ϋ������(16SC1)
	Differential(blurLImage, gradx, grady);

	Mat gradf;			//��׳�(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;		//��״T�Ȥα�פ�V(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_out, grady_out, gradm_out, gradd_out, gradf_out;		//��X��(8UC1 or 8UC3)
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradf, gradf_out);

	string gxOutfile = filepath + "\\" + infilename + "_5.1_GX.png";			//��X�μv�����(����)
	imwrite(gxOutfile, gradx_out);
	string gyOutfile = filepath + "\\" + infilename + "_5.2_GY.png";			//��X�μv�����(����)
	imwrite(gyOutfile, grady_out);
	string gmOutfile = filepath + "\\" + infilename + "_5.3_GM.png";			//��X�μv�����(�T��)
	imwrite(gmOutfile, gradm_out);
	string gdOutfile = filepath + "\\" + infilename + "_5.4_GD.png";			//��X�μv�����(��V)
	imwrite(gdOutfile, gradd_out);
	string gfOutfile = filepath + "\\" + infilename + "_5.5_GF.png";			//��X�μv�����(��)
	imwrite(gfOutfile, gradf_out);

	/*���βV�X�Ҧ�*/

	Mat divideLine;										//���βV�X�Ҧ�(8UC1)
	Divide(gradm, blurLImage, divideLine);

	Mat gradmDivide_out, gradfDivide_out;				//��X��(8UC1 or 8UC3)
	DrawAbsGraySystem(divideLine, gradmDivide_out);
	DrawColorSystem(divideLine, gradd, gradfDivide_out);

	string gradmDivideOutfile = filepath + "\\" + infilename + "_6.1_DIVIDEM.png";			//��X�Τ��βV�X�Ҧ�(�T��)
	imwrite(gradmDivideOutfile, gradmDivide_out);
	string gradfDivideOutfile = filepath + "\\" + infilename + "_6.2_DIVIDEF.png";			//��X�Τ��βV�X�Ҧ�(��)
	imwrite(gradfDivideOutfile, gradfDivide_out);

	/*�D���j�ȧ��*/
	
	Mat gradmNMS, graddNMS;			//�D�̤j�ȧ��(8UC1�B32FC1)
	NonMaximumSuppression(divideLine, gradd, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//��X��(8UC1 or 8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMSOutfile = filepath + "\\" + infilename + "_7.1_NMSM.png";			//�D�̤j�ȧ��(�T��)
	imwrite(gradmNMSOutfile, gradmNMS_out);
	string graddNMSOutfile = filepath + "\\" + infilename + "_7.2_NMSD.png";			//�D�̤j�ȧ��(��V)
	imwrite(graddNMSOutfile, graddNMS_out);
	string gradFNMSOutfile = filepath + "\\" + infilename + "_7.3_NMSF.png";			//�D�̤j�ȧ��(��)
	imwrite(gradFNMSOutfile, gradfNMS_out);

	/*�M������V�I*/
	
	Mat gradmCDD, graddCDD;			//�M������V�I(8UC1�B32FC1)
	ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//��X��(8UC1 or 8UC3)
	DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	DrawColorSystem(graddCDD, graddCDD_out);
	DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	string gradmCDDOutfile = filepath + "\\" + infilename + "_8.1_CDDM.png";			//�M������V�I(�T��)
	imwrite(gradmCDDOutfile, gradmCDD_out);
	string graddCDDOutfile = filepath + "\\" + infilename + "_8.2_CDDD.png";			//�M������V�I(��V)
	imwrite(graddCDDOutfile, graddCDD_out);
	string gradFCDDOutfile = filepath + "\\" + infilename + "_8.3_CDDF.png";			//�M������V�I(��)
	imwrite(gradFCDDOutfile, gradfCDD_out);

	/*��B�_�u�s�q*/

	Mat gradmWCBL, graddWCBL;			//�_�u�s�q(8UC1�B32FC1)
	ConnectBreakLine(gradmCDD, graddCDD, gradmWCBL, graddWCBL, 2, 3, 60, 1);

	Mat gradmWCBL_out, graddWCBL_out, gradfWCBL_out;		//��X��(8UC1 or 8UC3)
	DrawAbsGraySystem(gradmWCBL, gradmWCBL_out);
	DrawColorSystem(graddWCBL, graddWCBL_out);
	DrawColorSystem(gradmWCBL, graddWCBL, gradfWCBL_out);

	string gradmWCBLOutfile = filepath + "\\" + infilename + "_9.1_WCBLM.png";			//��B�_�u�s�q(�T��)
	imwrite(gradmWCBLOutfile, gradmWCBL_out);
	string graddWCBLOutfile = filepath + "\\" + infilename + "_9.2_WCBLD.png";			//��B�_�u�s�q(��V)
	imwrite(graddWCBLOutfile, graddWCBL_out);
	string gradWFCBLOutfile = filepath + "\\" + infilename + "_9.3_WCBLF.png";			//��B�_�u�s�q(��)
	imwrite(gradWFCBLOutfile, gradfWCBL_out);

	/*����֭�*/

	Mat edgeHT;		//����֭�(8UC1 and �G�ȼv��)
	HysteresisThreshold(gradmWCBL, edgeHT, 150, 50);
	string edgehtOutfile = filepath + "\\" + infilename + "_10_HT.png";			//����֭�
	imwrite(edgehtOutfile, edgeHT);

	/*�j���_�u�s�q*/

	Mat edgeFCBL;		//�j���_�u�s�q(8UC1)
	BWConnectBreakLine(gradmWCBL, graddWCBL, edgeHT, edgeFCBL, 2, 5, 90, 0);
	string edgefcblOutfile = filepath + "\\" + infilename + "_11_FCBL.png";			//�j���_�u�s�q
	imwrite(edgefcblOutfile, edgeFCBL);

    return 0;
}


