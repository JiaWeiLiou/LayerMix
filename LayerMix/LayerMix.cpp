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

	std::cout << "Please enter blur square size : ";
	int blurSize = 0;
	std::cin >> blurSize;

	if (blurSize % 2 == 0)
	{
		--blurSize;
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

	Mat blurImage;	//�ҽk�v��(8UC1)	
	GaussianBlur(grayImage, blurImage, Size(blurSize, blurSize), 0, 0);

	string bluroutfile = filepath + "\\" + infilename + "_1_BLUR.png";		//�ҽk�v��
	imwrite(bluroutfile, blurImage);

	/*�ϼh�V�X�Ҧ�*/

	Mat divideArea;		//���βV�X�Ҧ�(8UC1)
	Divide(grayImage, blurImage, divideArea);

	string divideaOutfile = filepath + "\\" + infilename + "_2_DIVIDEA.png";			//���βV�X�Ҧ�
	imwrite(divideaOutfile, divideArea);

	Mat hardmixArea;	//���L�|�X�V�X�Ҧ�(8UC1 and �G�ȼv��)
	HardMix(grayImage, divideArea, hardmixArea);			

	string hardmixaOutfile = filepath + "\\" + infilename + "_2_HARDMIXA.png";			//���L�|�X�V�X�Ҧ�
	imwrite(hardmixaOutfile, hardmixArea);

	/*�h���v�����T*/

	Mat clearWiteArea;		//�h���զ����T(8UC1 and �G�ȼv��)
	ClearNoise(hardmixArea, clearWiteArea, 20, 4, 1);

	string clearwOutfile = filepath + "\\" + infilename + "_3_CLEARW.png";			//�h���զ����T
	imwrite(clearwOutfile, clearWiteArea);

	Mat clearBlackArea;		//�h���¦����T(8UC1 and �G�ȼv��)
	ClearNoise(clearWiteArea, clearBlackArea, 20, 4, 0);

	string clearbOutfile = filepath + "\\" + infilename + "_3_CLEARB.png";			//�h���¦����T
	imwrite(clearbOutfile, clearBlackArea);


	/****���u���v���Ѩ�****/

	/*�p��v�����*/

	Mat gradx,grady;		//�����Ϋ������(16SC1)
	Differential(grayImage, gradx, grady);

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

	string gxOutfile = filepath + "\\" + infilename + "_4_GX.png";			//��X�μv�����(����)
	imwrite(gxOutfile, gradx_out);
	string gyOutfile = filepath + "\\" + infilename + "_4_GY.png";			//��X�μv�����(����)
	imwrite(gyOutfile, grady_out);
	string gmOutfile = filepath + "\\" + infilename + "_4_GM.png";			//��X�μv�����(�T��)
	imwrite(gmOutfile, gradm_out);
	string gdOutfile = filepath + "\\" + infilename + "_4_GD.png";			//��X�μv�����(��V)
	imwrite(gdOutfile, gradd_out);
	string gfOutfile = filepath + "\\" + infilename + "_4_GF.png";			//��X�μv�����(��)
	imwrite(gfOutfile, gradf_out);

	/*���βV�X�Ҧ�*/

	Mat divideLine;										//���βV�X�Ҧ�(8UC1)
	Divide(gradm, grayImage, divideLine);

	Mat gradmDivide_out, gradfDivide_out;				//��X��(8UC1 or 8UC3)
	DrawAbsGraySystem(divideLine, gradmDivide_out);
	DrawColorSystem(divideLine, gradd, gradfDivide_out);

	string gradmDivideOutfile = filepath + "\\" + infilename + "_5_DIVIDEM.png";			//��X�Τ��βV�X�Ҧ�(�T��)
	imwrite(gradmDivideOutfile, gradmDivide_out);
	string gradfDivideOutfile = filepath + "\\" + infilename + "_5_DIVIDEF.png";			//��X�Τ��βV�X�Ҧ�(��)
	imwrite(gradfDivideOutfile, gradfDivide_out);

	/*�D���j�ȧ��*/
	
	Mat gradmNMS, graddNMS;			//�D�̤j�ȧ��(8UC1�B32FC1)
	NonMaximumSuppression(divideLine, gradd, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//��X��(8UC1 or 8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMSOutfile = filepath + "\\" + infilename + "_6_NMSM.png";			//�D�̤j�ȧ��(�T��)
	imwrite(gradmNMSOutfile, gradmNMS_out);
	string graddNMSOutfile = filepath + "\\" + infilename + "_6_NMSD.png";			//�D�̤j�ȧ��(��V)
	imwrite(graddNMSOutfile, graddNMS_out);
	string gradFNMSOutfile = filepath + "\\" + infilename + "_6_NMSF.png";			//�D�̤j�ȧ��(��)
	imwrite(gradFNMSOutfile, gradfNMS_out);

	/*����֭�*/

	Mat edgeHT;		//����֭�(8UC1 and �G�ȼv��)
	HysteresisThreshold(gradmNMS, edgeHT, 150, 50);
	string edgehtOutfile = filepath + "\\" + infilename + "_7_HT.png";			//����֭�
	imwrite(edgehtOutfile, edgeHT);

    return 0;
}


