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

	Mat srcImage = imread(infile);	//��l��
	if (!srcImage.data) { printf("Oh�Ano�AŪ��srcImage���~~�I \n"); return false; }

	/*�N��Ϲ��ഫ���ǫ׹Ϲ�*/

	Mat grayImage;
	if (srcImage.type() != CV_8UC1)
	{
		cvtColor(srcImage, grayImage, CV_BGR2GRAY);

		string grayoutfile = filepath + "\\" + infilename + "_0_GRAY.png";		//�Ƕ��v��
		imwrite(grayoutfile, grayImage);
	}
	else
		grayImage = srcImage;

	/*�N�ǫ׹Ϲ����ȼҽk*/
	Mat blurImage;
	GaussianBlur(grayImage, blurImage, Size(blurSize, blurSize), 0, 0);

	string bluroutfile = filepath + "\\" + infilename + "_1_BLUR.png";		//�ҽk�v��
	imwrite(bluroutfile, blurImage);

	/*��󭱪��ϼh�V�X�Ҧ�*/

	Mat divideArea;
	Divide(grayImage, blurImage, divideArea);

	string divideaOutfile = filepath + "\\" + infilename + "_DIVIDEA.png";			//��󭱪��ϼh�V�X�Ҧ�
	imwrite(divideaOutfile, divideArea);

	Mat hardmixArea;
	HardMix(grayImage, divideArea, hardmixArea);

	string hardmixaOutfile = filepath + "\\" + infilename + "_HARDMIXA.png";			//��󭱪��ϼh�V�X�Ҧ�
	imwrite(hardmixaOutfile, hardmixArea);

	/*�h���v�����T*/

	Mat clearWiteArea,clearBlackArea;
	ClearNoise(hardmixArea, clearWiteArea, 20, 4, 1);

	string clearwOutfile = filepath + "\\" + infilename + "_CLEARW.png";			//��󭱪��ϼh�V�X�Ҧ�
	imwrite(clearwOutfile, clearWiteArea);

	ClearNoise(clearWiteArea, clearBlackArea, 20, 4, 0);

	string clearbOutfile = filepath + "\\" + infilename + "_CLEARB.png";			//��󭱪��ϼh�V�X�Ҧ�
	imwrite(clearbOutfile, clearBlackArea);

	/*�p��v�����*/

	Mat gradx,grady;		//16SC1
	Differential(grayImage, gradx, grady);

	Mat gradField;			//16SC2
	GradientField(gradx, grady, gradField);			//���X��׳�

	Mat gradm, gradd;		//32FC1
	CalculateGradient(gradField, gradm, gradd);		//�p���״T�ȤΤ�V

	Mat gradx_out, grady_out, gradm_out, gradd_out, gradf_out;		//8UC1
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradField, gradf_out);

	string gxOutfile = filepath + "\\" + infilename + "_GX.png";			//�v�����(����)
	imwrite(gxOutfile, gradx_out);
	string gyOutfile = filepath + "\\" + infilename + "_GY.png";			//�v�����(����)
	imwrite(gyOutfile, grady_out);
	string gmOutfile = filepath + "\\" + infilename + "_GM.png";			//�v�����(�T��)
	imwrite(gmOutfile, gradm_out);
	string gdOutfile = filepath + "\\" + infilename + "_GD.png";			//�v�����(��V)
	imwrite(gdOutfile, gradd_out);
	string gfOutfile = filepath + "\\" + infilename + "_GF.png";			//�v�����(��)
	imwrite(gfOutfile, gradf_out);

	/*�D�̤j�ȧ��*/

	Mat gradNMS;			//16SC2
	NonMaximumSuppression(gradField, gradNMS);

	Mat gradmNMS_out, gradfNMS_out;		//8UC1
	DrawAbsGraySystem(gradNMS, gradmNMS_out);
	DrawColorSystem(gradNMS, gradfNMS_out);

	string mNMSOutfile = filepath + "\\" + infilename + "_M_NMS.png";			//�D�̤j�ȧ��(�T��)
	imwrite(mNMSOutfile, gradmNMS_out);
	string fNMSOutfile = filepath + "\\" + infilename + "_F_NMS.png";			//�D�̤j�ȧ��(��)
	imwrite(fNMSOutfile, gradfNMS_out);

	/*�t���V�X�Ҧ�*/

	Mat gradDivide;			//8UC1
	Divide(gradmNMS_out, grayImage, gradDivide);		//�t���V�X�Ҧ�	

	string divideOutfile = filepath + "\\" + infilename + "_Divide.png";			//�t���V�X�Ҧ�
	imwrite(divideOutfile, gradDivide);

	/*����֭�*/
	Mat edgeHT;		//8UC1
	HysteresisThreshold(gradDivide, edgeHT, 150, 100);
	string edgehtOutfile = filepath + "\\" + infilename + "_HT.png";			//����֭�
	imwrite(edgehtOutfile, edgeHT);

    return 0;
}


