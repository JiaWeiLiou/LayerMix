// LayerMix.cpp : �w�q�D���x���ε{�����i�J�I�C
//

#include "stdafx.h"
#include <iostream>
#include <opencv2/opencv.hpp>  
#include <cmath>

#define UNKNOWN_FLOW_THRESH 1e9
#define PI 3.14159265359

using namespace std;
using namespace cv;

void LayerMix(InputArray _grayImage, InputArray _blurImage, OutputArray _mixImage);
void Differential(InputArray _grayImage, OutputArray _grad_x, OutputArray _grad_y, OutputArray _grad_mag);
void DrawAbsGraySystem(InputArray _field, OutputArray _grayField);

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

	Mat mixImageArea;
	LayerMix(grayImage, blurImage, mixImageArea);
	string lmaOutfile = filepath + "\\" + infilename + "_LMA.png";			//��󭱪��ϼh�V�X�Ҧ�
	imwrite(lmaOutfile, mixImageArea);

	/*�p��v�����*/
	Mat gradx,grady,gradm;
	Differential(grayImage, gradx, grady, gradm);
	DrawAbsGraySystem(gradx, gradx);
	DrawAbsGraySystem(grady, grady);
	DrawAbsGraySystem(gradm, gradm);

	string gxOutfile = filepath + "\\" + infilename + "_Gx.png";			//�v�����(����)
	imwrite(gxOutfile, gradx);
	string gyOutfile = filepath + "\\" + infilename + "_Gy.png";			//�v�����(����)
	imwrite(gyOutfile, grady);
	string gmOutfile = filepath + "\\" + infilename + "_Gm.png";			//�v�����(�T��)
	imwrite(gmOutfile, gradm);

	/*��פ��βV�X*/

    return 0;
}


void LayerMix(InputArray _grayImage, InputArray _blurImage, OutputArray _mixImage)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	Mat blurImage = _blurImage.getMat();
	CV_Assert(blurImage.type() == CV_8UC1);

	_mixImage.create(grayImage.size(), CV_8UC1);
	Mat mixImage = _mixImage.getMat();

	double divide = 0;
	for (int i = 0; i < grayImage.rows; ++i)
	{
		for (int j = 0; j < grayImage.cols; ++j)
		{
			divide = (double)grayImage.at<uchar>(i, j) / (double)blurImage.at<uchar>(i, j) > 1 ? 255 : ((double)grayImage.at<uchar>(i, j) / (double)blurImage.at<uchar>(i, j)) * 255.0;
			mixImage.at<uchar>(i, j) = divide + (double)blurImage.at<uchar>(i, j) < 255.0 ? 0 : 255;
		}
	}
}

void Differential(InputArray _grayImage, OutputArray _grad_x, OutputArray _grad_y, OutputArray _grad_mag) {

	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	_grad_x.create(grayImage.size(), CV_8UC1);
	Mat grad_x = _grad_x.getMat();

	_grad_y.create(grayImage.size(), CV_8UC1);
	Mat grad_y = _grad_y.getMat();

	_grad_mag.create(grayImage.size(), CV_8UC1);
	Mat grad_mag = _grad_mag.getMat();

	Mat grayImageRef;
	copyMakeBorder(grayImage, grayImageRef, 1, 1, 1, 1, BORDER_REPLICATE);
	float gradx_temp = 0;
	float grady_temp = 0;
	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j) {
			gradx_temp = ((float)grayImageRef.at<uchar>(i + 1, j) - (float)grayImageRef.at<uchar>(i + 1, j + 2))*0.5;
			grady_temp = ((float)grayImageRef.at<uchar>(i, j + 1) - (float)grayImageRef.at<uchar>(i + 2, j + 1))*0.5;

			grad_x.at<uchar>(i, j) = abs(gradx_temp);
			grad_y.at<uchar>(i, j) = abs(grady_temp);
			grad_mag.at<uchar>(i, j) = sqrt(pow(gradx_temp, 2) + pow(grady_temp, 2));
		}
}

void DrawAbsGraySystem(InputArray _field, OutputArray _grayField)
{
	Mat field = _field.getMat();

	_grayField.create(field.size(), CV_8UC1);
	Mat grayField = _grayField.getMat();

	// determine motion range:  
	double maxvalue = 0;

	// Find max flow to normalize fx and fy  
	for (int i = 0; i < field.rows; ++i)
		for (int j = 0; j < field.cols; ++j)
			maxvalue = maxvalue > field.at<uchar>(i, j) ? maxvalue : field.at<uchar>(i, j);

	for (int i = 0; i < field.rows; ++i)
		for (int j = 0; j < field.cols; ++j) {
			grayField.at<uchar>(i, j) = ((double)field.at<uchar>(i, j) / maxvalue) * 255;
		}
}