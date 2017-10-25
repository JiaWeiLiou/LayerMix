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

void LayerMix(InputArray _grayImage, OutputArray _mixImage,int blurSize);

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

	Mat mixImage;
	LayerMix(grayImage, mixImage, blurSize);
	string lmOutfile = filepath + "\\" + infilename + "_LM.png";		//�ɶ�(�|�X)
	imwrite(lmOutfile, mixImage);

    return 0;
}


void LayerMix(InputArray _grayImage, OutputArray _mixImage, int blurSize)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	_mixImage.create(grayImage.size(), CV_8UC1);
	Mat mixImage = _mixImage.getMat();

	Mat blurImage;
	medianBlur(grayImage, blurImage, blurSize);

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
