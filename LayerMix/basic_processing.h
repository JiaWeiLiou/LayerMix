#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

/*�M��ڵ��I*/
int findroot(int labeltable[], int label);

/*�M��s�q�u*/
int bwlabel(InputArray _binaryImg, OutputArray _labels, int nears);

/*�P�_�I������*/
// channel 1 put the type of points
//  0 -- None Point
//  1 -- Isolated Point
//  2 -- EndPoint 
//  3 -- Line Point
//  4 -- Out of Endpoint 
// channel 2 put the neighborhood direction at end of Line Point
//  0 -- None
//  1 - 8 -- Location of neighborhood point
//	+ - + - + - +
//	| 2 | 3 | 4 |
//	+ - + - + - +
//	| 1 | 0 | 5 |
//	+ - + - + - +
//	| 8 | 7 | 6 |
//	+ - + - + - +
void pointlabel(InputArray _gradm, OutputArray _labels);

/*�Ыئ���*/
void makecolorwheel(vector<Scalar> &colorwheel);

/*�N�Ϥ���H������V�����(��J��׳��α�פ�V)*/
void DrawColorSystem(InputArray _field, OutputArray _colorField);

/*�N�Ϥ���H������V�����(��J��״T�Ȥα�פ�V)*/
void DrawColorSystem(InputArray _gradm, InputArray _gradd, OutputArray _colorField);

/*�N�Ϥ���u�ʩԦ��åH�Ƕ������*/
void DrawAbsGraySystem(InputArray _field, OutputArray _grayField);

/*�ϼh�V�X�Ҧ�*/
void LayerMix(InputArray _grayImage, InputArray _blurImage, OutputArray _mixImage);

/*��󭱪����βV�X�Ҧ�*/
void Divide(InputArray _grayImage, InputArray _mixImage, OutputArray _divideImage);

/*���L�|�X�V�X�Ҧ�*/
void HardMix(InputArray _grayImage, InputArray _mixImage, OutputArray _hardmixImage);

/*�h����󭱪����T*/
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears = 4, bool BW = 0);

/*�����t��*/
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady);

/*���X�����Ϋ�����V��׬���׳�*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*�p���״T�ȤΤ�V*/
void CalculateGradient(InputArray _gradientField, OutputArray _gradm, OutputArray _gradd);

/*�D���j�ȧ��*/
void NonMaximumSuppression(InputArray _gradm, InputArray _gradd, OutputArray _gradmNMS, OutputArray _graddNMS);

/*�M������V�I*/
void ClearDifferentDirection(InputArray _gradm, InputArray _gradd, OutputArray _gradmCDD, OutputArray _graddCDD);

/*�T���_�u�s�q*/
void ConnectBreakLine(InputArray _gradm, InputArray _gradd, OutputArray _gradmCBL, OutputArray _graddCBL, int startSpace = 2, int endSpace = 5, int degree = 60, bool flag = 0);

/*����֭�*/
void HysteresisThreshold(InputArray _NMSgradientField_abs, OutputArray _HTedge, int upperThreshold, int lowerThreshold);

/*�G���_�u�s�q*/
void BWConnectBreakLine(InputArray _gradmWCBL, InputArray _graddWCBL, InputArray _edgeHT, OutputArray _edgeFCBL, int startSpace = 2, int endSpace = 100, int degree = 150, bool flag = 0);
