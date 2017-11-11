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
//  4 -- Bifurcation point
//  5 -- Out of Endpoint 

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

/*��׳��_�u�s�q*/
// startSpace -> �_�l�j�M�����Z
// endSpace   -> �����j�M�����Z
// degree     -> �e�Ի~�t����
// flagT = 0  -> �u�j�M���I
// flagT = 1  -> �i�j�M���I�B�u�q�I�B�����I
// flagD = 0  -> �u�j�M90�׽d��
// flagD = 1  -> �i�j�M180�׽d��
void ConnectBreakLine(InputArray _gradm, InputArray _gradd, OutputArray _gradmCBL, OutputArray _graddCBL, int startSpace = 2, int endSpace = 5, int degree = 60, int flagT = 0, bool flagD = 0);

/*����֭�*/
void HysteresisThreshold(InputArray _gradm, OutputArray _bwLine, int upperThreshold = 150, int lowerThreshold = 50);

/*�M���S�w�I*/
//  iter -- �R������
//  flagT = 0 -> �M���t���I
//  flagT = 1 -> �M�����I
void ClearSpecialPoint(InputArray _bwLine, OutputArray _bwLineCSP, int border = 0, int iter = 1, bool flagT = 0);

/*�G���_�u�s�q*/
// startSpace -> �_�l�j�M�����Z
// endSpace   -> �����j�M�����Z
// degree     -> �e�Ի~�t����
// flagT = 0  -> �u�j�M���I
// flagT = 1  -> �i�j�M���I�B�u�q�I�B�����I
// flagD = 0  -> �u�j�M90�׽d��
// flagD = 1  -> �i�j�M180�׽d��
void BWConnectBreakLine(InputArray _gradm, InputArray _gradd, InputArray _bwLine, OutputArray _gradmCBL, OutputArray _graddCBL, OutputArray _bwLineCBL, int startSpace, int endSpace, int degree, int flagT, bool flagD);
