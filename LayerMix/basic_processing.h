#pragma once
#include <opencv2/opencv.hpp> 

using namespace cv;

/*尋找根結點*/
int findroot(int labeltable[], int label);

/*尋找連通線*/
int bwlabel(InputArray _binaryImg, OutputArray _labels, int nears);

/*創建色環*/
void makecolorwheel(vector<Scalar> &colorwheel);

/*將圖片轉以色環方向場顯示*/
void DrawColorSystem(InputArray _field, OutputArray _colorField);

/*將圖片轉線性拉伸並以灰階值顯示*/
void DrawAbsGraySystem(InputArray _field, OutputArray _grayField);

/*圖層混合模式*/
void LayerMix(InputArray _grayImage, InputArray _blurImage, OutputArray _mixImage);

/*差分混合模式*/
void Divide(InputArray _grayImage, InputArray _mixImage, OutputArray _divideImage);

/*實色印疊合混合模式*/
void HardMix(InputArray _grayImage, InputArray _mixImage, OutputArray _hardmixImage);

/*去除基於面的雜訊*/
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears = 4, bool BW = 0);

/*中央差分*/
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady);

/*結合水平及垂直方向梯度為梯度場*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField);

/*計算梯度幅值及方向*/
void CalculateGradient(InputArray _gradientField, OutputArray _gradm, OutputArray _gradd);

/*非極大值抑制*/
void NonMaximumSuppression(InputArray _gradientField, OutputArray _NMSgradientField);

/*滯後閥值*/
void HysteresisThreshold(InputArray _NMSgradientField_abs, OutputArray _HTedge, int upperThreshold, int lowerThreshold);
