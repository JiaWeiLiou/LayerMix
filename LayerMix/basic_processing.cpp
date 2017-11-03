#include "stdafx.h"
#include "basic_processing.h"


/*尋找根結點*/
int findroot(int labeltable[], int label)
{
	int x = label;
	while (x != labeltable[x])
		x = labeltable[x];
	return x;
}

/*尋找連通線*/
int bwlabel(InputArray _binaryImg, OutputArray _labels, int nears)
{
	Mat binaryImg = _binaryImg.getMat();
	CV_Assert(binaryImg.type() == CV_8UC1);

	_labels.create(binaryImg.size(), CV_32SC1);
	Mat labels = _labels.getMat();
	labels = Scalar(0);

	if (nears != 4 && nears != 6 && nears != 8)
		nears = 8;

	int nobj = 0;    // number of objects found in image  

	int* labeltable = new int[binaryImg.rows*binaryImg.cols];		// initialize label table with zero  
	memset(labeltable, 0, binaryImg.rows*binaryImg.cols * sizeof(int));
	int ntable = 0;

	//	labeling scheme
	//	+ - + - + - +
	//	| D | C | E |
	//	+ - + - + - +
	//	| B | A |   |
	//	+ - + - + - +
	//	A is the center pixel of a neighborhood.In the 3 versions of connectedness :
	//	4 : A connects to B and C
	//	6 : A connects to B, C, and D
	//	8 : A connects to B, C, D, and E


	for (int i = 0; i < binaryImg.rows; i++)
		for (int j = 0; j < binaryImg.cols; j++)
			if (binaryImg.at<uchar>(i, j) == 255)   // if A is an object  
			{
				// get the neighboring labels B, C, D, and E
				int B, C, D, E;

				if (j == 0) { B = 0; }
				else { B = findroot(labeltable, labels.at<int>(i, j - 1)); }

				if (i == 0) { C = 0; }
				else { C = findroot(labeltable, labels.at<int>(i - 1, j)); }

				if (i == 0 || j == 0) { D = 0; }
				else { D = findroot(labeltable, labels.at<int>(i - 1, j - 1)); }

				if (i == 0 || j == binaryImg.cols - 1) { E = 0; }
				else { E = findroot(labeltable, labels.at<int>(i - 1, j + 1)); }

				if (nears == 4)		// apply 4 connectedness  
				{
					if (B && C)	// B and C are labeled  
					{
						if (B == C) { labels.at<int>(i, j) = B; }
						else
						{
							labeltable[C] = B;
							labels.at<int>(i, j) = B;
						}
					}
					else if (B) { labels.at<int>(i, j) = B; }            // B is object but C is not  
					else if (C) { labels.at<int>(i, j) = C; }            // C is object but B is not  
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; }	// B, C, D not object - new object label and put into table  
				}
				else if (nears == 6)	// apply 6 connected ness  
				{
					if (D) { labels.at<int>(i, j) = D; }              // D object, copy label and move on  
					else if (B && C)		// B and C are labeled  
					{
						if (B == C) { labels.at<int>(i, j) = B; }
						else
						{
							int tlabel = B < C ? B : C;
							labeltable[B] = tlabel;
							labeltable[C] = tlabel;
							labels.at<int>(i, j) = tlabel;
						}
					}
					else if (B) { labels.at<int>(i, j) = B; }        // B is object but C is not  	
					else if (C) { labels.at<int>(i, j) = C; }        // C is object but B is not  
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; } 	// B, C, D not object - new object label and put into table
				}
				else if (nears == 8)	// apply 8 connectedness  
				{
					if (B || C || D || E)
					{
						int tlabel;
						if (B) { tlabel = B; }
						else if (C) { tlabel = C; }
						else if (D) { tlabel = D; }
						else if (E) { tlabel = E; }

						labels.at<int>(i, j) = tlabel;

						if (B && B != tlabel) { labeltable[B] = tlabel; }
						if (C && C != tlabel) { labeltable[C] = tlabel; }
						if (D && D != tlabel) { labeltable[D] = tlabel; }
						if (E && E != tlabel) { labeltable[E] = tlabel; }
					}
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; } // label and put into table
				}
			}
			else { labels.at<int>(i, j) = 0; }	// A is not an object so leave it

			for (int i = 0; i <= ntable; i++)
				labeltable[i] = findroot(labeltable, i);	// consolidate component table  

			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					labels.at<int>(i, j) = labeltable[labels.at<int>(i, j)];	// run image through the look-up table  

			// count up the objects in the image  
			for (int i = 0; i <= ntable; i++)
				labeltable[i] = 0;		//clear all table label
			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					++labeltable[labels.at<int>(i, j)];		//calculate all label numbers

			labeltable[0] = 0;		//clear 0 label
			for (int i = 1; i <= ntable; i++)
				if (labeltable[i] > 0)
					labeltable[i] = ++nobj;	// number the objects from 1 through n objects  and reset label table

			// run through the look-up table again  
			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					labels.at<int>(i, j) = labeltable[labels.at<int>(i, j)];

			delete[] labeltable;
			labeltable = nullptr;
			return nobj;
}

/*判斷點的類型*/
void pointlabel(InputArray _gradm, OutputArray _labels)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	_labels.create(gradm.size(), CV_8UC2);
	Mat labels = _labels.getMat();

	Mat gradmRef;
	copyMakeBorder(gradm, gradmRef, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			vector<char> nearPoint;
			int num = 0;

			if (gradmRef.at<uchar>(i + 1, j + 1) != 0)
			{
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j + 1));		//0
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j));			//1
				nearPoint.push_back(gradmRef.at<uchar>(i, j));				//2
				nearPoint.push_back(gradmRef.at<uchar>(i, j + 1));			//3
				nearPoint.push_back(gradmRef.at<uchar>(i, j + 2));			//4
				nearPoint.push_back(gradmRef.at<uchar>(i + 1, j + 2));		//5
				nearPoint.push_back(gradmRef.at<uchar>(i + 2, j + 2));		//6
				nearPoint.push_back(gradmRef.at<uchar>(i + 2, j + 1));		//7
				nearPoint.push_back(gradmRef.at<uchar>(i + 2, j));			//8
				num = 9 - count(nearPoint.begin(), nearPoint.end(), 0);

				if (num == 1)		//Isolated Point
				{
					labels.at<Vec2b>(i, j)[0] = 1;
					labels.at<Vec2b>(i, j)[1] = 0;
				}
				else if (num == 2)		//End of Line Point 
				{
					labels.at<Vec2b>(i, j)[0] = 2;
					for (int k = 1; k <= 8; ++k)
						if (nearPoint[k] != 0)
						{
							labels.at<Vec2b>(i, j)[1] = k;
							break;
						}
				}
				else if (num == 3)
				{
					for (int k = 1; k <= 8; ++k)
						if (nearPoint[(k % 9)] != 0 && nearPoint[(k % 8) + 1] != 0) //1,2、2,3、...、8,1   End of Line Point 
						{
							labels.at<Vec2b>(i, j)[0] = 2;
							if (k % 2 == 1)		//只存取對角線的方向
								labels.at<Vec2b>(i, j)[1] = k + 1;
							else
								labels.at<Vec2b>(i, j)[1] = k;
							break;
						}
						else	//Line Point
						{
							labels.at<Vec2b>(i, j)[0] = 3;
							labels.at<Vec2b>(i, j)[1] = 0;
						}
				}
				else		//Line Point
				{
					labels.at<Vec2b>(i, j)[0] = 3;
					labels.at<Vec2b>(i, j)[1] = 0;
				}

			}
			else
			{
				labels.at<Vec2b>(i, j)[0] = 0;
				labels.at<Vec2b>(i, j)[1] = 0;
			}
		}
}

/*創建色環*/
void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;	//紅色(Red)     至黃色(Yellow)
	int YG = 15;	//黃色(Yellow)  至綠色(Green)
	int GC = 15;	//綠色(Green)   至青色(Cyan)
	int CB = 15;	//青澀(Cyan)    至藍色(Blue)
	int BM = 15;	//藍色(Blue)    至洋紅(Magenta)
	int MR = 15;	//洋紅(Magenta) 至紅色(Red)

	for (int i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (int i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (int i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (int i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (int i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (int i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

/*將圖片轉以色環方向場顯示(輸入梯度場或梯度方向)*/
void DrawColorSystem(InputArray _field, OutputArray _colorField)
{
	Mat field;
	Mat temp = _field.getMat();
	if (temp.type() == CV_16SC2) {
		temp.convertTo(field, CV_32FC2);
	}
	else {
		field = _field.getMat();
	}

	_colorField.create(field.size(), CV_8UC3);
	Mat colorField = _colorField.getMat();

	static vector<Scalar> colorwheel; //Scalar i,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:  
	int maxrad = -1;

	if (field.type() == CV_32FC1)
	{
		maxrad = 255;		//只有梯度方向無梯度幅值

		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;

				if (field.at<float>(i, j) == -1000.0f)		//用以顯示無梯度方向
				{
					for (int b = 0; b < 3; b++)
					{
						data[2 - b] = 255;
					}
				}
				else
				{
					float rad = maxrad;

					float angle = field.at<float>(i, j) / CV_PI;    //單位為-1至+1
					float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
					int k0 = (int)fk;
					int k1 = (k0 + 1) % colorwheel.size();
					float f = fk - k0;

					float col0 = 0.0f;
					float col1 = 0.0f;
					float col = 0.0f;
					for (int b = 0; b < 3; b++)
					{
						col0 = colorwheel[k0][b] / 255.0f;
						col1 = colorwheel[k1][b] / 255.0f;
						col = (1 - f) * col0 + f * col1;
						if (rad <= 1)
							col = 1 - rad * (1 - col); // increase saturation with radius  
						else
							col = col;  //out of range
						data[2 - b] = (int)(255.0f * col);
					}
				}
			}
	}
	else if (field.type() == CV_32FC2)
	{
		// Find max flow to normalize fx and fy  
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				Vec2f field_at_point = field.at<Vec2f>(i, j);
				float fx = field_at_point[0];
				float fy = field_at_point[1];
				float rad = sqrt(fx * fx + fy * fy);
				maxrad = maxrad > rad ? maxrad : rad;
			}

		maxrad = maxrad / 2;		//加深顯示結果(可取消此行)

		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;
				Vec2f field_at_point = field.at<Vec2f>(i, j);

				float fx = field_at_point[0];
				float fy = field_at_point[1];

				float rad = sqrt(fx * fx + fy * fy) / maxrad;

				float angle = atan2(fy, fx) / CV_PI;    //單位為-1至+1
				float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
				int k0 = (int)fk;
				int k1 = (k0 + 1) % colorwheel.size();
				float f = fk - k0;

				float col0 = 0.0f;
				float col1 = 0.0f;
				float col = 0.0f;
				for (int b = 0; b < 3; b++)
				{
					col0 = colorwheel[k0][b] / 255.0f;
					col1 = colorwheel[k1][b] / 255.0f;
					col = (1 - f) * col0 + f * col1;
					if (rad <= 1)
						col = 1 - rad * (1 - col); // increase saturation with radius  
					else
						col = col;  //out of range
					data[2 - b] = (int)(255.0f * col);
				}
			}
	}
}

/*將圖片轉以色環方向場顯示(輸入梯度幅值及梯度方向)*/
void DrawColorSystem(InputArray _gradm, InputArray _gradd, OutputArray _colorField)
{
	Mat gradm;
	Mat temp1 = _gradm.getMat();
	if (temp1.type() == CV_8UC1) { temp1.convertTo(gradm, CV_32FC1); }
	else { gradm = _gradm.getMat(); }

	Mat gradd;
	Mat temp2 = _gradd.getMat();
	if (temp2.type() == CV_8UC1) { temp2.convertTo(gradd, CV_32FC1); }
	else { gradd = _gradd.getMat(); }

	_colorField.create(gradm.size(), CV_8UC3);
	Mat colorField = _colorField.getMat();

	static vector<Scalar> colorwheel; //Scalar i,g,b  
	if (colorwheel.empty())
		makecolorwheel(colorwheel);

	// determine motion range:  
	int maxrad = -1;

	// Find max flow to normalize fx and fy  
	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			float rad = gradm.at<float>(i, j);
			maxrad = maxrad > rad ? maxrad : rad;
		}

	maxrad = maxrad / 2;		//加深顯示結果(可取消此行)

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;

			if (gradd.at<float>(i, j) == -1000.0f)		//用以顯示無梯度方向
			{
				for (int b = 0; b < 3; b++)
				{
					data[2 - b] = 255;
				}
			}
			else
			{
				float rad = gradm.at<float>(i, j) / maxrad;

				float angle = gradd.at<float>(i, j) / CV_PI;    //單位為-1至+1
				float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //計算角度對應之索引位置
				int k0 = (int)fk;
				int k1 = (k0 + 1) % colorwheel.size();
				float f = fk - k0;

				float col0 = 0.0f;
				float col1 = 0.0f;
				float col = 0.0f;
				for (int b = 0; b < 3; b++)
				{
					col0 = colorwheel[k0][b] / 255.0f;
					col1 = colorwheel[k1][b] / 255.0f;
					col = (1 - f) * col0 + f * col1;
					if (rad <= 1)
						col = 1 - rad * (1 - col); // increase saturation with radius  
					else
						col = col;  //out of range
					data[2 - b] = (int)(255.0f * col);
				}
			}
		}
}

/*將圖片轉線性拉伸並以灰階值顯示*/
void DrawAbsGraySystem(InputArray _field, OutputArray _grayField)
{
	Mat field;
	Mat temp = _field.getMat();
	if (temp.type() == CV_16SC2) { temp.convertTo(field, CV_32FC2); }
	else if (temp.type() == CV_16SC1) { temp.convertTo(field, CV_32FC1); }
	else { field = _field.getMat(); }

	_grayField.create(field.size(), CV_8UC1);
	Mat grayField = _grayField.getMat();



	// determine motion range:  
	float maxvalue = 0;

	if (field.type() == CV_8UC1)
	{
		// Find max value
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
				maxvalue = maxvalue > field.at<uchar>(i, j) ? maxvalue : field.at<uchar>(i, j);

		//linear stretch to 255
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
				grayField.at<uchar>(i, j) = ((float)field.at<uchar>(i, j) / maxvalue) * 255;
	}
	else if (field.type() == CV_32FC1)
	{
		// Find max value
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
				maxvalue = maxvalue > abs(field.at<float>(i, j)) ? maxvalue : abs(field.at<float>(i, j));

		//linear stretch to 255
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
				grayField.at<uchar>(i, j) = ((float)abs(field.at<float>(i, j)) / maxvalue) * 255;
	}
	else if (field.type() == CV_32FC2)
	{
		// Find max value
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				float fx = field.at<Vec2f>(i, j)[0];
				float fy = field.at<Vec2f>(i, j)[1];
				float absvalue = sqrt(fx * fx + fy * fy);
				maxvalue = maxvalue > absvalue ? maxvalue : absvalue;
			}

		//linear stretch to 255
		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				float fx = field.at<Vec2f>(i, j)[0];
				float fy = field.at<Vec2f>(i, j)[1];
				float absvalue = sqrt(fx * fx + fy * fy);
				grayField.at<uchar>(i, j) = (absvalue / maxvalue) * 255;
			}
	}
}

/*圖層混合模式*/
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
		for (int j = 0; j < grayImage.cols; ++j)
		{
			divide = (double)grayImage.at<uchar>(i, j) / (double)blurImage.at<uchar>(i, j) > 1 ? 255 : ((double)grayImage.at<uchar>(i, j) / (double)blurImage.at<uchar>(i, j)) * 255.0;
			mixImage.at<uchar>(i, j) = divide + (double)blurImage.at<uchar>(i, j) < 255.0 ? 0 : 255;
		}
}

/*分割混合模式*/
void Divide(InputArray _grayImage, InputArray _mixImage, OutputArray _divideImage)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	Mat mixImage = _mixImage.getMat();
	CV_Assert(mixImage.type() == CV_8UC1);

	_divideImage.create(grayImage.size(), CV_8UC1);
	Mat divideImage = _divideImage.getMat();

	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j)
			divideImage.at<uchar>(i, j) = (double)grayImage.at<uchar>(i, j) / (double)mixImage.at<uchar>(i, j) > 1 ? 255 : ((double)grayImage.at<uchar>(i, j) / (double)mixImage.at<uchar>(i, j)) * 255;
}

/*實色印疊合混合模式*/
void HardMix(InputArray _grayImage, InputArray _mixImage, OutputArray _hardmixImage)
{
	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	Mat mixImage = _mixImage.getMat();
	CV_Assert(mixImage.type() == CV_8UC1);

	_hardmixImage.create(grayImage.size(), CV_8UC1);
	Mat hardmixImage = _hardmixImage.getMat();

	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j)
			hardmixImage.at<uchar>(i, j) = grayImage.at<uchar>(i, j) + mixImage.at<uchar>(i, j) < 255 ? 0 : 255;
}

/*去除雜訊*/
void ClearNoise(InputArray _binaryImg, OutputArray _clearAreaImage, int noise, int nears, bool BW)
{
	Mat binaryImg = _binaryImg.getMat();
	CV_Assert(binaryImg.type() == CV_8UC1);

	_clearAreaImage.create(binaryImg.size(), CV_8UC1);
	Mat clearAreaImage = _clearAreaImage.getMat();

	Mat labels(binaryImg.size(), CV_32SC1, Scalar(0));

	// 0 claer black noise
	// 1 clear wite noise
	if (BW == 0)
	{
		for (int i = 0; i < binaryImg.rows; i++)
			for (int j = 0; j < binaryImg.cols; j++)
				if (binaryImg.at<uchar>(i, j) == 255) { binaryImg.at<uchar>(i, j) = 0; }
				else { binaryImg.at<uchar>(i, j) = 255; }
	}

	if (nears != 4 && nears != 6 && nears != 8)
		nears = 8;

	int nobj = 0;    // number of objects found in image  

	int* labeltable = new int[binaryImg.rows*binaryImg.cols];		// initialize label table with zero  
	memset(labeltable, 0, binaryImg.rows*binaryImg.cols * sizeof(int));
	int ntable = 0;

	//	labeling scheme
	//	+ - + - + - +
	//	| D | C | E |
	//	+ - + - + - +
	//	| B | A |   |
	//	+ - + - + - +
	//	A is the center pixel of a neighborhood.In the 3 versions of connectedness :
	//	4 : A connects to B and C
	//	6 : A connects to B, C, and D
	//	8 : A connects to B, C, D, and E


	for (int i = 0; i < binaryImg.rows; i++)
		for (int j = 0; j < binaryImg.cols; j++)
			if (binaryImg.at<uchar>(i, j) == 255)   // if A is an object  
			{
				// get the neighboring labels B, C, D, and E
				int B, C, D, E;

				if (j == 0) { B = 0; }
				else { B = findroot(labeltable, labels.at<int>(i, j - 1)); }

				if (i == 0) { C = 0; }
				else { C = findroot(labeltable, labels.at<int>(i - 1, j)); }

				if (i == 0 || j == 0) { D = 0; }
				else { D = findroot(labeltable, labels.at<int>(i - 1, j - 1)); }

				if (i == 0 || j == binaryImg.cols - 1) { E = 0; }
				else { E = findroot(labeltable, labels.at<int>(i - 1, j + 1)); }

				if (nears == 4)		// apply 4 connectedness  
				{
					if (B && C)	// B and C are labeled  
					{
						if (B == C) { labels.at<int>(i, j) = B; }
						else
						{
							labeltable[C] = B;
							labels.at<int>(i, j) = B;
						}
					}
					else if (B) { labels.at<int>(i, j) = B; }            // B is object but C is not  
					else if (C) { labels.at<int>(i, j) = C; }            // C is object but B is not  
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; }	// B, C, D not object - new object label and put into table  
				}
				else if (nears == 6)	// apply 6 connected ness  
				{
					if (D) { labels.at<int>(i, j) = D; }              // D object, copy label and move on  
					else if (B && C)		// B and C are labeled  
					{
						if (B == C) { labels.at<int>(i, j) = B; }
						else
						{
							int tlabel = B < C ? B : C;
							labeltable[B] = tlabel;
							labeltable[C] = tlabel;
							labels.at<int>(i, j) = tlabel;
						}
					}
					else if (B) { labels.at<int>(i, j) = B; }        // B is object but C is not  	
					else if (C) { labels.at<int>(i, j) = C; }        // C is object but B is not  
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; } 	// B, C, D not object - new object label and put into table
				}
				else if (nears == 8)	// apply 8 connectedness  
				{
					if (B || C || D || E)
					{
						int tlabel;
						if (B) { tlabel = B; }
						else if (C) { tlabel = C; }
						else if (D) { tlabel = D; }
						else if (E) { tlabel = E; }

						labels.at<int>(i, j) = tlabel;

						if (B && B != tlabel) { labeltable[B] = tlabel; }
						if (C && C != tlabel) { labeltable[C] = tlabel; }
						if (D && D != tlabel) { labeltable[D] = tlabel; }
						if (E && E != tlabel) { labeltable[E] = tlabel; }
					}
					else { labels.at<int>(i, j) = labeltable[ntable] = ++ntable; } // label and put into table
				}
			}
			else { labels.at<int>(i, j) = 0; }	// A is not an object so leave it

			for (int i = 0; i <= ntable; i++)
				labeltable[i] = findroot(labeltable, i);	// consolidate component table  

			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					labels.at<int>(i, j) = labeltable[labels.at<int>(i, j)];	// run image through the look-up table  

			// count up the objects in the image  
			for (int i = 0; i <= ntable; i++)
				labeltable[i] = 0;		//clear all table label
			for (int i = 0; i < binaryImg.rows; i++)
				for (int j = 0; j < binaryImg.cols; j++)
					++labeltable[labels.at<int>(i, j)];		//calculate all label numbers

			labeltable[0] = 0;		//clear 0 label
			for (int i = 1; i <= ntable; i++)
				if (labeltable[i] > noise) { labeltable[i] = 255; }	// number the objects from 1 through n objects  and reset label table
				else { labeltable[i] = 0; }

				// run through the look-up table again  
				for (int i = 0; i < binaryImg.rows; i++)
					for (int j = 0; j < binaryImg.cols; j++)
						clearAreaImage.at<uchar>(i, j) = labeltable[labels.at<int>(i, j)];

				delete[] labeltable;
				labeltable = nullptr;

				if (BW == 0)
				{
					for (int i = 0; i < clearAreaImage.rows; i++)
						for (int j = 0; j < clearAreaImage.cols; j++)
						{
							if (binaryImg.at<uchar>(i, j) == 255) { binaryImg.at<uchar>(i, j) = 0; }
							else { binaryImg.at<uchar>(i, j) = 255; }

							if (clearAreaImage.at<uchar>(i, j) == 255) { clearAreaImage.at<uchar>(i, j) = 0; }
							else { clearAreaImage.at<uchar>(i, j) = 255; }
						}
				}
}

/*中央差分*/
void Differential(InputArray _grayImage, OutputArray _gradx, OutputArray _grady) {

	Mat grayImage = _grayImage.getMat();
	CV_Assert(grayImage.type() == CV_8UC1);

	_gradx.create(grayImage.size(), CV_16SC1);
	Mat gradx = _gradx.getMat();

	_grady.create(grayImage.size(), CV_16SC1);
	Mat grady = _grady.getMat();

	Mat grayImageRef;
	copyMakeBorder(grayImage, grayImageRef, 1, 1, 1, 1, BORDER_REPLICATE);
	for (int i = 0; i < grayImage.rows; ++i)
		for (int j = 0; j < grayImage.cols; ++j) {
			gradx.at<short>(i, j) = ((float)-grayImageRef.at<uchar>(i + 1, j) + (float)grayImageRef.at<uchar>(i + 1, j + 2))*0.5;
			grady.at<short>(i, j) = ((float)-grayImageRef.at<uchar>(i, j + 1) + (float)grayImageRef.at<uchar>(i + 2, j + 1))*0.5;
		}
}

/*結合水平及垂直方向梯度為梯度場*/
void GradientField(InputArray _grad_x, InputArray _grad_y, OutputArray _gradientField) {

	Mat grad_x = _grad_x.getMat();
	CV_Assert(grad_x.type() == CV_16SC1);

	Mat grad_y = _grad_y.getMat();
	CV_Assert(grad_y.type() == CV_16SC1);

	_gradientField.create(grad_y.rows, grad_x.cols, CV_16SC2);
	Mat gradientField = _gradientField.getMat();

	for (int i = 0; i < grad_y.rows; ++i)
		for (int j = 0; j < grad_x.cols; ++j)
		{
			gradientField.at<Vec2s>(i, j)[0] = grad_x.at<short>(i, j);
			gradientField.at<Vec2s>(i, j)[1] = grad_y.at<short>(i, j);
		}

}

/*計算梯度幅值及方向*/
void CalculateGradient(InputArray _gradientField, OutputArray _gradm, OutputArray _gradd)
{
	Mat gradientField = _gradientField.getMat();
	CV_Assert(gradientField.type() == CV_16SC2);

	_gradm.create(gradientField.size(), CV_8UC1);
	Mat gradm = _gradm.getMat();

	_gradd.create(gradientField.size(), CV_32FC1);
	Mat gradd = _gradd.getMat();

	for (int i = 0; i < gradientField.rows; ++i)
		for (int j = 0; j < gradientField.cols; ++j)
		{
			short x = gradientField.at<Vec2s>(i, j)[0];
			short y = gradientField.at<Vec2s>(i, j)[1];
			gradm.at<uchar>(i, j) = sqrt(x*x + y*y);

			if (x == 0 && y == 0) { gradd.at<float>(i, j) = -1000.0f; }	//用以顯示無梯度方向
			else { gradd.at<float>(i, j) = atan2(y, x); }
		}
}

/*非極大值抑制*/
void NonMaximumSuppression(InputArray _gradm, InputArray _gradd, OutputArray _gradmNMS, OutputArray _graddNMS)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradd = _gradd.getMat();
	CV_Assert(gradd.type() == CV_32FC1);

	_gradmNMS.create(gradm.size(), CV_8UC1);
	Mat gradmNMS = _gradmNMS.getMat();

	_graddNMS.create(gradd.size(), CV_32FC1);
	Mat graddNMS = _graddNMS.getMat();

	Mat gradmRef;
	copyMakeBorder(gradm, gradmRef, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(0));

	float theta = 0.0f;			//目前像素的方向
	int amplitude = 0;			//目前像素的幅值
	int amplitude1 = 0;			//鄰域像素1的幅值
	int amplitude2 = 0;			//鄰域像素2的幅值
	float A1 = 0.0f;			//上臨域1幅值
	float A2 = 0.0f;			//上臨域2幅值
	float B1 = 0.0f;			//下臨域1幅值
	float B2 = 0.0f;			//下臨域2幅值
	float alpha = 0.0f;			//比例係數

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			amplitude = gradm.at<uchar>(i, j);
			theta = ((gradd.at<float>(i, j) + CV_PI) / CV_PI)*180.0f;

			if (gradd.at<float>(i, j) == -1000.0f)
			{
				gradmNMS.at<uchar>(i, j) = 0;
				graddNMS.at<float>(i, j) = -1000.0f;
				continue;
			}


			if ((theta >= 0.0f && theta < 45.0f) || (theta >= 180.0f && theta < 225.0f))
			{
				alpha = tan(theta* CV_PI / 180.0);
				A1 = gradmRef.at<uchar>(i + 1, j);
				A2 = gradmRef.at<uchar>(i, j);
				B1 = gradmRef.at<uchar>(i + 1, j + 2);
				B2 = gradmRef.at<uchar>(i + 2, j + 2);

			}
			else if ((theta >= 45.0f && theta < 90.0f) || (theta >= 225.0f && theta < 270.0f))
			{
				alpha = tan((90.0f - theta)* CV_PI / 180.0);
				A1 = gradmRef.at<uchar>(i, j + 1);
				A2 = gradmRef.at<uchar>(i, j);
				B1 = gradmRef.at<uchar>(i + 2, j + 1);
				B2 = gradmRef.at<uchar>(i + 2, j + 2);
			}
			else if ((theta >= 90.0f && theta < 135.0f) || (theta >= 270.0f && theta < 315.0f))
			{
				alpha = tan((theta - 90.0f)* CV_PI / 180.0);
				A1 = gradmRef.at<uchar>(i, j + 1);
				A2 = gradmRef.at<uchar>(i, j + 2);
				B1 = gradmRef.at<uchar>(i + 2, j + 1);
				B2 = gradmRef.at<uchar>(i + 2, j);
			}
			else if ((theta >= 135.0f && theta < 180.0f) || (theta >= 315.0f && theta <= 360.0f))
			{
				alpha = tan((180.0f - theta)* CV_PI / 180.0);
				A1 = gradmRef.at<uchar>(i + 1, j + 2);
				A2 = gradmRef.at<uchar>(i, j + 2);
				B1 = gradmRef.at<uchar>(i + 1, j);
				B2 = gradmRef.at<uchar>(i + 2, j);
			}

			amplitude1 = A1*(1 - alpha) + A2*alpha;
			amplitude2 = B1*(1 - alpha) + B2*alpha;

			if (amplitude > amplitude1 && amplitude > amplitude2)
			{
				gradmNMS.at<uchar>(i, j) = gradm.at<uchar>(i, j);
				graddNMS.at<float>(i, j) = gradd.at<float>(i, j);
			}
			else
			{
				gradmNMS.at<uchar>(i, j) = 0;
				graddNMS.at<float>(i, j) = -1000.0f;		//用以區分無角度
			}
		}
}

/*清除異方向點*/
void ClearDifferentDirection(InputArray _gradm, InputArray _gradd, OutputArray _gradmCDD, OutputArray _graddCDD)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradd = _gradd.getMat();
	CV_Assert(gradd.type() == CV_32FC1);

	_gradmCDD.create(gradm.size(), CV_8UC1);
	Mat gradmCDD = _gradmCDD.getMat();

	_graddCDD.create(gradd.size(), CV_32FC1);
	Mat graddCDD = _graddCDD.getMat();

	Mat mask(gradm.rows + 2, gradm.cols + 2, CV_8UC1, Scalar(255));

	float theta = 0.0f;			//目前像素的方向

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			theta = ((gradd.at<float>(i, j) + CV_PI) / CV_PI)*180.0f;

			if ((theta <= 360.0f && theta >= 337.5f) || (theta < 22.5f &&  theta >= 0.0f) || (theta >= 157.5f && theta < 202.5f))
				mask.at<uchar>(i + 1, j) = mask.at<uchar>(i + 1, j + 2) = 0;
			else if ((theta >= 22.5f && theta < 67.5f) || (theta >= 202.5f && theta < 247.5f))
				mask.at<uchar>(i, j) = mask.at<uchar>(i + 2, j + 2) = 0;
			else if ((theta >= 67.5f && theta < 112.5f) || (theta >= 247.5f && theta < 292.5f))
				mask.at<uchar>(i, j + 1) = mask.at<uchar>(i + 2, j + 1) = 0;
			else if ((theta >= 112.5f && theta < 157.5f) || (theta >= 292.5f && theta < 337.5f))
				mask.at<uchar>(i, j + 2) = mask.at<uchar>(i + 2, j) = 0;
		}

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
			if (mask.at<uchar>(i + 1, j + 1) == 255)
			{
				gradmCDD.at<uchar>(i, j) = gradm.at<uchar>(i, j);
				graddCDD.at<float>(i, j) = gradd.at<float>(i, j);
			}
			else
			{
				gradmCDD.at<uchar>(i, j) = 0;
				graddCDD.at<float>(i, j) = -1000.0f;
			}
}

/*斷線連通*/
void ConnectBreakLine(InputArray _gradm, InputArray _gradd, OutputArray _gradmCBL, OutputArray _graddCBL, int startSpace,int endSpace)
{
	Mat gradm = _gradm.getMat();
	CV_Assert(gradm.type() == CV_8UC1);

	Mat gradd = _gradd.getMat();
	CV_Assert(gradd.type() == CV_32FC1);

	_gradmCBL.create(gradm.size(), CV_8UC1);
	Mat gradmCBL = _gradmCBL.getMat();

	_graddCBL.create(gradd.size(), CV_32FC1);
	Mat graddCBL = _graddCBL.getMat();

	gradm.copyTo(gradmCBL);
	gradd.copyTo(graddCBL);

	Mat endPointMap;		//儲存點類型
	pointlabel(gradm, endPointMap);

	Mat nearPoint1(gradm.size(), CV_8UC1, Scalar(1));		//相鄰方向1
	Mat nearPoint2(gradm.size(), CV_8UC1, Scalar(1));		//相鄰方向2
	Mat nearPoint3(gradm.size(), CV_8UC1, Scalar(1));		//相鄰方向3

	if (startSpace != 2)	//查詢是否遮擋
	{
		for (int x = 2; x <= startSpace - 1; ++x)
		{
			Mat graddRef;
			copyMakeBorder(gradd, graddRef, x - 1, x - 1, x - 1, x - 1, BORDER_CONSTANT, Scalar(-1000.0f));

			for (int i = 1; i < gradd.rows - 1; ++i)		//不搜尋影像邊界
				for (int j = 1; j < gradd.cols - 1; ++j)		//不搜尋影像邊界
				{
					int ir = i + x - 1, jr = j + x - 1;		//reference index i,j for graddRef

					if (endPointMap.at<Vec2b>(i, j)[0] == 2)	//判斷是否為端點
					{

						bool flag1 = 1, flag2 = 1, flag3 = 1;	//相鄰flag

						if (endPointMap.at<Vec2b>(i, j)[1] == 1)		//8區域搜尋 - 1區
						{
							//N->NE
							if (nearPoint1.at<uchar>(i, j) == 1)
								for (int is = ir - x, js = jr, nowLocation = x; js <= jr + x - 1; ++js, ++nowLocation)
								{
									if (graddRef.at<float>(is, js) != -1000.0f)	{ flag1 = 0; }
								}

							//NE->SE
							if (nearPoint2.at<uchar>(i, j) == 1)
							for (int is = ir - x, js = jr + x, nowLocation = 0; is <= ir + x - 1; ++is, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f)	{ flag2 = 0; }
							}

							//SE->S
							if (nearPoint3.at<uchar>(i, j) == 1)
							for (int is = ir + x, js = jr + x, nowLocation = 0; js >= jr; --js, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f) { flag3 = 0; }
							}
						}
						else if (endPointMap.at<Vec2b>(i, j)[1] == 2)		//8區域搜尋 - 2區
						{
							nearPoint3.at<uchar>(i, j) = 0;
							//NE->SE
							if (nearPoint1.at<uchar>(i, j) == 1)
							for (int is = ir - x, js = jr + x, nowLocation = 0; is <= ir + x - 1; ++is, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f)	{ flag1 = 0; }
							}
							//SE->SW
							if (nearPoint2.at<uchar>(i, j) == 1)
							for (int is = ir + x, js = jr + x, nowLocation = 0; js >= jr - x + 1; --js, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag2 = 0; }
							}
							//SW
							if (graddRef.at<float>(ir + x, jr - x) != -1000.0f && nearPoint2.at<uchar>(i, j) == 1)
							{
								flag2 = 0;
							}
						}
						else if (endPointMap.at<Vec2b>(i, j)[1] == 3)		//8區域搜尋 - 3區
						{
							//E->SE
							if (nearPoint1.at<uchar>(i, j) == 1)
								for (int is = ir, js = jr + x, nowLocation = x; is <= ir + x - 1; ++is, ++nowLocation)
								{
									if (graddRef.at<float>(is, js) != -1000.0f){ flag1 = 0; }
								}

							//SE->SW
							if (nearPoint2.at<uchar>(i, j) == 1)
							for (int is = ir + x, js = jr + x, nowLocation = 0; js >= jr - x + 1; --js, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag2 = 0; }
							}

							//SW->W
							if (nearPoint3.at<uchar>(i, j) == 1)
							for (int is = ir + x, js = jr - x, nowLocation = 0; is >= ir; --is, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag3 = 0; }
							}
						}
						else if (endPointMap.at<Vec2b>(i, j)[1] == 4)		//8區域搜尋 - 4區
						{
							nearPoint3.at<uchar>(i, j) = 0;
							//SE->SW
							if (nearPoint1.at<uchar>(i, j) == 1)
							for (int is = ir + x, js = jr + x, nowLocation = 0; js >= jr - x + 1; --js, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag1 = 0; }
							}
							//SW->NW
							if (nearPoint2.at<uchar>(i, j) == 1)
							for (int is = ir + x, js = jr - x, nowLocation = 0; is >= ir - x + 1; --is, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag2 = 0; }
							}
							//NW
							if (graddRef.at<float>(ir - x, jr - x) != -1000.0f && nearPoint2.at<uchar>(i, j) == 1)
							{
								flag2 = 0;
							}
						}
						else if (endPointMap.at<Vec2b>(i, j)[1] == 5)		//8區域搜尋 - 5區
						{
							//S->SW
							if (nearPoint1.at<uchar>(i, j) == 1)
								for (int is = ir + x, js = jr, nowLocation = x; js >= jr - x + 1; --js, ++nowLocation)
								{
									if (graddRef.at<float>(is, js) != -1000.0f)	{ flag1 = 0; }
								}
							//SW->NW
							if (nearPoint2.at<uchar>(i, j) == 1)
							for (int is = ir + x, js = jr - x, nowLocation = 0; is >= ir - x + 1; --is, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag2 = 0; }
							}
							//NW->N
							if (nearPoint3.at<uchar>(i, j) == 1)
							for (int is = ir - x, js = jr - x, nowLocation = 0; js <= jr; ++js, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag3 = 0; }
							}
						}
						else if (endPointMap.at<Vec2b>(i, j)[1] == 6)		//8區域搜尋 - 6區
						{
							nearPoint3.at<uchar>(i, j) = 0;
							//SW->NW
							if (nearPoint1.at<uchar>(i, j) == 1)
							for (int is = ir + x, js = jr - x, nowLocation = 0; is >= ir - x + 1; --is, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag1 = 0; }
							}
							//NW->NE
							if (nearPoint2.at<uchar>(i, j) == 1)
							for (int is = ir - x, js = jr - x, nowLocation = 0; js <= jr + x - 1; ++js, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag2 = 0; }
							}
							//NE
							if (graddRef.at<float>(ir - x, jr + x) != -1000.0f && nearPoint2.at<uchar>(i, j) == 1)
							{
								flag2 = 0;
							}
						}
						else if (endPointMap.at<Vec2b>(i, j)[1] == 7)		//8區域搜尋 - 7區
						{
							//W->NW
							if (nearPoint1.at<uchar>(i, j) == 1)
								for (int is = ir, js = jr - x, nowLocation = x; is >= ir - x + 1; --is, ++nowLocation)
								{
									if (graddRef.at<float>(is, js) != -1000.0f)	{ flag1 = 0; }
								}
							//NW->NE
							if (nearPoint2.at<uchar>(i, j) == 1)
							for (int is = ir - x, js = jr - x, nowLocation = 0; js <= jr + x - 1; ++js, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag2 = 0; }
							}
							//NE->E
							if (nearPoint3.at<uchar>(i, j) == 1)
							for (int is = ir - x, js = jr + x, nowLocation = 0; is <= ir; ++is, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag3 = 0; }
							}
						}
						else if (endPointMap.at<Vec2b>(i, j)[1] == 8)		//8區域搜尋 - 8區
						{
							nearPoint3.at<uchar>(i, j) = 0;
							//NW->NE
							if (nearPoint1.at<uchar>(i, j) == 1)
							for (int is = ir - x, js = jr - x, nowLocation = 0; js <= jr + x - 1; ++js, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag1 = 0; }
							}
							//NE->SE
							for (int is = ir - x, js = jr + x, nowLocation = 0; is <= ir + x - 1; ++is, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f){ flag2 = 0; }
							}
							//SE
							if (graddRef.at<float>(ir + x, jr + x) != -1000.0f && nearPoint2.at<uchar>(i, j) == 1)
							{
								flag2 = 0;
							}
						}

						//修改端點類型
						if (flag1 == 0) { nearPoint1.at<uchar>(i, j) = 0; }
						if (flag2 == 0) { nearPoint2.at<uchar>(i, j) = 0; }
						if (flag3 == 0) { nearPoint3.at<uchar>(i, j) = 0; }
						if (nearPoint1.at<uchar>(i, j) == 0 && nearPoint2.at<uchar>(i, j) == 0 && nearPoint3.at<uchar>(i, j) == 0)
						{
							endPointMap.at<Vec2b>(i, j)[0] = 4;	//淘汰端點
						}
					}
				}
		}
	}
	for (int x = startSpace; x <= endSpace; ++x)
	{
		Mat graddRef;
		copyMakeBorder(gradd, graddRef, x - 1, x - 1, x - 1, x - 1, BORDER_CONSTANT, Scalar(-1000.0f));

		/*搜尋並連通線*/
		for (int i = 1; i < gradd.rows - 1; ++i)		//不搜尋影像邊界
			for (int j = 1; j < gradd.cols - 1; ++j)		//不搜尋影像邊界
			{
				int ir = i + x - 1, jr = j + x - 1;		//reference index i,j for graddRef

				if (endPointMap.at<Vec2b>(i, j)[0] == 2)	//判斷是否為端點
				{
					float theta0 = ((gradd.at<float>(i, j) + CV_PI) / CV_PI)*180.0f;	//目前端點角度
					float divtheta = 0.0f;		//搜索點相差角度
					float mintheta = 180.0f;	//最小相差角度

					char searchLocation = 0;	//四區域分類
					char k = 0;					//四區域分類中的位置( k = 0 : 2*x-1 )

					float connectgradm = 0;		//連通目標幅值
					float connectgradd = 0.0f;		//連通目標方向

					bool flag1 = 1, flag2 = 1, flag3 = 1;	//相鄰flag

					/*搜尋最佳點(八區域分類)*/
					if (endPointMap.at<Vec2b>(i, j)[1] == 1)		//8區域搜尋 - 1區
					{
						//N->NE
						if (nearPoint1.at<uchar>(i, j) == 1)
							for (int is = ir - x, js = jr, nowLocation = x; js <= jr + x - 1; ++js, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
								{
									flag1 = 0;

									divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
									divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
									if (divtheta < mintheta)
									{
										mintheta = divtheta;
										searchLocation = 2;
										k = nowLocation;
									}
								}
							}

						//NE->SE
						if (nearPoint2.at<uchar>(i, j) == 1)
						for (int is = ir - x, js = jr + x, nowLocation = 0; is <= ir + x - 1; ++is, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag2 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 3;
									k = nowLocation;
								}
							}
						}

						//SE->S
						if (nearPoint3.at<uchar>(i, j) == 1)
						for (int is = ir + x, js = jr + x, nowLocation = 0; js >= jr; --js, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag3 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 4;
									k = nowLocation;
								}
							}
						}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 2)		//8區域搜尋 - 2區
					{
						nearPoint3.at<uchar>(i, j) = 0;
						//NE->SE
						if (nearPoint1.at<uchar>(i, j) == 1)
						for (int is = ir - x, js = jr + x, nowLocation = 0; is <= ir + x - 1; ++is, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag1 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 3;
									k = nowLocation;
								}
							}
						}
						//SE->SW
						if (nearPoint2.at<uchar>(i, j) == 1)
						for (int is = ir + x, js = jr + x, nowLocation = 0; js >= jr - x + 1; --js, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag2 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 4;
									k = nowLocation;
								}
							}
						}
						//SW
						if (graddRef.at<float>(ir + x, jr - x) != -1000.0f && nearPoint2.at<uchar>(i, j) == 1 && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
						{
							flag2 = 0;

							divtheta = abs(((graddRef.at<float>(ir + x, jr - x) + CV_PI) / CV_PI)*180.0f - theta0);
							divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
							if (divtheta < mintheta)
							{
								mintheta = divtheta;
								searchLocation = 1;
								k = 0;
							}
						}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 3)		//8區域搜尋 - 3區
					{
						//E->SE
						if (nearPoint1.at<uchar>(i, j) == 1)
							for (int is = ir, js = jr + x, nowLocation = x; is <= ir + x - 1; ++is, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
								{
									flag1 = 0;

									divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
									divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
									if (divtheta < mintheta)
									{
										mintheta = divtheta;
										searchLocation = 3;
										k = nowLocation;
									}
								}
							}

						//SE->SW
						if (nearPoint2.at<uchar>(i, j) == 1)
						for (int is = ir + x, js = jr + x, nowLocation = 0; js >= jr - x + 1; --js, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag2 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 4;
									k = nowLocation;
								}
							}
						}

						//SW->W
						if (nearPoint3.at<uchar>(i, j) == 1)
						for (int is = ir + x, js = jr - x, nowLocation = 0; is >= ir; --is, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag3 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 1;
									k = nowLocation;
								}
							}
						}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 4)		//8區域搜尋 - 4區
					{
						nearPoint3.at<uchar>(i, j) = 0;
						//SE->SW
						if (nearPoint1.at<uchar>(i, j) == 1)
						for (int is = ir + x, js = jr + x, nowLocation = 0; js >= jr - x + 1; --js, ++nowLocation)
						{

							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag1 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 4;
									k = nowLocation;
								}
							}
						}
						//SW->NW
						if (nearPoint2.at<uchar>(i, j) == 1)
						for (int is = ir + x, js = jr - x, nowLocation = 0; is >= ir - x + 1; --is, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag2 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 1;
									k = nowLocation;
								}
							}
						}
						//NW
						if (graddRef.at<float>(ir - x, jr - x) != -1000.0f && nearPoint2.at<uchar>(i, j) == 1 && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
						{
							flag2 = 0;

							divtheta = abs(((graddRef.at<float>(ir - x, jr - x) + CV_PI) / CV_PI)*180.0f - theta0);
							divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
							if (divtheta < mintheta)
							{
								mintheta = divtheta;
								searchLocation = 2;
								k = 0;
							}
						}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 5)		//8區域搜尋 - 5區
					{
						//S->SW
						if (nearPoint1.at<uchar>(i, j) == 1)
							for (int is = ir + x, js = jr, nowLocation = x; js >= jr - x + 1; --js, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
								{
									flag1 = 0;

									divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
									divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
									if (divtheta < mintheta)
									{
										mintheta = divtheta;
										searchLocation = 4;
										k = nowLocation;
									}
								}
							}
						//SW->NW
						if (nearPoint2.at<uchar>(i, j) == 1)
						for (int is = ir + x, js = jr - x, nowLocation = 0; is >= ir - x + 1; --is, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag2 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 1;
									k = nowLocation;
								}
							}
						}
						//NW->N
						if (nearPoint3.at<uchar>(i, j) == 1)
						for (int is = ir - x, js = jr - x, nowLocation = 0; js <= jr; ++js, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag3 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 2;
									k = nowLocation;
								}
							}
						}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 6)		//8區域搜尋 - 6區
					{
						nearPoint3.at<uchar>(i, j) = 0;
						//SW->NW
						if (nearPoint1.at<uchar>(i, j) == 1)
						for (int is = ir + x, js = jr - x, nowLocation = 0; is >= ir - x + 1; --is, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag1 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 1;
									k = nowLocation;
								}
							}
						}
						//NW->NE
						if (nearPoint2.at<uchar>(i, j) == 1)
						for (int is = ir - x, js = jr - x, nowLocation = 0; js <= jr + x - 1; ++js, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag2 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 2;
									k = nowLocation;
								}
							}
						}
						//NE
						if (graddRef.at<float>(ir - x, jr + x) != -1000.0f && nearPoint2.at<uchar>(i, j) == 1 && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
						{
							flag2 = 0;

							divtheta = abs(((graddRef.at<float>(ir - x, jr + x) + CV_PI) / CV_PI)*180.0f - theta0);
							divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
							if (divtheta < mintheta)
							{
								mintheta = divtheta;
								searchLocation = 3;
								k = 0;
							}
						}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 7)		//8區域搜尋 - 7區
					{
						//W->NW
						if (nearPoint1.at<uchar>(i, j) == 1)
							for (int is = ir, js = jr - x, nowLocation = x; is >= ir - x + 1; --is, ++nowLocation)
							{
								if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
								{
									flag1 = 0;

									divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
									divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
									if (divtheta < mintheta)
									{
										mintheta = divtheta;
										searchLocation = 1;
										k = nowLocation;
									}
								}
							}
						//NW->NE
						if (nearPoint2.at<uchar>(i, j) == 1)
						for (int is = ir - x, js = jr - x, nowLocation = 0; js <= jr + x - 1; ++js, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag2 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 2;
									k = nowLocation;
								}
							}
						}
						//NE->E
						if (nearPoint3.at<uchar>(i, j) == 1)
						for (int is = ir - x, js = jr + x, nowLocation = 0; is <= ir; ++is, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag3 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 3;
									k = nowLocation;
								}
							}
						}
					}
					else if (endPointMap.at<Vec2b>(i, j)[1] == 8)		//8區域搜尋 - 8區
					{
						nearPoint3.at<uchar>(i, j) = 0;
						//NW->NE
						if (nearPoint1.at<uchar>(i, j) == 1)
						for (int is = ir - x, js = jr - x, nowLocation = 0; js <= jr + x - 1; ++js, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag1 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 2;
									k = nowLocation;
								}
							}
						}
						//NE->SE
						if (nearPoint2.at<uchar>(i, j) == 1)
						for (int is = ir - x, js = jr + x, nowLocation = 0; is <= ir + x - 1; ++is, ++nowLocation)
						{
							if (graddRef.at<float>(is, js) != -1000.0f && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
							{
								flag2 = 0;

								divtheta = abs(((graddRef.at<float>(is, js) + CV_PI) / CV_PI)*180.0f - theta0);
								divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
								if (divtheta < mintheta)
								{
									mintheta = divtheta;
									searchLocation = 3;
									k = nowLocation;
								}
							}
						}
						//SE
						if (graddRef.at<float>(ir + x, jr + x) != -1000.0f && nearPoint2.at<uchar>(i, j) == 1 && endPointMap.at<Vec2b>(i, j)[0] != 1 && endPointMap.at<Vec2b>(i, j)[0] != 4)
						{
							flag2 = 0;

							divtheta = abs(((graddRef.at<float>(ir + x, jr + x) + CV_PI) / CV_PI)*180.0f - theta0);
							divtheta = divtheta > 180 ? (360 - divtheta) : divtheta;
							if (divtheta < mintheta)
							{
								mintheta = divtheta;
								searchLocation = 4;
								k = 0;
							}
						}
					}

					//修改端點類型
					if (flag1 == 0) { nearPoint1.at<uchar>(i, j) = 0; }
					if (flag2 == 0) { nearPoint2.at<uchar>(i, j) = 0; }
					if (flag3 == 0) { nearPoint3.at<uchar>(i, j) = 0; }
					if (nearPoint1.at<uchar>(i, j) == 0 && nearPoint2.at<uchar>(i, j) == 0 && nearPoint3.at<uchar>(i, j) == 0)
					{
						endPointMap.at<Vec2b>(i, j)[0] = 4;	//淘汰的端點
					}

					//連通最佳點(四區域分類)
					if (searchLocation == 1 && mintheta <= 60)		//4區域連通 - SW區
					{
						connectgradm = gradm.at<uchar>(i + x - k, j - x);		//連通目標幅值
						connectgradd = gradd.at<float>(i + x - k, j - x);		//連通目標方向

						int step = 0;	//步數及權重(from 0 to x)
						int sign = 0;	//斜線偏左或偏右

						if (x - k > 0) { sign = 1; }
						else if (x - k < 0) { sign = -1; }

						//直線區
						for (int ic = i, jc = j; jc >= j - (x - abs(k - x)); --jc)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
						//斜線區
						for (int ic = i + sign, jc = j - (x - abs(k - x)) - 1; jc >= j - x; ic = ic + sign, --jc)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
					}
					else if (searchLocation == 2 && mintheta <= 60)		//4區域連通 - NW區
					{
						connectgradm = gradm.at<uchar>(i - x, j - x + k);		//連通目標幅值
						connectgradd = gradd.at<float>(i - x, j - x + k);		//連通目標方向

						int step = 0;	//步數及權重(from 0 to x)
						int sign = 0;	//斜線偏左或偏右

						if (x - k > 0) { sign = 1; }
						else if (x - k < 0) { sign = -1; }

						//直線區
						for (int ic = i, jc = j; ic >= i - (x - abs(k - x)); --ic)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
						//斜線區
						for (int ic = i - (x - abs(k - x)) - 1, jc = j - sign; ic >= i - x; --ic, jc = jc - sign)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
					}
					else if (searchLocation == 3 && mintheta <= 60)		//4區域連通 - NE區
					{
						connectgradm = gradm.at<uchar>(i - x + k, j + x);		//連通目標幅值
						connectgradd = gradd.at<float>(i - x + k, j + x);		//連通目標方向

						int step = 0;	//步數及權重(from 0 to x)
						int sign = 0;	//斜線偏左或偏右

						if (x - k > 0) { sign = 1; }
						else if (x - k < 0) { sign = -1; }

						//直線區
						for (int ic = i, jc = j; jc <= j + (x - abs(k - x)); ++jc)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
						//斜線區
						for (int ic = i - sign, jc = j + (x - abs(k - x)) + 1; jc <= j + x; ic = ic - sign, ++jc)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
					}
					else if (searchLocation == 4 && mintheta <= 60)		//4區域連通 - SE區
					{
						connectgradm = gradm.at<uchar>(i + x, j + x - k);		//連通目標幅值
						connectgradd = gradd.at<float>(i + x, j + x - k);		//連通目標方向

						int step = 0;	//步數及權重(from 0 to x)
						int sign = 0;	//斜線偏左或偏右

						if (x - k > 0) { sign = 1; }
						else if (x - k < 0) { sign = -1; }

						//直線區
						for (int ic = i, jc = j; ic <= i + (x - abs(k - x)); ++ic)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
						//斜線區
						for (int ic = i + (x - abs(k - x)) + 1, jc = j + sign; ic <= i + x; ++ic, jc = jc + sign)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
					}

				}
			}
	}
}

/*滯後閥值*/
void HysteresisThreshold(InputArray _NMSgradientField_abs, OutputArray _HTedge, int upperThreshold, int lowerThreshold)
{
	Mat NMSgradientField_abs = _NMSgradientField_abs.getMat();
	CV_Assert(NMSgradientField_abs.type() == CV_8UC1);

	_HTedge.create(NMSgradientField_abs.size(), CV_8UC1);
	Mat HTedge = _HTedge.getMat();

	Mat UT;		//上閥值二值化
	threshold(NMSgradientField_abs, UT, upperThreshold, 255, THRESH_BINARY);
	Mat LT;		//下閥值二值化
	threshold(NMSgradientField_abs, LT, lowerThreshold, 255, THRESH_BINARY);
	Mat MT;		//弱邊緣
	MT.create(NMSgradientField_abs.size(), CV_8UC1);
	for (int i = 0; i < NMSgradientField_abs.rows; ++i)
		for (int j = 0; j < NMSgradientField_abs.cols; ++j)
		{
			if (LT.at<uchar>(i, j) == 255 && UT.at<uchar>(i, j) == 0)
				MT.at<uchar>(i, j) = 255;
			else
				MT.at<uchar>(i, j) = 0;

			if (UT.at<uchar>(i, j) == 255)
				HTedge.at<uchar>(i, j) = 255;
			else
				HTedge.at<uchar>(i, j) = 0;
		}

	Mat labelImg;
	int labelNum = bwlabel(MT, labelImg, 8);
	labelNum = labelNum + 1;	// include label 0
	int* labeltable = new int[labelNum];		// initialize label table with zero  
	memset(labeltable, 0, labelNum * sizeof(int));

	for (int i = 0; i < NMSgradientField_abs.rows; ++i)
		for (int j = 0; j < NMSgradientField_abs.cols; ++j)
		{
			//+ - + - + - +
			//| B | C | D |
			//+ - + - + - +
			//| E | A | F |
			//+ - + - + - +
			//| G | H | I |
			//+ - + - + - +

			int B, C, D, E, F, G, H, I;

			if (i == 0 || j == 0) { B = 0; }
			else { B = UT.at<uchar>(i - 1, j - 1); }

			if (i == 0) { C = 0; }
			else { C = UT.at<uchar>(i - 1, j); }

			if (i == 0 || j == NMSgradientField_abs.cols - 1) { D = 0; }
			else { D = UT.at<uchar>(i - 1, j + 1); }

			if (j == 0) { E = 0; }
			else { E = UT.at<uchar>(i, j - 1); }

			if (j == NMSgradientField_abs.cols - 1) { F = 0; }
			else { F = UT.at<uchar>(i, j + 1); }

			if (i == NMSgradientField_abs.rows - 1 || j == 0) { G = 0; }
			else { G = UT.at<uchar>(i + 1, j - 1); }

			if (i == NMSgradientField_abs.rows - 1) { H = 0; }
			else { H = UT.at<uchar>(i + 1, j); }

			if (i == NMSgradientField_abs.rows - 1 || j == NMSgradientField_abs.cols - 1) { I = 0; }
			else { I = UT.at<uchar>(i + 1, j + 1); }

			// apply 8 connectedness  
			if (B || C || D || E || F || G || H || I)
			{
				++labeltable[labelImg.at<int>(i, j)];
			}
		}

	labeltable[0] = 0;		//clear 0 label

	for (int i = 0; i < labelImg.rows; i++)
		for (int j = 0; j < labelImg.cols; j++)
		{
			if (labeltable[labelImg.at<int>(i, j)] > 0)
			{
				HTedge.at<uchar>(i, j) = 255;
			}
		}
	delete[] labeltable;
	labeltable = nullptr;
}