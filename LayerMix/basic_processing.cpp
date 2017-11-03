#include "stdafx.h"
#include "basic_processing.h"


/*�M��ڵ��I*/
int findroot(int labeltable[], int label)
{
	int x = label;
	while (x != labeltable[x])
		x = labeltable[x];
	return x;
}

/*�M��s�q�u*/
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

/*�P�_�I������*/
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
						if (nearPoint[(k % 9)] != 0 && nearPoint[(k % 8) + 1] != 0) //1,2�B2,3�B...�B8,1   End of Line Point 
						{
							labels.at<Vec2b>(i, j)[0] = 2;
							if (k % 2 == 1)		//�u�s���﨤�u����V
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

/*�Ыئ���*/
void makecolorwheel(vector<Scalar> &colorwheel)
{
	int RY = 15;	//����(Red)     �ܶ���(Yellow)
	int YG = 15;	//����(Yellow)  �ܺ��(Green)
	int GC = 15;	//���(Green)   �ܫC��(Cyan)
	int CB = 15;	//�C��(Cyan)    ���Ŧ�(Blue)
	int BM = 15;	//�Ŧ�(Blue)    �ܬv��(Magenta)
	int MR = 15;	//�v��(Magenta) �ܬ���(Red)

	for (int i = 0; i < RY; i++) colorwheel.push_back(Scalar(255, 255 * i / RY, 0));
	for (int i = 0; i < YG; i++) colorwheel.push_back(Scalar(255 - 255 * i / YG, 255, 0));
	for (int i = 0; i < GC; i++) colorwheel.push_back(Scalar(0, 255, 255 * i / GC));
	for (int i = 0; i < CB; i++) colorwheel.push_back(Scalar(0, 255 - 255 * i / CB, 255));
	for (int i = 0; i < BM; i++) colorwheel.push_back(Scalar(255 * i / BM, 0, 255));
	for (int i = 0; i < MR; i++) colorwheel.push_back(Scalar(255, 0, 255 - 255 * i / MR));
}

/*�N�Ϥ���H������V�����(��J��׳��α�פ�V)*/
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
		maxrad = 255;		//�u����פ�V�L��״T��

		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;

				if (field.at<float>(i, j) == -1000.0f)		//�ΥH��ܵL��פ�V
				{
					for (int b = 0; b < 3; b++)
					{
						data[2 - b] = 255;
					}
				}
				else
				{
					float rad = maxrad;

					float angle = field.at<float>(i, j) / CV_PI;    //��쬰-1��+1
					float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //�p�⨤�׹��������ަ�m
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

		maxrad = maxrad / 2;		//�[�`��ܵ��G(�i��������)

		for (int i = 0; i < field.rows; ++i)
			for (int j = 0; j < field.cols; ++j)
			{
				uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;
				Vec2f field_at_point = field.at<Vec2f>(i, j);

				float fx = field_at_point[0];
				float fy = field_at_point[1];

				float rad = sqrt(fx * fx + fy * fy) / maxrad;

				float angle = atan2(fy, fx) / CV_PI;    //��쬰-1��+1
				float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //�p�⨤�׹��������ަ�m
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

/*�N�Ϥ���H������V�����(��J��״T�Ȥα�פ�V)*/
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

	maxrad = maxrad / 2;		//�[�`��ܵ��G(�i��������)

	for (int i = 0; i < gradm.rows; ++i)
		for (int j = 0; j < gradm.cols; ++j)
		{
			uchar *data = colorField.data + colorField.step[0] * i + colorField.step[1] * j;

			if (gradd.at<float>(i, j) == -1000.0f)		//�ΥH��ܵL��פ�V
			{
				for (int b = 0; b < 3; b++)
				{
					data[2 - b] = 255;
				}
			}
			else
			{
				float rad = gradm.at<float>(i, j) / maxrad;

				float angle = gradd.at<float>(i, j) / CV_PI;    //��쬰-1��+1
				float fk = (angle + 1.0) / 2.0 * (colorwheel.size() - 1);  //�p�⨤�׹��������ަ�m
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

/*�N�Ϥ���u�ʩԦ��åH�Ƕ������*/
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

/*�ϼh�V�X�Ҧ�*/
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

/*���βV�X�Ҧ�*/
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

/*���L�|�X�V�X�Ҧ�*/
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

/*�h�����T*/
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

/*�����t��*/
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

/*���X�����Ϋ�����V��׬���׳�*/
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

/*�p���״T�ȤΤ�V*/
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

			if (x == 0 && y == 0) { gradd.at<float>(i, j) = -1000.0f; }	//�ΥH��ܵL��פ�V
			else { gradd.at<float>(i, j) = atan2(y, x); }
		}
}

/*�D���j�ȧ��*/
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

	float theta = 0.0f;			//�ثe��������V
	int amplitude = 0;			//�ثe�������T��
	int amplitude1 = 0;			//�F�칳��1���T��
	int amplitude2 = 0;			//�F�칳��2���T��
	float A1 = 0.0f;			//�W�{��1�T��
	float A2 = 0.0f;			//�W�{��2�T��
	float B1 = 0.0f;			//�U�{��1�T��
	float B2 = 0.0f;			//�U�{��2�T��
	float alpha = 0.0f;			//��ҫY��

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
				graddNMS.at<float>(i, j) = -1000.0f;		//�ΥH�Ϥ��L����
			}
		}
}

/*�M������V�I*/
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

	float theta = 0.0f;			//�ثe��������V

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

/*�_�u�s�q*/
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

	Mat endPointMap;		//�x�s�I����
	pointlabel(gradm, endPointMap);

	Mat nearPoint1(gradm.size(), CV_8UC1, Scalar(1));		//�۾F��V1
	Mat nearPoint2(gradm.size(), CV_8UC1, Scalar(1));		//�۾F��V2
	Mat nearPoint3(gradm.size(), CV_8UC1, Scalar(1));		//�۾F��V3

	if (startSpace != 2)	//�d�߬O�_�B��
	{
		for (int x = 2; x <= startSpace - 1; ++x)
		{
			Mat graddRef;
			copyMakeBorder(gradd, graddRef, x - 1, x - 1, x - 1, x - 1, BORDER_CONSTANT, Scalar(-1000.0f));

			for (int i = 1; i < gradd.rows - 1; ++i)		//���j�M�v�����
				for (int j = 1; j < gradd.cols - 1; ++j)		//���j�M�v�����
				{
					int ir = i + x - 1, jr = j + x - 1;		//reference index i,j for graddRef

					if (endPointMap.at<Vec2b>(i, j)[0] == 2)	//�P�_�O�_�����I
					{

						bool flag1 = 1, flag2 = 1, flag3 = 1;	//�۾Fflag

						if (endPointMap.at<Vec2b>(i, j)[1] == 1)		//8�ϰ�j�M - 1��
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
						else if (endPointMap.at<Vec2b>(i, j)[1] == 2)		//8�ϰ�j�M - 2��
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
						else if (endPointMap.at<Vec2b>(i, j)[1] == 3)		//8�ϰ�j�M - 3��
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
						else if (endPointMap.at<Vec2b>(i, j)[1] == 4)		//8�ϰ�j�M - 4��
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
						else if (endPointMap.at<Vec2b>(i, j)[1] == 5)		//8�ϰ�j�M - 5��
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
						else if (endPointMap.at<Vec2b>(i, j)[1] == 6)		//8�ϰ�j�M - 6��
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
						else if (endPointMap.at<Vec2b>(i, j)[1] == 7)		//8�ϰ�j�M - 7��
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
						else if (endPointMap.at<Vec2b>(i, j)[1] == 8)		//8�ϰ�j�M - 8��
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

						//�ק���I����
						if (flag1 == 0) { nearPoint1.at<uchar>(i, j) = 0; }
						if (flag2 == 0) { nearPoint2.at<uchar>(i, j) = 0; }
						if (flag3 == 0) { nearPoint3.at<uchar>(i, j) = 0; }
						if (nearPoint1.at<uchar>(i, j) == 0 && nearPoint2.at<uchar>(i, j) == 0 && nearPoint3.at<uchar>(i, j) == 0)
						{
							endPointMap.at<Vec2b>(i, j)[0] = 4;	//�^�O���I
						}
					}
				}
		}
	}
	for (int x = startSpace; x <= endSpace; ++x)
	{
		Mat graddRef;
		copyMakeBorder(gradd, graddRef, x - 1, x - 1, x - 1, x - 1, BORDER_CONSTANT, Scalar(-1000.0f));

		/*�j�M�ós�q�u*/
		for (int i = 1; i < gradd.rows - 1; ++i)		//���j�M�v�����
			for (int j = 1; j < gradd.cols - 1; ++j)		//���j�M�v�����
			{
				int ir = i + x - 1, jr = j + x - 1;		//reference index i,j for graddRef

				if (endPointMap.at<Vec2b>(i, j)[0] == 2)	//�P�_�O�_�����I
				{
					float theta0 = ((gradd.at<float>(i, j) + CV_PI) / CV_PI)*180.0f;	//�ثe���I����
					float divtheta = 0.0f;		//�j���I�ۮt����
					float mintheta = 180.0f;	//�̤p�ۮt����

					char searchLocation = 0;	//�|�ϰ����
					char k = 0;					//�|�ϰ����������m( k = 0 : 2*x-1 )

					float connectgradm = 0;		//�s�q�ؼдT��
					float connectgradd = 0.0f;		//�s�q�ؼФ�V

					bool flag1 = 1, flag2 = 1, flag3 = 1;	//�۾Fflag

					/*�j�M�̨��I(�K�ϰ����)*/
					if (endPointMap.at<Vec2b>(i, j)[1] == 1)		//8�ϰ�j�M - 1��
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 2)		//8�ϰ�j�M - 2��
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 3)		//8�ϰ�j�M - 3��
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 4)		//8�ϰ�j�M - 4��
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 5)		//8�ϰ�j�M - 5��
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 6)		//8�ϰ�j�M - 6��
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 7)		//8�ϰ�j�M - 7��
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
					else if (endPointMap.at<Vec2b>(i, j)[1] == 8)		//8�ϰ�j�M - 8��
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

					//�ק���I����
					if (flag1 == 0) { nearPoint1.at<uchar>(i, j) = 0; }
					if (flag2 == 0) { nearPoint2.at<uchar>(i, j) = 0; }
					if (flag3 == 0) { nearPoint3.at<uchar>(i, j) = 0; }
					if (nearPoint1.at<uchar>(i, j) == 0 && nearPoint2.at<uchar>(i, j) == 0 && nearPoint3.at<uchar>(i, j) == 0)
					{
						endPointMap.at<Vec2b>(i, j)[0] = 4;	//�^�O�����I
					}

					//�s�q�̨��I(�|�ϰ����)
					if (searchLocation == 1 && mintheta <= 60)		//4�ϰ�s�q - SW��
					{
						connectgradm = gradm.at<uchar>(i + x - k, j - x);		//�s�q�ؼдT��
						connectgradd = gradd.at<float>(i + x - k, j - x);		//�s�q�ؼФ�V

						int step = 0;	//�B�Ƥ��v��(from 0 to x)
						int sign = 0;	//�׽u�����ΰ��k

						if (x - k > 0) { sign = 1; }
						else if (x - k < 0) { sign = -1; }

						//���u��
						for (int ic = i, jc = j; jc >= j - (x - abs(k - x)); --jc)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
						//�׽u��
						for (int ic = i + sign, jc = j - (x - abs(k - x)) - 1; jc >= j - x; ic = ic + sign, --jc)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
					}
					else if (searchLocation == 2 && mintheta <= 60)		//4�ϰ�s�q - NW��
					{
						connectgradm = gradm.at<uchar>(i - x, j - x + k);		//�s�q�ؼдT��
						connectgradd = gradd.at<float>(i - x, j - x + k);		//�s�q�ؼФ�V

						int step = 0;	//�B�Ƥ��v��(from 0 to x)
						int sign = 0;	//�׽u�����ΰ��k

						if (x - k > 0) { sign = 1; }
						else if (x - k < 0) { sign = -1; }

						//���u��
						for (int ic = i, jc = j; ic >= i - (x - abs(k - x)); --ic)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
						//�׽u��
						for (int ic = i - (x - abs(k - x)) - 1, jc = j - sign; ic >= i - x; --ic, jc = jc - sign)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
					}
					else if (searchLocation == 3 && mintheta <= 60)		//4�ϰ�s�q - NE��
					{
						connectgradm = gradm.at<uchar>(i - x + k, j + x);		//�s�q�ؼдT��
						connectgradd = gradd.at<float>(i - x + k, j + x);		//�s�q�ؼФ�V

						int step = 0;	//�B�Ƥ��v��(from 0 to x)
						int sign = 0;	//�׽u�����ΰ��k

						if (x - k > 0) { sign = 1; }
						else if (x - k < 0) { sign = -1; }

						//���u��
						for (int ic = i, jc = j; jc <= j + (x - abs(k - x)); ++jc)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
						//�׽u��
						for (int ic = i - sign, jc = j + (x - abs(k - x)) + 1; jc <= j + x; ic = ic - sign, ++jc)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
					}
					else if (searchLocation == 4 && mintheta <= 60)		//4�ϰ�s�q - SE��
					{
						connectgradm = gradm.at<uchar>(i + x, j + x - k);		//�s�q�ؼдT��
						connectgradd = gradd.at<float>(i + x, j + x - k);		//�s�q�ؼФ�V

						int step = 0;	//�B�Ƥ��v��(from 0 to x)
						int sign = 0;	//�׽u�����ΰ��k

						if (x - k > 0) { sign = 1; }
						else if (x - k < 0) { sign = -1; }

						//���u��
						for (int ic = i, jc = j; ic <= i + (x - abs(k - x)); ++ic)
						{
							endPointMap.at<Vec2b>(ic, jc)[0] = 3;
							endPointMap.at<Vec2b>(ic, jc)[1] = 0;
							gradmCBL.at<uchar>(ic, jc) = (float)gradm.at<uchar>(i, j)*(x - step) / x + connectgradm*step / x;
							graddCBL.at<float>(ic, jc) = atan2(sin(gradd.at<float>(i, j))*(x - step) + sin(connectgradd)*step, cos(gradd.at<float>(i, j))*(x - step) + cos(connectgradd)*step);
							++step;
						}
						//�׽u��
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

/*����֭�*/
void HysteresisThreshold(InputArray _NMSgradientField_abs, OutputArray _HTedge, int upperThreshold, int lowerThreshold)
{
	Mat NMSgradientField_abs = _NMSgradientField_abs.getMat();
	CV_Assert(NMSgradientField_abs.type() == CV_8UC1);

	_HTedge.create(NMSgradientField_abs.size(), CV_8UC1);
	Mat HTedge = _HTedge.getMat();

	Mat UT;		//�W�֭ȤG�Ȥ�
	threshold(NMSgradientField_abs, UT, upperThreshold, 255, THRESH_BINARY);
	Mat LT;		//�U�֭ȤG�Ȥ�
	threshold(NMSgradientField_abs, LT, lowerThreshold, 255, THRESH_BINARY);
	Mat MT;		//�z��t
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