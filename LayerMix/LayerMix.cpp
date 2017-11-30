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

	if (blurAreaSize % 2 == 0) { --blurAreaSize; }

	std::cout << "Please enter blur square size for Line : ";
	int blurLineSize = 0;
	std::cin >> blurLineSize;

	if (blurLineSize % 2 == 0) { --blurLineSize; }

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

		string gray_outfile = filepath + "\\" + infilename + "_0.0_GRAY.png";		//�Ƕ��v��(�Ƕ�)
		imwrite(gray_outfile, grayImage);
	}
	else { grayImage = srcImage; }

	Mat grayColor_out;		//��X��(8UC3)
	DrawColorBar(grayImage, grayColor_out);

	string grayColor_outfile = filepath + "\\" + infilename + "_0.1_GRAY.png";		//�Ƕ��v��(�m��)
	imwrite(grayColor_outfile, grayColor_out);

	/****��󭱪��v���Ѩ�****/

	/*�ҽk�Ƕ��v��*/

	Mat blurImage;	//�ҽk�v��(8UC1)	
	GaussianBlur(grayImage, blurImage, Size(blurAreaSize, blurAreaSize), 0, 0);

	string blurA_outfile = filepath + "\\" + infilename + "_1_BLURA.png";			//�ҽk�Ƕ��v��(�Ƕ�)
	imwrite(blurA_outfile, blurImage);

	/*�ϼh�V�X�Ҧ�*/

	Mat divideArea;		//���βV�X�Ҧ�(8UC1)
	DivideArea(grayImage, blurImage, divideArea);

	string divideA_outfile = filepath + "\\" + infilename + "_2.1_DIVIDEA.png";			//�����βV�X�Ҧ�
	imwrite(divideA_outfile, divideArea);

	Mat hardmixArea;	//���L�|�X�V�X�Ҧ�(8UC1(BW))
	HardMix(grayImage, divideArea, hardmixArea);

	string hardmixA_outfile = filepath + "\\" + infilename + "_2.2_HARDMIXA.png";		//���L�|�X�V�X�Ҧ�
	imwrite(hardmixA_outfile, hardmixArea);

	/*�h���v�����T*/

	Mat clearWiteArea;		//�h���զ����T(8UC1(BW))
	ClearNoise(hardmixArea, clearWiteArea, 20, 4, 1);

	string clearWA_outfile = filepath + "\\" + infilename + "_3.1_CLEARW.png";			//�h���զ����T
	imwrite(clearWA_outfile, clearWiteArea);

	Mat clearBlackArea;		//�h���¦����T(8UC1(BW))
	ClearNoise(clearWiteArea, clearBlackArea, 20, 4, 0);

	string clearBA_outfile = filepath + "\\" + infilename + "_3.2_CLEARB.png";			//�h���¦����T
	imwrite(clearBA_outfile, clearBlackArea);


	/****���u���v���Ѩ�****/

	/*�ҽk�Ƕ��v��*/

	Mat blurImageL;	//�ҽk�v��(8UC1)	
	GaussianBlur(grayImage, blurImageL, Size(blurLineSize, blurLineSize), 0, 0);

	string blurL_outfile = filepath + "\\" + infilename + "_4_BLURL.png";			//�ҽk�Ƕ��v��(�Ƕ�)
	imwrite(blurL_outfile, blurImageL);

	/*�p��v�����*/

	Mat gradx, grady;		//�����Ϋ������(16SC1)
	Differential(blurImageL, gradx, grady);

	Mat gradf;			//��׳�(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;		//��״T�Ȥα�פ�V(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_out, grady_out, gradm_out,gradd_out, gradf_out;		//��X��(8UC1�B8UC1�B8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradx, gradx_out);
	DrawAbsGraySystem(grady, grady_out);
	DrawAbsGraySystem(gradm, gradm_out);
	DrawColorSystem(gradd, gradd_out);
	DrawColorSystem(gradf, gradf_out);

	string gradx_outfile = filepath + "\\" + infilename + "_5.1_GX.png";			//��X�μv�����(����)
	imwrite(gradx_outfile, gradx_out);
	string grady_outfile = filepath + "\\" + infilename + "_5.2_GY.png";			//��X�μv�����(����)
	imwrite(grady_outfile, grady_out);
	string gradm_outfile = filepath + "\\" + infilename + "_5.3_GM.png";			//��X�μv�����(�T��)
	imwrite(gradm_outfile, gradm_out);
	string gradd_outfile = filepath + "\\" + infilename + "_5.4_GD.png";			//��X�μv�����(��V)
	imwrite(gradd_outfile, gradd_out);
	string gradf_outfile = filepath + "\\" + infilename + "_5.5_GF.png";			//��X�μv�����(��)
	imwrite(gradf_outfile, gradf_out);


	///*��V�ҽk*/

	//Mat graddBlur;	//�ҽk��V(8UC1)	
	//BlurDirection(gradd, graddBlur, 5);

	//Mat graddBlur_out, gradfBlur_out;		//��X��(8UC3�B8UC3)
	//DrawColorSystem(graddBlur, graddBlur_out);
	//DrawColorSystem(gradm, graddBlur, gradfBlur_out);

	//string graddBlur_outfile = filepath + "\\" + infilename + "_5.1_BLURD.png";			//�ҽk��V(��V)
	//imwrite(graddBlur_outfile, graddBlur_out);
	//string gradfBlur_outfile = filepath + "\\" + infilename + "_5.2_BLURF.png";			//�ҽk��V(��)
	//imwrite(gradfBlur_outfile, gradfBlur_out);

	/*�D���j�ȧ��*/

	Mat gradmNMS, graddNMS;			//�D�̤j�ȧ��(8UC1�B32FC1)
	NonMaximumSuppression(gradm, gradd, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMS_outfile = filepath + "\\" + infilename + "_6.1_NMSM.png";			//�D�̤j�ȧ��(�T��)
	imwrite(gradmNMS_outfile, gradmNMS_out);
	string graddNMS_outfile = filepath + "\\" + infilename + "_6.2_NMSD.png";			//�D�̤j�ȧ��(��V)
	imwrite(graddNMS_outfile, graddNMS_out);
	string gradfNMS_outfile = filepath + "\\" + infilename + "_6.3_NMSF.png";			//�D�̤j�ȧ��(��)
	imwrite(gradfNMS_outfile, gradfNMS_out);

	///*�M������V�I*/

	//Mat gradmCDD, graddCDD;			//�M������V�I(8UC1�B32FC1)
	//ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	//Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//��X��(8UC1�B8UC3�B8UC3)
	//DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	//DrawColorSystem(graddCDD, graddCDD_out);
	//DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	//string gradmCDD_outfile = filepath + "\\" + infilename + "_8.1_CDDM.png";			//�M������V�I(�T��)
	//imwrite(gradmCDD_outfile, gradmCDD_out);
	//string graddCDD_outfile = filepath + "\\" + infilename + "_8.2_CDDD.png";			//�M������V�I(��V)
	//imwrite(graddCDD_outfile, graddCDD_out);
	//string gradfCDD_outfile = filepath + "\\" + infilename + "_8.3_CDDF.png";			//�M������V�I(��)
	//imwrite(gradfCDD_outfile, gradfCDD_out);

	/*�ҽk�T��*/

	Mat gradmBlur;	//�ҽk�T��(8UC1)	
	GaussianBlur(gradmNMS, gradmBlur, Size(5, 5), 0, 0);

	string blurM_outfile = filepath + "\\" + infilename + "_7_BLURM.png";				//�ҽk�T��(�T��)
	imwrite(blurM_outfile, gradmBlur);

	/*�u���βV�X�Ҧ�*/

	Mat gradmDivide;										//�u���βV�X�Ҧ�(8UC1)
	DivideLine(gradmNMS, gradmBlur, gradmDivide);

	Mat gradmDivide_out, gradfDivide_out;		//��X��(8UC1�B8UC3)
	DrawAbsGraySystem(gradmDivide, gradmDivide_out);
	DrawColorSystem(gradmDivide, graddNMS, gradfDivide_out);

	string gradmDivide_outfile = filepath + "\\" + infilename + "_8.1_DIVIDEM.png";		//�u���βV�X�Ҧ�(�T��)
	imwrite(gradmDivide_outfile, gradmDivide_out);
	string gradfDivide_outfile = filepath + "\\" + infilename + "_8.2_DIVIDEF.png";		//�u���βV�X�Ҧ�(��)
	imwrite(gradfDivide_outfile, gradfDivide_out);

	/*�h���t���I*/

	Mat gradmCP, graddCP;	//�h���t���I(8UC1�B8UC3)
	ClearPoint(gradmDivide, graddNMS, gradmCP, graddCP);

	Mat gradmCP_out, graddCP_out, gradfCP_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmCP, gradmCP_out);
	DrawColorSystem(graddCP, graddCP_out);
	DrawColorSystem(gradmCP, graddCP, gradfCP_out);

	string gradmCP_outfile = filepath + "\\" + infilename + "_9.1_CPM.png";			//�h���t���I(�T��)
	imwrite(gradmCP_outfile, gradmCP_out);
	string graddCP_outfile = filepath + "\\" + infilename + "_9.2_CPD.png";			//�h���t���I(��V)
	imwrite(graddCP_outfile, graddCP_out);
	string gradfCP_outfile = filepath + "\\" + infilename + "_9.3_CPF.png";			//�h���t���I(��)
	imwrite(gradfCP_outfile, gradfCP_out);

	/*��ٺ��I�s�q*/

	Mat gradmSCL, graddSCL;			//�u��ٺ��I�s�q(8UC1�B32FC1)
	ConnectLine(gradmCP, graddCP, gradmSCL, graddSCL, 2, 5, 60, 0, 0);

	Mat gradmSCL_out, graddSCL_out, gradfSCL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmSCL, gradmSCL_out);
	DrawColorSystem(graddSCL, graddSCL_out);
	DrawColorSystem(gradmSCL, graddSCL, gradfSCL_out);

	string gradmSCL_outfile = filepath + "\\" + infilename + "_10.1_SCLM.png";			//��ٺ��I�s�q(�T��)
	imwrite(gradmSCL_outfile, gradmSCL_out);
	string graddSCL_outfile = filepath + "\\" + infilename + "_10.2_SCLD.png";			//��ٺ��I�s�q(��V)
	imwrite(graddSCL_outfile, graddSCL_out);
	string gradfSCL_outfile = filepath + "\\" + infilename + "_10.3_SCLF.png";			//��ٺ��I�s�q(��)
	imwrite(gradfSCL_outfile, gradfSCL_out);

	/*�������*/

	Mat gradmHC, graddHC;		//�������(8UC1�B32FC1)
	HysteresisCut(gradmSCL, graddSCL, clearBlackArea, gradmHC, graddHC);

	Mat gradmHC_out, graddHC_out, gradfHC_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmHC, gradmHC_out);
	DrawColorSystem(graddHC, graddHC_out);
	DrawColorSystem(gradmHC, graddHC, gradfHC_out);

	string gradmHC_outfile = filepath + "\\" + infilename + "_11.1_HCM.png";				//�������(�T��)
	imwrite(gradmHC_outfile, gradmHC_out);
	string graddHC_outfile = filepath + "\\" + infilename + "_11.2_HCD.png";				//�������(��V)
	imwrite(graddHC_outfile, graddHC_out);
	string gradfHC_outfile = filepath + "\\" + infilename + "_11.3_HCF.png";				//�������(��)
	imwrite(gradfHC_outfile, gradfHC_out);

	/*�G�Ȥ�*/

	Mat lineHT;		//�G�Ȥ�(8UC1(BW))
	threshold(gradmHC, lineHT, 1, 255, THRESH_BINARY);

	string LHT_outfile = filepath + "\\" + infilename + "_12_BW.png";					//�G�Ȥ�(�G��)
	imwrite(LHT_outfile, lineHT);

	/*�j����I�s�q*/

	Mat gradmACL, graddACL, lineACL;			//�u��ٺ��I�s�q(8UC1�B32FC1�B8UC1(BW))
	BWConnectLine(gradmSCL, graddSCL, lineHT, gradmACL, graddACL, lineACL, 2, 5, 90, 1, 1);

	Mat gradmACL_out, graddACL_out, gradfACL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmACL, gradmACL_out);
	DrawColorSystem(graddACL, graddACL_out);
	DrawColorSystem(gradmACL, graddACL, gradfACL_out);

	string lineACL_outfile = filepath + "\\" + infilename + "_13.0_ACL.png";			//�j����I�s�q(�G��)
	imwrite(lineACL_outfile, lineACL);
	string gradmACL_outfile = filepath + "\\" + infilename + "_13.1_ACLM.png";			//�j����I�s�q(�T��)
	imwrite(gradmACL_outfile, gradmACL_out);
	string graddACL_outfile = filepath + "\\" + infilename + "_13.2_ACLD.png";			//�j����I�s�q(��V)
	imwrite(graddACL_outfile, graddACL_out);
	string gradfACL_outfile = filepath + "\\" + infilename + "_13.3_ACLF.png";			//�j����I�s�q(��)
	imwrite(gradfACL_outfile, gradfACL_out);

	/*�h�����u*/

	Mat lineCNL;	//�h�����u(8UC1(BW))
	ClearNoise(lineACL, lineCNL, 20, 8, 1);

	string lineCNL_outfile = filepath + "\\" + infilename + "_14_CNL.png";				//�h�����u(�G��)
	imwrite(lineCNL_outfile, lineCNL);


	/****���X���P�u���Ѩ����G****/

	/*���X���P�u*/

	Mat edge;	//���X���P�u(8UC1(BW))
	BWCombine(clearBlackArea, lineACL, edge);

	Mat edgeLabel_out, edgeImage_out;		//��X��(8UC3�B8UC3)
	DrawLabel(edge, edgeLabel_out);
	DrawEdge(edge, srcImage, edgeImage_out);

	string edgeBW_outfile = filepath + "\\" + infilename + "_16.0_COMBINEBW.png";		//���X���P�u(�G��)
	imwrite(edgeBW_outfile, edge);
	string edgeL_outfile = filepath + "\\" + infilename + "_16.1_COMBINEL.png";			//���X���P�u(����)
	imwrite(edgeL_outfile, edgeLabel_out);
	string edgeI_outfile = filepath + "\\" + infilename + "_16.2_COMBINEI.png";			//���X���P�u(�|��)
	imwrite(edgeI_outfile, edgeImage_out);

	/*�Z���ഫ*/
	
	Mat object;			//����G�ȹϹ�(8UC1(BW))
	BWCombine(edge, object);

	Mat objectDT;		//�Z���ഫ(32FC1(BW))
	distanceTransform(object, objectDT, CV_DIST_L2, 3);

	Mat objectDT_out;		//��X��(8UC1)
	DrawAbsGraySystem(objectDT, objectDT_out);

	string objectDT_outfile = filepath + "\\" + infilename + "_17_DT.png";			//�Z���ഫ(�Ƕ�)
	imwrite(objectDT_outfile, objectDT_out);

	///*�������t��k����*/

	return 0;
}


