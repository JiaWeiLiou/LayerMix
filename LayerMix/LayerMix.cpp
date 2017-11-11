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

	if (blurAreaSize % 2 == 0)	{ --blurAreaSize; }

	std::cout << "Please enter blur square size for Line : ";
	int blurLineSize = 0;
	std::cin >> blurLineSize;

	if (blurLineSize % 2 == 0)	{ --blurLineSize; }

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

		string gray_outfile = filepath + "\\" + infilename + "_0_GRAY.png";		//�Ƕ��v��
		imwrite(gray_outfile, grayImage);
	}
	else
		grayImage = srcImage;


	/****��󭱪��v���Ѩ�****/

	/*�N�ǫ׹Ϲ������ҽk*/

	Mat blurAImage;	//�ҽk�v��(8UC1)	
	GaussianBlur(grayImage, blurAImage, Size(blurAreaSize, blurAreaSize), 0, 0);

	string blurA_outfile = filepath + "\\" + infilename + "_1_BLURA.png";		//�ҽk�v��
	imwrite(blurA_outfile, blurAImage);

	/*�ϼh�V�X�Ҧ�*/

	Mat divideArea;		//���βV�X�Ҧ�(8UC1)
	Divide(grayImage, blurAImage, divideArea);

	string divideA_outfile = filepath + "\\" + infilename + "_2.1_DIVIDEA.png";			//���βV�X�Ҧ�
	imwrite(divideA_outfile, divideArea);

	Mat hardmixArea;	//���L�|�X�V�X�Ҧ�(8UC1(BW))
	HardMix(grayImage, divideArea, hardmixArea);			

	string hardmixA_outfile = filepath + "\\" + infilename + "_2.2_HARDMIXA.png";			//���L�|�X�V�X�Ҧ�
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

	/*�N�ǫ׹Ϲ����ȼҽk*/

	Mat blurLImage;	//�ҽk�v��(8UC1)	
	medianBlur(grayImage, blurLImage, blurLineSize);

	string blurL_outfile = filepath + "\\" + infilename + "_4_BLURL.png";		//�ҽk�v��
	imwrite(blurL_outfile, blurLImage);

	/*�p��v�����*/

	Mat gradx,grady;		//�����Ϋ������(16SC1)
	Differential(blurLImage, gradx, grady);

	Mat gradf;			//��׳�(16SC2)
	GradientField(gradx, grady, gradf);

	Mat gradm, gradd;		//��״T�Ȥα�פ�V(32FC1)
	CalculateGradient(gradf, gradm, gradd);

	Mat gradx_out, grady_out, gradm_out, gradd_out, gradf_out;		//��X��(8UC1�B8UC1�B8UC1�B8UC3�B8UC3)
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

	/*���βV�X�Ҧ�*/

	Mat divideLine;										//���βV�X�Ҧ�(8UC1)
	Divide(gradm, blurLImage, divideLine);

	Mat gradmDivide_out, gradfDivide_out;				//��X��(8UC1�B8UC3)
	DrawAbsGraySystem(divideLine, gradmDivide_out);
	DrawColorSystem(divideLine, gradd, gradfDivide_out);

	string divideM_outfile = filepath + "\\" + infilename + "_6.1_DIVIDEM.png";			//��X�Τ��βV�X�Ҧ�(�T��)
	imwrite(divideM_outfile, gradmDivide_out);
	string divideF_outfile = filepath + "\\" + infilename + "_6.2_DIVIDEF.png";			//��X�Τ��βV�X�Ҧ�(��)
	imwrite(divideF_outfile, gradfDivide_out);

	/*�D���j�ȧ��*/
	
	Mat gradmNMS, graddNMS;			//�D�̤j�ȧ��(8UC1�B32FC1)
	NonMaximumSuppression(divideLine, gradd, gradmNMS, graddNMS);

	Mat gradmNMS_out, graddNMS_out, gradfNMS_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmNMS, gradmNMS_out);
	DrawColorSystem(graddNMS, graddNMS_out);
	DrawColorSystem(gradmNMS, graddNMS, gradfNMS_out);

	string gradmNMS_outfile = filepath + "\\" + infilename + "_7.1_NMSM.png";			//�D�̤j�ȧ��(�T��)
	imwrite(gradmNMS_outfile, gradmNMS_out);
	string graddNMS_outfile = filepath + "\\" + infilename + "_7.2_NMSD.png";			//�D�̤j�ȧ��(��V)
	imwrite(graddNMS_outfile, graddNMS_out);
	string gradfNMS_outfile = filepath + "\\" + infilename + "_7.3_NMSF.png";			//�D�̤j�ȧ��(��)
	imwrite(gradfNMS_outfile, gradfNMS_out);

	/*�M������V�I*/
	
	Mat gradmCDD, graddCDD;			//�M������V�I(8UC1�B32FC1)
	ClearDifferentDirection(gradmNMS, graddNMS, gradmCDD, graddCDD);

	Mat gradmCDD_out, graddCDD_out, gradfCDD_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmCDD, gradmCDD_out);
	DrawColorSystem(graddCDD, graddCDD_out);
	DrawColorSystem(gradmCDD, graddCDD, gradfCDD_out);

	string gradmCDD_outfile = filepath + "\\" + infilename + "_8.1_CDDM.png";			//�M������V�I(�T��)
	imwrite(gradmCDD_outfile, gradmCDD_out);
	string graddCDD_outfile = filepath + "\\" + infilename + "_8.2_CDDD.png";			//�M������V�I(��V)
	imwrite(graddCDD_outfile, graddCDD_out);
	string gradfCDD_outfile = filepath + "\\" + infilename + "_8.3_CDDF.png";			//�M������V�I(��)
	imwrite(gradfCDD_outfile, gradfCDD_out);

	/*�_�u�s�q*/

	Mat gradmCBL, graddCBL;			//�_�u�s�q(8UC1�B32FC1)
	ConnectBreakLine(gradmCDD, graddCDD, gradmCBL, graddCBL, 2, 3, 60, 0, 0);

	Mat gradmCBL_out, graddCBL_out, gradfCBL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmCBL, gradmCBL_out);
	DrawColorSystem(graddCBL, graddCBL_out);
	DrawColorSystem(gradmCBL, graddCBL, gradfCBL_out);

	string gradmCBL_outfile = filepath + "\\" + infilename + "_9.1_CBLM.png";			//�_�u�s�q(�T��)
	imwrite(gradmCBL_outfile, gradmCBL_out);
	string graddCBL_outfile = filepath + "\\" + infilename + "_9.2_CBLD.png";			//�_�u�s�q(��V)
	imwrite(graddCBL_outfile, graddCBL_out);
	string gradfCBL_outfile = filepath + "\\" + infilename + "_9.3_CBLF.png";			//�_�u�s�q(��)
	imwrite(gradfCBL_outfile, gradfCBL_out);

	/*����֭�*/

	Mat lineHT;		//����֭�(8UC1(BW))
	HysteresisThreshold(gradmCBL, lineHT, 150, 50);
	string LHT_outfile = filepath + "\\" + infilename + "_10_HT.png";			//����֭�
	imwrite(LHT_outfile, lineHT);

	/*�h���t���I*/

	Mat lineCIP;	//�h���t���I(8UC1(BW))
	ClearSpecialPoint(lineHT, lineCIP, 0, 1, 0);
	string LCIP_outfile = filepath + "\\" + infilename + "_11_CIP.png";			//�h���t���I
	imwrite(LCIP_outfile, lineCIP);

	/*�u��ٺ��I�s�q*/

	Mat gradmSSCBL, graddSSCBL, lineSSCBL;			//�u��ٺ��I�s�q(8UC1�B32FC1�B8UC1(BW))
	BWConnectBreakLine(gradmCBL, graddCBL, lineCIP, gradmSSCBL, graddSSCBL, lineSSCBL, 2, 10, 90, 0, 0);

	Mat gradmSSCBL_out, graddSSCBL_out, gradfSSCBL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmSSCBL, gradmSSCBL_out);
	DrawColorSystem(graddSSCBL, graddSSCBL_out);
	DrawColorSystem(gradmSSCBL, graddSSCBL, gradfSSCBL_out);

	string LSSCBL_outfile = filepath + "\\" + infilename + "_12.1.0_SSCBL.png";			//�u��ٺ��I�s�q(�G��)
	imwrite(LSSCBL_outfile, lineSSCBL);
	string gradmSSCBL_outfile = filepath + "\\" + infilename + "_12.1.1_SSCBLM.png";			//�u��ٺ��I�s�q(�T��)
	imwrite(gradmSSCBL_outfile, gradmSSCBL_out);
	string graddSSCBL_outfile = filepath + "\\" + infilename + "_12.1.2_SSCBLD.png";			//�u��ٺ��I�s�q(��V)
	imwrite(graddSSCBL_outfile, graddSSCBL_out);
	string gradfSSCBL_outfile = filepath + "\\" + infilename + "_12.1.3_SSCBLF.png";			//�u��ٺ��I�s�q(��)
	imwrite(gradfSSCBL_outfile, gradfSSCBL_out);

	/*�u�_�u�j��s�q*/

	Mat gradmSACBL, graddSACBL, lineSACBL;			//�u�_�u�j��s�q(8UC1�B32FC1�B8UC1(BW))
	BWConnectBreakLine(gradmSSCBL, graddSSCBL, lineSSCBL, gradmSACBL, graddSACBL, lineSACBL, 2, 5, 180, 1, 1);

	Mat gradmSACBL_out, graddSACBL_out, gradfSACBL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmSACBL, gradmSACBL_out);
	DrawColorSystem(graddSACBL, graddSACBL_out);
	DrawColorSystem(gradmSACBL, graddSACBL, gradfSACBL_out);

	string LSACBL_outfile = filepath + "\\" + infilename + "_12.2.0_SACBL.png";					//�u�_�u�j��s�q(�G��)
	imwrite(LSACBL_outfile, lineSACBL);
	string gradmSACBL_outfile = filepath + "\\" + infilename + "_12.2.1_SACBLM.png";			//�u�_�u�j��s�q(�T��)
	imwrite(gradmSACBL_outfile, gradmSACBL_out);
	string graddSACBL_outfile = filepath + "\\" + infilename + "_12.2.2_SACBLD.png";			//�u�_�u�j��s�q(��V)
	imwrite(graddSACBL_outfile, graddSACBL_out);
	string gradfSACBL_outfile = filepath + "\\" + infilename + "_12.2.3_SACBLF.png";			//�u�_�u�j��s�q(��)
	imwrite(gradfSACBL_outfile, gradfSACBL_out);

	/*�h�����T���I*/

	Mat lineCNP;	//�h���t���I(8UC1(BW))
	ClearSpecialPoint(lineSACBL, lineCNP, blurLineSize, 3, 1);
	string LCNP_outfile = filepath + "\\" + infilename + "_13_CIP.png";			//�h���t���I
	imwrite(LCNP_outfile, lineCNP);

	/*����ٺ��I�s�q*/

	Mat gradmLSCBL, graddLSCBL, lineLSCBL;			//����ٺ��I�s�q(8UC1�B32FC1�B8UC1(BW))
	BWConnectBreakLine(gradmSACBL, graddSACBL, lineCNP, gradmLSCBL, graddLSCBL, lineLSCBL, 2, 100, 90, 0, 0);

	Mat gradmLSCBL_out, graddLSCBL_out, gradfLSCBL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmLSCBL, gradmLSCBL_out);
	DrawColorSystem(graddLSCBL, graddLSCBL_out);
	DrawColorSystem(gradmLSCBL, graddLSCBL, gradfLSCBL_out);

	string LLSCBL_outfile = filepath + "\\" + infilename + "_14.1.0_LSCBL.png";					//����ٺ��I�s�q(�G��)
	imwrite(LLSCBL_outfile, lineLSCBL);
	string gradmLSCBL_outfile = filepath + "\\" + infilename + "_14.1.1_LSCBLM.png";			//����ٺ��I�s�q(�T��)
	imwrite(gradmLSCBL_outfile, gradmLSCBL_out);
	string graddLSCBL_outfile = filepath + "\\" + infilename + "_14.1.2_LSCBLD.png";			//����ٺ��I�s�q(��V)
	imwrite(graddLSCBL_outfile, graddLSCBL_out);
	string gradfLSCBL_outfile = filepath + "\\" + infilename + "_14.1.3_LSCBLF.png";			//����ٺ��I�s�q(��)
	imwrite(gradfLSCBL_outfile, gradfLSCBL_out);

	/*���_�u�j��s�q*/
	
	Mat gradmLACBL, graddLACBL, lineLACBL;			//���_�u�j��s�q(8UC1�B32FC1�B8UC1(BW))
	BWConnectBreakLine(gradmLSCBL, graddLSCBL, lineLSCBL, gradmLACBL, graddLACBL, lineLACBL, 2, 20, 90, 1, 0);

	Mat gradmLACBL_out, graddLACBL_out, gradfLACBL_out;		//��X��(8UC1�B8UC3�B8UC3)
	DrawAbsGraySystem(gradmLACBL, gradmLACBL_out);
	DrawColorSystem(graddLACBL, graddLACBL_out);
	DrawColorSystem(gradmLACBL, graddLACBL, gradfLACBL_out);

	string LLACBL_outfile = filepath + "\\" + infilename + "_14.2.0_LACBL.png";					//���_�u�j��s�q(�G��)
	imwrite(LLACBL_outfile, lineLACBL);
	string gradmLACBL_outfile = filepath + "\\" + infilename + "_14.2.1_LACBLM.png";			//���_�u�j��s�q(�T��)
	imwrite(gradmLACBL_outfile, gradmLACBL_out);
	string graddLACBL_outfile = filepath + "\\" + infilename + "_14.2.2_LACBLD.png";			//���_�u�j��s�q(��V)
	imwrite(graddLACBL_outfile, graddLACBL_out);
	string gradfLACBL_outfile = filepath + "\\" + infilename + "_14.2.3_LACBLF.png";			//���_�u�j��s�q(��)
	imwrite(gradfLACBL_outfile, gradfLACBL_out);

	/*�h����X���u�Ωt�߽u*/

	/*�������t��k����*/

    return 0;
}


