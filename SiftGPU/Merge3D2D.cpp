#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <assert.h>
#include <limits.h>
#include <omp.h>
#include <sys/time.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <istream>
#include <string>
#include <cmath>
#include <map>

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>
#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include "matrix.h"
#include "vector.h"

#include "verify_two_view_matches.h"
#include "feature_correspondence.h"
#include "camera_intrinsics_prior.h"
#include "estimate_twoview_info.h"
#include "twoview_info.h"

#include "SiftGPU.h"

using namespace std;
using namespace Eigen;

#define FREE_MYLIB dlclose
#define GET_MYPROC dlsym
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#define VOCAB_TREE_TH 	0.05
#define SAMPSON_ERR_5P 	2.25
#define INLIER_TH		30

#define CX	1632.0
#define CY	1224.0

typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> TransformType;
typedef std::vector<Eigen::Vector3d>                PointsType;

typedef struct _Point2D
{
	double x;
	double y;
	int camId;
	_Point2D(){}
	_Point2D(double x1, double y1, int camId1): x(x1), y(y1), camId(camId1)
	{}

}Point2D;

typedef struct _frame
{
	int 		frameIndex;
	string 		imgFileName;
	string 		siftFileName;
	bool		isValid;

	vector<long long int> 				pointIdList;
	vector<bool>					pointValidity;
	vector<float > *				siftDesc;
	vector<SiftGPU::SiftKeypoint> *			siftKey;

	_frame()
	{
		siftDesc = NULL;
		siftKey = NULL;
	}

	~_frame()
	{
			delete siftDesc;
			delete siftKey;
	}
}Frame;

typedef struct _match
{
	int firstFrameIndex;
	int secondFrameIndex;
	double rotMat[9];
	double tMat[3];
	double cMat[3];
	double eMat[9];
	double conf;
}Match;

typedef struct _PointAttr
{
	long long int pointId;
	double X;
	double Y;
	double Z;

	int R;
	int G;
	int B;

	_PointAttr()
	{
		pointId = -1;
		X = Y = Z = 0;
		R = G = B = 255;
	}

}PointAttr;


typedef struct _camera
{
	int camName;
	double rotMat[9];
	double tMat[3];
	double cMat[3];
	double focal;
	double k1;
	double k2;
}Camera;


typedef struct _camcorrespond
{
	int camName;
	double rotMat1[9];
	double cMat1[3];
	double rotMat2[9];
	double cMat2[3];
}CamCorrespond;

typedef struct _merge
{
	string firstCluster;
	string secondCluster;
	string newCluster; 
}MergeInfo;

void getCfromRT(double *Rot, double *Trans, double *Pos)
{
	double *tempRT = (double *)malloc(9 * sizeof(double));
	matrix_transpose(3, 3, Rot, tempRT);
	matrix_product(3, 3, 3, 1, tempRT, Trans, Pos);
	matrix_scale(3, 1, Pos, -1.0, Pos);
	free(tempRT);
}

void getTfromRC(double *Rot, double *Pos, double *Trans)
{
	matrix_product(3, 3, 3, 1, Rot, Pos, Trans);
	matrix_scale(3, 1, Trans, -1.0, Trans);
}



void printProgress (int currFrame, int totalFrame)
{
	double ratio = currFrame/ (double)totalFrame;
    int val = (int) (ratio * 100);
    int lpad = (int) (ratio * PBWIDTH);
    int rpad = PBWIDTH - lpad;
    printf ("\r%3d%% [%.*s%*s %d/%d]", val, lpad, PBSTR, rpad, "", currFrame, totalFrame);
    fflush (stdout);
}

static void split(vector<string> &toks, const string &s, const string &delims)
{
	toks.clear();

	string::const_iterator segment_begin = s.begin();
	string::const_iterator current = s.begin();
	string::const_iterator string_end = s.end();

	while (true)
	{
		if (current == string_end || delims.find(*current) != string::npos || *current == '\r')
		{
			if (segment_begin != current)
				toks.push_back(string(segment_begin, current));

			if (current == string_end || *current == '\r')
				break;

			segment_begin = current + 1;
		}

		current++;
	}

}

static void ReadNVM(const string &nvmFile, vector< Camera *> &cameraList, vector< PointAttr *> &ptAttrList, vector< vector<Point2D> *> &trackList,
			int color = 0)
{
	ifstream myfile;
	string line;
	vector <string> nvmToks;

	//Read nvm file
	myfile.open(nvmFile.c_str());

	//Read Header
	getline(myfile, line);

	//Read number of camera
	getline(myfile, line);
	int nCams1 = stoi(line);

	//Read cameras
	for ( int index = 0; index < nCams1; index ++ )
	{
		getline(myfile, line);
		split(nvmToks, line, " ");

		Camera *newCam = new Camera;

		newCam->camName = stoi(nvmToks[0]) + 1;
		newCam->focal = stod(nvmToks[1]);

		for ( int i = 0; i < 9; i ++ ) 
		{
			newCam->rotMat[i] = stod(nvmToks[i + 2]);
		}
		
		for ( int i = 0; i < 3; i ++ ) 
		{
			newCam->tMat[i] = stod(nvmToks[i + 11]);
		}

		getCfromRT(newCam->rotMat, newCam->tMat, newCam->cMat);

					
		newCam->k1 = stod(nvmToks[14]);
		newCam->k2 = stod(nvmToks[15]);

		cameraList.push_back(newCam);
	}

	//Read number of points
	getline(myfile, line);
	int nPoints1 = stoi(line);

	//Read points
	for ( int index = 0; index < nPoints1; index ++ )
	{
		getline(myfile, line);
		split(nvmToks, line, " ");

		PointAttr * newPtAttr = new PointAttr;
		vector<Point2D> * newList = new vector<Point2D>;

		newPtAttr->X = stod(nvmToks[0]);
		newPtAttr->Y = stod(nvmToks[1]);
		newPtAttr->Z = stod(nvmToks[2]);

		switch ( color )
		{
		case 0:
			newPtAttr->R = stoi(nvmToks[3]);
			newPtAttr->G = stoi(nvmToks[4]);
			newPtAttr->B = stoi(nvmToks[5]);
			break;
		case 1:
			newPtAttr->R = 255;
			newPtAttr->G = 0;
			newPtAttr->B = 0;
			break;
		case 2:
			newPtAttr->R = 0;
			newPtAttr->G = 255;
			newPtAttr->B = 0;
			break;
		case 3:
			newPtAttr->R = 0;
			newPtAttr->G = 255;
			newPtAttr->B = 0;
			break;
		default:
			newPtAttr->R = stoi(nvmToks[3]);
			newPtAttr->G = stoi(nvmToks[4]);
			newPtAttr->B = stoi(nvmToks[5]);
		}

		int nView = stoi(nvmToks[6]);
		int prevCamId = -1;

		for ( int i = 0; i < nView; i ++ ) 
		{	Point2D newPt;

			newPt.camId = stol(nvmToks[7 + i * 4 + 0]);
			newPtAttr->pointId = stol(nvmToks[7 + i * 4 + 1]);
			newPt.x = stod(nvmToks[7 + i * 4 + 2]);
			newPt.y = stod(nvmToks[7 + i * 4 + 3]);

			if ( prevCamId == newPt.camId ) continue;

			prevCamId = newPt.camId;
			newList->push_back(newPt);
		}

		trackList.push_back(newList);
		ptAttrList.push_back(newPtAttr);
	}
	myfile.close();
}

static void SaveNVM(const string &nvmFile, vector< Camera *> &cameraList, vector< PointAttr *> &ptAttrList, vector< vector<Point2D> *> &trackList)
{
	std::cout << "Saving model to " << nvmFile << "...\n"; 
	ofstream out(nvmFile);

	out << "NVM_V3_R9T\n" << cameraList.size() << '\n' << std::setprecision(12);

	for(size_t i = 0; i < cameraList.size(); ++i)
	{
		out << cameraList[i]->camName << ' ' << cameraList[i]->focal << ' ';

		for(int j  = 0; j < 9; ++j) out << cameraList[i]->rotMat[j] << ' ';

		out << cameraList[i]->tMat[0] << ' ' << cameraList[i]->tMat[1] << ' ' << cameraList[i]->tMat[2] << ' '
		    << cameraList[i]->k1 << ' ' << cameraList[i]->k2 << "\n"; 
	}

	out << ptAttrList.size() << '\n';

	for(size_t i = 0; i < ptAttrList.size(); ++i)
	{
		out << ptAttrList[i]->X << ' ' << ptAttrList[i]->Y << ' ' << ptAttrList[i]->Z << ' ' 
		    << ptAttrList[i]->R << ' ' << ptAttrList[i]->G << ' ' << ptAttrList[i]->B << ' '; 

		vector<Point2D> *tmp = trackList[i];
		out << tmp->size() << ' ';

		for(size_t j = 0; j < tmp->size(); ++j)    out << (*tmp)[j].camId << ' ' << ptAttrList[i]->pointId << ' ' <<  (*tmp)[j].x << ' ' << (*tmp)[j].y << ' ';

		out << '\n';
	}
}

static TransformType ComputeRT(const PointsType& src, const PointsType& dst)
{
	assert(src.size() == dst.size());
	int pairSize = src.size();
	Eigen::Vector3d center_src(0, 0, 0), center_dst(0, 0, 0);

	vector<double> Xsrc, Ysrc, Zsrc; 
	vector<double> Xdst, Ydst, Zdst; 
	for (int i=0; i<pairSize; ++i)
	{
		Xsrc.push_back(src[i][0]);
		Ysrc.push_back(src[i][1]);
		Zsrc.push_back(src[i][2]);

		Xdst.push_back(dst[i][0]);
		Ydst.push_back(dst[i][1]);
		Zdst.push_back(dst[i][2]);
	}

	int halfPairSize = pairSize / 2;

	sort (Xsrc.begin(), Xsrc.end());
	sort (Ysrc.begin(), Ysrc.end());	
	sort (Zsrc.begin(), Zsrc.end());
	center_src[0] = Xsrc[halfPairSize];
	center_src[1] = Ysrc[halfPairSize];
	center_src[2] = Zsrc[halfPairSize];

	sort (Xdst.begin(), Xdst.end());
	sort (Ydst.begin(), Ydst.end());	
	sort (Zdst.begin(), Zdst.end());
	center_dst[0] = Xdst[halfPairSize];
	center_dst[1] = Ydst[halfPairSize];
	center_dst[2] = Zdst[halfPairSize];

	Eigen::MatrixXd S(pairSize, 3), D(pairSize, 3);
	for (int i=0; i<pairSize; ++i)
	{
		for (int j=0; j<3; ++j)
			S(i, j) = src[i][j] - center_src[j];
		for (int j=0; j<3; ++j)
			D(i, j) = dst[i][j] - center_dst[j];
	}
	Eigen::MatrixXd Dt = D.transpose();
	Eigen::Matrix3d H = Dt * S;
	Eigen::Matrix3d W, U, V;

	JacobiSVD<Eigen::MatrixXd> svd;
	Eigen::MatrixXd H_(3, 3);

	for (int i=0; i<3; ++i)  
	{
		for(int j=0; j<3; ++j) H_(i, j) = H(i, j);
		{
			svd.compute(H_, Eigen::ComputeThinU | Eigen::ComputeThinV );
		}
	}

	if (!svd.computeU() || !svd.computeV()) {
		std::cerr << "decomposition error" << endl;
		return std::make_pair(Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
	}
	Eigen::Matrix3d Vt = svd.matrixV().transpose();
	Eigen::Matrix3d R = svd.matrixU()*Vt;

	Eigen::Vector3d t = center_dst - R*center_src;	
	
	return std::make_pair(R, t);
}

static  void ComputeRigidTransform(PointsType& src, PointsType& dst, double &scale, Eigen::Matrix3d &rot, Eigen::Vector3d &tran)
{
	//Determine the scale
	vector<double> scaleList;

	for ( int index = 0; index < src.size(); index ++ )
	{
		for ( int index1 = index + 1; index1 < src.size(); index1 ++ )
		{
			Eigen::Vector3d diffSrc = (src[index] - src[index1]);
			Eigen::Vector3d diffDst = (dst[index] - dst[index1]);

			double tmpScale = 0.0;
			if ( diffSrc.norm() != 0.0 )
			{
				tmpScale = diffDst.norm()/diffSrc.norm();
				scaleList.push_back(tmpScale);
			}
		}
	}

	//Calculate median of scales
	sort (scaleList.begin(), scaleList.end());
	int middleIndex = scaleList.size() / 2;
	scale = scaleList[middleIndex];

	//Correct scale for second
	for ( int index = 0; index < dst.size(); index ++ )
	{
		src[index] = scale * src[index];
	}	

	//Get best RT based on ransac
	TransformType bestRT;
	double minDiff;

	//Compute R & T
	TransformType rtTmp = ComputeRT(src, dst);

	rot = rtTmp.first;	
	tran = rtTmp.second;
	return;
}


int main(int argc, char *argv[])
{
	void * hsiftgpu = dlopen("./bin/libsiftgpu.so", RTLD_LAZY);

	if(hsiftgpu == NULL)
	{
		printf("Lib not found\n");
		return 0;
	}

	SiftGPU* (*pCreateNewSiftGPU)(int) = NULL;
	SiftMatchGPU* (*pCreateNewSiftMatchGPU)(int) = NULL;
	pCreateNewSiftGPU = (SiftGPU* (*) (int)) GET_MYPROC(hsiftgpu, "CreateNewSiftGPU");
	pCreateNewSiftMatchGPU = (SiftMatchGPU* (*)(int)) GET_MYPROC(hsiftgpu, "CreateNewSiftMatchGPU");
	SiftGPU* sift = pCreateNewSiftGPU(1);
	SiftMatchGPU* matcher = pCreateNewSiftMatchGPU(4096);
	char * argv_tmp[] = {"-fo", "-1",  "-v", "0","-cuda", "[0]"};//
	int argc_tmp = sizeof(argv_tmp)/sizeof(char*);
	sift->ParseParam(argc_tmp, argv_tmp);

	if ( argc < 2 )
	{
		printf("Invalid arguments\n ./a.out <Output DIR>\n");
		return 0;
	}

	string outDir = argv[1];
	string consolidedFile = outDir + "/consolided_result.txt";
	
	vector<Frame *> globalFrameList;
	int frameCount = 0;

	//Read consolided_result.txt
	ifstream myfile1, myfile;
	string line;
	vector <string> nvmToks;

	string keyListFile = outDir + "/KeyList.txt";

	//Read CX,CY
	vector<pair<float, float> > centreList;
	vector<string> siftFileList;
	myfile1.open(keyListFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");
		siftFileList.push_back(nvmToks[0]);
		float cx = stof(nvmToks[2]) / 2.0;
		float cy = stof(nvmToks[1]) / 2.0;
		centreList.push_back(make_pair(cx, cy));
	}
	myfile1.close();

	myfile1.open(consolidedFile.c_str());

	//Read no of images
	getline(myfile1, line);
	split(nvmToks, line, " ");
	int totalFrames = stoi(nvmToks[0]);

	printf("Loading SIFT features...\n");

	if (! getline(myfile1, line) )
	{
		printf("Unable to read line\n");
		return -1;
	}

	split(nvmToks, line, " ");

	int imgIndex = stoi(nvmToks[0]);
	const char *imgName = nvmToks[1].c_str();
	int nSift = stoi(nvmToks[2]);

	for ( int index = 0; index < totalFrames; index ++ )
	{

		Frame *newFrame = new Frame;
		newFrame->frameIndex = imgIndex;

		vector<float > *singleDesc = new vector<float>;
		vector<SiftGPU::SiftKeypoint> *singleKey = new vector<SiftGPU::SiftKeypoint>;

		while ( getline(myfile1, line) )
		{
			split(nvmToks, line, " ");

			if ( nvmToks.size() == 3 ) 
			{
				imgIndex = stoi(nvmToks[0]);
				imgName = nvmToks[1].c_str();
				nSift = stoi(nvmToks[2]);
				break;
			}

			long long int pointId = stol(nvmToks[0]);
			int nSift = stoi(nvmToks[0]);

			SiftGPU::SiftKeypoint skp;
			skp.y = stod(nvmToks[1]);
			skp.x = stod(nvmToks[2]);
			skp.s = stod(nvmToks[3]);
			skp.o = stod(nvmToks[4]);
			int siftLen = stoi(nvmToks[5]);
			singleKey->push_back(skp);

			for ( int dIndex = 0; dIndex < siftLen; dIndex ++ )
			{
				int d = stoi(nvmToks[6 + dIndex]);
				singleDesc->push_back(d/512.0f);
			}
			
			newFrame->pointIdList.push_back(pointId);
			newFrame->pointValidity.push_back(true);
		}

		newFrame->siftDesc = singleDesc;
		newFrame->siftKey = singleKey;
		globalFrameList.push_back(newFrame);

		frameCount ++;
		printProgress(frameCount, totalFrames);
	}
	myfile1.close();

	printf("\n");

	string clusterListFile = outDir + "/dendogram/fullclustermergeinfo.txt";

	vector<MergeInfo> clusterMergeList;
	myfile1.open(clusterListFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");

		MergeInfo newInfo;
		newInfo.firstCluster = nvmToks[1];
		newInfo.secondCluster = nvmToks[2];
		newInfo.newCluster = nvmToks[0];

		clusterMergeList.push_back(newInfo);
	}
	myfile1.close();

	for ( int cIndex = 0; cIndex < clusterMergeList.size(); cIndex ++)
	{
		//Read first nvm file
		string folderName1 = outDir + "/dendogram/cluster" + clusterMergeList[cIndex].firstCluster + "/";
		string nvmFile1 = folderName1 + "/outputVSFM_GB.nvm";
		cout << nvmFile1 << endl;
		myfile.open(nvmFile1.c_str());

		//Camera and point list
		vector< Camera *> 		cameraList1;
		vector< PointAttr *> 		ptAttrList1;
		vector< vector<Point2D> *> 	trackList1;

		ReadNVM(nvmFile1, cameraList1, ptAttrList1, trackList1, 1);

		//Read second nvm file
		string folderName2 = outDir + "/dendogram/cluster" + clusterMergeList[cIndex].secondCluster + "/";
		string nvmFile2 = folderName2 + "/outputVSFM_GB.nvm";
		cout << nvmFile2 << endl;
		myfile.open(nvmFile2.c_str());

		string dataFile = outDir + "/dendogram/cluster" + clusterMergeList[cIndex].newCluster + "/Data3D2D.txt";
		FILE *fpData = fopen(dataFile.c_str(), "w");

		//Camera and point list
		vector< Camera *> 		cameraList2;
		vector< PointAttr *> 		ptAttrList2;
		vector< vector<Point2D> *> 	trackList2;

		ReadNVM(nvmFile2, cameraList2, ptAttrList2, trackList2, 3);

		//Camera corresomdences
		vector< CamCorrespond *> camCorr;

		//Align the camera
		for ( int index1 = 0; index1 < cameraList1.size(); index1 ++ )
		{
			int camIndex = cameraList1[index1]->camName;
			Frame *imgFrame = NULL;

			cv::Mat distCoeffs(4, 1, cv::DataType<double>::type);
			distCoeffs.at<double>(0) = cameraList1[index1]->k1;
			distCoeffs.at<double>(1) = cameraList1[index1]->k2;
			distCoeffs.at<double>(2) = 0;
			distCoeffs.at<double>(3) = 0;

			cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
			cv::setIdentity(cameraMatrix);
			cameraMatrix.at<double>(cv::Point(0, 0)) = cameraList1[index1]->focal;
			cameraMatrix.at<double>(cv::Point(1, 1)) = cameraList1[index1]->focal;

			for ( int index2 = 0; index2 < globalFrameList.size(); index2 ++ )
			{
				if ( globalFrameList[index2]->frameIndex == camIndex )
				{
					imgFrame =  globalFrameList[index2];
					break;
				}
			}

			if ( imgFrame != NULL )
			{
				vector<cv::Point3f> objectPoints;
				vector<cv::Point2f> imagePoints;

				vector<SiftGPU::SiftKeypoint> *siftKey = imgFrame->siftKey;

				for ( int index2 = 0; index2 < imgFrame->pointIdList.size(); index2 ++ )
				{
					for ( int index3 = 0; index3 < ptAttrList2.size(); index3 ++ )
					{
						if ( ptAttrList2[index3]->pointId == imgFrame->pointIdList[index2] )
						{
							cv::Point3f tmp3Dpoints;
							cv::Point2f tmp2Dpoints;

							tmp3Dpoints.x = ptAttrList2[index3]->X;
							tmp3Dpoints.y = ptAttrList2[index3]->Y;
							tmp3Dpoints.z = ptAttrList2[index3]->Z;

							tmp2Dpoints.x = (*siftKey)[index2].x - centreList[camIndex - 1].first;
							tmp2Dpoints.y = (*siftKey)[index2].y - centreList[camIndex - 1].second;

							objectPoints.push_back(tmp3Dpoints);
							imagePoints.push_back(tmp2Dpoints);
							break;
						}
					}
				}
				
				bool useExtrinsicGuess = true;
				int iterationsCount = 500;
				float reprojectionError = 1.0;
				std::vector<double> inliers;

				cv::Mat rvec = cv::Mat(3, 3, CV_64F);
				cv::Mat tvec = cv::Mat(3, 1, CV_64F);

				if ( objectPoints.size() > 50 )
				{
					printf("%d \n", objectPoints.size());
				
					solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, 0.99, inliers, CV_ITERATIVE);
				
					Rodrigues(rvec, rvec);
					cv::Mat cvec = -1.0 * (rvec.t() * tvec);

					printf("CAM: %d\n", camIndex);
					/*printf("%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", 
						rvec.at<double>(0,0), rvec.at<double>(0,1), rvec.at<double>(0,2), 
						rvec.at<double>(1,0), rvec.at<double>(1,1), rvec.at<double>(1,2), 
						rvec.at<double>(2,0), rvec.at<double>(2,1), rvec.at<double>(2,2) );
					printf("%.9lf %.9lf %.9lf\n", cvec.at<double>(0), cvec.at<double>(1), cvec.at<double>(2));*/

					CamCorrespond *corr = new CamCorrespond;
					corr->camName = camIndex;
					memcpy(corr->rotMat1, cameraList1[index1]->rotMat, 9 * sizeof(double));
					memcpy(corr->cMat1, cameraList1[index1]->cMat, 3 * sizeof(double));

					corr->rotMat2[0] = rvec.at<double>(0,0);
					corr->rotMat2[1] = rvec.at<double>(0,1);
					corr->rotMat2[2] = rvec.at<double>(0,2);
					corr->rotMat2[3] = rvec.at<double>(1,0);
					corr->rotMat2[4] = rvec.at<double>(1,1);
					corr->rotMat2[5] = rvec.at<double>(1,2);
					corr->rotMat2[6] = rvec.at<double>(2,0);
					corr->rotMat2[7] = rvec.at<double>(2,1);
					corr->rotMat2[8] = rvec.at<double>(2,2);

					corr->cMat2[0] = cvec.at<double>(0);
					corr->cMat2[1] = cvec.at<double>(1);
					corr->cMat2[2] = cvec.at<double>(2);

					camCorr.push_back(corr);
				}
			}
		}

		//Align the camera
		fprintf(fpData, "         \n");
		int validCount = 0;

		for ( int index1 = 0; index1 < cameraList2.size(); index1 ++ )
		{
			int camIndex = cameraList2[index1]->camName;
			Frame *imgFrame = NULL;

			cv::Mat distCoeffs(4, 1, cv::DataType<double>::type);
			distCoeffs.at<double>(0) = cameraList2[index1]->k1;
			distCoeffs.at<double>(1) = cameraList2[index1]->k2;
			distCoeffs.at<double>(2) = 0;
			distCoeffs.at<double>(3) = 0;

			cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
			cv::setIdentity(cameraMatrix);
			cameraMatrix.at<double>(cv::Point(0, 0)) = cameraList2[index1]->focal;
			cameraMatrix.at<double>(cv::Point(1, 1)) = cameraList2[index1]->focal;

			for ( int index2 = 0; index2 < globalFrameList.size(); index2 ++ )
			{
				if ( globalFrameList[index2]->frameIndex == camIndex )
				{
					imgFrame =  globalFrameList[index2];
					break;
				}
			}

			if ( imgFrame != NULL )
			{
				vector<cv::Point3f> objectPoints;
				vector<cv::Point2f> imagePoints;

				vector<SiftGPU::SiftKeypoint> *siftKey = imgFrame->siftKey;

				for ( int index2 = 0; index2 < imgFrame->pointIdList.size(); index2 ++ )
				{
					for ( int index3 = 0; index3 < ptAttrList1.size(); index3 ++ )
					{
						if ( ptAttrList1[index3]->pointId == imgFrame->pointIdList[index2] )
						{
							cv::Point3f tmp3Dpoints;
							cv::Point2f tmp2Dpoints;

							tmp3Dpoints.x = ptAttrList1[index3]->X;
							tmp3Dpoints.y = ptAttrList1[index3]->Y;
							tmp3Dpoints.z = ptAttrList1[index3]->Z;

							tmp2Dpoints.x = (*siftKey)[index2].x - centreList[camIndex - 1].first;
							tmp2Dpoints.y = (*siftKey)[index2].y - centreList[camIndex - 1].second;
							

							objectPoints.push_back(tmp3Dpoints);
							imagePoints.push_back(tmp2Dpoints);
							break;
						}
					}
				}
				
				bool useExtrinsicGuess = true;
				int iterationsCount = 500;
				float reprojectionError = 1.0;
				std::vector<double> inliers;

				cv::Mat rvec = cv::Mat(3, 3, CV_64F);
				cv::Mat tvec = cv::Mat(3, 1, CV_64F);

				if ( objectPoints.size() > 50 )
				{
					printf("%d \n", objectPoints.size());
				
					solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, 0.99, inliers, CV_ITERATIVE);
				
					validCount ++;

					Rodrigues(rvec, rvec);
					cv::Mat cvec = -1.0 * (rvec.t() * tvec);

					printf("CAM: %d\n", camIndex);
					
					fprintf(fpData, "%d %lf\n", camIndex, cameraList2[index1]->focal);

					fprintf(fpData, "%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", 
						cameraList2[index1]->rotMat[0], cameraList2[index1]->rotMat[1], cameraList2[index1]->rotMat[2], 
						cameraList2[index1]->rotMat[3], cameraList2[index1]->rotMat[4], cameraList2[index1]->rotMat[5], 
						cameraList2[index1]->rotMat[6], cameraList2[index1]->rotMat[7], cameraList2[index1]->rotMat[8]);
					fprintf(fpData, "%.9lf %.9lf %.9lf\n", cameraList2[index1]->cMat[0], cameraList2[index1]->cMat[1], cameraList2[index1]->cMat[2]);

					fprintf(fpData, "%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", 
						rvec.at<double>(0,0), rvec.at<double>(0,1), rvec.at<double>(0,2), 
						rvec.at<double>(1,0), rvec.at<double>(1,1), rvec.at<double>(1,2), 
						rvec.at<double>(2,0), rvec.at<double>(2,1), rvec.at<double>(2,2) );
					fprintf(fpData, "%.9lf %.9lf %.9lf\n", cvec.at<double>(0), cvec.at<double>(1), cvec.at<double>(2));

					fprintf(fpData, "%d\n", objectPoints.size());

					for ( int tmpIndex = 0; tmpIndex < objectPoints.size(); tmpIndex ++ )
					{
						fprintf(fpData, "%lf %lf %lf %lf %lf\n",  
							objectPoints[tmpIndex].x, objectPoints[tmpIndex].y, objectPoints[tmpIndex].z, 
											imagePoints[tmpIndex].x, imagePoints[tmpIndex].y);
					}

					CamCorrespond *corr = new CamCorrespond;
					corr->camName = camIndex;
					memcpy(corr->rotMat2, cameraList2[index1]->rotMat, 9 * sizeof(double));
					memcpy(corr->cMat2, cameraList2[index1]->cMat, 3 * sizeof(double));

					corr->rotMat1[0] = rvec.at<double>(0,0);
					corr->rotMat1[1] = rvec.at<double>(0,1);
					corr->rotMat1[2] = rvec.at<double>(0,2);
					corr->rotMat1[3] = rvec.at<double>(1,0);
					corr->rotMat1[4] = rvec.at<double>(1,1);
					corr->rotMat1[5] = rvec.at<double>(1,2);
					corr->rotMat1[6] = rvec.at<double>(2,0);
					corr->rotMat1[7] = rvec.at<double>(2,1);
					corr->rotMat1[8] = rvec.at<double>(2,2);

					corr->cMat1[0] = cvec.at<double>(0);
					corr->cMat1[1] = cvec.at<double>(1);
					corr->cMat1[2] = cvec.at<double>(2);

					camCorr.push_back(corr);
				}
			}
		}

		//Find 3D-3D correspondences of cameras
		PointsType firstList, secondList;
		Eigen::Vector3d pt1, pt2;

		string rFile = outDir + "/dendogram/cluster" + clusterMergeList[cIndex].newCluster + "/RTS3D2D_Rij.txt";
		FILE *fpRij = fopen(rFile.c_str(), "w");

		for ( int tI = 0; tI < camCorr.size(); tI ++ )
		{
			pt1[0] = camCorr[tI]->cMat1[0];
			pt1[1] = camCorr[tI]->cMat1[1];
			pt1[2] = camCorr[tI]->cMat1[2];

			pt2[0] = camCorr[tI]->cMat2[0];
			pt2[1] = camCorr[tI]->cMat2[1];
			pt2[2] = camCorr[tI]->cMat2[2];

			firstList.push_back(pt1);
			secondList.push_back(pt2);


			double tempRT[9], Rres[9];
			matrix_transpose(3, 3, camCorr[tI]->rotMat1, tempRT);
			matrix_product(3, 3, 3, 3, tempRT, camCorr[tI]->rotMat2, Rres);

			fprintf(fpRij, "%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", 
				Rres[0], Rres[1], Rres[2], Rres[3], Rres[4], Rres[5], Rres[6], Rres[7], Rres[8]);


		}
		fclose(fpRij);

		//Calculate scale and RT
		double scale = 1.0; 
		TransformType RT;
		
		ComputeRigidTransform(secondList, firstList, scale, RT.first, RT.second);

		string rtsFile = outDir + "/dendogram/cluster" + clusterMergeList[cIndex].newCluster + "/RTS3D2D_SVD.txt";
		FILE *fpRTS = fopen(rtsFile.c_str(), "w");

		fprintf(fpRTS, "%.9lf\n%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n%lf %lf %lf\n", scale,
			RT.first(0, 0), RT.first(0, 1), RT.first(0, 2), RT.first(1, 0), RT.first(1, 1), RT.first(1, 2), 
			RT.first(2, 0), RT.first(2, 1), RT.first(2, 2), RT.second[0], RT.second[1], RT.second[2]);

		fclose(fpRTS);

		fseek(fpData, 0, SEEK_SET);
		fprintf(fpData, "%d", validCount);

		fclose(fpData);
	}

	return 0;
}


