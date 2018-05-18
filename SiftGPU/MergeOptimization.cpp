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
#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/video.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudaarithm.hpp>

#include <Eigen/SVD>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/Geometry>

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

#define CLAMP(x,mn,mx) (((x) < mn) ? mn : (((x) > mx) ? mx : (x)))

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define RAD2DEG(r) ((r) * (180.0 / M_PI))

typedef std::pair<Eigen::Matrix3d, Eigen::Vector3d> TransformType;
typedef std::vector<Eigen::Vector3d>                PointsType;

static int global_num_points;
static double *global_Rs = NULL;
static double *global_ts = NULL;
static v2_t *global_ps = NULL;

typedef struct _Point2D
{
	double x;
	double y;
	int camId;
	_Point2D(){}
	_Point2D(double x1, double y1, int camId1): x(x1), y(y1), camId(camId1)
	{}

}Point2D;

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

typedef struct _merge
{
	string firstCluster;
	string secondCluster;
	string newCluster; 
}MergeInfo;

static void getCfromRT(double *Rot, double *Trans, double *Pos)
{
	double *tempRT = (double *)malloc(9 * sizeof(double));
	matrix_transpose(3, 3, Rot, tempRT);
	matrix_product(3, 3, 3, 1, tempRT, Trans, Pos);
	matrix_scale(3, 1, Pos, -1.0, Pos);
	free(tempRT);
}

static void getTfromRC(double *Rot, double *Pos, double *Trans)
{
	matrix_product(3, 3, 3, 1, Rot, Pos, Trans);
	matrix_scale(3, 1, Trans, -1.0, Trans);
}

static bool sortbysec(const pair<int,int> &a,
              const pair<int,int> &b)
{
    return (a.second < b.second);
}

static bool sortbysecPair(const pair<Eigen::Vector3d, double> &a,
              const pair<Eigen::Vector3d, double> &b)
{
    return (a.second < b.second);
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

static TransformType ComputeRT(const PointsType& src, const PointsType& dst, const Eigen::Matrix3d &rot)
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

	/*for (int i=0; i<pairSize; ++i)
	{
		center_src += src[i];
		center_dst += dst[i];
	}

	center_src /= (double)pairSize;
	center_dst /= (double)pairSize;*/

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

	R = rot;

	Eigen::Vector3d t = center_dst - R*center_src;	

	vector<pair<Eigen::Vector3d, double> > tList;

	for (int i=0; i<pairSize; ++i)
	{
		Eigen::Vector3d Tij = dst[i] - R*src[i];
		tList.push_back(make_pair(Tij, Tij.norm()));
	}
	
	sort (tList.begin(), tList.end(), sortbysecPair);

	t = tList[tList.size()/2].first;

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
	TransformType rtTmp = ComputeRT(src, dst, rot);
	
	//Align source points
	vector<double> diffList;
	for ( int index1 = 0; index1 < dst.size(); index1 ++ )
	{
		Eigen::Vector3d pt;

		pt[0] = src[index1][0];
		pt[1] = src[index1][1];
		pt[2] = src[index1][2];

		pt = scale * pt;
		pt = rtTmp.first * pt + rtTmp.second;
		
		Eigen::Vector3d diff = pt - dst[index1];	
		diffList.push_back( diff.norm() );		
	}	

	/*vector<double> tmpDiff = diffList;
	sort (tmpDiff.begin(), tmpDiff.end());
	middleIndex = tmpDiff.size() / 2;
	double medianDiff = tmpDiff[middleIndex];
	double diffTh = medianDiff * 1.2;
	
	PointsType tmpSrc, tmpDst;
	for ( int index1 = 0; index1 < dst.size(); index1 ++ )
	{
		if (diffList[index1] < diffTh )
		{			
			tmpSrc.push_back(src[index1]);
			tmpDst.push_back(dst[index1]);
		}
	}	

	//Compute R & T
	bestRT = ComputeRT(tmpSrc, tmpDst);*/

	int nPtPerIter = 10;
	int nIter = 1000;//(int)((float)src.size() / (float) nPtPerIter + 0.5);

	for ( int index = 0; index < nIter; index ++ )
	{
		srand(index + 10000);
		if ( index % 10 == 0 )
		{
			nPtPerIter = src.size() > nPtPerIter * 2 ? nPtPerIter * 2 :  src.size() ;
		}

		PointsType tmpSrc, tmpDst;
		vector<int> tmpVector;
		for ( int i = 0; i < nPtPerIter; i ++ )
		{
			int rIndex = rand() % src.size();
			bool fFlag = false;

			for ( int tIndex = 0; tIndex < tmpVector.size(); tIndex ++ )
			{
				if ( rIndex == tmpVector[tIndex] ) 
				{
					fFlag = true;
					break;
				}
			}
			
			if ( fFlag ) rIndex = rand() % src.size();

			tmpVector.push_back(rIndex);
			
			tmpSrc.push_back(src[rIndex]);
			tmpDst.push_back(dst[rIndex]);
		}

		//Compute R & T
		TransformType rtTmp = ComputeRT(tmpSrc, tmpDst, rot);

		//Align all points
		double totalDiff = 0, medianDiff = 0;
		vector<double> diffList;
		for ( int index1 = 0; index1 < dst.size(); index1 ++ )
		{
			Eigen::Vector3d pt;

			pt[0] = src[index1][0];
			pt[1] = src[index1][1];
			pt[2] = src[index1][2];

			pt = scale * pt;
			pt = rtTmp.first * pt + rtTmp.second;
			
			Eigen::Vector3d diff = pt - dst[index1];	
			totalDiff += diff.norm();
			diffList.push_back( diff.norm() );		
		}	
		
		//Calculate median of diff
		sort (diffList.begin(), diffList.end());
		int middleIndex = diffList.size() / 2;
		medianDiff = diffList[middleIndex];//totalDiff / diffList.size();

		if ( index == 0 )
		{
			minDiff = medianDiff;
			bestRT = rtTmp;
		}

		if ( minDiff > medianDiff )
		{
			minDiff = medianDiff;	
			bestRT = rtTmp;
		}
	}

	rot = bestRT.first;	
	tran = bestRT.second;
	return;
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

		newCam->camName = stoi(nvmToks[0]);
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

int main(int argc, char *argv[])
{

	if ( argc < 2 )
	{
		printf("Invalid arguments\n ./a.out <Output DIR>\n");
		return 0;
	}

	string outDir = argv[1];
	int StIndexG = stoi(argv[2]);

	printf("\n\nLoading cluster details ...\n");

	string clusterListFile = outDir + "/dendogram/fullclustermergeinfo.txt";

	ifstream myfile;
	string line;
	vector <string> nvmToks;

	vector<MergeInfo> clusterMergeList;
	myfile.open(clusterListFile.c_str());
	while(getline(myfile, line))
	{
		split(nvmToks, line, " ");

		MergeInfo newInfo;
		newInfo.firstCluster = nvmToks[1];
		newInfo.secondCluster = nvmToks[2];
		newInfo.newCluster = nvmToks[0];

		clusterMergeList.push_back(newInfo);
	}
	myfile.close();

	for ( int cIndex = StIndexG; cIndex < clusterMergeList.size(); cIndex ++)
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

		//Camera and point list
		vector< Camera *> 		cameraList2;
		vector< PointAttr *> 		ptAttrList2;
		vector< vector<Point2D> *> 	trackList2;

		ReadNVM(nvmFile2, cameraList2, ptAttrList2, trackList2, 3);

		//Find 3D-3D correspondences
		PointsType firstList, secondList;
		vector<pair<int, int>> matchIndexList;

		for ( int index1 = 0; index1 < ptAttrList1.size(); index1 ++ )
		{
			for ( int index2 = 0; index2 < ptAttrList2.size(); index2 ++ )
			{
				if ( ptAttrList1[index1]->pointId == ptAttrList2[index2]->pointId )
				{
					Eigen::Vector3d pt1, pt2;

					pt1[0] = ptAttrList1[index1]->X;
					pt1[1] = ptAttrList1[index1]->Y;
					pt1[2] = ptAttrList1[index1]->Z;

					pt2[0] = ptAttrList2[index2]->X;
					pt2[1] = ptAttrList2[index2]->Y;
					pt2[2] = ptAttrList2[index2]->Z;

					ptAttrList1[index1]->R = 0;
					ptAttrList1[index1]->G = 0;
					ptAttrList1[index1]->B = 255;

					ptAttrList2[index2]->R = 255;
					ptAttrList2[index2]->G = 255;
					ptAttrList2[index2]->B = 255;

					firstList.push_back(pt1);
					secondList.push_back(pt2);

					matchIndexList.push_back(make_pair(index1, index2));

					break;
				} 
			}	
		}

		printf("\n#Correspondences[%s, %s]: %d\n", clusterMergeList[cIndex].firstCluster.c_str(), clusterMergeList[cIndex].secondCluster.c_str(), firstList.size());

		//Calculate scale and RT
		double scale = 1.0; 
		TransformType RT;
		
		string RTSFile = outDir + "/dendogram/cluster" + clusterMergeList[cIndex].newCluster + "/RTS2D2D.txt";
		ifstream myfile1;
		myfile1.open(RTSFile.c_str());

		getline(myfile1, line);
		split(nvmToks, line, " ");
		scale = stod(nvmToks[0]);

		getline(myfile1, line);
		split(nvmToks, line, " ");
		RT.first <<  stod(nvmToks[0]), stod(nvmToks[1]), stod(nvmToks[2]), stod(nvmToks[3]), stod(nvmToks[4]), stod(nvmToks[5]), stod(nvmToks[6]), stod(nvmToks[7]), stod(nvmToks[8]);

		getline(myfile1, line);
		split(nvmToks, line, " ");
		RT.second <<  stod(nvmToks[0]), stod(nvmToks[1]), stod(nvmToks[2]);

		myfile1.close();

		//Align the camera
		for ( int index2 = 0; index2 < cameraList2.size(); index2 ++ )
		{
			Eigen::Matrix3d R;
			R << cameraList2[index2]->rotMat[0], cameraList2[index2]->rotMat[1], cameraList2[index2]->rotMat[2],
			     cameraList2[index2]->rotMat[3], cameraList2[index2]->rotMat[4], cameraList2[index2]->rotMat[5],
			     cameraList2[index2]->rotMat[6], cameraList2[index2]->rotMat[7], cameraList2[index2]->rotMat[8];
			
			R = R * RT.first.transpose();

			cameraList2[index2]->rotMat[0] = R(0, 0);
			cameraList2[index2]->rotMat[1] = R(0, 1);
			cameraList2[index2]->rotMat[2] = R(0, 2);
			cameraList2[index2]->rotMat[3] = R(1, 0);
			cameraList2[index2]->rotMat[4] = R(1, 1);
			cameraList2[index2]->rotMat[5] = R(1, 2);
			cameraList2[index2]->rotMat[6] = R(2, 0);
			cameraList2[index2]->rotMat[7] = R(2, 1);
			cameraList2[index2]->rotMat[8] = R(2, 2);
		}	

		//Merge nvm
		for ( int index1 = 0; index1 < ptAttrList2.size(); index1 ++ )
		{
			vector<Point2D> *tmpList = trackList2[index1];
			for ( int index2 = 0; index2 < tmpList->size(); index2 ++ )
			{
				(*tmpList)[index2].camId +=  cameraList1.size();
			}
		}

		cameraList1.insert(cameraList1.end(), cameraList2.begin(), cameraList2.end());

		sort(matchIndexList.begin(), matchIndexList.end(), sortbysec);

		for ( int index1 = matchIndexList.size() - 1; index1 >= 0; index1 -- )
		{
			trackList1[matchIndexList[index1].first]->insert(trackList1[matchIndexList[index1].first]->end(), 
				trackList2[matchIndexList[index1].second]->begin(), trackList2[matchIndexList[index1].second]->end());

			trackList2.erase(trackList2.begin() + matchIndexList[index1].second);
			ptAttrList2.erase(ptAttrList2.begin() + matchIndexList[index1].second);
		}

		trackList1.insert(trackList1.end(), trackList2.begin(), trackList2.end());
		ptAttrList1.insert(ptAttrList1.end(), ptAttrList2.begin(), ptAttrList2.end());

		//Dump nvm
		string newFolderName = outDir + "/dendogram/cluster" + clusterMergeList[cIndex].newCluster + "/";

		if ( 0 != mkdir(newFolderName.c_str(), 0777))
		{
			printf("Unable to create dir.\n");
		}

		string nvmFile = newFolderName + "/outputVSFM_GB.nvm"; 
		SaveNVM(nvmFile, cameraList1, ptAttrList1, trackList1);

		//Dump files for optimization
		string newRT_file = newFolderName + "/RTglobalmapped_for_optimisation.txt";
		string newF_file = newFolderName + "/focal_for_optimisation.txt";
		string newM_file = newFolderName + "/Invmap.txt";
		string newOurs_file = newFolderName + "/ours_new.txt";

		FILE *fpR = fopen(newRT_file.c_str(), "w");
		FILE *fpF = fopen(newF_file.c_str(), "w");
		FILE *fpM = fopen(newM_file.c_str(), "w");
		FILE *fpOurs = fopen(newOurs_file.c_str(), "w");

		fprintf(fpF, "%d\n", cameraList1.size());

		for ( int index = 0; index < cameraList1.size(); index ++ )
		{
			fprintf(fpR, "%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", cameraList1[index]->rotMat[0], 
					cameraList1[index]->rotMat[1], cameraList1[index]->rotMat[2], cameraList1[index]->rotMat[3], 
					cameraList1[index]->rotMat[4], cameraList1[index]->rotMat[5], cameraList1[index]->rotMat[6], 
					cameraList1[index]->rotMat[7], cameraList1[index]->rotMat[8]);
			fprintf(fpR, "%.9lf %.9lf %.9lf\n", cameraList1[index]->cMat[0], cameraList1[index]->cMat[1], cameraList1[index]->cMat[2]);

			fprintf(fpF, "%d %.9lf %d\n", cameraList1[index]->camName, cameraList1[index]->focal, 2);
			fprintf(fpM, "%d %d\n", index, cameraList1[index]->camName);
		}
		fclose(fpR);
		fclose(fpF);
		fclose(fpM);

		for ( int i = 0; i < trackList1.size(); i ++ )
		{
			vector<Point2D> *list = trackList1[i];

			//if ( ptAttrList1[i]->X != 0.0 || ptAttrList1[i]->Y != 0.0 ||  ptAttrList1[i]->Z != 0.0 )
			{
				fprintf(fpOurs, "%lf %lf %lf %d %d %d %d ", ptAttrList1[i]->X, ptAttrList1[i]->Y, ptAttrList1[i]->Z, 
						ptAttrList1[i]->R, ptAttrList1[i]->G, ptAttrList1[i]->B, trackList1[i]->size());
		
				for ( int j = 0; j < trackList1[i]->size(); j ++ )
				{
					fprintf(fpOurs, "%d %lld %lf %lf ", (*list)[j].camId, ptAttrList1[i]->pointId, (*list)[j].x, (*list)[j].y);
				}

				fprintf(fpOurs, "\n");
			}
		}

		fclose(fpOurs);
		
		//Dump ply
		string plyFile = newFolderName + "/merged.ply"; 
		FILE *fpPly = fopen(plyFile.c_str(), "w");
		int pointCount = 0;

		fprintf(fpPly,"      \n");
		fprintf(fpPly,"                              \n");
		fprintf(fpPly,"                              \n");
		fprintf(fpPly,"property float x\n");
		fprintf(fpPly,"property float y\n");
		fprintf(fpPly,"property float z\n");
		fprintf(fpPly,"property uchar red\n");
		fprintf(fpPly,"property uchar green\n");
		fprintf(fpPly,"property uchar blue\n");
		fprintf(fpPly,"end_header\n");

		for ( int index1 = 0; index1 < ptAttrList1.size(); index1 ++ )
		{
			//if ( ptAttrList1[index1]->X < 100.0 )
			{
				fprintf(fpPly,"%.10lf %.10lf %.10lf %d %d %d\n", ptAttrList1[index1]->X, ptAttrList1[index1]->Y, ptAttrList1[index1]->Z,
									ptAttrList1[index1]->R, ptAttrList1[index1]->G, ptAttrList1[index1]->B);
				pointCount ++;
			}
		}

		for ( int index1 = 0; index1 < ptAttrList2.size(); index1 ++ )
		{
			//if ( ptAttrList2[index1]->X < 100.0 )
			{
				fprintf(fpPly,"%.10lf %.10lf %.10lf %d %d %d\n", ptAttrList2[index1]->X, ptAttrList2[index1]->Y, ptAttrList2[index1]->Z,
									ptAttrList2[index1]->R, ptAttrList2[index1]->G, ptAttrList2[index1]->B);
				pointCount ++;
			}
		}
		
		fseek(fpPly,0,SEEK_SET);
		fprintf(fpPly,"ply\n");
		fprintf(fpPly,"format ascii 1.0\n");
		fprintf(fpPly,"element vertex %d", pointCount);

		fclose(fpPly);

		//Run bundler
		string bundlerCmd = "/home/tcsuser/Desktop/SFM/bin/optimizer_all " + newFolderName + "/ 0 250 16 32 1 0";
		system(bundlerCmd.c_str());
		
		break;
	}
}

