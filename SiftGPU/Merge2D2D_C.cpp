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

#include "optimize_relative_position_with_known_rotation.h"
#include "camera.h"
#include "feature_correspondence.h"
#include "test_util.h"

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

	vector<double> tmpDiff = diffList;
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
	bestRT = ComputeRT(tmpSrc, tmpDst);

	/*int nPtPerIter = 10;
	int nIter = 100;//(int)((float)src.size() / (float) nPtPerIter + 0.5);

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
		TransformType rtTmp = ComputeRT(tmpSrc, tmpDst);

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
		medianDiff = diffList[middleIndex];

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
	}*/

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
			newPt.x = stol(nvmToks[7 + i * 4 + 2]);
			newPt.y = stol(nvmToks[7 + i * 4 + 3]);

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

static void init(const double* poses,
	const double* positions,
	int nObj, double* X)
{
	const int row = 3 * nObj;

	Eigen::MatrixXf A(row,3);;
	Eigen::MatrixXf B(row, 1);
	Eigen::MatrixXf t(3, 3);;;
	Eigen::MatrixXf S(3, 1);;

	for (int i = 0; i < nObj; i++)
	{
		A(3 * i + 0, 0) = 0;
		A(3 * i + 0, 1) = -1*poses[3*i+2];
		A(3 * i + 0, 2) = poses[3 * i + 1];

		A(3 * i + 1, 0) = poses[3 * i + 2];
		A(3 * i + 1, 1) = 0;
		A(3 * i + 1, 2) = -1*poses[3 * i + 0];

		A(3 * i + 2, 0) = -1*poses[3 * i + 1];
		A(3 * i + 2, 1) = poses[3 * i + 0];
		A(3 * i + 2, 2) = 0;

		B(3 * i + 0, 0) = poses[3 * i + 1]*positions[3*i+2] - poses[3 * i + 2] * positions[3 * i + 1];
		B(3 * i + 1, 0) = poses[3 * i + 2] * positions[3 * i + 0] - poses[3 * i + 0] * positions[3 * i + 2];
		B(3 * i + 2, 0) = poses[3 * i + 0] * positions[3 * i + 1] - poses[3 * i + 1] * positions[3 * i + 0];

		//cout <<  "POSE: " << poses[3*i] << "  " << poses[3*i + 1] << " " <<  poses[3*i + 2] << endl;
		//cout <<  "POS:  " << positions[3*i] << "  " << positions[3*i + 1] << " " <<  positions[3*i + 2] << endl << endl;
	}

	t = A.transpose()*A;

	S = t.inverse()*A.transpose()*B;

	X[0] = S(0);
	X[1] = S(1);
	X[2] = S(2);

	cout <<  "OUT:  " << X[0] << "  " << X[1] << " " <<  X[2] << endl;
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

	string orgPair = outDir + "/original_pairs5point.txt";
	vector< pair<int, int> > orgPairList;
	myfile.open(orgPair.c_str());
	while(getline(myfile, line))
	{
		split(nvmToks, line, " ");

		orgPairList.push_back(make_pair(stoi(nvmToks[0]), stoi(nvmToks[1])));
	}
	myfile.close();

	string rFile = outDir + "/R5Point.txt";
	vector<Eigen::Matrix3d> rList;
	myfile.open(rFile.c_str());
	while(getline(myfile, line))
	{
		split(nvmToks, line, " ");

		Eigen::Matrix3d R;
		R << stod(nvmToks[0]), stod(nvmToks[1]), stod(nvmToks[2]),
		      stod(nvmToks[3]), stod(nvmToks[4]), stod(nvmToks[5]),
		      stod(nvmToks[6]), stod(nvmToks[7]), stod(nvmToks[8]);

		rList.push_back(R);
	}
	myfile.close();

	string tFile = outDir + "/T5Point.txt";
	vector<Eigen::Vector3d> tList;
	myfile.open(tFile.c_str());
	while(getline(myfile, line))
	{
		split(nvmToks, line, " ");

		Eigen::Vector3d T;
		T << stod(nvmToks[0]), stod(nvmToks[1]), stod(nvmToks[2]);
		tList.push_back(T);
	}
	myfile.close();


	//Read CX,CY
	string keyListFile = outDir + "/KeyList.txt";
	vector<pair<float, float> > centreList;
	myfile.open(keyListFile.c_str());
	while(getline(myfile, line))
	{
		split(nvmToks, line, " ");
		float cx = stof(nvmToks[2]) / 2.0;
		float cy = stof(nvmToks[1]) / 2.0;
		centreList.push_back(make_pair(cx, cy));
	}
	myfile.close();


	myfile.open(outDir + "/matches_forRtinlier5point.txt");
	vector<string> matchRTVec;
	vector<pair<int,int>> orgM1Pair;
	vector<pair<int,int>> orgM2Pair;

	getline(myfile, line);
	int numTotalMatches = stoi(line);

	for (int i = 0; i < numTotalMatches; i++)
	{
		vector <string> toks;
		getline(myfile, line);
		split(toks, line, " ");
		int id1 = stoi(toks[0]);
		int id2 = stoi(toks[1]);
		int numMatches = stoi(toks[2]);

		if ( numMatches == 0 ) continue;

		orgM1Pair.push_back(make_pair(id1, id2));
		orgM2Pair.push_back(make_pair(numMatches, matchRTVec.size()));

		for (int j = 0; j < numMatches; j++)
		{
			string line2;
			getline(myfile, line2);
			matchRTVec.push_back(line2);
		}

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

		//Rotate
		ifstream myfile;
		string newFolderName = outDir + "/dendogram/cluster" + clusterMergeList[cIndex].newCluster + "/";
		string rGlobalFile = newFolderName + "/R.txt";
		string matchLocalFile = newFolderName + "/2DCorrespondences.txt";
		myfile.open(rGlobalFile.c_str());
		FILE *fpMatchLocal = fopen(matchLocalFile.c_str(), "w");
	
		vector<pair<int, int> > newCamIndexList;
		vector<Eigen::Vector3d> newPosList;

		vector<pair<int, int> > newPairList;
		vector<pair<Eigen::Vector3d, int>> newCijList;

		//Read 1st line
		getline(myfile, line);

		//Read 2nd line
		getline(myfile, line);
		split(nvmToks, line, " ");

		Eigen::Matrix3d R21;
		R21 << stod(nvmToks[0]), stod(nvmToks[1]), stod(nvmToks[2]),
		      stod(nvmToks[3]), stod(nvmToks[4]), stod(nvmToks[5]),
		      stod(nvmToks[6]), stod(nvmToks[7]), stod(nvmToks[8]);

		cout << R21 << endl;

		/*for ( int index1 = 0; index1 < cameraList1.size(); index1 ++ )
		{	
			newCamIndexList.push_back(cameraList1[index1]->camName);

			Eigen::Vector3d C1;
			C1[0] = cameraList2[index1]->cMat[0];
			C1[1] = cameraList2[index1]->cMat[1];
			C1[2] = cameraList2[index1]->cMat[2];

			newPosList.push_back(C1);
		}*/

		for ( int index1 = 0; index1 < cameraList2.size(); index1 ++ )
		{
			Eigen::Vector3d C1;

			C1[0] = cameraList2[index1]->cMat[0];
			C1[1] = cameraList2[index1]->cMat[1];
			C1[2] = cameraList2[index1]->cMat[2];

			for ( int index2 = index1 + 1; index2 < cameraList2.size(); index2 ++ )
			{
				Eigen::Vector3d C2;

				C2[0] = cameraList2[index2]->cMat[0];
				C2[1] = cameraList2[index2]->cMat[1];
				C2[2] = cameraList2[index2]->cMat[2];

				Eigen::Vector3d C12 = C2 - C1; 

				newPairList.push_back(make_pair(cameraList2[index1]->camName, cameraList2[index2]->camName));
				newCijList.push_back(make_pair(C12/C12.norm(), 10));
			}
		}

		//Align the camera
		for ( int index2 = 0; index2 < cameraList2.size(); index2 ++ )
		{
			Eigen::Matrix3d R;
			R << cameraList2[index2]->rotMat[0], cameraList2[index2]->rotMat[1], cameraList2[index2]->rotMat[2],
			     cameraList2[index2]->rotMat[3], cameraList2[index2]->rotMat[4], cameraList2[index2]->rotMat[5],
			     cameraList2[index2]->rotMat[6], cameraList2[index2]->rotMat[7], cameraList2[index2]->rotMat[8];
			
			R = R * R21.transpose();

			cameraList2[index2]->rotMat[0] = R(0, 0);
			cameraList2[index2]->rotMat[1] = R(0, 1);
			cameraList2[index2]->rotMat[2] = R(0, 2);
			cameraList2[index2]->rotMat[3] = R(1, 0);
			cameraList2[index2]->rotMat[4] = R(1, 1);
			cameraList2[index2]->rotMat[5] = R(1, 2);
			cameraList2[index2]->rotMat[6] = R(2, 0);
			cameraList2[index2]->rotMat[7] = R(2, 1);
			cameraList2[index2]->rotMat[8] = R(2, 2);

			Eigen::Vector3d pt2;

			pt2[0] = cameraList2[index2]->cMat[0];
			pt2[1] = cameraList2[index2]->cMat[1];
			pt2[2] = cameraList2[index2]->cMat[2];

			pt2 = R21 * pt2;
		
			cameraList2[index2]->cMat[0] = pt2[0];
			cameraList2[index2]->cMat[1] = pt2[1];
			cameraList2[index2]->cMat[2] = pt2[2];

			getTfromRC(cameraList2[index2]->rotMat, cameraList2[index2]->cMat, cameraList2[index2]->tMat);
		}

		//Cij
		vector <Eigen::Vector3d> newCam2Center;
		vector <Eigen::Vector3d> oldCam2Center;
		
		string tmpFile = newFolderName + "/tmp.txt";
		FILE *fpTmp = fopen(tmpFile.c_str(), "w");

		for ( int index2 = 0; index2 < cameraList2.size(); index2 ++ )
		{
			double* poses = (double*)calloc(orgPairList.size(), sizeof(double));
			double* posesPrev = (double*)calloc(orgPairList.size(), sizeof(double));
			double* positions = (double*)calloc(orgPairList.size(), sizeof(double));
			int nObj = 0;
			Eigen::Vector3d T12, C12, C12prev;

			for ( int index1 = 0; index1 < cameraList1.size(); index1 ++ )
			{	
				bool fFlag = false;

				for ( int index3 = 0; index3 < orgPairList.size(); index3 ++ )
				{
					if ( cameraList1[index1]->camName == orgPairList[index3].first &&
						cameraList2[index2]->camName == orgPairList[index3].second )
					{
						fFlag = true;
						C12 = -1 * rList[index3].transpose() * tList[index3];
						T12 = tList[index3];
						break;
					}
					else if ( cameraList1[index1]->camName == orgPairList[index3].second &&
						cameraList2[index2]->camName == orgPairList[index3].first )
					{
						fFlag = true;
						C12 = rList[index3].transpose() * tList[index3];
						T12 = -tList[index3];
						break;
					}

				}

				if ( fFlag )
				{
					{
						C12prev = C12;

						int numTotalMatches = orgM1Pair.size();

						for (int i = 0; i < numTotalMatches; i++)
						{
							int id1 = orgM1Pair[i].first;
							int id2 = orgM1Pair[i].second;
							int numMatches = orgM2Pair[i].first;
							int stIndexStr = orgM2Pair[i].second;

							if ( numMatches == 0 ) continue;

							if (cameraList1[index1]->camName == id1 && cameraList2[index2]->camName == id2)
							{
								fprintf(fpMatchLocal, "%d %d %d\n", id1, id2, numMatches);

								theia::Camera camera1, camera2;

								Eigen::Matrix3d rotation1, rotation2;
								rotation1 << cameraList1[index1]->rotMat[0], cameraList1[index1]->rotMat[1], cameraList1[index1]->rotMat[2], 
									cameraList1[index1]->rotMat[3], cameraList1[index1]->rotMat[4], cameraList1[index1]->rotMat[5], 
									cameraList1[index1]->rotMat[6], cameraList1[index1]->rotMat[7], cameraList1[index1]->rotMat[8];
								camera1.SetOrientationFromRotationMatrix(rotation1);

								rotation2 << cameraList2[index2]->rotMat[0], cameraList2[index2]->rotMat[1], cameraList2[index2]->rotMat[2], 
									cameraList2[index2]->rotMat[3], cameraList2[index2]->rotMat[4], cameraList2[index2]->rotMat[5], 
									cameraList2[index2]->rotMat[6], cameraList2[index2]->rotMat[7], cameraList2[index2]->rotMat[8];
								camera2.SetOrientationFromRotationMatrix(rotation2);

								double focal1 = cameraList1[index1]->focal;
								double focal2 = cameraList2[index2]->focal;
								vector<theia::FeatureCorrespondence> matches;
		
								for (int j = 0; j < numMatches; j++)
								{
									vector <string> toks2;
									string line2 = matchRTVec[stIndexStr + j];
									split(toks2, line2, " ");

									theia::FeatureCorrespondence match;

									match.feature1.x() = (stod(toks2[1]) - centreList[id1 - 1].first) / focal1;
									match.feature1.y() = (stod(toks2[2]) - centreList[id1 - 1].second) / focal1;
									match.feature2.x() = (stod(toks2[4]) - centreList[id2 - 1].first) / focal2;
									match.feature2.y() = (stod(toks2[5]) - centreList[id2 - 1].second) / focal2;
									matches.emplace_back(match);

									fprintf(fpMatchLocal, "%s\n", line2.c_str());
								}

								Eigen::Vector3d relative_position;

								relative_position << T12[0], T12[1], T12[2];

								fprintf(fpTmp, "\n\n********************* %d \n", matches.size());

								theia::OptimizeRelativePositionWithKnownRotation(
									matches,
									camera1.GetOrientationAsAngleAxis(),
									camera2.GetOrientationAsAngleAxis(),
									&relative_position);

								C12 = rotation1.transpose()*relative_position;

								matches.clear();								
							}
							else if( cameraList2[index2]->camName == id1 && cameraList1[index1]->camName == id2 )
							{

								fprintf(fpMatchLocal, "%d %d %d\n", id1, id2, numMatches);

								theia::Camera camera1, camera2;

								Eigen::Matrix3d rotation1, rotation2;
								rotation2 << cameraList1[index1]->rotMat[0], cameraList1[index1]->rotMat[1], cameraList1[index1]->rotMat[2], 
									cameraList1[index1]->rotMat[3], cameraList1[index1]->rotMat[4], cameraList1[index1]->rotMat[5], 
									cameraList1[index1]->rotMat[6], cameraList1[index1]->rotMat[7], cameraList1[index1]->rotMat[8];
								camera2.SetOrientationFromRotationMatrix(rotation2);

								rotation1 << cameraList2[index2]->rotMat[0], cameraList2[index2]->rotMat[1], cameraList2[index2]->rotMat[2], 
									cameraList2[index2]->rotMat[3], cameraList2[index2]->rotMat[4], cameraList2[index2]->rotMat[5], 
									cameraList2[index2]->rotMat[6], cameraList2[index2]->rotMat[7], cameraList2[index2]->rotMat[8];
								camera1.SetOrientationFromRotationMatrix(rotation1);

								double focal2 = cameraList1[index1]->focal;
								double focal1 = cameraList2[index2]->focal;
								vector<theia::FeatureCorrespondence> matches;
		
								for (int j = 0; j < numMatches; j++)
								{
									vector <string> toks2;
									string line2 = matchRTVec[stIndexStr + j];
									split(toks2, line2, " ");

									theia::FeatureCorrespondence match;

									match.feature1.x() = (stod(toks2[1]) - centreList[id1 - 1].first) / focal1;
									match.feature1.y() = (stod(toks2[2]) - centreList[id1 - 1].second) / focal1;
									match.feature2.x() = (stod(toks2[4]) - centreList[id2 - 1].first) / focal2;
									match.feature2.y() = (stod(toks2[5]) - centreList[id2 - 1].second) / focal2;
									matches.emplace_back(match);

									fprintf(fpMatchLocal, "%s\n", line2.c_str());
								}

								Eigen::Vector3d relative_position;

								relative_position << T12[0], T12[1], T12[2];

								cout << endl << "M:  " << matches.size() << endl << "T12: " << relative_position << endl;
								cout << "Rot: " << rotation1 << endl;
						
								fprintf(fpTmp, "\n\n********************* %d \n", matches.size());

								theia::OptimizeRelativePositionWithKnownRotation(
									matches,
									camera1.GetOrientationAsAngleAxis(),
									camera2.GetOrientationAsAngleAxis(),
									&relative_position);

								C12 = -(rotation1.transpose()*relative_position);
								cout << "T12_U: " << relative_position << endl;
								cout << "C12: " << C12 << endl << endl;


								matches.clear();				
							}

						}
					}

					fprintf(fpTmp, "1. %lf %lf %lf %lf %lf %lf\n", C12prev(0), C12prev(1), C12prev(2), C12(0), C12(1), C12(2));

					newPairList.push_back(make_pair(cameraList1[index1]->camName, cameraList2[index2]->camName));
					newCijList.push_back(make_pair(C12/C12.norm(), 1));

					poses[nObj*3 + 0] = C12(0);
					poses[nObj*3 + 1] = C12(1);
					poses[nObj*3 + 2] = C12(2);

					posesPrev[nObj*3 + 0] = C12prev(0);
					posesPrev[nObj*3 + 1] = C12prev(1);
					posesPrev[nObj*3 + 2] = C12prev(2);

					positions[nObj*3 + 0] = cameraList1[index1]->cMat[0];
					positions[nObj*3 + 1] = cameraList1[index1]->cMat[1];
					positions[nObj*3 + 2] = cameraList1[index1]->cMat[2];

					nObj ++;

					bool fFlag = false;
					for ( int tmp = 0; tmp < newCamIndexList.size(); tmp ++ )
					{
						if ( newCamIndexList[tmp].first == cameraList1[index1]->camName ) 
						{
							fFlag = true;
							break;
						}
					}

					if ( fFlag == false )
					{
						newCamIndexList.push_back(make_pair(cameraList1[index1]->camName, 10));

						Eigen::Vector3d C1;
						C1[0] = cameraList1[index1]->cMat[0];
						C1[1] = cameraList1[index1]->cMat[1];
						C1[2] = cameraList1[index1]->cMat[2];

						newPosList.push_back(C1);
					}
				}
			}

			double X[3] = {0.0};
			double XPrev[3] = {0.0};

			if ( nObj > 1 )
			{
				init(posesPrev, positions, nObj, XPrev);	
				init(poses, positions, nObj, X);	


	double *anglePrev = new double[nObj];
	double *angle = new double[nObj];
	for(int i = 0; i < nObj; i++)
	{
		double solve[3];
		solve[0] = XPrev[0]  - positions[3*i + 0];
		solve[1] = XPrev[1]  - positions[3*i + 1];
		solve[2] = XPrev[2]  - positions[3*i + 2];

		double length = solve[0]*solve[0]  + solve[1]*solve[1] + solve[2]*solve[2];
		length = sqrt(length);

		double dot = (solve[0]*posesPrev[3*i + 0] + solve[1]*posesPrev[3*i + 1] + solve[2]*posesPrev[3*i + 2])/length;
		anglePrev[i] = acos(dot)*180/3.14;

		solve[0] = X[0]  - positions[3*i + 0];
		solve[1] = X[1]  - positions[3*i + 1];
		solve[2] = X[2]  - positions[3*i + 2];

		length = solve[0]*solve[0]  + solve[1]*solve[1] + solve[2]*solve[2];
		length = sqrt(length);

		dot = (solve[0]*poses[3*i + 0] + solve[1]*poses[3*i + 1] + solve[2]*poses[3*i + 2])/length;
		angle[i] = acos(dot)*180/3.14;

		fprintf(fpTmp, "P [%.10lf %.10lf %.10lf] [%.10lf %.10lf %.10lf] [%.10lf %.10lf %.10lf] \n", XPrev[0], XPrev[1], XPrev[2], 
					positions[3*i + 0], positions[3*i + 1], positions[3*i + 2], posesPrev[3*i + 0], posesPrev[3*i + 1], posesPrev[3*i + 2]);
		fprintf(fpTmp, "C [%.10lf %.10lf %.10lf] [%.10lf %.10lf %.10lf] [%.10lf %.10lf %.10lf] \n", X[0], X[1], X[2], 
					positions[3*i + 0], positions[3*i + 1], positions[3*i + 2], poses[3*i + 0], poses[3*i + 1], poses[3*i + 2]);

		fprintf(fpTmp, "%lf %lf\n", anglePrev[i], angle[i]);
		fflush(fpTmp);
	}
	fprintf(fpTmp, "\n\n");

				for(int i = 0; i < nObj; i ++)
				{
					Eigen::Vector3d tmp, tmp1;
					tmp << X[0] - positions[nObj*i], X[1] - positions[nObj*i + 1], X[2] - positions[nObj*i + 2];
					tmp1 << poses[i*3], poses[i*3 + 1], poses[i*3 + 2];

					Eigen::Vector3d tmp2 = tmp/tmp.norm();
					cout << tmp2.dot(tmp1)*180/3.143 <<endl;
				}

				Eigen::Vector3d C;
			
				if ( std::isnan(X[0]) || std::isnan(X[1]) || std::isnan(X[2]))
				{
					C << 0, 0, 0;
				}
				else
				{
					C << X[0], X[1], X[2];
					newCam2Center.push_back(C);

					newCamIndexList.push_back(make_pair(cameraList2[index2]->camName, 1));
					newPosList.push_back(C);

					fprintf(fpTmp, "2. %lf %lf %lf\n", XPrev[0], XPrev[1], XPrev[2]);
					fprintf(fpTmp, "3. %lf %lf %lf\n\n", X[0], X[1], X[2]);

					C << cameraList2[index2]->cMat[0], cameraList2[index2]->cMat[1], cameraList2[index2]->cMat[2];  
					oldCam2Center.push_back(C);

					printf("## %d\n", index2);  
				}
			}

			free(poses);
			free(positions);
		}
	
		fclose(fpTmp);

		vector<double> scaleList;
		double scale;
		for ( int index = 0; index < oldCam2Center.size(); index ++ )
		{
			for ( int index1 = index + 1; index1 < oldCam2Center.size(); index1 ++ )
			{
				Eigen::Vector3d diffSrc  = (oldCam2Center[index] - oldCam2Center[index1]);

				Eigen::Vector3d diffDst = (newCam2Center[index] - newCam2Center[index1]);
				
				double tmpScale = 0.0;
				if ( diffSrc.norm() != 0.0 && diffDst.norm() != 0.0 )
				{
					tmpScale = diffDst.norm()/diffSrc.norm();
					scaleList.push_back(tmpScale);
					printf("S[%d %d] %lf\n", index, index1, tmpScale);
				}
			}
		}

		//Calculate median of scales
		sort (scaleList.begin(), scaleList.end());

		/*for ( int i = 0; i < scaleList.size(); i ++)
		{
			printf("S[%d] %lf\n", i, scaleList[i]);
		}*/
		
		int middleIndex = scaleList.size() / 2;
		scale = scaleList[middleIndex];
		cout << "Scale: " << scale << endl;

		//Calculate Translation
		PointsType firstList, secondList;

		vector<double> xList, yList, zList;
		for ( int index2 = 0; index2 < oldCam2Center.size(); index2 ++ )
		{
			xList.push_back(oldCam2Center[index2](0) * scale  - (newCam2Center[index2])(0));
			yList.push_back(oldCam2Center[index2](1) * scale  - (newCam2Center[index2])(1));
			zList.push_back(oldCam2Center[index2](2) * scale  - (newCam2Center[index2])(2));


			Eigen::Vector3d pt1, pt2;

			pt1 = newCam2Center[index2];

			pt2[0] = oldCam2Center[index2](0);
			pt2[1] = oldCam2Center[index2](1);
			pt2[2] = oldCam2Center[index2](2);

			firstList.push_back(pt1);
			secondList.push_back(pt2);
		}
		
		sort (xList.begin(), xList.end());
		sort (yList.begin(), yList.end());
		sort (zList.begin(), zList.end());

		middleIndex = xList.size() / 2;

		cout << "C: " << xList[middleIndex] << "   " << yList[middleIndex] << "   " << zList[middleIndex] << endl;

		Eigen::Vector3d C;
		C << xList[middleIndex], yList[middleIndex], zList[middleIndex];

		Eigen::Vector3d T = -C; 
		
		cout <<T <<endl;

		string outFile = newFolderName + "/RTS2D2D.txt";
		FILE *fpOut = fopen(outFile.c_str(), "w");

		fprintf(fpOut, "%.9lf\n%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n%lf %lf %lf\n", scale,
			R21(0, 0), R21(0, 1), R21(0, 2), R21(1, 0), R21(1, 1), R21(1, 2), 
			R21(2, 0), R21(2, 1), R21(2, 2), T[0], T[1], T[2]);

		fclose(fpOut);
		
		TransformType RT;
		ComputeRigidTransform(secondList, firstList, scale, RT.first, RT.second);
		cout << scale << endl;
		cout << RT.first << endl;
		cout << RT.second << endl;

		string camNameFile = newFolderName + "/org_cam_name.txt";
		string positionFile = newFolderName + "/pos.txt";
		string directionFile = newFolderName + "/dir.txt";
		string pairFile = newFolderName + "/pair.txt";

		FILE *fpCName = fopen(camNameFile.c_str(), "w");
		FILE *fpPos = fopen(positionFile.c_str(), "w");
		FILE *fpDir = fopen(directionFile.c_str(), "w");
		FILE *fpPair = fopen(pairFile.c_str(), "w");


		for ( int index1 = 0; index1 < newCamIndexList.size(); index1 ++ )
		{
			fprintf(fpCName, "%d\n", newCamIndexList[index1].first);
			fprintf(fpPos, "%lf %lf %lf %d\n", (newPosList[index1])[0], (newPosList[index1])[1], (newPosList[index1])[2], newCamIndexList[index1].second);
		}

		int nPairs = 0;
		for ( int index1 = 0; index1 < newPairList.size(); index1 ++ )
		{

			int pair1 = -1;
			int pair2 = -1;

			for ( int index2 = 0; index2 < newCamIndexList.size(); index2 ++ )
			{
				if ( newPairList[index1].first == newCamIndexList[index2].first )
				{
					pair1 = index2;
				}	

				if ( newPairList[index1].second == newCamIndexList[index2].first )
				{
					pair2 = index2;
				}
			}

			if ( pair1 == -1 || pair2 == -1 ) continue;

			nPairs ++;
		}

		fprintf(fpPair, "%d %d\n", newCamIndexList.size(), nPairs);
		for ( int index1 = 0; index1 < newPairList.size(); index1 ++ )
		{

			int pair1 = -1;
			int pair2 = -1;

			for ( int index2 = 0; index2 < newCamIndexList.size(); index2 ++ )
			{
				if ( newPairList[index1].first == newCamIndexList[index2].first )
				{
					pair1 = index2;
				}	

				if ( newPairList[index1].second == newCamIndexList[index2].first )
				{
					pair2 = index2;
				}
			}

			if ( pair1 == -1 || pair2 == -1 ) continue;

			fprintf(fpPair, "%d %d\n", pair1, pair2);
			fprintf(fpDir, "%lf %lf %lf %d\n", (newCijList[index1].first)[0], (newCijList[index1].first)[1], (newCijList[index1].first)[2], newCijList[index1].second);
		}

		fclose(fpCName);
		fclose(fpPos);
		fclose(fpDir);
		fclose(fpPair);
		fclose(fpMatchLocal);

		break;
	}
}

