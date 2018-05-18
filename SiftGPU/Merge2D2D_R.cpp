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
	//vector<int> nMatchList;
	myfile.open(orgPair.c_str());
	while(getline(myfile, line))
	{
		split(nvmToks, line, " ");

		orgPairList.push_back(make_pair(stoi(nvmToks[0]), stoi(nvmToks[1])));
		//nMatchList.push_back(stoi(nvmToks[2]));
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


		//-0.8042905836 -0.0329016290 -0.5933252787 -0.1423832657 0.9800508195 0.1386629191 0.5769266931 0.1960048017 -0.7929302935
		//-0.8042905836 -0.0329016290 -0.5933252787 -0.1423832657 0.9800508195 0.1386629191 0.5769266931 0.1960048017 -0.7929302935

		string newFolderName = outDir + "/dendogram/cluster" + clusterMergeList[cIndex].newCluster + "/";

		//Dump nvm
		if ( 0 != mkdir(newFolderName.c_str(), 0777))
		{
			printf("Unable to create dir.\n");
		}

		string rLocalFile = newFolderName + "/R5point.txt"; 
		string pairLocalFile = newFolderName + "/original_pairs5point.txt"; 


		FILE *fpRL = fopen( rLocalFile.c_str(), "w");
		FILE *fpPL = fopen( pairLocalFile.c_str(), "w");

		for ( int index1 = 0; index1 < cameraList1.size(); index1 ++ )
		{
			for ( int index2 = 0; index2 < cameraList2.size(); index2 ++ )
			{	
				int mode = 0;
				bool fFlag = false;
				Eigen::Matrix3d R2211;
				for ( int index3 = 0; index3 < orgPairList.size(); index3 ++ )
				{

					//if ( nMatchList[index3] <= 20 ) continue;

					if ( cameraList1[index1]->camName == orgPairList[index3].first &&
						cameraList2[index2]->camName == orgPairList[index3].second )
					{
						Eigen::Matrix3d R1122 = rList[index3];
						R2211 = R1122.transpose();
						fFlag = true;
						break;
					}
					else if ( cameraList1[index1]->camName == orgPairList[index3].second &&
						cameraList2[index2]->camName == orgPairList[index3].first )
					{
						R2211 = rList[index3];
						fFlag = true;
						mode = 1;
						break;
					}
				}

				if ( fFlag )
				{
					cout << "[" << cameraList1[index1]->camName << " "<< cameraList2[index2]->camName << "]" << endl;
					cout << R2211 << endl<<endl;

					Eigen::Matrix3d R11, R22, R21; 

					R11 << 	cameraList1[index1]->rotMat[0], cameraList1[index1]->rotMat[1], cameraList1[index1]->rotMat[2],
						cameraList1[index1]->rotMat[3], cameraList1[index1]->rotMat[4], cameraList1[index1]->rotMat[5],
						cameraList1[index1]->rotMat[6], cameraList1[index1]->rotMat[7], cameraList1[index1]->rotMat[8];

					R22 << 	cameraList2[index2]->rotMat[0], cameraList2[index2]->rotMat[1], cameraList2[index2]->rotMat[2],
						cameraList2[index2]->rotMat[3], cameraList2[index2]->rotMat[4], cameraList2[index2]->rotMat[5],
						cameraList2[index2]->rotMat[6], cameraList2[index2]->rotMat[7], cameraList2[index2]->rotMat[8];

					R21 = R11.transpose() * R2211 * R22;

					cout << R11 << endl<<endl;
					cout << R22 << endl<<endl;

					if ( R21.determinant() <= 0.0 ||  R2211.determinant() <= 0.0 ||  R22.determinant() <= 0.0)
					{
						FILE *fp = fopen("log.txt", "a");
						fprintf(fp, "%d [%d,%d] R21: %lf, R11: %lf, R22: %lf, R2211: %lf\n", 
							cameraList1[index1]->camName, index1, index2, R21.determinant(), R11.determinant(), R22.determinant(), R2211.determinant());
						fclose(fp);						
					}

					//R21 = R2211.transpose();

					fprintf(fpRL, "%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", 
						R21(0, 0), R21(0, 1), R21(0, 2), R21(1, 0), R21(1, 1), R21(1, 2), R21(2, 0), R21(2, 1), R21(2, 2));

					fprintf(fpPL, "%d %d %d\n", 0, 1, mode);
				}
			}
		}

		fclose(fpRL);
		fclose(fpPL);
		break;
	}
}

