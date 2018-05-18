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
#include <pthread.h>

#include "verify_two_view_matches.h"
#include "feature_correspondence.h"
#include "camera_intrinsics_prior.h"
#include "estimate_twoview_info.h"
#include "twoview_info.h"

#include "SiftGPU.h"

using namespace std;

#define FREE_MYLIB dlclose
#define GET_MYPROC dlsym
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#define VOCAB_TREE_TH 	0.005
#define SAMPSON_ERR_5P 	1.0
#define INLIER_TH	20

//#define printf(...)

/*#define FX 	517.3
#define FY	516.5
#define CX	2304.0
#define CY	1536.0*/

typedef struct _matches
{
		int firstIndex;
		int secondIndex;
		double confidenceFactor;
}Matches;

typedef struct _frame
{
	int 		frameIndex;
	string 		imgFileName;
	string 		siftFileName;
	bool		isValid;

	vector< int> 					pointId;
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

bool VerifyTwoViewMatches(
		const VerifyTwoViewMatchesOptions& options,
		const CameraIntrinsicsPrior& intrinsics1,
		const CameraIntrinsicsPrior& intrinsics2,
		const std::vector<FeatureCorrespondence>& correspondences,
		TwoViewInfo* twoview_info,
		std::vector<int>* inlier_indices) {
	if (correspondences.size() < options.min_num_inlier_matches) {
		return false;
	}

//	struct timeval stTime, enTime;
//	gettimeofday(&stTime, NULL);

	// Estimate the two view info. If we fail to estimate a two view info then do
	// not add this view pair to the verified matches.
	if (!EstimateTwoViewInfo(options.estimate_twoview_info_options,
			intrinsics1,
			intrinsics2,
			correspondences,
			twoview_info,
			inlier_indices)) {
		return false;
	}


//	gettimeofday(&enTime, NULL);
//	printf("Five Point: %lf ms\n", ((enTime.tv_sec * 1000.0 + enTime.tv_usec/1000.0)
//			- (stTime.tv_sec * 1000.0 + stTime.tv_usec/1000.0)));


	if (inlier_indices->size() < options.min_num_inlier_matches) {
		return false;
	}
	return true;
}

static void GetPoseChange5Point(vector<FeatureCorrespondence> &correspondences, double avgFocal1, double avgFocal2,
		TwoViewInfo &twoview_info, vector<int> &inlier_indices, FILE *fpR, FILE *fpT, FILE *fpE)
{
	VerifyTwoViewMatchesOptions options;

	options.bundle_adjustment = false;
	options.min_num_inlier_matches = 10;
	options.estimate_twoview_info_options.max_sampson_error_pixels = SAMPSON_ERR_5P;
	options.estimate_twoview_info_options.max_ransac_iterations = 2000;

	CameraIntrinsicsPrior intrinsics1, intrinsics2;

	//setting K for image1
	intrinsics1.focal_length.value = avgFocal1;
	intrinsics1.focal_length.is_set = true;
	intrinsics1.principal_point[0].is_set = true;
	intrinsics1.principal_point[0].value = 0.0;
	intrinsics1.principal_point[1].is_set = true;
	intrinsics1.principal_point[1].value = 0.0;
	intrinsics1.aspect_ratio.is_set = true;
	intrinsics1.aspect_ratio.value = 1.0;
	intrinsics1.skew.is_set = true;
	intrinsics1.skew.value = 0.0;

	//setting K for image2
	intrinsics2.focal_length.value = avgFocal2;
	intrinsics2.focal_length.is_set = true;
	intrinsics2.principal_point[0].is_set = true;
	intrinsics2.principal_point[0].value = 0.0;
	intrinsics2.principal_point[1].is_set = true;
	intrinsics2.principal_point[1].value = 0.0;
	intrinsics2.aspect_ratio.is_set = true;
	intrinsics2.aspect_ratio.value = 1.0;
	intrinsics2.skew.is_set = true;
	intrinsics2.skew.value = 0.0;

	bool ret = VerifyTwoViewMatches(options, intrinsics1, intrinsics2, correspondences, &twoview_info, &inlier_indices);

	if ( inlier_indices.size() <= INLIER_TH )
	{
		inlier_indices.clear();
		return;
	}

	fprintf(fpR, "%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n", twoview_info.rotationmat_2(0, 0), twoview_info.rotationmat_2(0, 1), twoview_info.rotationmat_2(0, 2), twoview_info.rotationmat_2(1, 0), twoview_info.rotationmat_2(1, 1), twoview_info.rotationmat_2(1, 2), twoview_info.rotationmat_2(2, 0), twoview_info.rotationmat_2(2, 1), twoview_info.rotationmat_2(2, 2));
	fprintf(fpE, "%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n", twoview_info.essential_mat(0, 0), twoview_info.essential_mat(0, 1), twoview_info.essential_mat(0, 2), twoview_info.essential_mat(1, 0), twoview_info.essential_mat(1, 1), twoview_info.essential_mat(1, 2), twoview_info.essential_mat(2, 0), twoview_info.essential_mat(2, 1), twoview_info.essential_mat(2, 2));
	fprintf(fpT, "%.10lf %.10lf %.10lf\n", twoview_info.translation_2(0), twoview_info.translation_2(1), twoview_info.translation_2(2));

	return;
}

static void Run5Point(vector<FeatureCorrespondence> &correspondences, double avgFocal,
		TwoViewInfo &twoview_info, vector<int> &inlier_indices, double *rMat, double *tMat, double *eMat)
{
	VerifyTwoViewMatchesOptions options;

	options.bundle_adjustment = false;
	options.min_num_inlier_matches = 10;
	options.estimate_twoview_info_options.max_sampson_error_pixels = SAMPSON_ERR_5P;

	CameraIntrinsicsPrior intrinsics1, intrinsics2;

	//setting K for image1
	intrinsics1.focal_length.value = avgFocal;
	intrinsics1.focal_length.is_set = true;
	intrinsics1.principal_point[0].is_set = true;
	intrinsics1.principal_point[0].value = 0.0;
	intrinsics1.principal_point[1].is_set = true;
	intrinsics1.principal_point[1].value = 0.0;
	intrinsics1.aspect_ratio.is_set = true;
	intrinsics1.aspect_ratio.value = 1.0;
	intrinsics1.skew.is_set = true;
	intrinsics1.skew.value = 0.0;

	//setting K for image2
	intrinsics2.focal_length.value = avgFocal;
	intrinsics2.focal_length.is_set = true;
	intrinsics2.principal_point[0].is_set = true;
	intrinsics2.principal_point[0].value = 0.0;
	intrinsics2.principal_point[1].is_set = true;
	intrinsics2.principal_point[1].value = 0.0;
	intrinsics2.aspect_ratio.is_set = true;
	intrinsics2.aspect_ratio.value = 1.0;
	intrinsics2.skew.is_set = true;
	intrinsics2.skew.value = 0.0;

	bool ret = VerifyTwoViewMatches(options, intrinsics1, intrinsics2, correspondences, &twoview_info, &inlier_indices);

	rMat[0] = twoview_info.rotationmat_2(0, 0);
	rMat[1] = twoview_info.rotationmat_2(0, 1);
	rMat[2] = twoview_info.rotationmat_2(0, 2);
	rMat[3] = twoview_info.rotationmat_2(1, 0);
	rMat[4] = twoview_info.rotationmat_2(1, 1);
	rMat[5] = twoview_info.rotationmat_2(1, 2);
	rMat[6] = twoview_info.rotationmat_2(2, 0);
	rMat[7] = twoview_info.rotationmat_2(2, 1);
	rMat[8] = twoview_info.rotationmat_2(2, 2);


	eMat[0] = twoview_info.essential_mat(0, 0);
	eMat[1] = twoview_info.essential_mat(0, 1);
	eMat[2] = twoview_info.essential_mat(0, 2);
	eMat[3] = twoview_info.essential_mat(1, 0);
	eMat[4] = twoview_info.essential_mat(1, 1);
	eMat[5] = twoview_info.essential_mat(1, 2);
	eMat[6] = twoview_info.essential_mat(2, 0);
	eMat[7] = twoview_info.essential_mat(2, 1);
	eMat[8] = twoview_info.essential_mat(2, 2);

	tMat[0] = twoview_info.translation_2(0);
	tMat[1] = twoview_info.translation_2(1);
	tMat[2] = twoview_info.translation_2(2);
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

//#define MULTI_TH 1

#ifdef MULTI_TH

#define N_THREADS 10
#define MAX_SIFT 20000

typedef struct _data_container
{
	int Id;
	vector<Frame *> *globalFrameList;	
	vector<string > *siftFileList;
}DataContainer;

void *ReadSIFT(void *data_ptr)
{
	DataContainer *dataContainer = (DataContainer *)data_ptr;
	int thId = dataContainer->Id;
	vector<string > *siftFileList = dataContainer->siftFileList;
	vector<Frame *> *globalFrameList = dataContainer->globalFrameList;

	printf("TH[%d] spawned\n", pthread_self());

	int listSize = siftFileList->size();
	int perTh = listSize / N_THREADS;

	int stIndex = thId * perTh;
	int enIndex = stIndex + perTh;

	int pointCount = stIndex * MAX_SIFT;
	int nSift = 0, siftLen = 0;

	if ( thId == N_THREADS - 1 ) enIndex = listSize;

	for( int sIndex = stIndex; sIndex < enIndex; sIndex ++ )
	{
		FILE *fp = fopen((*siftFileList)[sIndex].c_str(), "r");
		if ( fp == NULL)
		{
			printf("Unable to open SIFT file %s\n", (*siftFileList)[sIndex].c_str());
			return NULL;
		}

		nSift = 0;
		siftLen = 0;
		//fscanf(fp, "%d %d", &nSift, &siftLen);
		fread ((char* )&nSift, sizeof(int), 1, fp);
		fread ((char* )&siftLen, sizeof(int), 1, fp);
		nSift = nSift > MAX_SIFT ? MAX_SIFT: nSift;		

		Frame *newFrame = new Frame;
		newFrame->frameIndex = sIndex;
		newFrame->isValid = false;
		newFrame->siftFileName = (*siftFileList)[sIndex];

		vector<float > *singleDesc = new vector<float>;
		vector<SiftGPU::SiftKeypoint> *singleKey = new vector<SiftGPU::SiftKeypoint>;

		for(int index = 0; index < nSift; index ++)
		{
			SiftKeypoint skp;
			//fscanf(fp, "%f %f %f %f\n", &skp.y, &skp.x, &skp.s, &skp.o);
			fread ((char* )&skp.y, sizeof(float), 1, fp);
			fread ((char* )&skp.x, sizeof(float), 1, fp);
			fread ((char* )&skp.s, sizeof(float), 1, fp);
			fread ((char* )&skp.o, sizeof(float), 1, fp);

			singleKey->push_back(skp);

			for ( int index1 = 0; index1 < siftLen; index1 ++ )
			{
				//int d;
				//fscanf(fp, "%d ", &d);
				//singleDesc->push_back(d/512.0f);

				float f;
				fread ((char* )&f, sizeof(float), 1, fp);
				singleDesc->push_back(f);
			}

			newFrame->pointId.push_back(pointCount);
			newFrame->pointValidity.push_back(false);
			pointCount ++;
		}

		newFrame->siftDesc = singleDesc;
		newFrame->siftKey = singleKey;
		(*globalFrameList)[sIndex]= newFrame;
		fclose(fp);
	}

	printf("TH[%d] exiting\n", pthread_self());

	return NULL;
}

#endif


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
	
	char * argv_tmp[] = {"-fo", "-1", "-tc2", "7680", "-v", "0", "-da","-nomc"};
	//char * argv_tmp[] = {"-fo", "-1", "-da", "-v", "1"};//
	//char * argv_tmp[] = {"-fo", "-1",  "-v", "0","-cuda", "[0]","-maxd","6000"};//
	SiftMatchGPU* matcher = pCreateNewSiftMatchGPU(4096);
	int argc_tmp = sizeof(argv_tmp)/sizeof(char*);
	sift->ParseParam(argc_tmp, argv_tmp);

	if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return 0;

	if ( argc < 3 )
	{
		printf("Invalid arguments\n ./a.out <Match File> <Output DIR>\n");
		return 0;
	}

	string matchFile = argv[1];
	string outDir = argv[2];

	//int m1 = stoi(argv[3]) - 1;
	//int m2 = stoi(argv[4]) - 1;

	string inlierFile = outDir + "/matches_forRtinlier5point.txt";
	string graphFile = outDir + "/graph.txt";
	string rFile = outDir + "/R5Point.txt";
	string tFile = outDir + "/T5Point.txt";
	string eFile = outDir + "/E5Point.txt";
	string orgPair = outDir + "/original_pairs5point.txt";
	string consolidedFile = outDir + "/consolided_result.txt";
	string focalFile = outDir + "/list_focal.txt";
	string keyListFile = outDir + "/KeyList.txt";
	string siftMatchFile = outDir + "/siftmatch.txt";

	ifstream myfile1;
	string line;
	vector <string> nvmToks;

	FILE *fpMatch = fopen(inlierFile.c_str(), "w");
	FILE *fpGraph = fopen(graphFile.c_str(), "w");
	FILE *fpR = fopen(rFile.c_str(), "w");
	FILE *fpE = fopen(eFile.c_str(), "w");
	FILE *fpT = fopen(tFile.c_str(), "w");
	FILE *fpPairs = fopen(orgPair.c_str(), "w");
	FILE *fpResult = fopen(consolidedFile.c_str(), "w");
	FILE *fpSiftMatch = fopen(siftMatchFile.c_str(), "w");

	if ( fpMatch == NULL || fpGraph == NULL || fpR == NULL ||
		 fpE == NULL || fpT == NULL || fpPairs == NULL || fpResult == NULL)
	{
		printf("Unable to open output file. Exiting.\n");
		return -1;
	}

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

	int totalFrames = siftFileList.size();
//	cout<<totalFrames
	vector<Frame *> globalFrameList(totalFrames);
	multimap<int,pair<int, int> > globalPtIndexList;

	int pointCount = 0;
	int nSift = 0, siftLen;

	printf("Loading SIFT features...\n");

#ifdef MULTI_TH
	/* this variable is our reference to the second thread */
	pthread_t th_desc[N_THREADS];
	DataContainer dataPtr[N_THREADS];
	
	for ( int i = 0; i < N_THREADS; i ++ )
	{
		dataPtr[i].Id = i;
		dataPtr[i].siftFileList = &siftFileList;
		dataPtr[i].globalFrameList = &globalFrameList;

		if(pthread_create(th_desc + i, NULL, ReadSIFT, dataPtr + i)) 
		{
			printf("Error creating thread[%d]\n", i);
			return 1;
		}
	}

	for ( int i = 0; i < N_THREADS; i ++ )
	{
		pthread_join(th_desc[i], NULL);
	}
#else
	for( int sIndex = 0; sIndex < siftFileList.size(); sIndex ++ )
	{
		FILE *fp = fopen(siftFileList[sIndex].c_str(), "r");
		if ( fp == NULL)
		{
			printf("Unable to open SIFT file %s\n", siftFileList[sIndex].c_str());
			return -1;
		}

		nSift = 0;
		siftLen = 0;
		fscanf(fp, "%d %d", &nSift, &siftLen);
		//fread ((char* )&nSift, sizeof(int), 1, fp);
		//fread ((char* )&siftLen, sizeof(int), 1, fp);

		Frame *newFrame = new Frame;
		newFrame->frameIndex = sIndex;
		newFrame->isValid = false;
		newFrame->siftFileName = siftFileList[sIndex];

		vector<float > *singleDesc = new vector<float>;
		vector<SiftGPU::SiftKeypoint> *singleKey = new vector<SiftGPU::SiftKeypoint>;
		int nDuplicate = 0;
		for(int index = 0; index < nSift; index ++)
		{
			SiftKeypoint skp;
			fscanf(fp, "%f %f %f %f\n", &skp.y, &skp.x, &skp.s, &skp.o);
			//fread ((char* )&skp.y, sizeof(float), 1, fp);
			//fread ((char* )&skp.x, sizeof(float), 1, fp);
			//fread ((char* )&skp.s, sizeof(float), 1, fp);
			//fread ((char* )&skp.o, sizeof(float), 1, fp);
			
			singleKey->push_back(skp);

			for ( int index1 = 0; index1 < siftLen; index1 ++ )
			{
				int d;
				fscanf(fp, "%d ", &d);
				singleDesc->push_back(d/512.0f);

				//float f;
				//fread ((char* )&f, sizeof(float), 1, fp);
				//singleDesc->push_back(f);
			}

			newFrame->pointId.push_back(pointCount);
			newFrame->pointValidity.push_back(false);
			globalPtIndexList.insert(pair<int, pair<int, int> >(pointCount, pair<int, int>(sIndex, index) ));

			pointCount ++;
		}

		newFrame->siftDesc = singleDesc;
		newFrame->siftKey = singleKey;
		globalFrameList[sIndex]= newFrame;
		fclose(fp);

		printProgress(sIndex, totalFrames);
	}
#endif

	printf("Total points %d\n", globalPtIndexList.size());

	//Read list focal
	vector<double> avgFocalList;
	myfile1.open(focalFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");

		double focal = stod(nvmToks[2]);
		avgFocalList.push_back(focal);
		//printf("Focal: %lf\n", focal);
	}
	myfile1.close();

	//must call once
	//matcher->SetLanguage(SiftMatchGPU::SIFTMATCH_GLSL);
	matcher->VerifyContextGL();
	
	matcher->SetMaxSift(20000);

	printf("\n\nMatching and generating graph...\n");

	vector< Matches > matchesList;
	myfile1.open(matchFile.c_str());
	while(getline(myfile1, line))
	{
		//cout<<line<<endl;
		split(nvmToks, line, " ");
		Matches match;
		match.firstIndex = stoi(nvmToks[0]);
		match.secondIndex = stoi(nvmToks[1]);
		match.confidenceFactor = stod(nvmToks[2]);
		matchesList.push_back(match);
	}
	myfile1.close();

	int totalMatches = matchesList.size();
	fprintf(fpSiftMatch, "%d\n", totalMatches);

	for ( int mIndex = 0; mIndex < totalMatches; mIndex ++ )
	{
		int firstIndex = matchesList[mIndex].firstIndex;
		int secondIndex = matchesList[mIndex].secondIndex;
		double confidenceFactor = matchesList[mIndex].confidenceFactor;

		if ( firstIndex >= secondIndex || confidenceFactor < VOCAB_TREE_TH /*|| firstIndex != m1 || secondIndex != m2*/ )
		{
			fprintf(fpSiftMatch, "%d %d 0\n", firstIndex + 1, secondIndex + 1);
			continue;
		}

		globalFrameList[firstIndex]->isValid = true;
		globalFrameList[secondIndex]->isValid = true;

		vector<SiftGPU::SiftKeypoint> *keyList1 = globalFrameList[firstIndex]->siftKey;
		vector<SiftGPU::SiftKeypoint> *keyList2 = globalFrameList[secondIndex]->siftKey;

		vector<float > *deskList1 = globalFrameList[firstIndex]->siftDesc;
		vector<float > *deskList2 = globalFrameList[secondIndex]->siftDesc;

		vector< int> &pointIdList1 = globalFrameList[firstIndex]->pointId;
		vector< int> &pointIdList2 = globalFrameList[secondIndex]->pointId;

		vector<bool> &valList1 = globalFrameList[firstIndex]->pointValidity;
		vector<bool> &valList2 = globalFrameList[secondIndex]->pointValidity;

		int num1 = keyList1->size();
		int num2 = keyList2->size();

		//Set descriptors to match, the first argument must be either 0 or 1
		//if you want to use more than 4096 or less than 4096
		//call matcher->SetMaxSift() to change the limit before calling setdescriptor
		matcher->SetDescriptors(0, num1, &(*deskList1)[0]); //image 1
		matcher->SetDescriptors(1, num2, &(*deskList2)[0]); //image 2

		//match and get result.
		int (*match_buf)[2] = new int[num1][2];
		//use the default thresholds. Check the declaration in SiftGPU.h

		int num_match = matcher->GetSiftMatch(num1, match_buf);
		std::cout << num_match << " sift matches were found;\n";

		if ( num_match <= 0 )
		{
			fprintf(fpSiftMatch, "%d %d 0\n", firstIndex + 1, secondIndex + 1);
			continue;
		}

		//Correspondences for five point
		vector<FeatureCorrespondence> correspondences;
		vector<pair<int,int> > remap;

		int nDuplicate = 0;
		fprintf(fpSiftMatch, "%d %d %d\n", firstIndex + 1, secondIndex + 1, num_match);

		//enumerate all the feature matches
		for(int i  = 0; i < num_match; ++i)
		{
			//How to get the feature matches:
			SiftGPU::SiftKeypoint &key1 = (*keyList1)[match_buf[i][0]];
			SiftGPU::SiftKeypoint &key2 = (*keyList2)[match_buf[i][1]];
			//key1 in the first image matches with key2 in the second image

			fprintf(fpSiftMatch, "%d %f %f %d %f %f\n", 
				pointIdList1[i], key1.x, key1.y, pointIdList2[i], key2.x, key2.y);

			FeatureCorrespondence tmp;
			tmp.feature1.x() = key1.x - centreList[firstIndex].first;
			tmp.feature1.y() = key1.y - centreList[firstIndex].second;
			tmp.feature2.x() = key2.x - centreList[secondIndex].first;
			tmp.feature2.y() = key2.y - centreList[secondIndex].second;

			bool duplicate = false;

			for ( int tIndex = 0; tIndex < correspondences.size(); tIndex ++ )
			{
				FeatureCorrespondence tskp = correspondences[tIndex];

				if ( (int)(tmp.feature1.x() * 100) == (int)(tskp.feature1.x() * 100)  && 
				     (int)(tmp.feature1.y() * 100) == (int)(tskp.feature1.y() * 100) ) 
				{
					duplicate = true;
				}

				if ( (int)(tmp.feature2.x() * 100) == (int)(tskp.feature2.x() * 100)  && 
				     (int)(tmp.feature2.y() * 100) == (int)(tskp.feature2.y() * 100) ) 
				{
					duplicate = true;
				}
			}

			if ( duplicate == false)
			{
				correspondences.push_back(tmp);
				remap.push_back(make_pair(match_buf[i][0], match_buf[i][1]));
			}else
			{
				nDuplicate ++;
			}

		}

		printf("\n%d/%d duplicate matches\n", nDuplicate, num_match);

		TwoViewInfo twoview_info;
		std::vector<int> inlier_indices;
		GetPoseChange5Point(correspondences, avgFocalList[firstIndex], avgFocalList[secondIndex], twoview_info, inlier_indices, fpR, fpT, fpE);

		printf("Inliers: %d\n", inlier_indices.size());


		if ( inlier_indices.size() <= INLIER_TH )
		{
			continue;
		}

		fprintf(fpPairs, "%d %d\n", firstIndex + 1, secondIndex + 1);

		std::cout << inlier_indices.size() << " inliers found;\n";
		double costFactor = (avgFocalList[firstIndex] * avgFocalList[secondIndex] * twoview_info.cost) / ((double)inlier_indices.size() * SAMPSON_ERR_5P);

		fprintf(fpGraph, "%d %d %0.10lf %d\n", firstIndex, secondIndex, costFactor, inlier_indices.size());

#if 0
		for (int i = 0; i < inlier_indices.size(); i ++)
		{
			int loc = inlier_indices[i];
			int index1 = remap[loc].first;
			int index2 = remap[loc].second;

			printf("[%d:%d %d:%d] [%d %d]\n", firstIndex, index1, secondIndex, index2, pointIdList1[index1], pointIdList2[index2]);

			if ( valList1[index1] == false && valList2[index2] == false )
			{
				printf("1. %d:%d -> %d:%d:%d\n", firstIndex, pointIdList1[index1], secondIndex, index2, pointIdList2[index2]);

				pointIdList2[index2] = pointIdList1[index1];
				valList1[index1] = true;
				valList2[index2] = true;
			}else if ( valList1[index1] == true && valList2[index2] == false )
			{
				printf("2. %d:%d -> %d:%d:%d\n", firstIndex, pointIdList1[index1], secondIndex, index2, pointIdList2[index2]);

				pointIdList2[index2] = pointIdList1[index1];
				valList2[index2] = true;
			}else if ( valList1[index1] == false && valList2[index2] == true )
			{
				printf("3. %d:%d <- %d:%d:%d\n", firstIndex, pointIdList1[index1], secondIndex, index2, pointIdList2[index2]);

				pointIdList1[index1] = pointIdList2[index2];
				valList1[index1] = true;
			}else if ( pointIdList1[index1] != pointIdList2[index2] )
			{
				int searchId = pointIdList2[index2];
				pointIdList2[index2] = pointIdList1[index1];

				if ( searchId != pointIdList1[index1] )
				{
					for ( int j = 0; j < globalFrameList.size(); j ++ )
					{
						for ( int k = 0; k < globalFrameList[j]->pointId.size(); k ++ )
						{
							if ( searchId == globalFrameList[j]->pointId[k] )
							{
								printf("4. %d:%d -> %d:%d:%d\n", firstIndex, pointIdList1[index1], j, k, globalFrameList[j]->pointId[k]);
								globalFrameList[j]->pointId[k] = pointIdList1[index1];
								//break;
							}
						}
					}
				}
			}

		}
#else
		for (int i = 0; i < inlier_indices.size(); i++)
		{
			int loc = inlier_indices[i];
			int index1 = remap[loc].first;
			int index2 = remap[loc].second;

			//printf("[%d:%d %d:%d] [%d %d]\n", firstIndex, index1, secondIndex, index2, pointIdList1[index1], pointIdList2[index2]);

			if ( valList1[index1] == false && valList2[index2] == false )
			{
				//printf("1. %d:%d -> %d:%d:%d\n", firstIndex, pointIdList1[index1], secondIndex, index2, pointIdList2[index2]);

				multimap<int, pair<int, int> >::iterator it;
				it = globalPtIndexList.find(pointIdList2[index2]);
				pair<int, int> indexPair = it->second;
				globalPtIndexList.erase (it);
				globalPtIndexList.insert(pair<int, pair<int, int> >(pointIdList1[index1], indexPair));

				pointIdList2[index2] = pointIdList1[index1];
				valList1[index1] = true;
				valList2[index2] = true;
			}else if ( valList1[index1] == true && valList2[index2] == false )
			{
				//printf("2. %d:%d -> %d:%d:%d\n", firstIndex, pointIdList1[index1], secondIndex, index2, pointIdList2[index2]);

				multimap<int, pair<int, int> >::iterator it;
				it = globalPtIndexList.find(pointIdList2[index2]);
				pair<int, int> indexPair = it->second;
				globalPtIndexList.erase (it);
				globalPtIndexList.insert(pair<int, pair<int, int> >(pointIdList1[index1], indexPair));

				pointIdList2[index2] = pointIdList1[index1];
				valList2[index2] = true;
			}else if ( valList1[index1] == false && valList2[index2] == true )
			{
				//printf("3. %d:%d <- %d:%d:%d\n", firstIndex, pointIdList1[index1], secondIndex, index2, pointIdList2[index2]);

				multimap<int, pair<int, int> >::iterator it;
				it = globalPtIndexList.find(pointIdList1[index1]);
				pair<int, int> indexPair = it->second;
				globalPtIndexList.erase (it);
				globalPtIndexList.insert(pair<int, pair<int, int> >(pointIdList2[index2], indexPair));

				pointIdList1[index1] = pointIdList2[index2];
				valList1[index1] = true;
			}else
			{
				int searchId = pointIdList2[index2];
				int replaceId = pointIdList1[index1];

				if ( searchId != replaceId )
				{
				    	pair< multimap<int, pair<int, int> >::iterator, multimap<int, pair<int, int> >::iterator> ret;
				    	ret = globalPtIndexList.equal_range(searchId);
				
					vector< pair<int, int> > indexPairList;

				    	for (multimap<int, pair<int, int> >::iterator it=ret.first; it!=ret.second; ++it)
					{
				      		pair<int, int> indexPair = it->second;

						//printf("4. %d:%d -> %d:%d:%d\n", firstIndex, pointIdList1[index1], indexPair.first, indexPair.second, globalFrameList[indexPair.first]->pointId[indexPair.second]);

						globalFrameList[indexPair.first]->pointId[indexPair.second] = replaceId;
						indexPairList.push_back(indexPair);
					}

					globalPtIndexList.erase (ret.first, ret.second);

					for (int itmp = 0; itmp < indexPairList.size(); ++ itmp)
					{
						globalPtIndexList.insert(pair<int, pair<int, int> >(replaceId, indexPairList[itmp]));
					}
				}
			}
		}
#endif

		// clean up..
		delete[] match_buf;

		printProgress(firstIndex, totalFrames);
		printf("\n");
	}

	printf("\n\nFinding matches...\n");

	fprintf(fpMatch, "%d\n", totalMatches);
	myfile1.open(matchFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");

		int firstIndex = stoi(nvmToks[0]);
		int secondIndex = stoi(nvmToks[1]);
		double confidenceFactor = stod(nvmToks[2]);

		if ( firstIndex >= secondIndex || confidenceFactor < VOCAB_TREE_TH)
		{
			fprintf(fpMatch, "%d %d 0\n", firstIndex + 1, secondIndex + 1);
			continue;
		}

		vector<pair<int, int> > indexPair;
		for ( int j = 0; j < globalFrameList[firstIndex]->pointId.size(); j ++ )
		{
			for ( int k = 0; k < globalFrameList[secondIndex]->pointId.size(); k ++ )
			{
				if ( globalFrameList[firstIndex]->pointId[j] == globalFrameList[secondIndex]->pointId[k] )
				{
					indexPair.push_back(make_pair(j, k));
					break;
				}
			}
		}

		vector<SiftGPU::SiftKeypoint> *keyList1 = globalFrameList[firstIndex]->siftKey;
		vector<SiftGPU::SiftKeypoint> *keyList2 = globalFrameList[secondIndex]->siftKey;

		int num_match = indexPair.size();

		fprintf(fpMatch, "%d %d %d\n", firstIndex + 1, secondIndex + 1, num_match);

		for ( int j = 0; j < num_match; j ++ )
		{
					fprintf(fpMatch, "%d %f %f %d %f %f\n", 
						globalFrameList[firstIndex]->pointId[indexPair[j].first], (*keyList1)[indexPair[j].first].x, (*keyList1)[indexPair[j].first].y, 
						globalFrameList[secondIndex]->pointId[indexPair[j].second], (*keyList2)[indexPair[j].second].x, (*keyList2)[indexPair[j].second].y);
		}

		printProgress(firstIndex, totalFrames);
	}
	myfile1.close();

	printf("\n\nSaving matches...\n");

	//Put number of images
	fprintf(fpResult, "%d\n", globalFrameList.size());
	siftLen = 128;
	for ( int index = 0; index < globalFrameList.size(); index ++ )
	{
		int pointCount1 = 0;

		for ( int sIndex = 0; sIndex < globalFrameList[index]->pointValidity.size(); sIndex ++ )
		{
			pointCount1 ++;
		}

		fprintf(fpResult, "%d %s %d\n", index + 1, globalFrameList[index]->siftFileName.c_str(), pointCount1);

		for ( int sIndex = 0; sIndex < globalFrameList[index]->pointValidity.size(); sIndex ++ )
		{
			fprintf(fpResult, "%d ",  globalFrameList[index]->pointId[sIndex]);

			SiftGPU::SiftKeypoint &skp = (*globalFrameList[index]->siftKey)[sIndex];

			fprintf(fpResult, "%f %f %f %f %d ", skp.y, skp.x, skp.s, skp.o, siftLen);

			int stIndex = sIndex * siftLen;
			for ( int index1 = 0; index1 < siftLen; index1 ++ )
			{
				int d = ((int)floor(0.5 + 512.0f * (*globalFrameList[index]->siftDesc)[stIndex + index1]));
				fprintf(fpResult, "%d ", d);
			}
			fprintf(fpResult, "\n");
		}

		printProgress(index, totalFrames);
	}
	printf("\n\nComplete\n");

	fclose(fpMatch);
	fclose(fpGraph);
	fclose(fpR);
	fclose(fpE);
	fclose(fpT);
	fclose(fpPairs);
	fclose(fpResult);
	fclose(fpSiftMatch);

	for ( int  index = 0; index < globalFrameList.size(); index ++ )
	{
		delete globalFrameList[index];
	}

	//delete sift;
	delete matcher;

	FREE_MYLIB(hsiftgpu);
	return 1;
}
