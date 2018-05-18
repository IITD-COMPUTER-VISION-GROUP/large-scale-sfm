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

#include "matrix.h"
#include "vector.h"

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

#define VOCAB_TREE_TH 	0.05
#define SAMPSON_ERR_5P 	1//2.25
#define INLIER_TH	15

#define CX	2304.0
#define CY	1536.0

typedef struct _Point2D
{
	double x;
	double y;
	int camId;

	_Point2D(double x1, double y1, int camId1): x(x1), y(y1), camId(camId1)
	{}

}Point2D;

typedef struct _frame
{
	int 		frameIndex;
	string 		imgFileName;
	string 		siftFileName;
	bool		isValid;

	vector<int> 				pointIdList;
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

//CHANGE
static int global_num_points;
static double *global_Rs = NULL;
static double *global_ts = NULL;
static v2_t *global_ps = NULL;

/* Project a point onto an image */
static v2_t project(double *R, double *t0, double *P)
{
    double tmp[3], tmp2[3];
    v2_t result;

    /* Rigid transform */
    matrix_product331(R, P, tmp);
    matrix_sum(3, 1, 3, 1, tmp, t0, tmp2);

    /* Perspective division */
    Vx(result) = tmp2[0] / tmp2[2];
    Vy(result) = tmp2[1] / tmp2[2];

    return result;
}

static void triangulate_n_residual(const int *m, const int *n, double *x, double *fvec, double *iflag)
{
    int i;

    for(i = 0; i < global_num_points; i++)
    {
        int Roff = 9 * i;
        int toff = 3 * i;

        /* Project the point into the view */
        v2_t p = project(global_Rs + Roff, global_ts + toff, x);

        fvec[2 * i + 0] = Vx(global_ps[i]) - Vx(p);
        fvec[2 * i + 1] = Vy(global_ps[i]) - Vy(p);
    }
}

/* Find the point with the smallest squared projection error */
static v3_t triangulate_n(int num_points, v2_t *p, double *R, double *t, double *error_out,double *error_out1)
{
    int num_eqs = 2 * num_points;
    int num_vars = 3;

    double *A = (double *) malloc(sizeof(double) * num_eqs * num_vars);
    double *b = (double *) malloc(sizeof(double) * num_eqs);
    double *x = (double *) malloc(sizeof(double) * num_vars);

    int i;
    double error;

    v3_t r;

    for(i = 0; i < num_points; i++)
    {
        int Roff = 9 * i;
        int row = 6 * i;
        int brow = 2 * i;
        int toff = 3 * i;

        A[row + 0] = R[Roff + 0] - Vx(p[i]) * R[Roff + 6];
        A[row + 1] = R[Roff + 1] - Vx(p[i]) * R[Roff + 7];
        A[row + 2] = R[Roff + 2] - Vx(p[i]) * R[Roff + 8];

        A[row + 3] = R[Roff + 3] - Vy(p[i]) * R[Roff + 6];
        A[row + 4] = R[Roff + 4] - Vy(p[i]) * R[Roff + 7];
        A[row + 5] = R[Roff + 5] - Vy(p[i]) * R[Roff + 8];

        b[brow + 0] = t[toff + 2] * Vx(p[i]) - t[toff + 0];
        b[brow + 1] = t[toff + 2] * Vy(p[i]) - t[toff + 1];
    }

    /* Find the least squares result */
    dgelsy_driver(A, b, x, num_eqs, num_vars, 1);

    error = 0.0;
    for(i = 0; i < num_points; i++)
    {
        double dx, dy;
        int Roff = 9 * i;
        int toff = 3 * i;
        double pp[3];

        /* Compute projection error */
        matrix_product331(R + Roff, x, pp);
        pp[0] += t[toff + 0];
        pp[1] += t[toff + 1];
        pp[2] += t[toff + 2];

        dx = pp[0] / pp[2] - Vx(p[i]);
        dy = pp[1] / pp[2] - Vy(p[i]);

        error += (dx * dx + dy * dy);
    }

    error = sqrt(error) / num_points;

    // printf("[triangulate_n] Error [before polishing]: %0.3e\n", error);

    /* Run a non-linear optimization to refine the result */
    global_num_points = num_points;
    global_ps = p;
    global_Rs = R;  global_ts = t;
    lmdif_driver((void *)triangulate_n_residual, num_eqs, num_vars, x, 1.0e-5);

    error = 0.0;
    for(i = 0; i < num_points; i++)
    {
        double dx, dy;
        int roff = 9 * i;
        int toff = 3 * i;
        double pp[3];

        ///* compute projection error
        matrix_product331(R + roff, x, pp);
        pp[0] += t[toff + 0];
        pp[1] += t[toff + 1];
        pp[2] += t[toff + 2];

        dx = pp[0] / pp[2] - Vx(p[i]);
        dy = pp[1] / pp[2] - Vy(p[i]);

        error_out1[i] = sqrt(dx * dx + dy * dy);

        error += (dx * dx + dy * dy);

        if ( pp[2] < 0.0001 )
        {
            error = 10000000.0;
        }
    }

    error = sqrt(error) / num_points;

    //   // printf("[triangulate_n] Error [after polishing]: %0.3e\n", error);

    if(error_out != NULL)
    {
        *error_out = error;
    }

    r = v3_new(x[0], x[1], x[2]);

    free(A);
    free(b);
    free(x);

    return r;
}

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

static void GetPoseChange5Point(vector<FeatureCorrespondence> &correspondences, double avgFocal,
		TwoViewInfo &twoview_info, vector<int> &inlier_indices, FILE *fpR, FILE *fpT, FILE *fpE)
{
	VerifyTwoViewMatchesOptions options;

	options.bundle_adjustment = false;
	options.min_num_inlier_matches = 10;
	options.estimate_twoview_info_options.max_sampson_error_pixels = SAMPSON_ERR_5P;
	options.estimate_twoview_info_options.max_ransac_iterations = 1000;

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

	fprintf(fpR, "%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n", twoview_info.rotationmat_2(0, 0), twoview_info.rotationmat_2(0, 1), twoview_info.rotationmat_2(0, 2), twoview_info.rotationmat_2(1, 0), twoview_info.rotationmat_2(1, 1), twoview_info.rotationmat_2(1, 2), twoview_info.rotationmat_2(2, 0), twoview_info.rotationmat_2(2, 1), twoview_info.rotationmat_2(2, 2));
	fprintf(fpE, "%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n", twoview_info.essential_mat(0, 0), twoview_info.essential_mat(0, 1), twoview_info.essential_mat(0, 2), twoview_info.essential_mat(1, 0), twoview_info.essential_mat(1, 1), twoview_info.essential_mat(1, 2), twoview_info.essential_mat(2, 0), twoview_info.essential_mat(2, 1), twoview_info.essential_mat(2, 2));
	fprintf(fpT, "%.10lf %.10lf %.10lf\n", twoview_info.translation_2(0), twoview_info.translation_2(1), twoview_info.translation_2(2));

}

static void Run5Point(vector<FeatureCorrespondence> &correspondences, double avgFocal1, double avgFocal2,
		TwoViewInfo &twoview_info, vector<int> &inlier_indices, double *rMat, double *tMat, double *eMat)
{
	VerifyTwoViewMatchesOptions options;

	options.bundle_adjustment = false;
	options.min_num_inlier_matches = 10;
	options.estimate_twoview_info_options.max_sampson_error_pixels = SAMPSON_ERR_5P;

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

	string inlierFile = outDir + "/matches_forRtinlier5point.txt";
	string graphFile = outDir + "/graph.txt";
	string rFile = outDir + "/R5Point.txt";
	string tFile = outDir + "/T5Point.txt";
	string eFile = outDir + "/E5Point.txt";
	string orgPair = outDir + "/original_pairs5point.txt";
	string consolidedFile = outDir + "/consolided_result.txt";
	string clusterListFile = outDir + "/dendogram/clustersizefile.txt";
	string keyListFile = outDir + "/KeyList.txt";

	vector<Frame *> globalFrameList;
	int frameCount = 0;

	//Read consolided_result.txt
	ifstream myfile1;
	string line;
	vector <string> nvmToks;

	myfile1.open(consolidedFile.c_str());

	//Read no of images
	getline(myfile1, line);
	split(nvmToks, line, " ");
	int totalFrames = stoi(nvmToks[0]);

#if 1

	/*if (! getline(myfile1, line) )
	{
		printf("Unable to read line\n");
		return -1;
	}

	split(nvmToks, line, " ");

	cout << line << endl;

	int imgIndex = stoi(nvmToks[0]);
	const char *imgName = nvmToks[1].c_str();
	int nSift = stoi(nvmToks[2]);

	for ( int index = 0; index < totalFrames; index ++ )
	{
		int c = 0;		
		while ( getline(myfile1, line) )
		{
			split(nvmToks, line, " ");

			if ( nvmToks.size() == 3 ) 
			{
				printf("%d %s %d %d\n", imgIndex, imgName, nSift, c);
				imgIndex = stoi(nvmToks[0]);
				imgName = nvmToks[1].c_str();
				nSift = stoi(nvmToks[2]);
				break;
			}
			c++;	
		}

	}
	return 1;*/

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

			int pointId = stoi(nvmToks[0]);

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
#else

	printf("Loading SIFT features...\n");

	for ( int index = 0; index < totalFrames; index ++ )
	{
		if (! getline(myfile1, line) )
		{
			printf("Unable to read line\n");
			return -1;
		}

		split(nvmToks, line, " ");

		int imgIndex = stoi(nvmToks[0]);
		const char *imgName = nvmToks[1].c_str();
		int nSift = stoi(nvmToks[2]);

		Frame *newFrame = new Frame;
		newFrame->frameIndex = imgIndex;

		vector<float > *singleDesc = new vector<float>;
		vector<SiftGPU::SiftKeypoint> *singleKey = new vector<SiftGPU::SiftKeypoint>;

		for ( int sIndex = 0; sIndex < nSift; sIndex ++ )
		{
			if (! getline(myfile1, line) )
			{
				printf("Unable to read line\n");
				return -1;
			}

			split(nvmToks, line, " ");

			int pointId = stoi(nvmToks[0]);

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
#endif

	printf("\n\nLoading matches...\n");
	printProgress(1, 3);

	vector<Match *>	globalMatchList;

	myfile1.open(orgPair.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");
		Match *newMatch = new Match;

		newMatch->firstFrameIndex = stoi(nvmToks[0]);
		newMatch->secondFrameIndex = stoi(nvmToks[1]);

		globalMatchList.push_back(newMatch);
	}
	myfile1.close();

	int mIndex = 0;
	myfile1.open(eFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");
		Match *newMatch = globalMatchList[mIndex ++];

		for ( int i = 0; i < 9; i ++ )
		{
			newMatch->eMat[i] = stod(nvmToks[i]);
		}
	}
	myfile1.close();

	printProgress(2, 3);

	mIndex = 0;
	myfile1.open(rFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");
		Match *newMatch = globalMatchList[mIndex ++];

		for ( int i = 0; i < 9; i ++ )
		{
			newMatch->rotMat[i] = stod(nvmToks[i]);
		}
	}
	myfile1.close();

	mIndex = 0;
	myfile1.open(tFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");
		Match *newMatch = globalMatchList[mIndex ++];

		for ( int i = 0; i < 3; i ++ )
		{
			newMatch->tMat[i] = stod(nvmToks[i]);
		}

		getCfromRT(newMatch->rotMat, newMatch->tMat, newMatch->cMat);
	}
	myfile1.close();

	//Read list focal
	string focalFile = outDir + "/list_focal.txt";
	vector<double> avgFocalList;
	myfile1.open(focalFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");

		double focal = stod(nvmToks[2]);
		avgFocalList.push_back(focal);
	}
	myfile1.close();

	//Read CX,CY
	vector<pair<float, float> > centreList;
	myfile1.open(keyListFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");
		float cx = stof(nvmToks[2]) / 2.0;
		float cy = stof(nvmToks[1]) / 2.0;
		centreList.push_back(make_pair(cx, cy));
	}
	myfile1.close();

	printProgress(3, 3);

	printf("\n\nLoading cluster details and expanding matches...\n");

	vector<string> clusterFileNameList;
	myfile1.open(clusterListFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");
		clusterFileNameList.push_back(nvmToks[0]);
		//CHANGE
	}
	myfile1.close();

	//must call once
	matcher->VerifyContextGL();

	vector<vector<Match *>*> clusterWiseMatchList;

	for ( int cIndex = 0; cIndex < clusterFileNameList.size(); cIndex ++)
	{
		string fileName = outDir + "/dendogram/" + clusterFileNameList[cIndex] + ".txt";
		string lfFile = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/listsize_focal1.txt";
		string clmFile = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/cluster_list_map.txt";

		FILE *fpLF = fopen(lfFile.c_str(), "w");
		FILE *fpCLM = fopen(clmFile.c_str(), "w");

		vector<int> indexList;
		vector<Match *> *clusterWiseMatch = new vector<Match *>;

		myfile1.open(fileName.c_str());
		while(getline(myfile1, line))
		{
			split(nvmToks, line, " ");
			indexList.push_back(stoi(nvmToks[0]));
		}
		myfile1.close();

		std::sort (indexList.begin(), indexList.end());

		for (int i = 0; i < indexList.size(); i ++)
		{
			printf("%d\n", indexList[i]);
		}

		int nMatches = globalMatchList.size();
		//Search matches for this cluster
		for ( int mIndex = 0; mIndex < nMatches; mIndex ++ )
		{
			Match *match = globalMatchList[mIndex];

			bool f1 = false, f2 = false;

			for (int i = 0; i < indexList.size(); i ++)
			{
				if ( match->firstFrameIndex == indexList[i] )
				{
					f1 = true;
				}
				if ( match->secondFrameIndex == indexList[i] )
				{
					f2 = true;
					break;
				}
			}

			if ( f1 && f2 )
			{
				clusterWiseMatch->push_back(match);

				//
				/*int firstIndex = match->firstFrameIndex - 1;
				int secondIndex = match->secondFrameIndex - 1;

				vector<SiftGPU::SiftKeypoint> *keyList1 = globalFrameList[firstIndex]->siftKey;
				vector<SiftGPU::SiftKeypoint> *keyList2 = globalFrameList[secondIndex]->siftKey;

				vector<float > *deskList1 = globalFrameList[firstIndex]->siftDesc;
				vector<float > *deskList2 = globalFrameList[secondIndex]->siftDesc;

				vector<long long int> &pointIdList1 = globalFrameList[firstIndex]->pointIdList;
				vector<long long int> &pointIdList2 = globalFrameList[secondIndex]->pointIdList;

				vector<bool> &valList1 = globalFrameList[firstIndex]->pointValidity;
				vector<bool> &valList2 = globalFrameList[secondIndex]->pointValidity;

				int num1 = keyList1->size();
				int num2 = keyList2->size();

				int nMatches = 0;
				
				for ( int l = 0; l < pointIdList1.size(); l ++ )
				{
					for ( int m = 0; m < pointIdList2.size(); m ++ )
					{
						if ( pointIdList1[l] == pointIdList2[m] )
						{
							nMatches ++;
							break;
						}
					}
				}

				if ( nMatches <= INLIER_TH )
				{
					continue;
				}

				
				fprintf(fpInliers,"%d %d %d\n", firstIndex + 1, secondIndex + 1, nMatches);
			
				for ( int l = 0; l < pointIdList1.size(); l ++ )
				{
					for ( int m = 0; m < pointIdList2.size(); m ++ )
					{
						if ( pointIdList1[l] == pointIdList2[m] )
						{
							SiftGPU::SiftKeypoint &key1 = (*keyList1)[l];
							SiftGPU::SiftKeypoint &key2 = (*keyList2)[m];

							fprintf(fpInliers, "%lld %lf %lf %lld %lf %lf\n", 
							pointIdList1[l], key1.x - CX, key1.y - CY,
							pointIdList2[m], key2.x - CX, key2.y - CY);

							break;
						}
					}
				}*/
			}
		}

		//Create new matches
		int nPriorMatches = clusterWiseMatch->size();

		for (int i = 0; i < indexList.size(); i ++)
		{
			fprintf(fpLF, "%d 0 %f\n", indexList[i] - 1, avgFocalList[indexList[i] - 1]);
			fprintf(fpCLM, "%d\n", indexList[i] - 1);

			for (int j = i + 1; j < indexList.size()/*( i + 10 >= indexList.size() ? indexList.size() : i + 10)*/; j ++)
			{
				int firstIndex = indexList[i] - 1;
				int secondIndex = indexList[j] - 1;

				vector<SiftGPU::SiftKeypoint> *keyList1 = globalFrameList[firstIndex]->siftKey;
				vector<SiftGPU::SiftKeypoint> *keyList2 = globalFrameList[secondIndex]->siftKey;

				vector<float > *deskList1 = globalFrameList[firstIndex]->siftDesc;
				vector<float > *deskList2 = globalFrameList[secondIndex]->siftDesc;

				vector<int> &pointIdList1 = globalFrameList[firstIndex]->pointIdList;
				vector<int> &pointIdList2 = globalFrameList[secondIndex]->pointIdList;

				vector<bool> &valList1 = globalFrameList[firstIndex]->pointValidity;
				vector<bool> &valList2 = globalFrameList[secondIndex]->pointValidity;

				int num1 = keyList1->size();
				int num2 = keyList2->size();

				bool f1 = false;
				for ( int k = 0; k < nPriorMatches; k ++ )
				{
					if ( indexList[i] == (*clusterWiseMatch)[k]->firstFrameIndex &&
							indexList[j] == (*clusterWiseMatch)[k]->secondFrameIndex)
					{
						f1 = true;
						break;
					}
				}

				if ( true == f1 ) 
				{
					continue;
				}

				//Create new match for thus cluster
				Match *match = new Match;
				match->firstFrameIndex = indexList[i];
				match->secondFrameIndex = indexList[j];

				//Set descriptors to match, the first argument must be either 0 or 1
				//if you want to use more than 4096 or less than 4096
				//call matcher->SetMaxSift() to change the limit before calling setdescriptor
				matcher->SetDescriptors(0, num1, &(*deskList1)[0]); //image 1
				matcher->SetDescriptors(1, num2, &(*deskList2)[0]); //image 2

				//match and get result.
				int (*match_buf)[2] = new int[num1][2];
				//use the default thresholds. Check the declaration in SiftGPU.h
				int num_match = matcher->GetSiftMatch(num1, match_buf);
				//std::cout << num_match << " sift matches were found;\n";

				if ( num_match <= INLIER_TH )
				{
					continue;
				}

				//Correspondences for five point
				vector<FeatureCorrespondence> correspondences;

				//enumerate all the feature matches
				for(int i  = 0; i < num_match; ++i)
				{
					//How to get the feature matches:
					SiftGPU::SiftKeypoint &key1 = (*keyList1)[match_buf[i][0]];
					SiftGPU::SiftKeypoint &key2 = (*keyList2)[match_buf[i][1]];
					//key1 in the first image matches with key2 in the second image

					FeatureCorrespondence tmp;
					tmp.feature1.x() = key1.x - centreList[firstIndex].first;
					tmp.feature1.y() = key1.y - centreList[firstIndex].second;
					tmp.feature2.x() = key2.x - centreList[secondIndex].first;
					tmp.feature2.y() = key2.y - centreList[secondIndex].second;
					correspondences.push_back(tmp);
				}

				TwoViewInfo twoview_info;
				std::vector<int> inlier_indices;
				Run5Point(correspondences, avgFocalList[firstIndex], avgFocalList[secondIndex], twoview_info, inlier_indices, match->rotMat, match->tMat, match->eMat);
				getCfromRT(match->rotMat, match->tMat, match->cMat);

				if ( inlier_indices.size() < INLIER_TH )
				{
					delete match;
					continue;
				}

				for (int k = 0; k < inlier_indices.size(); k ++)
				{
					int loc = inlier_indices[k];
					int pointId1 = pointIdList1[match_buf[loc][0]];
					int pointId2 = pointIdList2[match_buf[loc][1]];

					if ( pointId1 == pointId2 ) continue;

					for ( int l = 0; l < globalFrameList.size(); l ++ )
					{
						for ( int m = 0; m < globalFrameList[l]->pointIdList.size(); m ++ )
						{
							if ( pointId2 == globalFrameList[l]->pointIdList[m] )
							{
								globalFrameList[l]->pointIdList[m] = pointId1;
								break;
							}
						}
					}

				}

				//CHANGE
#if 0
					//Dump files for optimization
					string newRT_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/RTglobalmapped_for_optimisation.txt";
					string newF_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/focal_for_optimisation.txt";
					string newM_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/Invmap.txt";
					string newOurs_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/ours_new.txt";

					FILE *fpR = fopen(newRT_file.c_str(), "w");
					FILE *fpF = fopen(newF_file.c_str(), "w");
					FILE *fpM = fopen(newM_file.c_str(), "w");
					FILE *fpOurs = fopen(newOurs_file.c_str(), "w");

					fprintf(fpR, "1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n0.0 0.0 0.0\n");
					fprintf(fpR, "%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", 
						match->rotMat[0], match->rotMat[1], match->rotMat[2], 
						match->rotMat[3], match->rotMat[4], match->rotMat[5],
						match->rotMat[6], match->rotMat[7], match->rotMat[8]);
					fprintf(fpR, "%.9lf %.9lf %.9lf\n", match->cMat[0], match->cMat[1], match->cMat[2]);
					
					fprintf(fpF, "2\n1 %.9lf 2\n2 %.9lf 2\n",avgFocalList[firstIndex], avgFocalList[secondIndex]);
					fprintf(fpM, "0 1\n1 2\n");

					fclose(fpR);
					fclose(fpF);
					fclose(fpM);


					for (int k = 0; k < inlier_indices.size(); k ++)
					{
						int loc = inlier_indices[k];
						long long int pointId1 = pointIdList1[match_buf[loc][0]];
						long long int pointId2 = pointIdList2[match_buf[loc][1]];

						v2_t pv[2];
						double Rs[9 * 2];
						double ts[3 * 2];
						cv::Point2f unDistPt;
  						double *error_array = (double*)calloc(2, sizeof(double));
						double error_curr = 0.0;

						pv[0].p[0] = correspondences[loc].feature1.x() / avgFocalList[firstIndex];
						pv[0].p[1] = correspondences[loc].feature1.y() / avgFocalList[firstIndex];
						pv[1].p[0] = correspondences[loc].feature2.x() / avgFocalList[secondIndex];
						pv[1].p[1] = correspondences[loc].feature2.y() / avgFocalList[secondIndex];

						Rs[0] = 1.0; Rs[1] = 0.0; Rs[2] = 0.0; 
						Rs[3] = 0.0; Rs[4] = 1.0; Rs[5] = 0.0; 
						Rs[6] = 0.0; Rs[7] = 0.0; Rs[8] = 1.0; 
						ts[0] = 0.0; ts[1] = 0.0; ts[2] = 0.0; 
						memcpy(Rs + 9, match->rotMat, 9 * sizeof(double));
						memcpy(ts + 3, match->tMat, 3 * sizeof(double));

						v3_t cloud_point = triangulate_n(2, pv, Rs, ts, &error_curr, error_array);

						if(error_curr < (0.01))
						{
							fprintf(fpOurs, "%lf %lf %lf 255 255 255 2 0 %lld %lf %lf 1 %lld %lf %lf\n", 
								cloud_point.p[0], cloud_point.p[1], cloud_point.p[2],
								pointId1, correspondences[loc].feature1.x(), correspondences[loc].feature1.y(),
								pointId2, correspondences[loc].feature2.x(), correspondences[loc].feature2.y());
						}

						if ( pointId1 == pointId2 ) continue;

						for ( int l = 0; l < globalFrameList.size(); l ++ )
						{
							for ( int m = 0; m < globalFrameList[l]->pointIdList.size(); m ++ )
							{
								if ( pointId2 == globalFrameList[l]->pointIdList[m] )
								{
									globalFrameList[l]->pointIdList[m] = pointId1;
									break;
								}
							}
						}
					}

					fclose(fpOurs);	
					string bundlerCmd = "./bin/optimizer_tmp " + outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/ 1 50 16 32 0.5 2";
					cout << bundlerCmd << endl;
					system(bundlerCmd.c_str());

					//Read NVM
					string nvmFile = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/outputVSFM_GB.nvm";	
					ifstream myfile;

					//Read nvm file
					myfile.open(nvmFile.c_str());

					//Read Header
					getline(myfile, line);

					//Read number of camera
					getline(myfile, line);
					int nCams1 = stoi(line);

					if ( nCams1 == 2 )
					{
						Eigen::Matrix3d R1, R2, R12;
						Eigen::Vector3d T1, T2, C1, C2, T12, C12;

						getline(myfile, line);
						split(nvmToks, line, " ");

						R1 << stod(nvmToks[2]), stod(nvmToks[3]), stod(nvmToks[4]),
						     stod(nvmToks[5]), stod(nvmToks[6]), stod(nvmToks[7]),
						     stod(nvmToks[8]), stod(nvmToks[9]), stod(nvmToks[10]);
						
						T1 << stod(nvmToks[11]), stod(nvmToks[12]), stod(nvmToks[13]);

						getline(myfile, line);
						split(nvmToks, line, " ");

						R2 << stod(nvmToks[2]), stod(nvmToks[3]), stod(nvmToks[4]),
						     stod(nvmToks[5]), stod(nvmToks[6]), stod(nvmToks[7]),
						     stod(nvmToks[8]), stod(nvmToks[9]), stod(nvmToks[10]);
						
						T2 << stod(nvmToks[11]), stod(nvmToks[12]), stod(nvmToks[13]);

						R12 = R2 * R1.transpose();
						T12 = T2 - R2 * R1.transpose() * T1;

						printf("5Point:\nRij %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", 
							match->rotMat[0], match->rotMat[1], match->rotMat[2], 
							match->rotMat[3], match->rotMat[4], match->rotMat[5],
							match->rotMat[6], match->rotMat[7], match->rotMat[8]);
						printf("Tij %.9lf %.9lf %.9lf\n", match->tMat[0], match->tMat[1], match->tMat[2]);

						match->rotMat[0] = R12(0, 0);
						match->rotMat[1] = R12(0, 1);
						match->rotMat[2] = R12(0, 2);
						match->rotMat[3] = R12(1, 0);
						match->rotMat[4] = R12(1, 1);
						match->rotMat[5] = R12(1, 2);
						match->rotMat[6] = R12(2, 0);
						match->rotMat[7] = R12(2, 1);
						match->rotMat[8] = R12(2, 2);

						match->tMat[0] = T12[0]/T12.norm();
						match->tMat[1] = T12[1]/T12.norm();
						match->tMat[2] = T12[2]/T12.norm();

						printf("BA:\nRij %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", 
							match->rotMat[0], match->rotMat[1], match->rotMat[2], 
							match->rotMat[3], match->rotMat[4], match->rotMat[5],
							match->rotMat[6], match->rotMat[7], match->rotMat[8]);
						printf("Tij %.9lf %.9lf %.9lf\n", match->tMat[0], match->tMat[1], match->tMat[2]);

						C1 = -R1.transpose() * T1;
						C2 = -R2.transpose() * T2;
						C12 = C2 - C1;

						match->cMat[0] = C12[0]/C12.norm();
						match->cMat[1] = C12[1]/C12.norm();
						match->cMat[2] = C12[2]/C12.norm();

						//getCfromRT(match->rotMat, match->tMat, match->cMat);

						//getchar();
					}
					
				//END
#endif

				clusterWiseMatch->push_back(match);

				/*fprintf(fpInliers,"%d %d %d\n", firstIndex + 1, secondIndex + 1, inlier_indices.size());
				
				for (int k = 0; k < inlier_indices.size(); k ++)
				{
					int loc = inlier_indices[k];
					long long int pointId1 = pointIdList1[match_buf[loc][0]];
					long long int pointId2 = pointIdList2[match_buf[loc][1]];

					fprintf(fpInliers, "%lld %lf %lf %lld %lf %lf\n", 
							pointId1, correspondences[loc].feature1.x(), correspondences[loc].feature1.y(),
							pointId2, correspondences[loc].feature2.x(), correspondences[loc].feature2.y());

					for ( int l = 0; l < globalFrameList.size(); l ++ )
					{
						for ( int m = 0; m < globalFrameList[l]->pointIdList.size(); m ++ )
						{
							if ( pointId2 == globalFrameList[l]->pointIdList[m] )
							{
								globalFrameList[l]->pointIdList[m] = pointId1;
								break;
							}
						}
					}

				}*/

			}

		}
		fclose(fpLF);
		fclose(fpCLM);

		/*for ( int i = 0; i < clusterWiseMatch.size(); i ++ )
		{
			fprintf(fpPair, "%d %d\n", clusterWiseMatch[i]->firstFrameIndex, clusterWiseMatch[i]->secondFrameIndex);
			fprintf(fpRot, "%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n", clusterWiseMatch[i]->rotMat[0], clusterWiseMatch[i]->rotMat[1], clusterWiseMatch[i]->rotMat[2],
					clusterWiseMatch[i]->rotMat[3], clusterWiseMatch[i]->rotMat[4], clusterWiseMatch[i]->rotMat[5], clusterWiseMatch[i]->rotMat[6], clusterWiseMatch[i]->rotMat[7],
					clusterWiseMatch[i]->rotMat[8]);
			fprintf(fpE, "%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n", clusterWiseMatch[i]->eMat[0], clusterWiseMatch[i]->eMat[1], 
					clusterWiseMatch[i]->eMat[2], clusterWiseMatch[i]->eMat[3], clusterWiseMatch[i]->eMat[4], clusterWiseMatch[i]->eMat[5], 
					clusterWiseMatch[i]->eMat[6], clusterWiseMatch[i]->eMat[7], clusterWiseMatch[i]->eMat[8]);
			fprintf(fpT, "%.10lf %.10lf %.10lf\n", clusterWiseMatch[i]->tMat[0], clusterWiseMatch[i]->tMat[1], clusterWiseMatch[i]->tMat[2]);
			fprintf(fpC, "%.10lf %.10lf %.10lf\n", clusterWiseMatch[i]->cMat[0], clusterWiseMatch[i]->cMat[1], clusterWiseMatch[i]->cMat[2]);
		}*/
		
		clusterWiseMatchList.push_back(clusterWiseMatch);

		printProgress(cIndex+1, clusterFileNameList.size());
	}


	printf("\n\nDumping matches...\n");
	for ( int cIndex = 0; cIndex < clusterFileNameList.size(); cIndex ++)
	{
		string fileName = outDir + "/dendogram/" + clusterFileNameList[cIndex] + ".txt";
		vector<int> indexList;

		myfile1.open(fileName.c_str());
		while(getline(myfile1, line))
		{
			split(nvmToks, line, " ");
			indexList.push_back(stoi(nvmToks[0]));
		}
		myfile1.close();

		std::sort (indexList.begin(), indexList.end());

		vector<Match *> &clusterWiseMatch = *(clusterWiseMatchList[cIndex]);

		string pairFileTmp = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/original_pairs5point.txt";
		string rotFileTmp = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/R5point.txt";
		string tFileTmp = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/T5point.txt";
		string cFileTmp = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/C5point.txt";
		string eFileTmp = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/E5point.txt";
		string inlierFileTmp = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/matches_forRtinlier5point.txt";
		string worldFile = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/world1.txt";


		FILE *fpPair = fopen(pairFileTmp.c_str(), "w");
		FILE *fpRot = fopen(rotFileTmp.c_str(), "w");
		FILE *fpT = fopen(tFileTmp.c_str(), "w");
		FILE *fpC = fopen(cFileTmp.c_str(), "w");
		FILE *fpE = fopen(eFileTmp.c_str(), "w");
		FILE *fpInliers = fopen(inlierFileTmp.c_str(), "w");
		FILE *fpWorld = fopen(worldFile.c_str(), "w");

		vector<int> ptIdList;
		vector<vector<Point2D> *> trackList;

		int totalMatches = 0;
		for ( int i = 0; i < clusterWiseMatch.size(); i ++ )
		{
			int firstIndex = clusterWiseMatch[i]->firstFrameIndex - 1;
			int secondIndex = clusterWiseMatch[i]->secondFrameIndex - 1;

			vector<SiftGPU::SiftKeypoint> *keyList1 = globalFrameList[firstIndex]->siftKey;
			vector<SiftGPU::SiftKeypoint> *keyList2 = globalFrameList[secondIndex]->siftKey;

			vector<float > *deskList1 = globalFrameList[firstIndex]->siftDesc;
			vector<float > *deskList2 = globalFrameList[secondIndex]->siftDesc;

			vector<int> &pointIdList1 = globalFrameList[firstIndex]->pointIdList;
			vector<int> &pointIdList2 = globalFrameList[secondIndex]->pointIdList;

			vector<bool> &valList1 = globalFrameList[firstIndex]->pointValidity;
			vector<bool> &valList2 = globalFrameList[secondIndex]->pointValidity;

			int num1 = keyList1->size();
			int num2 = keyList2->size();

			int nMatches = 0;
			
			for ( int l = 0; l < pointIdList1.size(); l ++ )
			{
				for ( int m = 0; m < pointIdList2.size(); m ++ )
				{
					if ( pointIdList1[l] == pointIdList2[m] )
					{
						nMatches ++;
						break;
					}
				}
			}

			if ( nMatches <= INLIER_TH )
			{
				continue;
			}
			totalMatches ++;
		}
		fprintf(fpInliers,"%d\n", totalMatches);			

		for ( int i = 0; i < clusterWiseMatch.size(); i ++ )
		{
			int firstLocalId = 0, secondLocalId = 0; 

			for ( int j = 0; j < indexList.size(); j ++ )
			{
				if ( clusterWiseMatch[i]->firstFrameIndex == indexList[j] )  firstLocalId = j + 1;

				if ( clusterWiseMatch[i]->secondFrameIndex == indexList[j] )  secondLocalId = j + 1;
			}

			int firstIndex = clusterWiseMatch[i]->firstFrameIndex - 1;
			int secondIndex = clusterWiseMatch[i]->secondFrameIndex - 1;

			vector<SiftGPU::SiftKeypoint> *keyList1 = globalFrameList[firstIndex]->siftKey;
			vector<SiftGPU::SiftKeypoint> *keyList2 = globalFrameList[secondIndex]->siftKey;

			vector<float > *deskList1 = globalFrameList[firstIndex]->siftDesc;
			vector<float > *deskList2 = globalFrameList[secondIndex]->siftDesc;

			vector<int> &pointIdList1 = globalFrameList[firstIndex]->pointIdList;
			vector<int> &pointIdList2 = globalFrameList[secondIndex]->pointIdList;

			vector<bool> &valList1 = globalFrameList[firstIndex]->pointValidity;
			vector<bool> &valList2 = globalFrameList[secondIndex]->pointValidity;

			int num1 = keyList1->size();
			int num2 = keyList2->size();

			int nMatches = 0;
			
			for ( int l = 0; l < pointIdList1.size(); l ++ )
			{
				for ( int m = 0; m < pointIdList2.size(); m ++ )
				{
					if ( pointIdList1[l] == pointIdList2[m] )
					{
						nMatches ++;
						break;
					}
				}
			}

			if ( nMatches <= INLIER_TH )
			{
				continue;
			}

			fprintf(fpPair, "%d %d\n", firstLocalId, secondLocalId);
			fprintf(fpRot, "%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n", clusterWiseMatch[i]->rotMat[0], clusterWiseMatch[i]->rotMat[1], clusterWiseMatch[i]->rotMat[2],
					clusterWiseMatch[i]->rotMat[3], clusterWiseMatch[i]->rotMat[4], clusterWiseMatch[i]->rotMat[5], clusterWiseMatch[i]->rotMat[6], clusterWiseMatch[i]->rotMat[7],
					clusterWiseMatch[i]->rotMat[8]);
			fprintf(fpE, "%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n", clusterWiseMatch[i]->eMat[0], clusterWiseMatch[i]->eMat[1], 
					clusterWiseMatch[i]->eMat[2], clusterWiseMatch[i]->eMat[3], clusterWiseMatch[i]->eMat[4], clusterWiseMatch[i]->eMat[5], 
					clusterWiseMatch[i]->eMat[6], clusterWiseMatch[i]->eMat[7], clusterWiseMatch[i]->eMat[8]);
			fprintf(fpT, "%.10lf %.10lf %.10lf\n", clusterWiseMatch[i]->tMat[0], clusterWiseMatch[i]->tMat[1], clusterWiseMatch[i]->tMat[2]);
			fprintf(fpC, "%.10lf %.10lf %.10lf\n", clusterWiseMatch[i]->cMat[0], clusterWiseMatch[i]->cMat[1], clusterWiseMatch[i]->cMat[2]);

			fprintf(fpInliers,"%d %d %d\n", firstLocalId - 1, secondLocalId - 1, nMatches);	
		
			for ( int l = 0; l < pointIdList1.size(); l ++ )
			{
				for ( int m = 0; m < pointIdList2.size(); m ++ )
				{
					if ( pointIdList1[l] == pointIdList2[m] )
					{
						SiftGPU::SiftKeypoint &key1 = (*keyList1)[l];
						SiftGPU::SiftKeypoint &key2 = (*keyList2)[m];

						fprintf(fpInliers, "%lld %lf %lf %lld %lf %lf\n", 
						pointIdList1[l], key1.x - centreList[firstIndex].first, key1.y - centreList[firstIndex].second,
						pointIdList2[m], key2.x - centreList[secondIndex].first, key2.y - centreList[secondIndex].second);

						break;
					}
				}
			}
		}

		fclose(fpPair);
		fclose(fpRot);
		fclose(fpT);
		fclose(fpC);
		fclose(fpE);
		fclose(fpInliers);

		
		vector < int > removeList;

		for (int i = 0; i < indexList.size(); i ++)
		{
			int firstIndex = indexList[i] - 1;

			vector<SiftGPU::SiftKeypoint> *keyList1 = globalFrameList[firstIndex]->siftKey;
			vector<int> &pointIdList1 = globalFrameList[firstIndex]->pointIdList;

			for ( int j = 0; j < pointIdList1.size(); j ++ )
			{
				int searchId = pointIdList1[j];
				bool foundFlag = false;
				for ( int k = 0; k < ptIdList.size(); k ++ )
				{
					if ( searchId == ptIdList[k] )
					{
						foundFlag = true;
						bool camFoundF = false;
						for(int l = 0; l < trackList[k]->size(); l ++ )
						{
							if( (*(trackList[k]))[l].camId == firstIndex + 1 ) camFoundF = true;
						}

						if ( camFoundF == false )
						{
							trackList[k]->push_back(Point2D((*keyList1)[j].x - centreList[firstIndex].first, (*keyList1)[j].y - centreList[firstIndex].second, firstIndex + 1));
						}else
						{
							removeList.push_back(k);
						}

						break;
					}
				}
					
				if ( false == foundFlag )
				{
					vector<Point2D> *newVec = new vector<Point2D>;
					newVec->push_back(Point2D((*keyList1)[j].x - centreList[firstIndex].first, (*keyList1)[j].y - centreList[firstIndex].second, firstIndex + 1));
					trackList.push_back(newVec);
					ptIdList.push_back(searchId);
				}

			}
		}

		for( int tmpI = 0; tmpI < removeList.size(); tmpI ++ )
		{
			trackList[removeList[tmpI]] = NULL;
		}
		int removeCounter = 0;

		for ( int i = 0; i < ptIdList.size(); i ++ )
		{
			if ( trackList[i] != NULL )
			{
				if ( trackList[i]->size() >= 2 )
				{ 
					fprintf(fpWorld, "0.0 0.0 0.0 255 255 255 %d ", trackList[i]->size());
			
					for ( int j = 0; j < trackList[i]->size(); j ++ )
					{
						fprintf(fpWorld, "%d %lld %lf %lf ", (*(trackList[i]))[j].camId - 1, ptIdList[i], (*(trackList[i]))[j].x, (*(trackList[i]))[j].y);
					}

					fprintf(fpWorld, "\n");
				}
			}else
			{
				removeCounter ++;
			}
		}

		printf("%d/%d tracks removed due to duplicacy\n", removeCounter, ptIdList.size());

		fclose(fpWorld);

		printProgress(cIndex+1, clusterFileNameList.size());
	}

	printf("\n\n");
}


