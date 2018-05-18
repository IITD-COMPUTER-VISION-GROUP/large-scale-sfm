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
//using namespace cv;

#define FREE_MYLIB dlclose
#define GET_MYPROC dlsym
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#define VOCAB_TREE_TH 	0.05
#define SAMPSON_ERR_5P 	1.0
#define INLIER_TH	30
#define TH_2D3D		50

#define CLAMP(x,mn,mx) (((x) < mn) ? mn : (((x) > mx) ? mx : (x)))

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define RAD2DEG(r) ((r) * (180.0 / M_PI))


static int global_num_points;
static double *global_Rs = NULL;
static double *global_ts = NULL;
static v2_t *global_ps = NULL;

typedef struct _Point2D
{
	double x;
	double y;
	int camId;
	bool validity;

	_Point2D(){}
	_Point2D(double x1, double y1, int camId1): x(x1), y(y1), camId(camId1), validity(true)
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
	int 	camName;
	double 	rotMat[9];
	double 	tMat[3];
	double 	cMat[3];
	double 	focal;
	double 	k1;
	double 	k2;
	bool 	validity;
}Camera;

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

static cv::Point2f unDistortPoint(const cv::Point2f pt, double k1)
{
    if(k1 == 0)
        return pt;

    const double t2 = pt.y * pt.y;
    const double t3 = t2 * t2 * t2;
    const double t4 = pt.x * pt.x;
    const double t7 = k1 * (t2 + t4);

    if(k1 > 0)
    {
        const double t8 = 1.0 / t7;
        const double t10 = t3 / (t7 * t7);
        const double t14 = sqrt(t10 * (0.25 + t8 / 27.0));
        const double t15 = t2 * t8 * pt.y * 0.5;
        const double t17 = pow(t14 + t15, 1.0 / 3.0);
        const double t18 = t17 - t2 * t8 / (t17 * 3.0);
        return cv::Point2f(t18 * pt.x / pt.y, t18);
    }
    else
    {
        const double t9 = t3 / (t7 * t7 * 4.0);
        const double t11 = t3 / (t7 * t7 * t7 * 27.0);
        const std::complex<double> t12 = t9 + t11;
        const std::complex<double> t13 = sqrt(t12);
        const double t14 = t2 / t7;
        const double t15 = t14 * pt.y * 0.5;
        const std::complex<double> t16 = t13 + t15;
        const std::complex<double> t17 = pow(t16, 1.0 / 3.0);
        const std::complex<double> t18 = (t17 + t14 / (t17 * 3.0)) * std::complex<double>(0.0, sqrt(3.0));
        const std::complex<double> t19 = -0.5 * (t17 + t18) + t14 / (t17 * 6.0);
        return cv::Point2f(t19.real() * pt.x / pt.y, t19.real());
    }
}

static double ComputeRayAngle(v2_t p_norm, v2_t q_norm, double *R1, double *R2, double *t1, double *t2)
{
    double R1_inv[9], R2_inv[9];

    matrix_transpose(3, 3, (double *) R1, R1_inv);
    matrix_transpose(3, 3, (double *) R2, R2_inv);

    double p_w[3], q_w[3];

    double pv[3] = { Vx(p_norm), Vy(p_norm), 1.0 };
    double qv[3] = { Vx(q_norm), Vy(q_norm), 1.0 };

    double Rpv[3], Rqv[3];

    matrix_product331(R1_inv, pv, Rpv);
    matrix_product331(R2_inv, qv, Rqv);

    matrix_sum(3, 1, 3, 1, Rpv, (double *) t1, p_w);
    matrix_sum(3, 1, 3, 1, Rqv, (double *)t2, q_w);

    /* Subtract out the camera center */
    double p_vec[3], q_vec[3];
    matrix_diff(3, 1, 3, 1, p_w, (double *) t1, p_vec);
    matrix_diff(3, 1, 3, 1, q_w, (double *) t2, q_vec);

    /* Compute the angle between the rays */
    double dot;
    matrix_product(1, 3, 3, 1, p_vec, q_vec, &dot);

    double mag = matrix_norm(3, 1, p_vec) * matrix_norm(3, 1, q_vec);

    return acos(CLAMP(dot / mag, -1.0 + 1.0e-8, 1.0 - 1.0e-8));
}

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

    vector<double> errList;

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
	errList.push_back(sqrt(dx * dx + dy * dy));

        error += (dx * dx + dy * dy);

        if ( pp[2] < 0.0001 )
        {
            error = 10000000.0;
        }
    }

    error = sqrt(error) / num_points;

    std::sort (errList.begin(), errList.end());

    if(error_out != NULL && errList.size() > 0)
    {
        *error_out = error;
	//*error_out = errList[errList.size()/2];
    }

    r = v3_new(x[0], x[1], x[2]);

    free(A);
    free(b);
    free(x);

    return r;
}

int main(int argc, char *argv[])
{

	if ( argc < 2 )
	{
		printf("Invalid arguments\n ./a.out <Output DIR>\n");
		return 0;
	}

	string outDir = argv[1];

	string clusterListFile = outDir + "/dendogram/clustersizefile.txt";

	printf("\n\nLoading cluster details and expanding matches...\n");

	ifstream myfile1;
	string line;
	vector <string> nvmToks;

	vector<string> clusterFileNameList;
	myfile1.open(clusterListFile.c_str());
	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");
		clusterFileNameList.push_back(nvmToks[0]);
	}
	myfile1.close();

	for ( int cIndex = 0; cIndex < clusterFileNameList.size(); cIndex ++)
	{
		string fileName = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/list_focal.txt";

		cout << fileName << endl;
		vector<int> indexList;
			
		vector<Camera *> camList;		

		myfile1.open(fileName.c_str());
		while(getline(myfile1, line))
		{
			Camera *newCam = new Camera;
			split(nvmToks, line, " ");
			indexList.push_back(stoi(nvmToks[0]));

			newCam->camName = stoi(nvmToks[0]);
			newCam->validity = true;
			camList.push_back(newCam);
		}
		myfile1.close();
		std::sort (indexList.begin(), indexList.end());

		string RFile = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/R.txt";
		string CFile = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/C.txt";
		string WFile = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/world1.txt";
		string focalFile = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/list_focal.txt";
		string resecFile = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/resec.txt";

		myfile1.open(RFile.c_str());
		int tIndex = 0;
		while(getline(myfile1, line))
		{
			split(nvmToks, line, " ");

			for ( int i = 0; i < 9; i ++ ) 
			{
				camList[tIndex]->rotMat[i] = stod(nvmToks[i]);
			}

			tIndex ++;
		}
		myfile1.close();

		myfile1.open(CFile.c_str());
		tIndex = 0;
		while(getline(myfile1, line))
		{
			split(nvmToks, line, " ");
			double *tmpC = new double[3];
			
			for ( int i = 0; i < 3; i ++ ) 
			{			
				camList[tIndex]->cMat[i] = stod(nvmToks[i]);
			}

			getTfromRC(camList[tIndex]->rotMat, camList[tIndex]->cMat, camList[tIndex]->tMat);

			tIndex ++;
		}
		myfile1.close();
	
		//Read list focal
		vector<pair<int, int> > resecCamIndexList;
		myfile1.open(resecFile.c_str());
		while(getline(myfile1, line))
		{
			split(nvmToks, line, " ");

			int index = stoi(nvmToks[0]);
			resecCamIndexList.push_back(make_pair(index, -1));
		}
		myfile1.close();

		//Create map index list
		vector<int> mapIndex(indexList[indexList.size() - 1] + 1, -1);
		
		for ( int index = 0; index < indexList.size(); index ++ )
		{
			bool rFound = false;
			for ( int index1 = 0; index1 < resecCamIndexList.size(); index1 ++ )
			{
				if ( resecCamIndexList[index1].first == indexList[index] )
				{
					resecCamIndexList[index1].second = index;
					rFound = true;
					break;
				}
			}	

			if ( rFound == false )
			{
				mapIndex[indexList[index]] = index;
			}
		}

		//Read list focal
		vector<int> focalIndex;
		myfile1.open(focalFile.c_str());
		int maxIndex = 0;
		tIndex = 0;
		while(getline(myfile1, line))
		{
			split(nvmToks, line, " ");

			double focal = stod(nvmToks[2]);

			int index = stoi(nvmToks[0]);
			focalIndex.push_back(index);

			if ( maxIndex < index )
			{
				maxIndex = index;
			}
			camList[tIndex]->focal = focal;
			camList[tIndex]->k1 = 0;
			camList[tIndex]->k2 = 0;
			tIndex ++;
		}
		myfile1.close();

		vector<double> focalListLocal(maxIndex + 1, 0.0);
		for(tIndex = 0; tIndex < camList.size(); tIndex ++ )
		{
			focalListLocal[focalIndex[tIndex]] = camList[tIndex]->focal;

			printf("%d %lf %lf %lf %lf %lf %lf %lf\n", camList[tIndex]->camName, 
			camList[tIndex]->rotMat[0], camList[tIndex]->rotMat[1], camList[tIndex]->rotMat[2], 
			camList[tIndex]->cMat[0], camList[tIndex]->cMat[1], camList[tIndex]->cMat[2], camList[tIndex]->focal);
		}

		vector<PointAttr *> ptAttrList;
		vector<vector<Point2D> *> trackList;

		myfile1.open(WFile.c_str());
		while(getline(myfile1, line))
		{
			split(nvmToks, line, " ");
			
			PointAttr * newPtAttr = new PointAttr;
			vector<Point2D> * newList = new vector<Point2D>;

			newPtAttr->R = stoi(nvmToks[3]);
			newPtAttr->G = stoi(nvmToks[4]);
			newPtAttr->B = stoi(nvmToks[5]);
		
			int nView = stoi(nvmToks[6]);
			int prevCamId = -1;

			for ( int i = 0; i < nView; i ++ ) 
			{	Point2D newPt;

				newPt.camId = stol(nvmToks[7 + i * 4 + 0]);
				newPtAttr->pointId = stol(nvmToks[7 + i * 4 + 1]);
				newPt.x = stod(nvmToks[7 + i * 4 + 2]);
				newPt.y = stod(nvmToks[7 + i * 4 + 3]);

				if ( prevCamId == newPt.camId ) 
				{
					continue;
				}

				bool foundFlag = false;
				for(int tIndex = 0; tIndex < focalIndex.size(); tIndex ++ )
				{
					if ( focalIndex[tIndex] == newPt.camId )
					{
						foundFlag = true;
						break;
					}
				}

				if ( foundFlag == false) 
				{
					continue;
				}

				prevCamId = newPt.camId;
				newList->push_back(newPt);
			}

			trackList.push_back(newList);
			ptAttrList.push_back(newPtAttr);
		}
		myfile1.close();

		/************ 1st level Triangulation ***********************/
		vector<pair<int, int> > resecCamPointList;
		{
			cv::Point2f distPt, unDistPt;
			int count = 0, count1 = 0, count2 = 0, count3 = 0;
//			int = numpoint
			double sum = 0;
			int nProj_Sum = 0;
			for ( int pIndex = 0; pIndex < trackList.size(); pIndex ++ )
			{
				vector<Point2D> *list = trackList[pIndex];
				PointAttr *ptAttr = ptAttrList[pIndex];

				v2_t pv[list->size()];
				double Rs[9 * list->size()];
				double ts[3 * list->size()];
				double fs[list->size()];
				double ks[list->size()];

				int nProjToTriangulation = 0;
				vector <int> camIdIndex;
			
				double avgFocal = 0;

				for ( int vIndex = 0; vIndex < list->size(); vIndex ++ )
				{

					if ( mapIndex[(*list)[vIndex].camId] == -1 )
					{
						resecCamPointList.push_back(make_pair((*list)[vIndex].camId, ptAttr->pointId));
						continue;
					}

					double trans[3];
					distPt.x = (*list)[vIndex].x / focalListLocal[(*list)[vIndex].camId];
					distPt.y = (*list)[vIndex].y / focalListLocal[(*list)[vIndex].camId];
				
					avgFocal += focalListLocal[(*list)[vIndex].camId];

					unDistPt = unDistortPoint(distPt, 0);

			    		pv[nProjToTriangulation].p[0] = unDistPt.x;
					pv[nProjToTriangulation].p[1] = unDistPt.y;
		
					memcpy(Rs + 9 * nProjToTriangulation, camList[mapIndex[(*list)[vIndex].camId]]->rotMat, 9 * sizeof(double));
					memcpy(ts + 3 * nProjToTriangulation, camList[mapIndex[(*list)[vIndex].camId]]->tMat, 3 * sizeof(double));
					fs[nProjToTriangulation] = focalListLocal[(*list)[vIndex].camId];
					ks[nProjToTriangulation] = camList[mapIndex[(*list)[vIndex].camId]]->k1;

					camIdIndex.push_back(mapIndex[(*list)[vIndex].camId]);

					nProjToTriangulation ++;
				}

				avgFocal /= (double)nProjToTriangulation;

				if(nProjToTriangulation >= 3)
				{
				    count1 ++;

				    double *error_array = (double*)calloc(nProjToTriangulation, sizeof(double));
				    bool conditioned = false;
				    double mean = 0, std;
				    double error_curr = 0.0;

				    for(int m = 0; m < nProjToTriangulation; m++)
				    {
					for(int n = m + 1; n < nProjToTriangulation; n++)
					{
					    int index1 = camIdIndex[m];
					    int index2 = camIdIndex[n];

					    double val = ComputeRayAngle(pv[m], pv[n], camList[index1]->rotMat, camList[index2]->rotMat, camList[index1]->cMat, camList[index2]->cMat);
					    if(RAD2DEG(val) > 1.0)
					    {
						conditioned = true;
						break;
					    }
					}
				    }
				    
				    if(conditioned)
				    {
					count2 ++;
					v3_t cloud_point;

					cloud_point = triangulate_n(nProjToTriangulation, pv, Rs, ts, &error_curr, error_array);

					/*printf("%d\n", nProjToTriangulation);
					for ( int iT = 0; iT < nProjToTriangulation; iT ++ )
					{
						printf("\n%d %lf %lf\n", iT, pv[iT].p[0], pv[iT].p[1]);
						printf("%lf %lf %lf\n", Rs[iT * 9 + 0], Rs[iT * 9 + 1], Rs[iT * 9 + 2]);
						printf("%lf %lf %lf\n", ts[iT * 3 + 0], ts[iT * 3 + 1], ts[iT * 3 + 2]);
						printf("%lf\n", (error_array[iT] * error_array[iT] * fs[iT] * fs[iT])); 
					}*/

					if(error_curr < 0.01/*(10.0/avgFocal)*/)
					{
					    count3 ++;
					    std = 0;
					    mean = 0;

					    for(int ei = 0; ei < nProjToTriangulation; ei++)
					    {
						mean += error_array[ei];
						std = std + error_array[ei] * error_array[ei];
						sum += (error_array[ei] * error_array[ei] * fs[ei] * fs[ei]); 
					    }
					    nProj_Sum += nProjToTriangulation;
					    mean = mean / nProjToTriangulation;

					    std = std / nProjToTriangulation;

					    std = std - mean * mean;
					    std = sqrt(std);
					    
					    ptAttrList[pIndex]->X = cloud_point.p[0];
					    ptAttrList[pIndex]->Y = cloud_point.p[1];
					    ptAttrList[pIndex]->Z = cloud_point.p[2];
					    count++;
					}
				    }
				}
				camIdIndex.clear();
			}
			printf("%d %d %d [%d/%d] %lf\n\n", count1, count2, count3, count, trackList.size(), sum/(double)nProj_Sum);

			string newRT_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/RTglobalmapped_for_optimisation.txt";
			string newF_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/focal_for_optimisation.txt";
			string newM_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/Invmap.txt";
			string newOurs_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/ours_new.txt";
			string clusterDir = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/";
			string distortionFile = clusterDir + "/Distortion.txt";

			FILE *fpR = fopen(newRT_file.c_str(), "w");
			FILE *fpF = fopen(newF_file.c_str(), "w");
			FILE *fpM = fopen(newM_file.c_str(), "w");
			FILE *fpOurs = fopen(newOurs_file.c_str(), "w");
			FILE *fpDist = fopen(distortionFile.c_str(), "w");

			fprintf(fpF, "%d\n", indexList.size());

			for ( int index = 0; index < indexList.size(); index ++ )
			{
				fprintf(fpR, "%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", 
						camList[index]->rotMat[0], camList[index]->rotMat[1], camList[index]->rotMat[2],
						camList[index]->rotMat[3], camList[index]->rotMat[4], camList[index]->rotMat[5], 
						camList[index]->rotMat[6], camList[index]->rotMat[7], camList[index]->rotMat[8]);
				fprintf(fpR, "%.9lf %.9lf %.9lf\n", camList[index]->cMat[0], camList[index]->cMat[1], camList[index]->cMat[2]);

				fprintf(fpF, "%d %.9lf %d\n", indexList[index], camList[index]->focal, 2);
				fprintf(fpM, "%d %d\n", index, indexList[index]);
				fprintf(fpDist, "0.000\n");
			}
			fclose(fpR);
			fclose(fpF);
			fclose(fpM);
			fclose(fpDist);

			for ( int i = 0; i < trackList.size(); i ++ )
			{
				vector<Point2D> *list = trackList[i];

				if ( ptAttrList[i]->X != 0.0 || ptAttrList[i]->Y != 0.0 ||  ptAttrList[i]->Z != 0.0 )
				{

					int nProjToTriangulation = 0;
					for ( int j = 0; j < trackList[i]->size(); j ++ )
					{
						if ( mapIndex[(*list)[j].camId] == -1 ) continue;

						nProjToTriangulation ++;
					}

					if ( nProjToTriangulation > 0 )
					{
					
						fprintf(fpOurs, "%lf %lf %lf 255 255 255 %d ", ptAttrList[i]->X, ptAttrList[i]->Y, ptAttrList[i]->Z, nProjToTriangulation);
		
						for ( int j = 0; j < trackList[i]->size(); j ++ )
						{
							if ( mapIndex[(*list)[j].camId] == -1 ) continue;

							fprintf(fpOurs, "%d %lld %lf %lf ", mapIndex[(*list)[j].camId], ptAttrList[i]->pointId, (*list)[j].x, (*list)[j].y);
						}
			
						fprintf(fpOurs, "\n");
					}
				}
			}

			fclose(fpOurs);
		}

		//Run bundler
		string clusterDir = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/";
		//string bundlerCmd = "../bin/optimizer_all " + clusterDir + " -1 250 1600 3200 0.5 3";
		string bundlerCmd = "../bin/optimizer_all " + clusterDir + " 0 300 16 32 0.5 0";
		system(bundlerCmd.c_str());
		
		//getchar();

		/************ 2nd label Triangulation ***********************/
		{
			//Read first nvm file
			string nvmFile1 = clusterDir + "/outputVSFM_GB.nvm";

			//Camera and point list
			vector< Camera *> 		cameraList1;
			vector< PointAttr *> 		ptAttrList1;
			vector< vector<Point2D> *> 	trackList1;

			ReadNVM(nvmFile1, cameraList1, ptAttrList1, trackList1);


			//vector<int> mapIndex1(cameraList1[cameraList1.size() - 1]->camName + 1, -1);
			vector<int> mapIndex1(camList[camList.size() - 1]->camName + 1, -1);

			for ( int tIndex = 0; tIndex < cameraList1.size(); tIndex ++ )
			{
				mapIndex1[cameraList1[tIndex]->camName] = tIndex;
			}

			//Resection
			if ( 0 )
			{
				vector<cv::Point3f> objectPoints;
				vector<cv::Point2f> imagePoints;

				cv::Mat distCoeffs(4, 1, cv::DataType<double>::type);
				distCoeffs.at<double>(0) = 0;
				distCoeffs.at<double>(1) = 0;
				distCoeffs.at<double>(2) = 0;
				distCoeffs.at<double>(3) = 0;

				cv::Mat cameraMatrix(3, 3, cv::DataType<double>::type);
				cv::setIdentity(cameraMatrix);
				//Change
				cameraMatrix.at<double>(cv::Point(0, 0)) = camList[resecCamIndexList[0].second]->focal;
				cameraMatrix.at<double>(cv::Point(1, 1)) = camList[resecCamIndexList[0].second]->focal;

				bool useExtrinsicGuess = true;
				int iterationsCount = 500;
				float reprojectionError = 1.0;
				vector<double> inliers;

				for ( int index2 = 0; index2 < resecCamPointList.size(); index2 ++ )
				{
					for ( int index3 = 0; index3 < ptAttrList1.size(); index3 ++ )
					{
						if ( ptAttrList1[index3]->pointId == resecCamPointList[index2].second )
						{
							cv::Point3f tmp3Dpoints;
							cv::Point2f tmp2Dpoints;

							tmp3Dpoints.x = ptAttrList1[index3]->X;
							tmp3Dpoints.y = ptAttrList1[index3]->Y;
							tmp3Dpoints.z = ptAttrList1[index3]->Z;

							vector<Point2D> *list = trackList1[index3];
							for ( int vIndex = 0; vIndex < list->size(); vIndex ++ )
							{
								if ( (*list)[vIndex].camId == resecCamPointList[index2].first )
								{
									tmp2Dpoints.x = (*list)[vIndex].x;
									tmp2Dpoints.y = (*list)[vIndex].y;
									break;
								}
							}

							objectPoints.push_back(tmp3Dpoints);
							imagePoints.push_back(tmp2Dpoints);
							break;
						}
					}
				}


				cv::Mat rvec = cv::Mat(3, 3, CV_64F);
				cv::Mat tvec = cv::Mat(3, 1, CV_64F);


				if ( objectPoints.size() > TH_2D3D )
				{
					solvePnPRansac(objectPoints, imagePoints, cameraMatrix, distCoeffs, rvec, tvec, useExtrinsicGuess, iterationsCount, reprojectionError, 0.99, inliers, CV_ITERATIVE);
			
					Rodrigues(rvec, rvec);
					cv::Mat cvec = -1.0 * (rvec.t() * tvec);

					cout << camList[resecCamIndexList[0].second]->focal << endl;
					cout << objectPoints.size() << endl;
					cout << rvec << endl;
					cout << tvec << endl;
				}
			}

			cv::Point2f distPt, unDistPt;
			int count = 0, count1 = 0, count2 = 0, count3 = 0;
			double sum = 0;
			int nProj_Sum = 0;

			for ( int pIndex = 0; pIndex < trackList.size(); pIndex ++ )
			{
				vector<Point2D> *list = trackList[pIndex];

				v2_t pv[list->size()];
				double Rs[9 * list->size()];
				double ts[3 * list->size()];
				double fs[list->size()];
				double ks[list->size()];
				int nProjToTriangulation = 0;
				vector <int> camIdIndex;
				ptAttrList[pIndex]->X = 0;
			    	ptAttrList[pIndex]->Y = 0;
			    	ptAttrList[pIndex]->Z = 0;

				double avgFocal = 0;

				for ( int vIndex = 0; vIndex < list->size(); vIndex ++ )
				{
					if ( mapIndex1[(*list)[vIndex].camId] == -1 ) continue;
					
					double trans[3];
					distPt.x = (*list)[vIndex].x / cameraList1[mapIndex1[(*list)[vIndex].camId]]->focal;
					distPt.y = (*list)[vIndex].y / cameraList1[mapIndex1[(*list)[vIndex].camId]]->focal;
				
					avgFocal += focalListLocal[(*list)[vIndex].camId];

					unDistPt = unDistortPoint(distPt, 0/*cameraList1[mapIndex1[(*list)[vIndex].camId]]->k1*/);

			    		pv[nProjToTriangulation].p[0] = unDistPt.x;
					pv[nProjToTriangulation].p[1] = unDistPt.y;
		
					memcpy(Rs + 9 * nProjToTriangulation, cameraList1[mapIndex1[(*list)[vIndex].camId]]->rotMat, 9 * sizeof(double));
					memcpy(ts + 3 * nProjToTriangulation, cameraList1[mapIndex1[(*list)[vIndex].camId]]->tMat, 3 * sizeof(double));
					fs[nProjToTriangulation] = cameraList1[mapIndex1[(*list)[vIndex].camId]]->focal;
					ks[nProjToTriangulation] = cameraList1[mapIndex1[(*list)[vIndex].camId]]->k1;

					camIdIndex.push_back(mapIndex1[(*list)[vIndex].camId]);

					nProjToTriangulation ++;
				}

				avgFocal /= (double)nProjToTriangulation;

				if(nProjToTriangulation >= 2)
				{
				    count1 ++;

				    double *error_array = (double*)calloc(nProjToTriangulation, sizeof(double));
				    bool conditioned = false;
				    double mean = 0, std;
				    double error_curr = 0.0;

				    for(int m = 0; m < nProjToTriangulation; m++)
				    {
					for(int n = m + 1; n < nProjToTriangulation; n++)
					{
					    int index1 = camIdIndex[m];
					    int index2 = camIdIndex[n];

					    double val = ComputeRayAngle(pv[m], pv[n], cameraList1[index1]->rotMat, cameraList1[index2]->rotMat, cameraList1[index1]->cMat, cameraList1[index2]->cMat);
					    if(RAD2DEG(val) > 0.5)
					    {
						conditioned = true;
						break;
					    }
					}
				    }

				    if(conditioned)
				    {
					count2 ++;
					v3_t cloud_point;

					cloud_point = triangulate_n(nProjToTriangulation, pv, Rs, ts, &error_curr, error_array);

					if(error_curr < 0.01/*(20.0/avgFocal)*/)
					{
					    count3 ++;
					    std = 0;
					    mean = 0;

					    for(int ei = 0; ei < nProjToTriangulation; ei++)
					    {
						mean += error_array[ei];
						std = std + error_array[ei] * error_array[ei];
						sum += (error_array[ei] * error_array[ei] * fs[ei] * fs[ei]); 
					    }

					    nProj_Sum += nProjToTriangulation;

					    mean = mean / nProjToTriangulation;

					    std = std / nProjToTriangulation;

					    std = std - mean * mean;
					    std = sqrt(std);
					    
					    ptAttrList[pIndex]->X = cloud_point.p[0];
					    ptAttrList[pIndex]->Y = cloud_point.p[1];
					    ptAttrList[pIndex]->Z = cloud_point.p[2];
					    count++;
					}
				    }
				}
				camIdIndex.clear();
			}
			
			printf("%d %d %d [%d/%d] %lf\n\n", count1, count2, count3, count, trackList.size(), sum/(double)nProj_Sum);

			string newRT_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/RTglobalmapped_for_optimisation.txt";
			string newF_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/focal_for_optimisation.txt";
			string newM_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/Invmap.txt";
			string newOurs_file = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/ours_new.txt";
			string clusterDir = outDir + "/dendogram/" + clusterFileNameList[cIndex] + "/";
			string distortionFile = clusterDir + "/Distortion.txt";

			FILE *fpR = fopen(newRT_file.c_str(), "w");
			FILE *fpF = fopen(newF_file.c_str(), "w");
			FILE *fpM = fopen(newM_file.c_str(), "w");
			FILE *fpOurs = fopen(newOurs_file.c_str(), "w");
			FILE *fpDist = fopen(distortionFile.c_str(), "w");

			fprintf(fpF, "%d\n", cameraList1.size());

			for ( int index = 0; index < cameraList1.size(); index ++ )
			{
				fprintf(fpR, "%.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf %.9lf\n", 
						cameraList1[index]->rotMat[0], cameraList1[index]->rotMat[1], cameraList1[index]->rotMat[2],
						cameraList1[index]->rotMat[3], cameraList1[index]->rotMat[4], cameraList1[index]->rotMat[5], 
						cameraList1[index]->rotMat[6], cameraList1[index]->rotMat[7], cameraList1[index]->rotMat[8]);
				fprintf(fpR, "%.9lf %.9lf %.9lf\n", cameraList1[index]->cMat[0], cameraList1[index]->cMat[1], cameraList1[index]->cMat[2]);

				fprintf(fpF, "%d %.9lf %d\n", cameraList1[index]->camName, cameraList1[index]->focal, 2);
				fprintf(fpM, "%d %d\n", index, indexList[index]);
				fprintf(fpDist, "0.000\n"/*cameraList1[index]->k1*/);
			}
			fclose(fpR);
			fclose(fpF);
			fclose(fpM);
			fclose(fpDist);

			for ( int i = 0; i < trackList.size(); i ++ )
			{
				vector<Point2D> *list = trackList[i];

				if ( ptAttrList[i]->X != 0.0 || ptAttrList[i]->Y != 0.0 ||  ptAttrList[i]->Z != 0.0 )
				{
					int nProjToTriangulation = 0;
					for ( int j = 0; j < trackList[i]->size(); j ++ )
					{
						if ( mapIndex1[(*list)[j].camId] == -1 ) continue;

						nProjToTriangulation ++;
					}

					if ( nProjToTriangulation > 0 )
					{
						fprintf(fpOurs, "%lf %lf %lf 255 255 255 %d ", ptAttrList[i]->X, ptAttrList[i]->Y, ptAttrList[i]->Z, nProjToTriangulation);
		
						for ( int j = 0; j < trackList[i]->size(); j ++ )
						{
							if ( mapIndex1[(*list)[j].camId] == -1 ) continue;

							fprintf(fpOurs, "%d %lld %lf %lf ", mapIndex1[(*list)[j].camId], ptAttrList[i]->pointId, (*list)[j].x, (*list)[j].y);
						}
	
						fprintf(fpOurs, "\n");
					}
				}
			}

			fclose(fpOurs);

			if ( (sum/(double)nProj_Sum) < 100.00 )
			{
				bundlerCmd = "../bin/optimizer_all " + clusterDir + " 0 50 16 32 1 0";
			}
			else
			{
				bundlerCmd = "../bin/optimizer_all " + clusterDir + " 1 300 16 32 1 0";
			}
			
			//Run bundler
			system(bundlerCmd.c_str());
		}
	}
}

