// resection.cpp : Defines the entry point for the console application.
// Brojeshwar Bhowmick

#include <cstdio>
#include <iostream>
#include <fstream>
#include <istream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <omp.h>
#include "ceres/ceres.h"
#include "ceres/rotation.h"
#include <opencv2/opencv.hpp>




#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>  
#include <istream>
#include <string>
#include <stdio.h>
#include <stdlib.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <glog/logging.h>
#include <algorithm>
#include <vector>



using Eigen::AngleAxisd;
using Eigen::Map;
using Eigen::Matrix3d;
using Eigen::Quaterniond;
using Eigen::Vector2d;
using Eigen::Vector3d;

using namespace cv;
using namespace std;
// FILE _iob[] = { *stdin, *stdout, *stderr };
// //extern "C" FILE * __cdecl __iob_func(void) { return _iob; }
// extern "C" FILE * __iob_func(void) { return _iob; }

// extern void split(vector<string> &toks, const string &s, const string &delims);

// struct Error1
// {
// 	Error1(double observed_x, double observed_y)
// 		: observed_x(observed_x), observed_y(observed_y) {}

// 	template <typename T>
// 	bool operator()(const T* const camera, const T* const point, T* residuals) const
// 	{
// 		T p[3];
// 		T temp[3];

// 		temp[0] = point[0] - camera[0];
// 		temp[1] = point[1] - camera[1];
// 		temp[2] = point[2] - camera[2];

// 		T w[3];
// 		w[0] = camera[3];
// 		w[1] = camera[4];
// 		w[2] = camera[5];

// 		ceres::AngleAxisRotatePoint(w, temp, p);
// 		T xp = p[0] / p[2];
// 		T yp = p[1] / p[2];

// 		//T r2 = xp*xp + yp*yp;
// 		T distortion = T(1.0);//T(1.0) + r2  * (l1 + l2  * r2);

// 		//const T& focal = camera[6];
// 		const T& focal = T(667.0);
// 		T predicted_x = focal * distortion * xp;
// 		T predicted_y = focal * distortion * yp;
// 		residuals[0] = predicted_x - T(observed_x);
// 		residuals[1] = predicted_y - T(observed_y);
// 		return true;
// 	}
// 	static ceres::CostFunction* Create(const double observed_x, const double observed_y)
// 	{
// 		return (new ceres::AutoDiffCostFunction<Error1, 2, 6, 3>(new Error1(observed_x, observed_y)));
// 	}

// 	double observed_x;
// 	double observed_y;
// };



// void CalculateT(vector<Point3f> *meshPoints, vector<Point2f> *imagePoints,  Eigen::MatrixXf &R, cv::Mat &T, double focal, cv::Mat &newT)
// {
//  	int ptSize = 3;
//     int ptSizeX2 = ptSize << 1;

//     Eigen::MatrixXf A(6, 3);;
//     Eigen::MatrixXf B(6, 1);
//     Eigen::MatrixXf t(3, 3);;;
//     Eigen::MatrixXf S(3, 1);;
//     Eigen::MatrixXf P(3, 1);;
//     Eigen::MatrixXf p(3, 1);;
//     int nImgPoints = imagePoints->size();
//     int nMeshPoints = meshPoints->size();

//     vector<int>inliers, finalInlier;
//     double finalreproj = 0.0, avgProj;
// 	int maxNum = 0;

// 	cout << R << endl << endl;

// 	printf("Check 1.1\n");
    
//     for ( int index = 0; index < 500; index ++ )
//     {
//         avgProj = 0.0;
//         int id[ptSize];
//         Eigen::MatrixXf X(3, 1), V(3, 1);
        

//         for ( int rIndex = 0; rIndex < ptSize; rIndex ++ )
//         {
//             id[rIndex] = rand() % nMeshPoints;

//             for ( int k = 0; k < rIndex; k ++ )
//             {
//                 if (  id[rIndex] ==  id[k] )
//                 {
//                     id[rIndex] = rand() % nMeshPoints;
//                 }
//             }

//             X(0) = (*meshPoints)[id[rIndex]].x;
//             X(1) = (*meshPoints)[id[rIndex]].y;
//             X(2) = (*meshPoints)[id[rIndex]].z;

//             V = R * X;

//             A(rIndex*2, 0) = -1;
//             A(rIndex*2, 1) = 0;
//             A(rIndex*2, 2) = (*imagePoints)[id[rIndex]].x/focal;

//             A(rIndex*2 + 1, 0) = 0;
//             A(rIndex*2 + 1, 1) = -1;
//             A(rIndex*2 + 1, 2) = (*imagePoints)[id[rIndex]].y/focal;

//             B(rIndex*2, 0) = V(0) - V(2) * (*imagePoints)[id[rIndex]].x /focal;
//             B(rIndex*2 + 1, 0) = V(1) - V(2) * (*imagePoints)[id[rIndex]].y /focal;
//         }
        
//         ///printf("Check 1.2 %d\n", index);

//         t = A.transpose()*A;
//         S = t.inverse()*A.transpose()*B;
//         // cout<<"New T is "<<endl;
//         // cout<< S <<endl;
//         double minError = 4.0;

//         inliers.clear();
//         for ( int rIndex = 0; rIndex < nImgPoints; rIndex ++ )
//         {
//             P(0) = (*meshPoints)[rIndex].x;
//             P(1) = (*meshPoints)[rIndex].y;
//             P(2) = (*meshPoints)[rIndex].z;

// 			//printf("Check 1.3 %d %d\n", index, rIndex);

//             p = R*P + S;

// 			//printf("Check 1.4 %d %d\n", index, rIndex);

//             double reprojErr = sqrt((p(0)*focal/p(2) - (*imagePoints)[rIndex].x) * (p(0)*focal/p(2) - (*imagePoints)[rIndex].x) +
//                             (p(1) * focal/p(2) - (*imagePoints)[rIndex].y)* (p(1) * focal/p(2) - (*imagePoints)[rIndex].y));

//             if(reprojErr < minError)
//             {
//                 inliers.push_back(rIndex);
//                 avgProj += reprojErr;
//             }

//             //printf("Reproj %lf\n", reprojErr);

//         }

//         if(maxNum < inliers.size())
//         {
//             maxNum = inliers.size();
//             finalInlier = inliers;
//             finalreproj = avgProj/maxNum;
//             cout<< " DDDDD "<<endl;
//             newT.at<double>(0) = S(0);
//             newT.at<double>(1) = S(1);
//             newT.at<double>(2) = S(2);

//         }

//          //printf("Check 1.3 %d\n", index);
//     }

//     printf("Reprojection %lf %d\n", finalreproj, maxNum);
// }

// int resectionUpdate(vector<Point3f> *meshPoints, vector<Point2f> *imagePoints, double inputfocal, cv::Mat &Rin, cv::Mat &T,
//                     float maxReprojectionError, int minInliersCount, vector<int> *inliers, cv::Mat *initRvec, cv::Mat *initTvec,
//                     vector<double> *Rret, vector<double> *Tret, double *updatedfocal, double *reprojError,cv::Mat &Rij_mat)
// {
//     cv::Mat newT(3, 1, cv::DataType<double>::type);
//     cv::Mat C(3, 1, cv::DataType<double>::type);
//     cv::Mat Pt_2d(3, 1, cv::DataType<double>::type);
//     cv::Mat Pt_3d(3, 1, cv::DataType<double>::type);
//     cv::Mat R(3, 3, cv::DataType<double>::type);
// 	cv::Mat RT(3, 3, cv::DataType<double>::type);

// 	cv::Mat R_computed(3, 3, cv::DataType<double>::type);
// 	cv::Mat T_computed(3, 1, cv::DataType<double>::type);

// 	cv::Mat rvec;

//     Eigen::MatrixXf Ri(3,3), Rij(3,3), Rj(3,3);

//     for ( int i = 0; i < 3; i ++ )
//     {
//         for ( int j = 0; j < 3 ; j ++ )
//         {
//             Ri(i, j) = Rin.at<double>(i, j);
//             //Rij(i, j) = Rij.at<double>(i, j);
//         }
//     }

//     // Rij(0, 0) = 0.9999982410;
//     // Rij(0, 1) = 0.0016608251;
//     // Rij(0, 2) = 0.0008715348;
//     // Rij(1, 0) = -0.0016617689; 
//     // Rij(1, 1) = 0.9999980325; 
//     // Rij(1, 2) = 0.0010832994;
//     // Rij(2, 0) = -0.0008697339; 
//     // Rij(2, 1) = -0.0010847458;
//     // Rij(2, 2) = 0.9999990334;



//     Rij(0, 0) = Rij_mat.at<double>(0, 0);
//     Rij(0, 1) = Rij_mat.at<double>(0, 1);;
//     Rij(0, 2) = Rij_mat.at<double>(0, 2);;
//     Rij(1, 0) = Rij_mat.at<double>(1, 0);; 
//     Rij(1, 1) = Rij_mat.at<double>(1, 1);; 
//     Rij(1, 2) = Rij_mat.at<double>(1, 2);;
//     Rij(2, 0) = Rij_mat.at<double>(2, 0);; 
//     Rij(2, 1) = Rij_mat.at<double>(2, 1);;
//     Rij(2, 2) = Rij_mat.at<double>(2, 2);;



//     Rj = Rij * Ri.transpose();

// 	//cout << Ri <<endl;
// 	//cout << Rij <<endl;
//     //cout << Rj <<endl;

//     FILE *ptr = fopen("Reproj1.txt", "w");

//     cv::Mat distCoeffs = Mat::zeros(4, 1, CV_64FC1);

//     //cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
//     //distCoeffs.at<double>(0) = 0.0;
//     //distCoeffs.at<double>(1) = 0.0;
//     //distCoeffs.at<double>(2) = 0.0;
//     //distCoeffs.at<double>(3) = 0.0;

//     Mat cameraMatrix(3, 3, CV_64FC1);

//     //cv::Mat cameraMatrix(3,3,cv::DataType<double>::type);
//     //cv::setIdentity(cameraMatrix);

//     cameraMatrix.at<double>(0, 0) = inputfocal;
//     cameraMatrix.at<double>(0, 1) = 0.f;
//     cameraMatrix.at<double>(0, 2) = 0.f;
//     cameraMatrix.at<double>(1, 0) = 0.f;
//     cameraMatrix.at<double>(1, 1) = inputfocal;
//     cameraMatrix.at<double>(1, 2) = 0.f;
//     cameraMatrix.at<double>(2, 0) = 0.f;
//     cameraMatrix.at<double>(2, 1) = 0.f;
//     cameraMatrix.at<double>(2, 2) = 1.0;

//     printf("Check 1\n");

//     CalculateT(meshPoints, imagePoints, Rj, T, inputfocal, newT);

// 	printf("Check 2\n");

// 	//Non lin


// 	#if 1
// 	//cv::Mat ttvec = (*tvec) * 1.0;

// 	cout<< newT <<endl;
// 	cv::Mat ttvec = (newT) * 1.0;
// 	ceres::Problem problem;

// 	double *param = new double[6];

// 	//Rodrigues(*rvec, R);


// 	R.at<double>(0, 0) = Rj(0, 0);
//     R.at<double>(0, 1) = Rj(0, 1);
//     R.at<double>(0, 2) = Rj(0, 2);
//     R.at<double>(1, 0) = Rj(1, 0);
//     R.at<double>(1, 1) = Rj(1, 1);
//     R.at<double>(1, 2) = Rj(1, 2);
//     R.at<double>(2, 0) = Rj(2, 0);
//     R.at<double>(2, 1) = Rj(2, 1);
//     R.at<double>(2, 2) = Rj(2, 2);



// 	cv::transpose(R, RT);
// 	cout<< RT <<endl;
// 	cout<< ttvec <<endl;

// 	cv::gemm(RT, ttvec, 1, NULL, 0, C);

// 	C.at<double>(0) = C.at<double>(0)*-1;
// 	C.at<double>(1) = C.at<double>(1)*-1;
// 	C.at<double>(2) = C.at<double>(2)*-1;

// 	param[0] = C.at<double>(0);
// 	param[1] = C.at<double>(1);
// 	param[2] = C.at<double>(2);
// 	Rodrigues(R, rvec);
// 	param[3] = (rvec).at<double>(0);
// 	param[4] = (rvec).at<double>(1);
// 	param[5] = (rvec).at<double>(2);

// 	//param[6] = cameraMatrix.at<double>(0, 0);

// 	vector<double>point3d;

// 	for(int i = 0; i < (int)meshPoints->size(); i++)
// 	{
// 		point3d.push_back((*meshPoints)[i].x);
// 		point3d.push_back((*meshPoints)[i].y);
// 		point3d.push_back((*meshPoints)[i].z);
// 	}

// 	for (int i = 0; i < (int)meshPoints->size(); ++i)
// 	{
// 		// Each Residual block takes a point and a camera as input and outputs a 2
// 		// dimensional residual. Internally, the cost function stores the observed
// 		// image location and compares the reprojection against the observation.

// 		ceres::CostFunction* cost_function = Error1::Create((*imagePoints)[i].x, (*imagePoints)[i].y);
// 		//int camid = camidx[i];
// 		//int pointid = ptidx[i];
// 		problem.AddResidualBlock(cost_function, new ceres::HuberLoss(10000000) /* squared loss */, param, &point3d[0] + i * 3);
// 	}


// 	ceres::Solver::Options options;
// 	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
// 	options.minimizer_progress_to_stdout = false;
// 	options.max_linear_solver_iterations = 200;

// 	ceres::Solver::Summary summary;
// 	//printf("Solving\n");
// 	ceres::Solve(options, &problem, &summary);
// 	cout << "Done" << endl;
// 	//std::cout << summary.FullReport() << "\n";

// 	double r[9];
// 	double w[3];
// 	w[0] = param[3];
// 	w[1] = param[4];
// 	w[2] = param[5];
// 	ceres::AngleAxisToRotationMatrix(w, r);
// 	(*Rret)[0] = r[0];
// 	(*Rret)[1] = r[3];
// 	(*Rret)[2] = r[6];
// 	(*Rret)[3] = r[1];
// 	(*Rret)[4] = r[4];
// 	(*Rret)[5] = r[7];
// 	(*Rret)[6] = r[2];
// 	(*Rret)[7] = r[5];
// 	(*Rret)[8] = r[8];

// 	R_computed.at<double>(0, 0)	= r[0];
// 	R_computed.at<double>(0, 1) = r[3];
// 	R_computed.at<double>(0, 2) = r[6];
// 	R_computed.at<double>(1, 0) = r[1];
// 	R_computed.at<double>(1, 1) = r[4];
// 	R_computed.at<double>(1, 2) = r[7];
// 	R_computed.at<double>(2, 0) = r[2];
// 	R_computed.at<double>(2, 1) = r[5];
// 	R_computed.at<double>(2, 2) = r[8];
	

	

	

// 	C.at<double>(0) = param[0];
// 	C.at<double>(1) = param[1];
// 	C.at<double>(2) = param[2];

// 	T_computed = (-1.0)*(R_computed)*C;
// 	(*Tret)[0] = T_computed.at<double>(0);
// 	(*Tret)[1] = T_computed.at<double>(1);
// 	(*Tret)[2] = T_computed.at<double>(2);

	
// 	*updatedfocal = param[6];
// 	delete[]param;
// 	//Rodrigues(rvec, *initRvec);
// 	cout << "Done 2222" << endl;

// 	cout << T_computed << endl;

// 	// check reprojection error

// 	double sum= 0.0;
// 	double minError = 2.0;
// 	int inlCount = 0;
// 	//vector<int>inliers;

// 	for ( int rIndex = 0; rIndex < (int)meshPoints->size(); rIndex ++ )
//         {
//             Pt_3d.at<double>(0) = (*meshPoints)[rIndex].x;
//             Pt_3d.at<double>(1) = (*meshPoints)[rIndex].y;
//             Pt_3d.at<double>(2) = (*meshPoints)[rIndex].z;

// 			//printf("Check 1.3 %d %d\n", index, rIndex);

//            Pt_2d = R_computed*Pt_3d + T_computed;

// 			//printf("Check 1.4 %d %d\n", index, rIndex);

//             double reprojErr = sqrt((Pt_2d.at<double>(0) *inputfocal/Pt_2d.at<double>(2)  - (*imagePoints)[rIndex].x) * (Pt_2d.at<double>(0) *inputfocal/Pt_2d.at<double>(2)  - (*imagePoints)[rIndex].x) +
//                             (Pt_2d.at<double>(1)  * inputfocal/Pt_2d.at<double>(2)  - (*imagePoints)[rIndex].y)* (Pt_2d.at<double>(1)* inputfocal/Pt_2d.at<double>(2) - (*imagePoints)[rIndex].y));

//             if(reprojErr < minError)
//             {
//                 (*inliers).push_back(rIndex);
//                 //avgProj += reprojErr;
//                 sum = sum+ reprojErr;
//                 inlCount++;
//             }

            
//             //printf("Reproj %lf\n", reprojErr);

//         }
	
// 	cout << "Avg Reproj Error for inliers " << (sum/ (int)meshPoints->size())<< endl;
// 	cout << "minInliersCount  " << inlCount<< endl;
// #else
//   #if 0
// 	//Rodrigues(*rvec, *rvec);
// 	Rodrigues(Rj, *rvec);
// 	(*Rret)[0] = R.at<double>(0, 0);
// 	(*Rret)[1] = R.at<double>(0, 1);
// 	(*Rret)[2] = R.at<double>(0, 2);
// 	(*Rret)[3] = R.at<double>(1, 0);
// 	(*Rret)[4] = R.at<double>(1, 1);
// 	(*Rret)[5] = R.at<double>(1, 2);
// 	(*Rret)[6] = R.at<double>(2, 0);
// 	(*Rret)[7] = R.at<double>(2, 1);
// 	(*Rret)[8] = R.at<double>(2, 1);
//     //CHECK FOR C 
// 	(*Tret)[0] = newT.at<double>(0));
// 	(*Tret)[1] = newT.at<double>(1));
// 	(*Tret)[2] = newT.at<double>(2));
// 	*focal = cameraMatrix.at<double>(0, 0);
// #endif
// #endif
    
//     return 1;
// }

// bool computeRT(vector<cv::Point3f> *meshPoints, vector<double> &point3d, vector<cv::Point2f> *imagePoints, cv::Mat &distCoeffs, cv::Mat cameraMatrix, cv::Mat *rvec, cv::Mat *tvec, float maxReprojectionError, int minInliersCount, vector<int> *inliers, vector<double> *R2, vector<double> *C2, double *focal)
// {
// 	bool status = false;

// 	cv::Mat C(3, 1, cv::DataType<double>::type);
// 	cv::Mat R(3, 3, cv::DataType<double>::type);
// 	cv::Mat RT(3, 3, cv::DataType<double>::type);

// 	//bool useExtrinsicGuess = true;
// 	bool useExtrinsicGuess = false;
// 	int iterationsCount = 500;

// 	//Rodrigues(*rvec, *rvec);

// #if 1
// 	for(int i = 0; i < (int)(*imagePoints).size(); i++)
// 	{
// 		//cout << "B4 " << i << "  " << (*imagePoints)[i].x <<  "  " << (*imagePoints)[i].y << endl;
// 	}

// 	solvePnPRansac(*meshPoints, *imagePoints, cameraMatrix, distCoeffs, *rvec, *tvec, useExtrinsicGuess, iterationsCount, maxReprojectionError, 0.99, *inliers, 0);
// 	//solvePnPRansac(*meshPoints, *imagePoints, cameraMatrix, distCoeffs, *rvec, *tvec, useExtrinsicGuess, iterationsCount, maxReprojectionError, -1, *inliers, CV_EPNP);

// 	if((int)(*inliers).size() < minInliersCount)
// 	{
// 		cout << "ERROR from solvePnPRansac()" << endl << "number of inliers " << (*inliers).size() << ", which is less than the threshold " << minInliersCount << endl;
// 		//return(status);
// 	}
// 	cout << "number of inliers " << (*inliers).size() << endl;

// 	//for(int i = 0; i < (int)(*inliers).size(); i++)
// 	for(int i = 0; i < 10; i++)
// 	{
// 		cout << "index " << i << " is " << (*inliers)[i] << "  " << (*imagePoints)[i].x <<  "  " << (*imagePoints)[i].y << endl;
// 	}
//         //getchar();
// #endif

// #if 0
// 	cv::Mat ttvec = (*tvec) * 1.0;
// 	ceres::Problem problem;

// 	double *param = new double[6];

// 	Rodrigues(*rvec, R);

// 	cv::transpose(R, RT);

// 	cv::gemm(RT, ttvec, 1, NULL, 0, C);

// 	C.at<double>(0) = C.at<double>(0)*-1;
// 	C.at<double>(1) = C.at<double>(1)*-1;
// 	C.at<double>(2) = C.at<double>(2)*-1;

// 	param[0] = C.at<double>(0);
// 	param[1] = C.at<double>(1);
// 	param[2] = C.at<double>(2);
// 	param[3] = (*rvec).at<double>(0);
// 	param[4] = (*rvec).at<double>(1);
// 	param[5] = (*rvec).at<double>(2);

// 	//param[6] = cameraMatrix.at<double>(0, 0);

// 	for (int i = 0; i < (int)meshPoints->size(); ++i)
// 	{
// 		// Each Residual block takes a point and a camera as input and outputs a 2
// 		// dimensional residual. Internally, the cost function stores the observed
// 		// image location and compares the reprojection against the observation.

// 		ceres::CostFunction* cost_function = Error1::Create((*imagePoints)[i].x, (*imagePoints)[i].y);
// 		//int camid = camidx[i];
// 		//int pointid = ptidx[i];
// 		problem.AddResidualBlock(cost_function, new ceres::HuberLoss(10000000) /* squared loss */, param, &point3d[0] + i * 3);
// 	}


// 	ceres::Solver::Options options;
// 	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
// 	options.minimizer_progress_to_stdout = false;
// 	options.max_linear_solver_iterations = 200;

// 	ceres::Solver::Summary summary;
// 	//printf("Solving\n");
// 	ceres::Solve(options, &problem, &summary);
// 	//cout << "Done" << endl;
// 	//std::cout << summary.FullReport() << "\n";

// 	double r[9];
// 	double w[3];
// 	w[0] = param[3];
// 	w[1] = param[4];
// 	w[2] = param[5];
// 	ceres::AngleAxisToRotationMatrix(w, r);
// 	(*R2)[0] = r[0];
// 	(*R2)[1] = r[3];
// 	(*R2)[2] = r[6];
// 	(*R2)[3] = r[1];
// 	(*R2)[4] = r[4];
// 	(*R2)[5] = r[7];
// 	(*R2)[6] = r[2];
// 	(*R2)[7] = r[5];
// 	(*R2)[8] = r[8];

// 	(*C2)[0] = param[0];
// 	(*C2)[1] = param[1];
// 	(*C2)[2] = param[2];
// 	*focal = 667;
// 	delete[]param;
// 	Rodrigues(*rvec, *rvec);
// #else
// #if 0
// 	Rodrigues(*rvec, *rvec);

// 	(*R2)[0] = (double)((*rvec).at<double>(cv::Point(0, 0)));
// 	(*R2)[1] = (double)((*rvec).at<double>(cv::Point(1, 0)));
// 	(*R2)[2] = (double)((*rvec).at<double>(cv::Point(2, 0)));
// 	(*R2)[3] = (double)((*rvec).at<double>(cv::Point(0, 1)));
// 	(*R2)[4] = (double)((*rvec).at<double>(cv::Point(1, 1)));
// 	(*R2)[5] = (double)((*rvec).at<double>(cv::Point(2, 1)));
// 	(*R2)[6] = (double)((*rvec).at<double>(cv::Point(0, 2)));
// 	(*R2)[7] = (double)((*rvec).at<double>(cv::Point(1, 2)));
// 	(*R2)[8] = (double)((*rvec).at<double>(cv::Point(2, 2)));

// 	(*C2)[0] = (double)((*tvec).at<double>(0));
// 	(*C2)[1] = (double)((*tvec).at<double>(1));
// 	(*C2)[2] = (double)((*tvec).at<double>(2));
// 	*focal = cameraMatrix.at<double>(0, 0);
// #endif
// #endif
// 	status = true;
// 	return(status);
// }


// bool resection(vector<Point3f> *meshPoints, vector<Point2f> *imagePoints, double inputfocal, float maxReprojectionError, int minInliersCount, vector<int> *inliers, cv::Mat *initRvec, cv::Mat *initTvec, vector<double> *Rret, vector<double> *Cret, double *updatedfocal, double *reprojError)
// {
// 	FILE *ptr = fopen("Reproj.txt", "w");

// 	cv::Mat distCoeffs = Mat::zeros(4, 1, CV_64FC1);

// 	//cv::Mat distCoeffs(4,1,cv::DataType<double>::type);
// 	//distCoeffs.at<double>(0) = 0.0;
// 	//distCoeffs.at<double>(1) = 0.0;
// 	//distCoeffs.at<double>(2) = 0.0;
// 	//distCoeffs.at<double>(3) = 0.0;

// 	Mat cameraMatrix(3, 3, CV_64FC1);

// 	//cv::Mat cameraMatrix(3,3,cv::DataType<double>::type);
// 	//cv::setIdentity(cameraMatrix);	

// 	cameraMatrix.at<double>(0, 0) = inputfocal;
// 	cameraMatrix.at<double>(0, 1) = 0.f;
// 	cameraMatrix.at<double>(0, 2) = 0.f;
// 	cameraMatrix.at<double>(1, 0) = 0.f;
// 	cameraMatrix.at<double>(1, 1) = inputfocal;
// 	cameraMatrix.at<double>(1, 2) = 0.f;
// 	cameraMatrix.at<double>(2, 0) = 0.f;
// 	cameraMatrix.at<double>(2, 1) = 0.f;
// 	cameraMatrix.at<double>(2, 2) = 1.0;

// 	vector<double>point3d;

// 	for(int i = 0; i < (int)meshPoints->size(); i++)
// 	{
// 		point3d.push_back((*meshPoints)[i].x);
// 		point3d.push_back((*meshPoints)[i].y);
// 		point3d.push_back((*meshPoints)[i].z);
// 	}

// 	cv::Mat backR, backT;
// 	(*initRvec).copyTo(backR);
// 	(*initTvec).copyTo(backT);

// 	bool status = computeRT(meshPoints, point3d, imagePoints, distCoeffs, cameraMatrix, initRvec, initTvec, maxReprojectionError, minInliersCount, inliers, Rret, Cret, updatedfocal);

// 	if(status == true)
// 	{
// #if 0
// 		double sum = 0;
// 		for(int i = 0; i < (int)meshPoints->size(); i++)
// 		{
// 			double temp[3], p[3];

// 			temp[0] = (double)((*meshPoints)[i].x) - (*Cret)[0];
// 			temp[1] = (double)((*meshPoints)[i].y) - (*Cret)[1];
// 			temp[2] = (double)((*meshPoints)[i].z) - (*Cret)[2];

// 			double w[3];
// 			double r1[9] = {(*Rret)[0], (*Rret)[3], (*Rret)[6], (*Rret)[1], (*Rret)[4], (*Rret)[7], (*Rret)[2], (*Rret)[5], (*Rret)[8]};

// 			ceres::RotationMatrixToAngleAxis(r1, w);
// 			ceres::AngleAxisRotatePoint(w, temp, p);

// 			double xp = p[0] / p[2];
// 			double yp = p[1] / p[2];
// 			double predicted_x = (*updatedfocal) *  xp;
// 			double predicted_y = (*updatedfocal) *  yp;

// 			double errorx = (predicted_x - (*imagePoints)[i].x);
// 			double errory = (predicted_y - (*imagePoints)[i].y);

// 			double error = sqrt(errorx*errorx  + errory*errory);

// 			sum += error;
// 			if(i == 0)
// 			{
// 				printf("association (%lf %lf %lf) (%lf %lf) -> %lf\n", (*meshPoints)[i].x, (*meshPoints)[i].y, (*meshPoints)[i].z, (*imagePoints)[i].x, (*imagePoints)[i].y, error);
// 				printf("projection in cam (%lf %lf)\n", predicted_x, predicted_y);
// 				//getchar();
// 			}
// 		}
// 		sum = sum/(*meshPoints).size();

// 		*reprojError = sum;
// #else
// 		cv::Mat rMat;
// 		Rodrigues(*initRvec, rMat);
// 		vector<Point2f> projectedImagePoints;

// 		cout << "Rvec" << endl << (*initRvec) << endl << "Rmat" << endl << rMat << endl << "Tvec" << endl << (*initTvec) << endl;

// 		projectPoints((*meshPoints), (*initRvec), (*initTvec), cameraMatrix, distCoeffs, projectedImagePoints, noArray(), 0);
// 		//projectPoints((*meshPoints), backR, backT, cameraMatrix, distCoeffs, projectedImagePoints, noArray(), 0);

// 		double sum = 0;
// 		for(int j = 0; j < (int)inliers->size(); j++)
// 		{
// 			int i = (*inliers)[j];

// 			double errorx = (projectedImagePoints[i].x - (*imagePoints)[i].x);
// 			double errory = (projectedImagePoints[i].y - (*imagePoints)[i].y);

// 			double error = sqrt(errorx*errorx  + errory*errory);

// 			sum += error;
// 			//printf("Error %lf\n",error);

// 			fprintf(ptr, "%d\t%lf\n", i, error);
// 			fflush(ptr);

// 			if(i == 0)
// 			{
// 				printf("association (%lf %lf %lf) (%lf %lf) -> %lf\n", (*meshPoints)[i].x, (*meshPoints)[i].y, (*meshPoints)[i].z, (*imagePoints)[i].x, (*imagePoints)[i].y, error);
// 				printf("projection in cam (%lf %lf)\n", projectedImagePoints[i].x, projectedImagePoints[i].y);
// 				//getchar();
// 			}
// 		}
// 		sum = sum/(*inliers).size();

// 		*reprojError = sum;
// #endif
// 		if(*reprojError > maxReprojectionError)
// 		{
// 			printf("Reprojection error is %lf, very high. ERROR\n", *reprojError);
// 			status = false;
// 		}
// 	}

// 	fclose(ptr);

// 	return(status);
// }

void split(vector<string> &toks, const string &s, const string &delims)
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
double cameraparam[9];




struct Error2
{
	//imagePoints_deform[i].x, imagePoints_deform[i].y, mean_s[frontIndex[i]].begin(),eigenvec[3*frontIndex[i]].begin(),eigenvec[3*frontIndex[i]+1].begin(),eigenvec[3*frontIndex[i]+2].begin()

	Error2(double observed_x, double observed_y, int i,  double *mean,  const double*camparam )
		: observed_x(observed_x), observed_y(observed_y) ,i(i), mean(mean),camera(camparam){}

	template <typename T>
	bool operator()(const T* const param, T* residuals) const
	{
		T p[3];
		T w[3];
		 T temp[3];
		 p[0] = T(mean[0]);
		 p[1] = T(mean[1]);
		 p[2] = T(mean[2]);//{mean[0],mean[1],mean[2]};

		

		//temp[0] = mean[0]  + scale[0]* eig1[0] + scale[1]* eig1[1]+ scale[0]* eig1[0]+ scale[0]* eig1[0];//- camera[0];
		//temp[1] = mean[1] ;//- camera[1];
		//temp[2] = mean[2] ;//- camera[2];

		 p[0] = p[0] - param[3];
		p[1] = p[1] - param[4];
		p[2] = p[2] - param[5];
		
		w[0]= param[0];
		w[1]=param[1];
		w[2]=param[2];
		ceres::AngleAxisRotatePoint(w, p,temp);
		
		T xp = temp[0] / temp[2];
		T yp = temp[1] / temp[2];

		//T r2 = xp*xp + yp*yp;
		T distortion = T(1.0);//T(1.0) + r2  * (l1 + l2  * r2);

		const T& focal = T(camera[0]);
		const T cx = T(camera[1]);
		const T cy = T(camera[2]);

		T predicted_x = focal * distortion * xp + cx;
		T predicted_y = focal * distortion * yp + cy;
		if(i ==0)
		{
			residuals[0] = (predicted_x - T(observed_x))*T(10);

			residuals[1] =  (predicted_y - T(observed_y))*T(10);
		}
		else
		{
			residuals[0] = predicted_x - T(observed_x);

			residuals[1] =  predicted_y - T(observed_y);

		}
		//residuals[1] =  predicted_y - T(observed_y);
		//cout<< residuals[0] << " "<<residuals[1] <<endl;
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x, const double observed_y, int i,  double *mean, const double *camparam)
	{
		return (new ceres::AutoDiffCostFunction<Error2, 2, 6>(new Error2(observed_x, observed_y, i, mean, camparam)));
	}

	double observed_x;
	double observed_y;
	//double *line;
	 double *mean;
	int i;
	const double *camera;

};




struct Error1
{
	//imagePoints_deform[i].x, imagePoints_deform[i].y, mean_s[frontIndex[i]].begin(),eigenvec[3*frontIndex[i]].begin(),eigenvec[3*frontIndex[i]+1].begin(),eigenvec[3*frontIndex[i]+2].begin()

	Error1(double observed_x, double observed_y, int ii, double *mean,  const double *eig1, const double* eig2, const double * eig3, const double*camparam )
		: observed_x(observed_x), observed_y(observed_y) ,ii(ii),mean(mean),eig1(eig1),eig2(eig2), eig3(eig3),camerafixed(camparam){}

	template <typename T>
	bool operator()(const T* const scale, const T* const camera, T* residuals) const
	{
		T p[3];
		 T temp[3];
		 temp[0] = T(mean[0]);
		 temp[1] = T(mean[1]);
		 temp[2] = T(mean[2]);//{mean[0],mean[1],mean[2]};

		for(int i = 0; i < 6; i++)
		{
				temp[0] = temp[0] + scale[i]* eig1[i];
				temp[1] = temp[1] + scale[i]* eig2[i];
				temp[2] = temp[2] + scale[i]* eig3[i];
		}

		//temp[0] = mean[0]  + scale[0]* eig1[0] + scale[1]* eig1[1]+ scale[0]* eig1[0]+ scale[0]* eig1[0];//- camera[0];
		//temp[1] = mean[1] ;//- camera[1];
		//temp[2] = mean[2] ;//- camera[2];
		//if(ii)

		T w[3];
		w[0] = T(camera[0]);
		w[1] = T(camera[1]);
		w[2] = T(camera[2]);

		temp[0] = temp[0] - camera[3];
		temp[1] = temp[1] - camera[4];
		temp[2] = temp[2] - camera[5];

		ceres::AngleAxisRotatePoint(w, temp, p);
		// p[0] = p[0] + camera[6];
		// p[1] = p[1] + camera[7];
		// p[2] = p[2] + camera[8];
		

		T xp = p[0] / p[2];
		T yp = p[1] / p[2];

		//T r2 = xp*xp + yp*yp;
		T distortion = T(1.0);//T(1.0) + r2  * (l1 + l2  * r2);

		const T& focal = T(camerafixed[0]);
		const T cx = T(camerafixed[1]);
		const T cy = T(camerafixed[2]);

		T predicted_x = focal * distortion * xp + cx;
		T predicted_y = focal * distortion * yp + cy;

		
			residuals[0] = predicted_x - T(observed_x);

			residuals[1] =  predicted_y - T(observed_y);
			// residuals[2] =  camera[3] - T(34);
			// residuals[3] =  camera[4] - T(4);
			// residuals[4] =  camera[5] - T(0);
			// if(i ==0)
			// {
			// 	residuals[3] = (temp[0] - mean[0]);
			// 	residuals[4] = temp[1] - mean[1];
			// 	residuals[5] = temp[2] - mean[2];
			// }
			
		
		//cout<< residuals[0] << " "<<residuals[1] <<endl;
		return true;
	}
	static ceres::CostFunction* Create(const double observed_x, const double observed_y,int i, double *mean, const double *eig1, const double* eig2, const double * eig3, const double *camparam)
	{
		return (new ceres::AutoDiffCostFunction<Error1, 2, 6,6>(new Error1(observed_x, observed_y,i,  mean, eig1, eig2, eig3, camparam)));
	}

	double observed_x;
	double observed_y;
	//double *line;
	 double *mean;
	const double *eig1;
	const double *eig2;
	const double *eig3;
	const double *camerafixed;
	int ii;
	// int i;
};





struct ErrorRT
{
	//imagePoints_deform[i].x, imagePoints_deform[i].y, mean_s[frontIndex[i]].begin(),eigenvec[3*frontIndex[i]].begin(),eigenvec[3*frontIndex[i]+1].begin(),eigenvec[3*frontIndex[i]+2].begin()

	ErrorRT(  double *x1,  double* x2)
		: x1(x1), x2(x2) {}

	template <typename T>
	bool operator()( const T* const camera, T* residuals) const
	{
		T p[3];
		T s = T(camera[6]);
		T temp[3];
		temp[0] = s*T(x2[0]);
		temp[1] = s*T(x2[1]);
		temp[2] =  s*T(x2[2]);//{mean[0],mean[1],mean[2]};

		// for(int i = 0; i < 6; i++)
		// {
		// 		temp[0] = temp[0] + scale[i]* eig1[i];
		// 		temp[1] = temp[1] + scale[i]* eig2[i];
		// 		temp[2] = temp[2] + scale[i]* eig3[i];
		// }

		//temp[0] = mean[0]  + scale[0]* eig1[0] + scale[1]* eig1[1]+ scale[0]* eig1[0]+ scale[0]* eig1[0];//- camera[0];
		//temp[1] = mean[1] ;//- camera[1];
		//temp[2] = mean[2] ;//- camera[2];
		//if(ii)

		T w[3];
		w[0] = T(camera[0]);
		w[1] = T(camera[1]);
		w[2] = T(camera[2]);

		// temp[0] = temp[0] - s*camera[3];
		// temp[1] = temp[1] - s*camera[4];
		// temp[2] = temp[2] - s*camera[5];

		ceres::AngleAxisRotatePoint(w, temp, p);
		 p[0] = p[0] + T(camera[3]);
		 p[1] = p[1] + T(camera[4]);
		 p[2] = p[2] + T(camera[5]);
		

		
		
		residuals[0] = T(x1[0]) - p[0];
		residuals[1] = T(x1[1]) - p[1];
		residuals[2] = T(x1[2]) - p[2];

			// residuals[2] =  camera[3] - T(34);
			// residuals[3] =  camera[4] - T(4);
			// residuals[4] =  camera[5] - T(0);
			// if(i ==0)
			// {
			// 	residuals[3] = (temp[0] - mean[0]);
			// 	residuals[4] = temp[1] - mean[1];
			// 	residuals[5] = temp[2] - mean[2];
			// }
			
		
		//cout<< residuals[0] << " "<<residuals[1] <<" " <<residuals[2]<<endl;
		return true;
	}
	static ceres::CostFunction* Create( double *x1,  double* x2)
	{
		return (new ceres::AutoDiffCostFunction<ErrorRT, 3, 7>(new ErrorRT( x1, x2)));
	}


	 double *x1;
	 double *x2;
	int ii;
	// int i;
};





int ind_count = 0;
vector<int>ind2;
void onMouse(int evt, int x, int y, int flags, void* param) {
    if(evt == CV_EVENT_LBUTTONDOWN) {
        std::vector<cv::Point>* ptPtr = (std::vector<cv::Point>*)param;
    	//cv::Point* ptPtr = (cv::Point*)param;
          //  ptPtr->x = x;
            //ptPtr->y = y;
        ptPtr->push_back(cv::Point(x,y));

        ind2.push_back(ind_count);
       // cout<<ind_count<<endl;
        ind_count++;

        //cout<<"push"<<ptPtr->size()<<endl;
    }
}


int main(int argc, char *argv[])
{

	ifstream myfile,myfile1;
	string line,line1;
	vector <string> nvmToks1 ,nvmToks2;
	string path, filenameim2;

	string fileNameim = argv[1];

	myfile.open(fileNameim+"/3DCorrespondences.txt");
	getline(myfile, line);
	printf("%s\n", line.c_str());

	int num = stoi(line);

	double *data = (double*)calloc(num*6, sizeof(double));
	int ind = 0;
	while(getline(myfile, line))
	{
		
		vector<double>tmp;
		
		split(nvmToks1, line, " ");

		for(int i = 0 ; i < nvmToks1.size();i++)
		{
			//tmp.push_back(stod(nvmToks1[i]));
			data[ind] = stod(nvmToks1[i]);
			ind++;
		}

		
	}

	myfile.close();

	double w[3];
	double r[9];
	double r1[9];
	double t[3];
	double p[3];
	double temp[3];

	myfile.open(fileNameim+"/RTS2DR3DT.txt");

	getline(myfile, line);
	double scale = stod(line);

	getline(myfile, line);
	split(nvmToks1, line, " ");

	for(int i = 0 ; i < nvmToks1.size();i++)
	{
		r1[i] = stod(nvmToks1[i]);
	}
	
	r[0] = r1[0];
	r[1] = r1[3];
	r[2] = r1[6];
	r[3] = r1[1];
	r[4] = r1[4];
	r[5] = r1[7];
	r[6] = r1[2];
	r[7] = r1[5];
	r[8] = r1[8];

	//cout << r[0]<<" "<<r[1]<<" "<<r[2]<<" "<<r[3]<<" "<<r[4]<<" "<<r[5]<<" "<<r[6]<<" "<<r[7]<<" "<<r[8]<<endl;

	getline(myfile, line);

	split(nvmToks1, line, " ");

	for(int i = 0 ; i < nvmToks1.size();i++)
	{
		t[i] = stod(nvmToks1[i]);
	}

	getline(myfile, line);
	double ransacErr = stod(line);
	//cout<<ransacErr<<endl;

	ceres::RotationMatrixToAngleAxis(r, w);
	
	double w2[3];

	ceres::RotationMatrixToAngleAxis(r1, w2);

	double c[3];
	ceres::AngleAxisRotatePoint(w2, t, c);
	
	c[0] = -1*c[0];
	c[1] = -1*c[1];
	c[2] = -1*c[2];

	double camera[7];

	camera[0] = w[0];
	camera[1] = w[1];
	camera[2] = w[2];
	camera[3] = t[0];
	camera[4] = t[1];
	camera[5] = t[2];
	camera[6] = scale;

	
	ceres::Problem problem;
	double norm = 0, norm2 = 0;

	double error[num][6];
	vector<int>remove;
	
	for (int i = 0; i < num; ++i)
	{

		p[0] = data[i*6 + 3]* scale;// =  data[i*6 + 3]* scale;  
		p[1] = data[i*6 + 4]* scale;//  = data[i*6 + 4]* scale; 
		p[2] = data[i*6 + 5]* scale;// = data[i*6 + 5]* scale; 

		ceres::AngleAxisRotatePoint(w, p, temp);
		//cout<<temp[0]<<" "<<temp[1]<< " "<<temp[2]<<endl;
		//cout<<scale<<endl;
		//cout<<t[0]<<" "<<t[1]<<" "<<t[2]<<endl;
		temp[0] += t[0];
		temp[1] += t[1];
		temp[2] += t[2];
		temp[0] = data[i*6 + 0] - temp[0];
		temp[1] = data[i*6 + 1] - temp[1];
		temp[2] = data[i*6 + 2] - temp[2];

		norm += sqrt(temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2] );
		norm2 = sqrt(temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2] );
	
		//cout << data[i*6 + 0]<<" "<<data[i*6 + 1]<<" "<<data[i*6 + 2]<<" "<<data[i*6 + 3]<<" " <<data[i*6 + 4]<< " " <<data[i*6 + 5]<<endl;	
		//cout << norm2<<endl;

		if(norm2 > ransacErr)
		{
			remove.push_back(i);
		}

		error[i][0] = data[i*6 + 0] - temp[0];
		error[i][1] = data[i*6 + 1] - temp[1];
		error[i][2] = data[i*6 + 2] - temp[2];
	}
	
	int count = 0;
	for (int i = 0; i < num; ++i)
	{

		// Each Residual block takes a point and a camera as input and outputs a 2
		// dimensional residual. Internally, the cost function stores the observed

		// image location and compares the reprojection against the observation.
		
		std::vector<int>::iterator it;

  		it = find (remove.begin(), remove.end(), i);
 		 if (it != remove.end())
			continue;
		
		count++;

		ceres::CostFunction* cost_function = ErrorRT::Create(data+i*6, data+i*6+3);
		//int camid = camidx[i];
		//int pointid = ptidx[i];
		problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.0001) /* squared loss */, camera);
	}
	
	cout <<count<<" "<<num<<endl;
	printf("1. Average: %lf\n", norm/num);

	ceres::Solver::Options options;
	// options.logging_type = ceres::SILENT;
	// options.minimizer_progress_to_stdout = false;
	options.linear_solver_type = ceres::ITERATIVE_SCHUR;
	options.minimizer_progress_to_stdout = true;
	options.max_linear_solver_iterations = 200;

	ceres::Solver::Summary summary;
	//printf("Solving\n");
	ceres::Solve(options, &problem, &summary);

	w[0] = camera[0];
	w[1] = camera[1];
	w[2] = camera[2];
	ceres::AngleAxisToRotationMatrix(w, r);

	norm = 0;
	scale = camera[6];

	double t2[3], c2[3];
	
	c2[0] = camera[3];
	c2[1] = camera[4];
	c2[2] = camera[5];

	w2[0] =camera[0]; 
	w2[1] =camera[1];
	w2[2] =camera[2];

	ceres::AngleAxisRotatePoint(w2,c2, t2);

	t2[0] = -1*t2[0];
	t2[1] = -1*t2[1];
	t2[2] = -1*t2[2];

	//camera[3] = t2[0];
	//camera[4] = t2[1];
	//camera[5] = t2[2];

	FILE *fp = fopen("debug.txt", "w");
	for (int i = 0; i < num; ++i)
	{

		p[0] = data[i*6 + 3] * scale;  
		p[1] = data[i*6 + 4] * scale; 
		p[2] = data[i*6 + 5] * scale; 

		ceres::AngleAxisRotatePoint(w, p, temp);

		temp[0] += camera[3];
		temp[1] += camera[4];
		temp[2] += camera[5];
		temp[0] = data[i*6 + 0] - temp[0];
		temp[1] = data[i*6 + 1] - temp[1];
		temp[2] = data[i*6 + 2] - temp[2];

		norm += sqrt(temp[0] * temp[0] + temp[1] * temp[1] + temp[2] * temp[2] );

		error[i][3] = data[i*6 + 0] - temp[0];
		error[i][4] = data[i*6 + 1] - temp[1];
		error[i][5] = data[i*6 + 2] - temp[2];

		fprintf(fp, "%lf %lf %lf %lf %lf %lf\n", error[i][0], error[i][1], error[i][2], error[i][3], error[i][4], error[i][5]);
	}
	fclose(fp);

	printf("2. Average: %lf\n", norm/num);

	string fileOut = fileNameim + "/RTSRefined_1.txt";

	FILE *fpOut = fopen(fileOut.c_str(), "w");
	fprintf(fpOut, "%.10lf\n%.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf %.10lf\n%.10lf %.10lf %.10lf\n", 
			camera[6], r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8], camera[3], camera[4], camera[5] );
	fclose(fpOut);
}


