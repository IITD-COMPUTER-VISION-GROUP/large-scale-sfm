// cameraLocation.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <ceres/rotation.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <glog/logging.h>
#include <algorithm>
#include<string>
#include<fstream>
//#include<istream>
//#include "gtest/gtest.h"
#include "math_util.h"
#include "random.h"
#include "optimize_relative_position_with_known_rotation.h"
#include "camera.h"
#include "feature_correspondence.h"
#include "test_util.h"
using namespace theia;
using namespace std;
RandomNumberGenerator rng(52);

Camera RandomCamera() {
	Camera camera;
	camera.SetPosition(rng.RandVector3d());
	camera.SetOrientationFromAngleAxis(0.2 * rng.RandVector3d());
	camera.SetImageSize(1000, 1000);
	camera.SetFocalLength(800);
	camera.SetPrincipalPoint(500.0, 500.0);
	return camera;
}
void GetRelativeTranslationFromCameras(const Camera& camera1,
	const Camera& camera2,
	Eigen::Vector3d* relative_position) {
	const Eigen::Vector3d rotated_relative_position =
		camera2.GetPosition() - camera1.GetPosition();
	ceres::AngleAxisRotatePoint(camera1.GetOrientationAsAngleAxis().data(),
		rotated_relative_position.data(),
		relative_position->data());
	relative_position->normalize();
}

void TestOptimization(const Camera& camera1,
	const Camera& camera2,
	const std::vector<Eigen::Vector3d>& world_points,
	const double kPixelNoise,
	const double kTranslationNoise,
	const double kTolerance) {
	// Project points and create feature correspondences.
	std::vector<FeatureCorrespondence> matches;
	for (int i = 0; i < world_points.size(); i++) {
		const Eigen::Vector4d point = world_points[i].homogeneous();
		FeatureCorrespondence match;
		camera1.ProjectPoint(point, &match.feature1);
		camera2.ProjectPoint(point, &match.feature2);
		AddNoiseToProjection(kPixelNoise, &rng, &match.feature1);
		AddNoiseToProjection(kPixelNoise, &rng, &match.feature2);

		// Undo the calibration.
		match.feature1 =
			camera1.PixelToNormalizedCoordinates(match.feature1).hnormalized();
		match.feature2 =
			camera2.PixelToNormalizedCoordinates(match.feature2).hnormalized();
		matches.emplace_back(match);
	}

	Eigen::Vector3d relative_position;
	GetRelativeTranslationFromCameras(camera1, camera2, &relative_position);

	const Eigen::Vector3d gt_relative_position = relative_position;

	// Add noise to relative translation.
	const Eigen::AngleAxisd translation_noise(
		DegToRad(rng.RandGaussian(0.0, kTranslationNoise)),
		Eigen::Vector3d(rng.RandDouble(-1.0, 1.0),
			rng.RandDouble(-1.0, 1.0),
			rng.RandDouble(-1.0, 1.0)));
	relative_position = translation_noise * relative_position;

	CHECK(OptimizeRelativePositionWithKnownRotation(
		matches,
		camera1.GetOrientationAsAngleAxis(),
		camera2.GetOrientationAsAngleAxis(),
		&relative_position));

	const double translation_error = RadToDeg(
		acos(Clamp(gt_relative_position.dot(relative_position), -1.0, 1.0)));
	//std::cout << "hi" << std::endl;
//	EXPECT_LT(translation_error, kTolerance)
	//std::cout	<< "GT Position = " << gt_relative_position.transpose()
		//<< "\nEstimated position = " << relative_position.transpose() << translation_error << std::endl;
}

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


int main(int argc, char **argv)
{
	static const double kTolerance = 1e-6;
	static const double kPixelNoise = 0.0;
	static const double kTranslationNoise = 0.0;
	static const int kNumPoints = 100;
	std::vector<Eigen::Vector3d> points(kNumPoints);

	//FILE *fp = fopen("RC_estimates.txt", "r");
	ifstream myfile1, mylist1;
	string s2;
	vector<string> toks1, toks2;
	vector<vector<double>>R;
	string path =argv[1];
	vector<double>focal;
	string focalfile = path + "/list_focal.txt";
	myfile1.open(focalfile);
	while (getline(myfile1, s2)) {
		split(toks1, s2, " ");
		//cout << s2;
		if (toks1.size() > 2)
			focal.push_back(stod(toks1[2]));
		else
			focal.push_back(stod(toks1[1]));

		cout << focal[focal.size() - 1] << endl;
	}
	myfile1.close();	

	myfile1.open(path + "/R.txt");
	while (getline(myfile1, s2)) {
		split(toks1, s2, " ");
		vector<double>rtmp;
		for (int i = 0; i < 9; i++)
		{
			rtmp.push_back(stod(toks1[i]));
		}
		R.push_back(rtmp);
	}
	myfile1.close();

	vector<pair<int, int>> ids;
	vector<pair<int, int>> pairs;
	string pairfile = path + "/original_pairs_updated.txt";
	mylist1.open(pairfile);
	while (getline(mylist1, s2)) {
		pair<int, int> tmp;
		split(toks1, s2, " ");
		tmp.first = stoi(toks1[0])-1;
		tmp.second = stoi(toks1[1])-1;
		pairs.push_back(tmp);
	}

	mylist1.close();
	//cout << pairs.size() << endl;
	//cout << R.size() << endl;

	string Tfile = path + "/T5Point_updated.txt";
	
	mylist1.open(Tfile);
	//vector<pair<int, int>> pairs;
	vector<vector<double>> T_for_optimization;
	int index = 0;

	while (getline(mylist1, s2)) {
		pair<int, int> tmp;
		split(toks1, s2, " ");
		int id1 = pairs[index].first;
		int id2 = pairs[index].second;

		//if ( id2 >= 160 ) break;

		pairs.push_back(tmp);
		vector<double>rtmp;
		vector<double>ttmp;
		//getline(mylist1, s2);
		//split(toks2, s2, " ");
		
		/*for (int i = 0; i < 3; i++)
		{
			ttmp.push_back(stod(toks2[i]));
			
		}*/

		Eigen::Matrix3d rotation1_t,rotation2_t;
		rotation2_t << R[id2][0], R[id2][3], R[id2][6], R[id2][1], R[id2][4], R[id2][7], R[id2][2], R[id2][5], R[id2][8];
		rotation1_t << R[id1][0], R[id1][3], R[id1][6], R[id1][1], R[id1][4], R[id1][7], R[id1][2], R[id1][5], R[id1][8];
		Eigen::Vector3d tij;
		Eigen::Vector3d cij_ours;
		Eigen::Vector3d tij_optimized;
		
		tij << stod(toks1[0]), stod(toks1[1]), stod(toks1[2]);
		cij_ours = -1 * rotation2_t*tij;
		//cij_ours << 0.4599, 0.0847, 0.8839;
		tij_optimized = rotation1_t*cij_ours;
		//tij_optimized << 0 , 0 , 0;
		//cout << tij_optimized[0] << tij_optimized[1] << tij_optimized[2] << endl;
		ttmp.push_back(tij.x());
		ttmp.push_back(tij.y());
		ttmp.push_back(tij.z());

		//Camera camera1;

		//Eigen::Matrix3d rotation;
		//rotation << R[id1][0], R[id1][1], R[id1][2], R[id1][3], R[id1][4], R[id1][5], R[id1][6], R[id1][7], R[id1][8];
		////rotation << R[id1][0], R[id1][3], R[id1][6], R[id1][1], R[id1][4], R[id1][7], R[id1][2], R[id1][5], R[id1][8];
		//camera1.SetOrientationFromRotationMatrix(rotation);
		//ceres::AngleAxisRotatePoint(camera1.GetOrientationAsAngleAxis().data(),
		//	cij_ours.data(),
		//	tij_optimized.data());
		//ttmp.push_back(tij_optimized.x());
		//ttmp.push_back(tij_optimized.y());
		//ttmp.push_back(tij_optimized.z());


		T_for_optimization.push_back(ttmp);
		/*Camera camera1, camera2;

		Eigen::Matrix3d rotation;
		rotation << R[id1][0], R[id1][1], R[id1][2], R[id1][3], R[id1][4], R[id1][5], R[id1][6], R[id1][7], R[id1][8];
		camera1.SetOrientationFromRotationMatrix(rotation);

		rotation << R[id2][0], R[id2][1], R[id2][2], R[id2][3], R[id2][4], R[id2][5], R[id2][6], R[id2][7], R[id2][8];
		camera2.SetOrientationFromRotationMatrix(rotation);
		Eigen::Vector3d relative_position;

		relative_position << stod(toks2[0]) , stod(toks2[1]) , stod(toks2[2]);*/
		

		/*OptimizeRelativePositionWithKnownRotation(
			matches,
			camera1.GetOrientationAsAngleAxis(),
			camera2.GetOrientationAsAngleAxis(),
			&relative_position)*/

		index++;

	}
	//cout << "T Size : " << T_for_optimization.size() << endl;
	string line;
	myfile1.open(path + "matches_forRT_filtered.txt");
	getline(myfile1, line);
	//cout << line << endl;
	int numTotalMatches = stoi(line);

	string rPos = path + "C5point_updated.txt";
	FILE *fp = fopen(rPos.c_str(), "w");
	index = 0;

	cout << "numTotalMatches " << numTotalMatches << endl;

	for (int i = 0; i < numTotalMatches; i++)
	{
		vector <string> toks;
		getline(myfile1, line);
		split(toks, line, " ");
		int id1 = stoi(toks[0]);
		int id2 = stoi(toks[1]);
		int numMatches = stoi(toks[2]);

		printf("%d %d %d\n", id1, id2, focal.size());

		Camera camera1, camera2;

		Eigen::Matrix3d rotation;
		rotation << R[id1][0], R[id1][1], R[id1][2], R[id1][3], R[id1][4], R[id1][5], R[id1][6], R[id1][7], R[id1][8];
		//rotation << R[id1][0], R[id1][3], R[id1][6], R[id1][1], R[id1][4], R[id1][7], R[id1][2], R[id1][5], R[id1][8];
		camera1.SetOrientationFromRotationMatrix(rotation);

		rotation << R[id2][0], R[id2][1], R[id2][2], R[id2][3], R[id2][4], R[id2][5], R[id2][6], R[id2][7], R[id2][8];
		//rotation << R[id2][0], R[id2][3], R[id2][6], R[id2][1], R[id2][4], R[id2][7], R[id2][2], R[id2][5], R[id2][8];
		camera2.SetOrientationFromRotationMatrix(rotation);

		double focal1, focal2;

		focal1 = focal[id1];
		focal2 = focal[id2];
		std::vector<FeatureCorrespondence> matches;
		
		if ( numMatches != 0 )
		{
			for (int j = 0; j < numMatches; j++)
			{
				vector <string> toks2;
				string line2;
				getline(myfile1, line2);
				split(toks2, line2, " ");

				FeatureCorrespondence match;

				FeatureCorrespondence tmp;
				match.feature1.x() = stod(toks2[1]) / focal1;
				match.feature1.y() = stod(toks2[2]) / focal1;
				match.feature2.x() = stod(toks2[4]) / focal2;
				match.feature2.y() = stod(toks2[5]) / focal2;
				matches.emplace_back(match);
			}

			Eigen::Vector3d relative_position;

			relative_position << T_for_optimization[index][0], T_for_optimization[index][1], T_for_optimization[index][2];

			OptimizeRelativePositionWithKnownRotation(
				matches,
				camera1.GetOrientationAsAngleAxis(),
				camera2.GetOrientationAsAngleAxis(),
				&relative_position);
	

			Eigen::Matrix3d rotation1, rotation2;
			rotation2 << R[id2][0], R[id2][1], R[id2][2], R[id2][3], R[id2][4], R[id2][5], R[id2][6], R[id2][7], R[id2][8];
			rotation1 << R[id1][0], R[id1][1], R[id1][2], R[id1][3], R[id1][4], R[id1][5], R[id1][6], R[id1][7], R[id1][8];

			Eigen::Vector3d cij_ours;

			cij_ours = rotation1.transpose()*relative_position;

			fprintf(fp, "%lf %lf %lf\n", cij_ours.x(), cij_ours.y(), cij_ours.z());

			matches.clear();

			index ++;
		}
	}
	
	printf("Index = %d\n", index);
	fclose(fp);
}

