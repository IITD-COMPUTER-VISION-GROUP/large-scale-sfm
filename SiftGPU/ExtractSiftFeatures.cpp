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
#include "SiftGPU.h"

#include <opencv2/opencv.hpp>

using namespace std;

#define FREE_MYLIB dlclose
#define GET_MYPROC dlsym
#define PBSTR "||||||||||||||||||||||||||||||||||||||||||||||||||||||||||||"
#define PBWIDTH 60

#define FX 	517.3
#define FY	516.5

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
	SiftMatchGPU* matcher = pCreateNewSiftMatchGPU(20000);

	char * argv_tmp[] = {"-fo", "-1", "-tc2", "7680", "-v", "0", "-da","-nomc"};

 	//char * argv_tmp[] = {"-fo", "-1", "-loweo", "-da", "-v", "0", "-w", "3", "-maxd", "5120"};

	//char * argv_tmp[] = {"-fo", "-1", "-da", "-v", "0", "-maxd", "4096"};//
	//char * argv_tmp[] = {"-fo", "-1", "-da",  "-v", "0", "-cuda", "[0]"};//
	int argc_tmp = sizeof(argv_tmp)/sizeof(char*);
	sift->ParseParam(argc_tmp, argv_tmp);

	if(sift->CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) return 0;

	if ( argc < 3 )
	{
		printf("Invalid arguments\n ./a.out <input folder rgb.txt> <output folder>\n");
		return 0;
	}

	string inputPath = argv[1];
	string rgbTxt = inputPath + "/rgb.txt";
	string outputPath = argv[2];
	string outKeyPath = outputPath + "/sift/";
	string keyFileList = outputPath + "/KeyList.txt";
	string keyFileList1 = outputPath + "/KeyList1.txt";
	//string focalFile = outputPath + "/list_focal.txt";

	if ( 0 != mkdir(outKeyPath.c_str(), 0777))
	{
		printf("Unable to create dir.\n");
	}

	ifstream myfile1;
	string line;
	myfile1.open(rgbTxt.c_str());
	vector <string> nvmToks;
	int totalFrames = 0;

	while(getline(myfile1, line))
	{
		totalFrames ++;
	}
	myfile1.close();
	myfile1.open(rgbTxt.c_str());

	vector<float > descriptors(1);
	vector<SiftGPU::SiftKeypoint> keys(1);
	int num = 0;

	FILE *fp = fopen(keyFileList.c_str(), "w");
	if ( NULL == fp )
	{
		printf("Unable to create list file.\n");
		return -1;
	}

	FILE *fp1 = fopen(keyFileList1.c_str(), "w");
	if ( NULL == fp1 )
	{
		printf("Unable to create list file.\n");
		return -1;
	}

	/*FILE *fp1 = fopen(focalFile.c_str(), "w");
	if ( NULL == fp1 )
	{
		printf("Unable to create focal file.\n");
		return -1;
	}*/


	int frameCount = 0;
	printf("\n");

	long long int nSiftPoints = 0;

	while(getline(myfile1, line))
	{
		split(nvmToks, line, " ");
		string imgName, siftFileName;

		if ( nvmToks.size()  == 1 )
		{
			imgName = inputPath + "/rgb/" + nvmToks[0];
		}else
		{
			imgName = inputPath + "/rgb/" + nvmToks[1];
		}

		size_t lastindex = nvmToks[0].find_last_of("."); 
		siftFileName = nvmToks[0].substr(0, lastindex); 
		siftFileName = "sift/" + siftFileName + ".sift";
		string siftFullFileName = outputPath + siftFileName;

		if(sift->RunSIFT(imgName.c_str()))
		{
			sift->SaveSIFT(siftFullFileName.c_str());

			nSiftPoints += sift->GetFeatureNum();
		}

		cv::Mat image = cv::imread(imgName.c_str());
		cv::Size s = image.size();
		int rows = s.height;
		int cols = s.width;

		fprintf(fp, "%s %d %d %d\n", siftFullFileName.c_str(), rows, cols, sift->GetFeatureNum());
		fflush(fp);

		fprintf(fp1, "%s\n", siftFullFileName.c_str(), rows, cols, sift->GetFeatureNum());
		fflush(fp1);

		frameCount ++;

		printProgress(frameCount, totalFrames);
	}
	printf("\nDone.\n");

	fclose(fp);
	fclose(fp1);
	myfile1.close();

	string cntFile = outputPath + "/sift_count.txt";
	FILE *fpCnt = fopen(cntFile.c_str(), "w");
	fprintf(fpCnt, "%lld", nSiftPoints);
	fclose(fpCnt);

	delete sift;
	delete matcher;

	FREE_MYLIB(hsiftgpu);
	return 1;
}
