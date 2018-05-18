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

using namespace std;

#define TH_NUM_IMG 10

typedef struct _merge
{
	string firstCluster;
	string secondCluster;
	string newCluster; 
}MergeInfo;

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
	if ( argc < 2 )
	{
		printf("Invalid arguments\n ./a.out <Output DIR>\n");
		return 0;
	}

	string outDir = argv[1];

	string dendogramFolder = outDir + "/dendogram/";
	string clusterListFile = dendogramFolder + "/clustersizefile.txt";
	string mergeListFile = dendogramFolder + "/fullclustermergeinfo.txt";
	
	ifstream myfile1;
	string line;
	vector <string> nvmToks;

	vector<pair<string, int> > clusterList;
	myfile1.open(clusterListFile.c_str());	
	while ( getline(myfile1, line) )
	{
		split(nvmToks, line, " ");
		clusterList.push_back(make_pair(nvmToks[0], stoi(nvmToks[1])));
	}
	myfile1.close();

	
	vector<MergeInfo> mergeList;
	myfile1.open(mergeListFile.c_str());	
	while ( getline(myfile1, line) )
	{
		split(nvmToks, line, " ");
		MergeInfo mI;	
		mI.firstCluster = nvmToks[1];
		mI.secondCluster = nvmToks[2];
		mI.newCluster = nvmToks[0];
		mergeList.push_back(mI);
	}
	myfile1.close();


	for( int i = clusterList.size() - 1; i >= 0 ; i -- )
	{
		if( clusterList[i].second < TH_NUM_IMG )
		{
			for( int j = 0; j < mergeList.size(); j ++ )
			{
				string firstCluster = "cluster" + mergeList[j].firstCluster;
				string secondCluster = "cluster" + mergeList[j].secondCluster;

				if ( firstCluster == clusterList[i].first )
				{
					for ( int k = 0; k < clusterList.size(); k ++ )
					{
						if ( clusterList[k].first == secondCluster )
						{
							string clusterImgList = dendogramFolder + secondCluster + ".txt";
							string clusterImg1st = dendogramFolder + firstCluster + ".txt";
							string cmd = "cp " + clusterImgList + " " + clusterImgList + "_bak";
							system(cmd.c_str());
							cmd = "cat " + clusterImg1st + " >> " + clusterImgList;
							system(cmd.c_str());

							clusterList[k].second += clusterList[i].second;
							
							break;	
						}
					} 

					for ( int k = j + 1; k < mergeList.size(); k ++ )
					{
						if( mergeList[k].firstCluster == mergeList[j].newCluster )
						{
							mergeList[k].firstCluster = mergeList[j].secondCluster;
							break;
						}

						if( mergeList[k].secondCluster == mergeList[j].newCluster )
						{
							mergeList[k].secondCluster = mergeList[j].secondCluster;
							break;
						}
					}

					mergeList.erase(mergeList.begin() + j);

					break;
				}

				if ( secondCluster == clusterList[i].first )
				{

					for ( int k = 0; k < clusterList.size(); k ++ )
					{
						if ( clusterList[k].first == firstCluster )
						{
							string clusterImgList = dendogramFolder + firstCluster + ".txt";
							string clusterImg2nd = dendogramFolder + secondCluster + ".txt";
							string cmd = "cp " + clusterImgList + " " + clusterImgList + "_bak";
							system(cmd.c_str());
							cmd = "cat " + clusterImg2nd + " >> " + clusterImgList;
							system(cmd.c_str());

							clusterList[k].second += clusterList[i].second;
							
							break;	
						}
					} 

					for( int k = j + 1; k < mergeList.size(); k ++ )
					{
						if( mergeList[k].firstCluster == mergeList[j].newCluster )
						{
							mergeList[k].firstCluster = mergeList[j].firstCluster;
							break;
						}

						if( mergeList[k].secondCluster == mergeList[j].newCluster )
						{
							mergeList[k].secondCluster = mergeList[j].firstCluster;
							break;
						}
					}

					mergeList.erase(mergeList.begin() + j);

					break;
				}
			}
		
			clusterList.erase(clusterList.begin() + i);
		}
	}

	//Bak files
	string cmd = "cp " + clusterListFile + " " + clusterListFile + "_bak";
	system(cmd.c_str());
	cmd = "cp " + mergeListFile + " " + mergeListFile + "_bak";
	system(cmd.c_str());

	FILE *fp = fopen(clusterListFile.c_str(), "w");
	for ( int j = 0; j < clusterList.size(); j ++ )
	{
		fprintf(fp, "%s %d\n", clusterList[j].first.c_str(), clusterList[j].second);
	}	
	fclose(fp);

	fp = fopen(mergeListFile.c_str(), "w");
	for ( int j = 0; j < mergeList.size(); j ++ )
	{
		fprintf(fp, "%s %s %s\n", mergeList[j].newCluster.c_str(), mergeList[j].firstCluster.c_str(), mergeList[j].secondCluster.c_str());
	}	
	fclose(fp);
}












	/*for( int i = 0; i < mergeList.size(); i ++ )
	{
		bool f1 = false, f2 = false;
		string firstCluster = "cluster" + mergeList[i].firstCluster;
		string secondCluster = "cluster" + mergeList[i].secondCluster;
		
		for ( int j = 0; j < clusterList.size(); j ++ )
		{
			if ( firstCluster == clusterList[j].first )
			{
				f1 = true;	
			}

			if ( secondCluster == clusterList[j].first )
			{
				f2 = true;	
			}
		}

		if ( f1 && f2 )
		{
			string clusterCommon = dendogramFolder + "cluster_common_" + mergeList[i].firstCluster + "_" + mergeList[i].secondCluster + "_" + mergeList[i].newCluster + ".txt";
			
			myfile1.open(clusterCommon.c_str());
			vector<int> list1, list2;	
			while ( getline(myfile1, line) )
			{
				split(nvmToks, line, " ");
				bool f3 = false, f4 = false;
				for ( int k = 0; k < list1.size(); k ++ )
				{
					if ( stoi(nvmToks[0]) == list1[k] )
					{
						f3 = true;
						break;
					}
				}

				if ( !f3 ) list1.push_back(stoi(nvmToks[0]));
		
				for ( int k = 0; k < list2.size(); k ++ )
				{
					if ( stoi(nvmToks[1]) == list2[k] )
					{
						f4 = true;
						break;
					}
				}

				if ( !f4 ) list2.push_back(stoi(nvmToks[1]));
			}
			myfile1.close();

			cout << list1.size() << "  " << list2.size() << "  " << mergeList[i].firstCluster << "   " << mergeList[i].secondCluster << endl;

			string firstClusterFile = dendogramFolder + firstCluster + ".txt";
			FILE *fp = fopen(firstClusterFile.c_str(), "a");
			for ( int k = 0; k < (list2.size() > 5 ? 5 : list2.size()); k ++ )
			{
				fprintf(fp, "%d 1\n", list2[k]);
			}
			fclose(fp);

			string secondClusterFile = dendogramFolder + secondCluster + ".txt";
			fp = fopen(secondClusterFile.c_str(), "a");
			for ( int k = 0; k < (list1.size() > 5 ? 5 : list1.size()); k ++ )
			{
				fprintf(fp, "%d 2\n", list1[k]);
			}
			fclose(fp);
		} 
		else if ( f1 )
		{
			string clusterCommon = dendogramFolder + "cluster_common_" + mergeList[i].firstCluster + "_" + mergeList[i].secondCluster + "_" + mergeList[i].newCluster + ".txt";
			
			myfile1.open(clusterCommon.c_str());
			vector<int> list2;	
			while ( getline(myfile1, line) )
			{
				split(nvmToks, line, " ");
				bool f4 = false;
				for ( int k = 0; k < list2.size(); k ++ )
				{
					if ( stoi(nvmToks[1]) == list2[k] )
					{
						f4 = true;
						break;
					}
				}

				if ( !f4 ) list2.push_back(stoi(nvmToks[1]));
			}
			myfile1.close();

			cout << list2.size() << "  " << mergeList[i].firstCluster << "   " << mergeList[i].secondCluster << endl;

			string firstClusterFile = dendogramFolder + firstCluster + ".txt";
			FILE *fp = fopen(firstClusterFile.c_str(), "a");
			for ( int k = 0; k < (list2.size() > 10 ? 10 : list2.size()); k ++ )
			{
				fprintf(fp, "%d 1\n", list2[k]);
			}
			fclose(fp);
		}
	}	
}*/


