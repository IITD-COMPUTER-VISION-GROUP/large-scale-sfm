#include <string.h>
#include "utils.h"
#include <list>

using namespace std;
int whatToCalibrate;
int nActualCameras;        /**< Actual number of views in dataset. */
int nWorkingCameras;       /**< Number of views with which we are working in current iteration. */

// Variables for maintaining world points and their projections
string strWorldPtsProjectionFile = "worldPtProjections.txt";
int nWorldPts;                                 /**< Number of world points.  */
int nFeatureCorrs;                             /**< Maximum number of feature correspondences for a single world point. */
Feature3Vec vecOfWorldPts;   
Feature3Vec vecOfWorldPtsTmp;   /**< Current copy of the world points. */
vector<Feature3Vec> vecOfWorldPtProjs;         /**< Current copy of the world point projections. */
vector<vector <int>> vecOfsiftids;
// Variables for maintaining camera parameters
string strIntrinsicParamsFile;
string strExtrinsicParamsFile;

int nProjPtParams;
int nValidWorldPtProjs;
int nSingleCamParams, nWorldPtParams;
bool amCalibratingStructure, amCalibratingCameraIntrinsics, amCalibratingCameraOrientations, amCalibratingCameraPositions;
//FeatureList3 worldPtsArr;        /**< Array containing the world point estimates. */
//FeatureList* featureCorrsArr;    /**< Array of feature correspondence arrays. */
char *pProjVisibilityMask = NULL;       /**< Binary array indicating whether pt i is visible in image j or not. */
double *pWorldPtProjections;     /**< Array containing all valid world point projections. */
double *pInitialParamEstimate;   /**< Initial parameter vector consisting of camera parameters and projection pts. */
double *pInitialParamEstimate2;
double *global_last_ws;
double *global_last_Rs;
double *global_Rs;
double curErrorThresh = 1.0;
bool isFirstIteration = true;
double *pCurrentCameraEstimate = NULL;   /**< Current camera parameter vector. */
double *pCurrentWorldPtEstimate = NULL;   /**< Current world point parameter vector. */
int *pProjectionCount = NULL;
 Feature3Vec color;                     /**< Current copy of the world points. */
 Feature3Vec color1; 
 vector<Feature3Vec> vecOfWorldPtProjs1; 
 vector<Feature3Vec> vecOfWorldPtProjs2; 
  Feature3Vec vecOfWorldPts1;
// Utility matrices
  int num_observation_projection = 0;

int verbosityLevel;                   /**< Verbosity level of sparse bundle adjustment. */
double *pMinimizationOptions;         /**< Options configured for performing sparse bundle adjustment based minimization. */
double *pOutputInfoOptions;           /**< After performing sparse bundle adjustment, contains info regarding outcome of minimization. */

int num_projections = 0;

vector<int>inValidWorld;
vector<int>revmapworld;
vector<int>inValidCam;
 map<long, long>currentWorldIndex;

 vector<int>remapcam;

 vector<int>remapworld;

 vector<pair<int,string>>inv_mapping;

 void resetall()
 {
     inValidWorld.clear();
     inValidCam.clear();
     remapcam.clear();
       remapworld.clear();
       inv_mapping.clear();
       currentWorldIndex.clear();
 }


// Computes the camera matrix and its components (camera intrinsics and extrinsics). 



// Constructs a table of correspondences (with invalid corrs., if required).

// Loads a matrix from specified file

/* Find the median in a set of doubles (*/
double median_copy(int n, double *arr) {
    return kth_element_copy(n, n / 2, arr);
}

/* Find the kth element without changing the array */
double kth_element_copy(int n, int k, double *arr) {
    double *arr_copy = (double *)malloc(sizeof(double) * n);
    double kth_best;

    memcpy(arr_copy, arr, sizeof(double) * n);
    kth_best = kth_element(n, k, arr_copy);
    free(arr_copy);

    return kth_best;
}

/* Find the kth element in an unordered list of doubles */
double kth_element(int n, int k, double *arr) {
    if (k >= n) {
	printf("[kth_element] Error: k should be < n\n");
	return 0.0;
    } else {
	int split = partition(n, arr);
	if (k == split)
	    return arr[split];
	else if (k < split)
	    return kth_element(split, k, arr);
	else
	    return kth_element(n - split - 1, k - split - 1, arr + split + 1);
    }
}

/* Return the closest integer to x, rounding up */
int iround(double x) {
    if (x < 0.0) {
	return (int) (x - 0.5);
    } else {
	return (int) (x + 0.5);
    }
}

static int partition(int n, double *arr) {
    int pivot_idx = n / 2;
    double pivot = arr[pivot_idx];
    double tmp;
    int i, store_index;

    /* Swap pivot and end of array */
    tmp = arr[n-1];
    arr[n-1] = pivot;
    arr[pivot_idx] = tmp;
    
    store_index = 0;
    for (i = 0; i < n-1; i++) {
	if (arr[i] < pivot) {
	    tmp = arr[store_index];
	    arr[store_index] = arr[i];
	    arr[i] = tmp;
	    store_index++;
	}
    }
    
    tmp = arr[store_index];
    arr[store_index] = arr[n-1];
    arr[n-1] = tmp;
    
    return store_index;
}
