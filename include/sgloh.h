#ifndef SGLOH_H
#define SGLOH_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>

namespace SGloh
{

const float PI = 3.141592653589793;
float GetM(float q, float x);
float CalculateBin(int r, int d, int i, int m, int n, bool psi, float sigma, cv::Mat& gradients, cv::KeyPoint origin);
float CalculateBinPlus(int r, int d, int i, int m, int n, int v, bool psi, float sigma, cv::Mat& gradients, cv::KeyPoint origin);


struct sGLOH_Options
{
	int m; // number of angular segmentations
	int n; // number of radial segmentations
	int v; // number of bins for sGLOH+
	bool psi; // unique center bin or not
	float sigma; // sigma value to force
};

void detectAndCompute(cv::Mat& _image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray _descriptors, sGLOH_Options options);
void calculate_sGLOH_Descriptor(int m, int n, bool psi, float sigma, cv::Mat& gradients, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray _descriptors);
void calculate_sGLOH_Plus_Descriptor(int m, int n, int v, bool psi, float sigma, cv::Mat& gradients, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray _descriptors);

/*static Ptr<sGLOH> create(int _nfeatures = 0, int _nOctaveLayers = 3,
	float _contrastThreshold = 0.04, float _edgeThreshold = 10.0, float _sigma = 0.7);*/
}

#endif
