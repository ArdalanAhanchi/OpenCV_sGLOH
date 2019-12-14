#ifndef SGLOH_H
#define SGLOH_H

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>

#include "gradient.h"
#include "options.h"
#include "kp.h"

namespace SGloh
{

const float PI = 3.141592653589793;

float GetM(float q, float x);
float CalculateBin(int r, int d, int i, int m, int n, bool psi, float sigma, cv::Mat& gradients, cv::KeyPoint origin);
float CalculateBinPlus(int r, int d, int i, int m, int n, int v, bool psi, float sigma, cv::Mat& gradients, cv::KeyPoint origin);
void detectAndCompute(cv::Mat& _image, std::vector<cv::KeyPoint>& keypoints, cv::Mat& _descriptors, Options options);
void calculate_sGLOH_Descriptor(int m, int n, bool psi, float sigma, cv::Mat& gradients, std::vector<cv::KeyPoint>& keypoints, cv::Mat& _descriptors);
void calculate_sGLOH_Plus_Descriptor(int m, int n, int v, bool psi, float sigma, cv::Mat& gradients, std::vector<cv::KeyPoint>& keypoints, cv::Mat& _descriptors);
void rotateDescriptors(cv::Mat descriptors, cv::Mat& rotated, Options options);
/*static Ptr<sGLOH> create(int _nfeatures = 0, int _nOctaveLayers = 3,
	float _contrastThreshold = 0.04, float _edgeThreshold = 10.0, float _sigma = 0.7);*/
}

#endif
