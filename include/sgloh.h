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

/** Value of PI used in calculations. */
const float PI = 3.141592653589793;


/**
 *  A function which calculates the value of the descriptor at the given bin.
 *  after the calculations are finished, the value of the final results of the
 *  current bin (At origin) will be returned.
 */
float calculateBin(int r, int d, int i, int m, int n, bool psi,
    float sigma, cv::Mat& gradients, cv::KeyPoint origin);


/**
 *  A function which extracts keypoints (Based on the options, the extraction
 *  method will be different. Then it will calculate the descriptors.
 *  If keypoints are provided, only the descriptors will be calculated.
 *
 *  @param _image The source image used in this operation.
 *  @param keypoints The vector of keypoint which will be populated (if it isn't)
 *  @param options The options required to run this detector.
 */
void detectAndCompute(cv::Mat& _image, std::vector<cv::KeyPoint>& keypoints,
    cv::Mat& _descriptors, Options options);


/**
 *  The base function which will calculate the sGLOH descriptor based on the
 *  passed parameters. It will populate the descriptors matrix with the proper
 *  values for the descriptor. Prior to calling this functions, the keypoints
 *  should be calculated.
 */
void calculate_sGLOH_Descriptor(int m, int n, bool psi, float sigma,
    cv::Mat& gradients, std::vector<cv::KeyPoint>& keypoints, cv::Mat& _descriptors);


/**
 *  A function which Discretely rotates a descriptor vector by one increment of 2*PI/m
 *  The original options should be passed to it so it can access the m value.
 *  The rotated descriptor will be written to rotated matrix.
 */
void rotateDescriptors(cv::Mat descriptors, cv::Mat& rotated, Options options);


/**
 *  A function which calculates the M values specified in the original paper.
 *  If the value of x is smaller than half of q, it return x, otherwise it will
 *  return q-x.
 */
float getM(float q, float x);

}

#endif
