// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <ctime>
#include <iostream>

#include "sgloh.h"
#include "sGLOH_Options.h"
#include "kp.h"
#include "gradient.h"

using namespace cv;



int main(int argc, char** argv)
{
	std::cout << "Hello OpenCV!" << std::endl;
	// for each circular feature patch f there is a descriptor H
	// for all pixels in f determine via gaussian the gradient magnitude Gm and orientation Gd
	// steps for creating descriptor H:
	// first define n, m, and psi, where n is the number of descriptor rings, m is the number of regions per ring, and psi is whether or not the
	// innermost ring has one or m regions (0 or 1 respectively)
	// second, each region is broken up into m histogram bins
	// for each region R sub (r, d) where r is defined as the integer set from 1 to n and d is defined as the integer set from 0 to m - 1
	// the ith histogram bin value h sub (r, d) is defined as
	// sum for each pixel p in R sub (r, d) (Gm(p) *
	Mat image = imread("foreground.jpg");
	Mat testImage1 = imread("foreground.jpg");
	Mat testImage2 = imread("foreground.jpg");
	Mat testImage3 = imread("testImage3.jpg");
	Mat testImage4 = imread("testImage4.jpg");
	//namedWindow("image");
	//imshow("image", image);
	//waitKey(0);
	const int n = 2;
	const int m = 8;
	float q = 1.0;
	const bool psi = false;
	float radius = 7;
	float sigma = 1.6;
	//float H[n + 1][m][m];
	const int length = m * (m * n + 1 + (m - 1) * (psi ? 0 : 1));
	//float aych[length];
	//float aych[length];
	//aych.create(length, 1, CV_64F);
	std::vector<KeyPoint> points;
	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10.0, sigma);
	//sGLOH* essGLOH;// = new sGLOH();
	//sGLOH_Options options;
	//options.setOptions(8, 2, 0, false, 0.7f);
	/*sift->detect(image, points);*/
	Mat siftDescriptors;
	Mat emptyMask;
	// detectAndCompute instead of just detect to allow quality comparison between SIFT descriptor and sGLOH descriptor
	time_t startSIFT = std::time(NULL);
	sift->detectAndCompute(testImage1, emptyMask, points, siftDescriptors);
	time_t stopSIFT = std::time(NULL);
	std::cout << "SIFT took this long to detect and compute:\t\t";
	std::cout << (stopSIFT - startSIFT) << std::endl;
	std::srand(std::time(NULL));
	//Mat descriptorsFinal;
	//OutputArray descriptors = descriptorsFinal;
	//descriptors.create((int)points.size(), length, CV_32F);
	//descriptorsFinal = descriptors.getMat();
	//Mat gradients = Mat(image.size(), CV_64FC2);
	//for (int x = 0; x < gradients.cols; x++)
	//{
	//	for (int y = 0; y < gradients.rows; y++)
	//	{
	//		float magnitude = std::rand() % 4 + ((float)(std::rand() % 100) / 100);
	//		float orientation = (2 * PI / 360) * (std::rand() % 360 + ((float)(std::rand() % 100) / 100));
	//		Vec2d current = Vec2d(magnitude, orientation);
	//
	//		gradients.at<Vec2d>(y, x)[0] = current[0];
	//		gradients.at<Vec2d>(y, x)[1] = current[1];
	//	}
	//}
	//for (int keypoint = 59; keypoint < 60/*(int)points.size()*/; keypoint++)
	//{
	//	int counter = 0;
	//	if (psi)
	//	{
	//		for (int i = 0; i < m; i++)
	//		{
	//			int index = counter + ((i + 0) % m);
	//			//std::cout << index << std::endl;
	//			aych[index] = CalculateBin(0, 0, i, m, n, psi, sigma, gradients, points[keypoint]);
	//			//std::cout << aych[index] << std::endl;
	//			//H[0][0][i] = aych[index];
	//			//H[0][0][i] = CalculateBin(0, 0, i, m, n, sigma, gradients, points[keypoint]);
	//		}
	//		counter += m;
	//	}
	//	else
	//	{
	//		for (int d = 0; d < m; d++)
	//		{
	//			for (int i = 0; i < m; i++)
	//			{
	//				int index = counter + ((i + d) % m);
	//				//std::cout << index << std::endl;
	//				aych[index] = CalculateBin(0, d, i, m, n, psi, sigma, gradients, points[keypoint]);
	//				//std::cout << aych[index] << std::endl;
	//				//H[0][d][i] = aych[index];
	//				//H[0][d][i] = CalculateBin(0, d, i, m, n, sigma, gradients, points[keypoint]);

	//			}
	//			counter += m;
	//		}
	//	}

	//	for (int r = 1; r <= n; r++)
	//	{
	//		for (int d = 0; d < m; d++)
	//		{
	//			for (int i = 0; i < m; i++)
	//			{
	//				int index = counter + ((i + d) % m);
	//				//std::cout << index << std::endl;
	//				aych[index] = CalculateBin(r, d, i, m, n, psi, sigma, gradients, points[keypoint]);
	//				//std::cout << aych[index] << std::endl;
	//				//H[r][d][i] = aych[index];
	//				//H[r][d][i] = CalculateBin(r, d, i, m, n, sigma, gradients, points[keypoint]);

	//			}
	//			counter += m;
	//		}
	//	}

	//	// reduce descriptor vector to unit length
	//	float sum = 0;
	//	for (int i = 0; i < length; i++)
	//	{
	//		sum += std::pow(aych[i], 2);
	//	}
	//	float norm = std::sqrt(sum);
	//	for (int i = 0; i < length; i++)
	//	{
	//		aych[i] = aych[i] / norm;
	//		//std::cout << aych.at<float>(i) << std::endl;
	//	}
	//}
	Mat emm1, emm2;
	std::vector<KeyPoint> tP1, tP2;
	time_t start_sGLOH = std::time(NULL);
	sGLOH::detectAndCompute(testImage1, tP1, emm1, m, n, psi, sigma);
	sGLOH::detectAndCompute(testImage2, tP2, emm2, m, n, psi, sigma);
	//calculate_sGLOH_Descriptor(m, n, psi, sigma, gradients, points, emm);
	time_t stop_sGLOH = std::time(NULL);
	std::cout << "sGLOH took this long to detect and compute:\t\t";
	std::cout << (stop_sGLOH - start_sGLOH) << std::endl;
	std::cout << "end of file";
	//std::vector<DMatch> matches;

	//for (int i = 0; i < emm1.rows; i++)
	//{
	//	DMatch curr = DMatch();
	//	curr.distance = 1000000;
	//	for (int j = 0; j < emm2.rows; j++)
	//	{
	//		// get distance
	//		float sumSquares = 0;
	//		for (int k = 0; k < emm1.cols; k++)
	//		{
	//			sumSquares += std::pow(emm1.at<float>(i, k) - emm2.at<float>(j, k), 2);
	//		}
	//		float tempDistance = std::sqrt(sumSquares);
	//		if (tempDistance < curr.distance)
	//		{
	//			curr.distance = tempDistance;
	//			curr.queryIdx = i;
	//			curr.trainIdx = j;
	//		}
	//	}
	//	if (curr.distance < 1000000)
	//	{
	//		matches.push_back(curr);
	//	}
	//}
	time_t start_Match = std::time(NULL);
	Ptr<BFMatcher> bruteForceMatcher = BFMatcher::create();
	std::vector<std::vector<DMatch>> matches;
	matches.resize((size_t)m);
	size_t sizeSum = 0;
	for (int h = 0; h < m; h++)
	{
		//bruteForceMatcher->match(emm1, emm2, matches[i]);
		for (int i = 0; i < emm1.rows; i++)
		{
			DMatch curr = DMatch();
			curr.distance = 1000000;
			for (int j = 0; j < emm2.rows; j++)
			{
				// get distance
				float sumSquares = 0;
				for (int k = 0; k < emm1.cols; k++)
				{
					float testF1 = emm1.at<float>(i, k);
					float testF2 = emm2.at<float>(j, k);
					sumSquares += std::pow(emm1.at<float>(i, k) - emm2.at<float>(j, k), 2);
				}
				float tempDistance = std::sqrt(sumSquares);
				if (tempDistance < curr.distance)
				{
					curr.distance = tempDistance;
					curr.queryIdx = i;
					curr.trainIdx = j;
				}
			}
			if (curr.distance < 1000000)
			{
				matches[h].push_back(curr);
			}
		}
		sGLOH::rotateDescriptors(emm1.clone(), emm1, m, n, psi, sigma);
	}
	std::vector<DMatch> bestMatches;
	//bestMatches.resize(sizeSum);
	int i = 0;
	for (int j = 0; j < m; j++)
	{
		for (int k = 0; k < (int)matches[j].size(); k++)
		{
			DMatch curr(matches[j][k]);
			for (int l = j + 1; l < m; l++)
			{
				for (int m = 0; m < (int)matches[l].size(); m++)
				{
					if (curr.queryIdx == matches[l][m].queryIdx &&
						curr.trainIdx == matches[l][m].trainIdx &&
						matches[l][m] < curr)
					{
						curr = matches[l][m];
					}
				}
			}
			if (curr.distance >= 0)
			{
				bestMatches.push_back(curr);
			}
		}
	}
	time_t stop_Match = std::time(NULL);
	std::cout << "Brute force matching took this long:\t\t";
	std::cout << (stop_Match - start_Match) << std::endl;
	std::cout << "end of file";

	//Ptr<BFMatcher> bruteForceMatcher = BFMatcher::create();
	//std::vector<std::vector<DMatch>> matches;
	//bruteForceMatcher->knnMatch(emm1, emm2, matches, 5);
	Mat matchedImage1, matchedImage2, matchedImage3, matchedImage4, matchedImage5;
	drawMatches(testImage1, tP1, testImage2, tP2, bestMatches, matchedImage1);
	imshow("matches1", matchedImage1);
	namedWindow("matches1", WINDOW_NORMAL);
	waitKey(0);
	//drawMatches(testImage1, tP1, testImage2, tP2, matches[1], matchedImage2);
	//imshow("matches2", matchedImage2);
	//namedWindow("matches2", WINDOW_NORMAL);
	//waitKey(0);
	//drawMatches(testImage1, tP1, testImage2, tP2, matches[2], matchedImage3);
	//imshow("matches3", matchedImage3);
	//namedWindow("matches3", WINDOW_NORMAL);
	//waitKey(0);
	//drawMatches(testImage1, tP1, testImage2, tP2, matches[3], matchedImage4);
	//imshow("matches4", matchedImage4);
	//namedWindow("matches4", WINDOW_NORMAL);
	//waitKey(0);
	//drawMatches(testImage1, tP1, testImage2, tP2, matches[4], matchedImage5);
	//imshow("matches5", matchedImage5);
	//namedWindow("matches5", WINDOW_NORMAL);
	//waitKey(0);
	imwrite("result1.jpg", matchedImage1);
	//imwrite("result2.jpg", matchedImage2);
	//imwrite("result3.jpg", matchedImage3);
	//imwrite("result4.jpg", matchedImage4);
	//imwrite("result5.jpg", matchedImage5);
//	delete essGLOH;
	/*std::cout << descriptorsFinal.row(59) << std::endl;
    Call the test function to make some good ol pyramids.
	test(img);*/
}
