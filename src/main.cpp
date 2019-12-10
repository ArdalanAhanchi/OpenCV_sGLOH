// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>
#include <ctime>
#include <iostream>

#include "kp.cpp"
#include "sgloh.cpp"

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
	Mat testImage1 = imread("testImage1.jpg");
	Mat testImage2 = imread("testImage2.jpg");
	Mat testImage3 = imread("testImage3.jpg");
	Mat testImage4 = imread("testImage4.jpg");
	//namedWindow("image");
	//imshow("image", image);
	//waitKey(0);
	
	const int n = 2;
	const int m = 8;
	double q = 1.0;
	const bool psi = false;
	double radius = 7;
	double sigma = 1.6;
	//double H[n + 1][m][m];
	const int length = m * (m * n + 1 + (m - 1) * (psi ? 0 : 1));
	//double aych[length];
	//double aych[length];
	//aych.create(length, 1, CV_64F);
	std::vector<KeyPoint> points;
	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10.0, sigma);
	sGLOH* essGLOH = new sGLOH();
	sGLOH::sGLOH_Options options;
	options.m = 8;
	options.n = 2;
	options.psi = false;
	options.v = 0;
	options.sigma = 0.7;
	/*sift->detect(image, points);*/
	Mat siftDescriptors;
	Mat emptyMask;
	// detectAndCompute instead of just detect to allow quality comparison between SIFT descriptor and sGLOH descriptor
	time_t startSIFT = std::time(NULL);
	sift->detectAndCompute(image, emptyMask, points, siftDescriptors);
	time_t stopSIFT = std::time(NULL);
	std::cout << "SIFT took this long to detect and compute:\t\t";
	std::cout << (stopSIFT - startSIFT) << std::endl;
	std::srand(std::time(NULL));
	Mat descriptorsFinal;
	OutputArray descriptors = descriptorsFinal;
	descriptors.create((int)points.size(), length, CV_64F);
	descriptorsFinal = descriptors.getMat();
	Mat gradients = Mat(image.size(), CV_64FC2);
	//for (int x = 0; x < gradients.cols; x++)
	//{
	//	for (int y = 0; y < gradients.rows; y++)
	//	{
	//		double magnitude = std::rand() % 4 + ((double)(std::rand() % 100) / 100);
	//		double orientation = (2 * PI / 360) * (std::rand() % 360 + ((double)(std::rand() % 100) / 100));
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
	//	double sum = 0;
	//	for (int i = 0; i < length; i++)
	//	{
	//		sum += std::pow(aych[i], 2);
	//	}
	//	double norm = std::sqrt(sum);
	//	for (int i = 0; i < length; i++)
	//	{
	//		aych[i] = aych[i] / norm;
	//		//std::cout << aych.at<double>(i) << std::endl;
	//	}
	//}
	Mat emm;
	time_t start_sGLOH = std::time(NULL);
	essGLOH->detectAndCompute(image, points, siftDescriptors, options);
	//calculate_sGLOH_Descriptor(m, n, psi, sigma, gradients, points, emm);
	time_t stop_sGLOH = std::time(NULL);
	std::cout << "sGLOH took this long to detect and compute:\t\t";
	std::cout << (stop_sGLOH - start_sGLOH) << std::endl;
	std::cout << "end of file";
	delete essGLOH;
	//std::cout << descriptorsFinal.row(59) << std::endl;
    //Call the test function to make some good ol pyramids.
	//test(img);
}
