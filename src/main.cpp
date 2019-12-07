// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>
#include <ctime>
#include <iostream>

#include "gloh.cpp"

using namespace cv;

const double PI = 3.141592653589793;

struct GradientPixel
{
	int x;
	int y;
	double magnitude;
	double orientation;
};

GradientPixel CalculateGradient(Point2d pixel)
{
	// do stuff
	GradientPixel foo;
	foo.x = 0;
	foo.y = 0;
	foo.magnitude = 0;
	foo.orientation = 0;
	return foo;
}

double GetM(double q, double x)
{
	if (x < q / 2)
		return x;
	return q - x;
}

double CalculateBin(int r, int d, int i, int m, int n, bool psi, double sigma, Mat &gradients, KeyPoint origin)
{
	// get the x and y ranges for the image patch around the keypoint
	int xRange[] = { (int)std::floorf(origin.pt.x) - (int)(origin.size / 2), (int)std::floorf(origin.pt.x) + (int)(origin.size / 2) + 1 };
	int yRange[] = { (int)std::floorf(origin.pt.y) - (int)(origin.size / 2), (int)std::floorf(origin.pt.y) + (int)(origin.size / 2) + 1 };
	
	// floor and ceiling the ranges if they exceed the bounds of the image
	/*xRange[0] = xRange[0] < 0 ? 0 : xRange[0];
	yRange[0] = yRange[0] < 0 ? 0 : yRange[0];
	xRange[1] = xRange[1] > gradients.cols - 1 ? gradients.cols - 1 : xRange[1];
	yRange[1] = yRange[1] > gradients.rows - 1 ? gradients.rows - 1 : yRange[1];
	*/
	int count = 0;

	// limit operations to the current ring and slice
	double ringStart = r * (origin.size / 2) / (n + 1);
	double ringEnd = (r + 1) * (origin.size / 2) / (n + 1);
	double sliceStart = d * 2 * PI / m;
	double sliceEnd = (d + 1) * 2 * PI / m;
	if (!psi)
	{
		sliceStart = 0;
		sliceEnd = 2 * PI;
	}
	double result = 0;
	for (int k = xRange[0]; k <= xRange[1]; k++)
	{
		for (int j = yRange[0]; j <= yRange[1]; j++)
		{
			double rho = (std::sqrt((pow((double)k - (double)origin.pt.x, 2) + pow((double)j - (double)origin.pt.y, 2))));
			double theta = std::atan2((double)j - (double)origin.pt.y, (double)k - (double)origin.pt.x) + PI;
			if (rho > ringStart && rho <= ringEnd && theta > sliceStart && theta <= sliceEnd)
			{
				double power = -std::pow(GetM(2 * PI, gradients.at<Vec2d>(j, k)[1] - (2 * PI * i / m)), 2) / std::pow(2 * sigma, 2);
				
				result += gradients.at<Vec2d>(j, k)[0] * std::exp(power);
				count++;
			}
		}
	}
	/*for (int a = 0; a < gradientsCount; a++)
	{
		GradientPixel current = gradients[a];
		double rho = std::log(std::sqrt(std::pow(current.x, 2) + std::pow(current.y, 2)));
		double theta = 0;
		theta = std::atan2((double)current.y, (double)current.x) + PI;
		
		if (rho >= ringStart && rho < ringEnd && theta >= sliceStart && theta < sliceEnd)
		{
			double power = -std::pow(GetM(2 * PI, current.orientation - (2 * PI * i / m)), 2) / std::pow(2 * sigma, 2);
			result += current.magnitude * std::exp(power);
			count++;
		}
	}*/
	result *= (1 / std::sqrt(2 * PI) * sigma);
	return result;
}


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
	namedWindow("image");
	imshow("image", image);
	waitKey(0);
	
	const int n = 2;
	const int m = 8;
	double q = 1.0;
	const bool psi = false;
	double radius = 7;
	double sigma = 1.6;
	double H[n + 1][m][m];
	const int length = m * (m * n + 1 + (m - 1) * (psi ? 1 : 0));
	//double aych[length];
	double aych[length];
	//aych.create(length, 1, CV_64F);
	std::vector<KeyPoint> points;
	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10.0, sigma);
	/*sift->detect(image, points);*/
	Mat siftDescriptors;
	Mat emptyMask;
	// detectAndCompute instead of just detect to allow quality comparison between SIFT descriptor and sGLOH descriptor
	sift->detectAndCompute(image, emptyMask, points, siftDescriptors);
	std::srand(std::time(NULL));
	Mat descriptorsFinal;
	OutputArray descriptors = descriptorsFinal;
	descriptors.create((int)points.size(), length, CV_64F);
	descriptorsFinal = descriptors.getMat();
	Mat gradients = Mat(image.size(), CV_64FC2);
	for (int x = 0; x < gradients.cols; x++)
	{
		for (int y = 0; y < gradients.rows; y++)
		{
			double magnitude = std::rand() % 4 + ((double)(std::rand() % 100) / 100);
			double orientation = (2 * PI / 360) * (std::rand() % 360 + ((double)(std::rand() % 100) / 100));
			Vec2d current = Vec2d(magnitude, orientation);
			
			gradients.at<Vec2d>(y, x)[0] = current[0];
			gradients.at<Vec2d>(y, x)[1] = current[1];
		}
	}
	for (int keypoint = 59; keypoint < 60/*(int)points.size()*/; keypoint++)
	{
		int counter = 0;
		if (psi)
		{
			for (int d = 0; d < m; d++)
			{
				for (int i = 0; i < m; i++)
				{
					int index = counter + ((i + d) % m);
					//std::cout << index << std::endl;
					aych[index] = CalculateBin(0, d, i, m, n, psi, sigma, gradients, points[keypoint]);
					//std::cout << aych[index] << std::endl;
					//H[0][d][i] = aych[index];
					//H[0][d][i] = CalculateBin(0, d, i, m, n, sigma, gradients, points[keypoint]);

				}
				counter += m;
			}
		}
		else
		{
			for (int i = 0; i < m; i++)
			{
				int index = counter + ((i + 0) % m);
				//std::cout << index << std::endl;
				aych[index] = CalculateBin(0, 0, i, m, n, psi, sigma, gradients, points[keypoint]);
				//std::cout << aych[index] << std::endl;
				//H[0][0][i] = aych[index];
				//H[0][0][i] = CalculateBin(0, 0, i, m, n, sigma, gradients, points[keypoint]);
			}
			counter += m;
		}

		for (int r = 1; r <= n; r++)
		{
			for (int d = 0; d < m; d++)
			{
				for (int i = 0; i < m; i++)
				{
					int index = counter + ((i + d) % m);
					//std::cout << index << std::endl;
					aych[index] = CalculateBin(r, d, i, m, n, true, sigma, gradients, points[keypoint]);
					//std::cout << aych[index] << std::endl;
					//H[r][d][i] = aych[index];
					//H[r][d][i] = CalculateBin(r, d, i, m, n, sigma, gradients, points[keypoint]);

				}
				counter += m;
			}
		}

		// reduce descriptor vector to unit length
		double sum = 0;
		for (int i = 0; i < length; i++)
		{
			sum += aych[i];
		}
		for (int i = 0; i < length; i++)
		{
			aych[i] = aych[i] * q / sum;
			//std::cout << aych.at<double>(i) << std::endl;
		}
		double* singleDescriptor = descriptorsFinal.ptr<double>(keypoint);
		singleDescriptor = aych;
	}
	std::cout << descriptorsFinal.row(59) << std::endl;
    //Call the test function to make some good ol pyramids.
	//test(img);
}
