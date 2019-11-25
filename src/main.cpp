// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>
#include <iostream>

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

double CalculateBin(int r, int d, int i, int m, int n, double radius, double sigma, GradientPixel gradients[], int gradientsCount)
{
	int px = 50;
	int py = 50;
	int count = 0;
	double ringStart = r * radius / (n + 1);
	double ringEnd = (r + 1) * radius / (n + 1);
	double sliceStart = d * 2 * PI / m;
	double sliceEnd = (d + 1) * 2 * PI / m;
	double result = 0;
	for (int a = 0; a < gradientsCount; a++)
	{
		GradientPixel current = gradients[a];
		double rho = std::log(std::sqrt(std::pow(current.x, 2) + std::pow(current.y, 2)));
		double theta = 0;
		if (current.x > 0 && current.y > 0)
		{
			theta = std::atan((double)((double)current.y / (double)current.x));
		}
		else if (current.x == 0 && current.y > 0)
		{
			theta = PI / 2;
		}
		/*else if (current.x == 0 && current.y < 0)
		{
			theta = 3 * PI / 2;
		}*/
		if (rho >= ringStart && rho < ringEnd && theta >= sliceStart && theta < sliceEnd)
		{
			double power = -std::pow(GetM(2 * PI, current.orientation - (2 * PI * i / m)), 2) / std::pow(2 * sigma, 2);
			result += current.magnitude * std::exp(power);
			count++;
		}
	}
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
	const int m = 4;
	bool psi = false; 
	double radius = 5.4;
	double sigma = 1.6;
	double H[n + 1][m][m];
	GradientPixel gradients[100];
	std::vector<KeyPoint> points;
	Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10.0, sigma);
	/*sift->detect(image, points);*/
	Mat siftDescriptors;
	Mat emptyMask;
	// detectAndCompute instead of just detect to allow quality comparison between SIFT descriptor and sGLOH descriptor
	sift->detectAndCompute(image, emptyMask, points, siftDescriptors);
	
	for (int x = 0; x < 10; x++)
	{
		for (int y = 0; y < 10; y++)
		{
			GradientPixel current;
			current.x = x;
			current.y = y;
			current.magnitude = std::rand() % 4 + ((double)(std::rand() % 100) / 100);
			current.orientation = (2 * PI / 360) * (std::rand() % 360 + ((double)(std::rand() % 100) / 100));
			gradients[y * 10 + x] = current;
		}
	}

	if (psi)
	{
		for (int d = 0; d < m; d++)
		{
			for (int i = 0; i < m; i++)
			{
				H[0][d][i] = CalculateBin(0, d, i, m, n, radius, sigma, gradients, 100);
			}
		}
	}
	else
	{
		for (int i = 0; i < m; i++)
		{
			H[0][0][i] = CalculateBin(0, 0, i, m, n, radius, sigma, gradients, 100);
		}
		
		for (int d = 1; d < m; d++)
		{
			for (int i = 0; i < m; i++)
			{
				H[0][d][i] = 0;
			}
		}
	}

	for (int r = 1; r <= n; r++)
	{
		for (int d = 0; d < m; d++)
		{
			for (int i = 0; i < m; i++)
			{
				H[r][d][i] = CalculateBin(r, d, i, m, n, radius, sigma, gradients, 100);
			}
		}
	}
	
	
}
