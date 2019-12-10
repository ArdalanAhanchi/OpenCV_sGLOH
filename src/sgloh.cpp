#include "sgloh.h"
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
using namespace cv;

float sGLOH::GetM(float q, float x)
{
	if (x < q / 2)
		return x;
	return q - x;
}

float sGLOH::CalculateBin(int r, int d, int i, int m, int n, bool psi, float sigma, Mat& gradients, KeyPoint origin)
{
	// get the x and y ranges for the image patch around the keypoint
	// floor the range borders if they exceed the bounds of the image
	int xRange[] = { (int)std::floorf(origin.pt.x) - (int)(origin.size / 2), (int)std::floorf(origin.pt.x) + (int)(origin.size / 2) + 1 };
	int yRange[] = { (int)std::floorf(origin.pt.y) - (int)(origin.size / 2), (int)std::floorf(origin.pt.y) + (int)(origin.size / 2) + 1 };


	// limit operations to the current ring and slice
	float ringStart = r * (origin.size / 2) / (n + 1);
	float ringEnd = (r + 1) * (origin.size / 2) / (n + 1);
	float sliceStart = d * 2 * PI / m;
	float sliceEnd = (d + 1) * 2 * PI / m;
	if (psi && r == 0 && d == 0)
	{
		sliceStart = 0;
		sliceEnd = 2 * PI;
	}
	float result = 0;
	for (int k = xRange[0]; k <= xRange[1]; k++)
	{
		for (int j = yRange[0]; j <= yRange[1]; j++)
		{
			float rho = (std::sqrt((pow((float)k - (float)origin.pt.x, 2) + pow((float)j - (float)origin.pt.y, 2))));
			float theta = std::atan2((float)j - (float)origin.pt.y, (float)k - (float)origin.pt.x) + PI;
			if (rho > ringStart&& rho <= ringEnd && theta > sliceStart&& theta <= sliceEnd)
			{
				float power = -std::pow(GetM(2 * PI, gradients.at<float>(j, k, 1) - (2 * PI * i / m)), 2) / std::pow(2 * sigma, 2);

				result += gradients.at<float>(j, k, 0) * std::exp(power);
			}
		}
	}

	result *= (1 / std::sqrt(2 * PI) * sigma);
	return result;
}

//non-functional
//float sGLOH::CalculateBinPlus(int r, int d, int i, int m, int n, int v, bool psi, float sigma, Mat& gradients, KeyPoint origin)
//{
//	// get the x and y ranges for the image patch around the keypoint
//	int xRange[] = { (int)std::floorf(origin.pt.x) - (int)(origin.size / 2), (int)std::floorf(origin.pt.x) + (int)(origin.size / 2) + 1 };
//	int yRange[] = { (int)std::floorf(origin.pt.y) - (int)(origin.size / 2), (int)std::floorf(origin.pt.y) + (int)(origin.size / 2) + 1 };
//
//	int count = 0;
//	float z = 2 * PI / m;
//	float sigmaC = sigma / (2 * PI / m);
//	float sigmaLine = (z / v) * sigmaC;
//	// limit operations to the current ring and slice
//	float ringStart = r * (origin.size / 2) / (n + 1);
//	float ringEnd = (r + 1) * (origin.size / 2) / (n + 1);
//	float sliceStart = d * 2 * PI / m;
//	float sliceEnd = (d + 1) * 2 * PI / m;
//	if (psi && r == 0 && d == 0)
//	{
//		sliceStart = 0;
//		sliceEnd = 2 * PI;
//	}
//	float result = 0;
//	for (int k = xRange[0]; k <= xRange[1]; k++)
//	{
//		for (int j = yRange[0]; j <= yRange[1]; j++)
//		{
//			float rho = (std::sqrt((pow((float)k - (float)origin.pt.x, 2) + pow((float)j - (float)origin.pt.y, 2))));
//			float theta = std::atan2((float)j - (float)origin.pt.y, (float)k - (float)origin.pt.x) + PI;
//			if (rho > ringStart&& rho <= ringEnd && theta > sliceStart&& theta <= sliceEnd)
//			{
//
//				float x = gradients.at<Vec2d>(j, k)[1];
//				float xzfloor = (float)std::floorf((float)x / (float)z);
//				float Sm = x - z * xzfloor;
//				float mLine = z * i / v;
//				float Mz = GetM(z, Sm - mLine);
//				float power = -std::pow(Mz, 2) / std::pow(2 * sigmaLine, 2);
//
//				result += gradients.at<Vec2d>(j, k)[0] * std::exp(power);
//				count++;
//			}
//		}
//	}
//	result *= (1 / std::sqrt(2 * PI) * sigmaLine);
//	return result;
//}

void sGLOH::detectAndCompute(InputArray _image, std::vector<KeyPoint>& keypoints, OutputArray _descriptors, sGLOH_Options options)
{

	size_t ksize = keypoints.size();
	if ((int) ksize < 1)
	{
		// get points first
		Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create(0, 3, 0.04, 10.0, options.sigma);
		Mat empty, unused;
		sift->detectAndCompute(_image, empty, keypoints, unused);
	}
	else
	{
		
	}
	// get gradients from image
	Mat greyscale;
	cvtColor(_image, greyscale, COLOR_BGR2GRAY);
	int dimensions[] = { _image.getMat().rows, _image.getMat().cols, 2 };
	Mat gradients = Mat(3, dimensions, CV_32F, Scalar::all(0));
	Mat dx, dy;
	spatialGradient(greyscale, dx, dy);
	for (int i = 0; i < gradients.rows; i++)
	{
		for (int j = 0; j < gradients.cols; j++)
		{
			int Gx = dx.at<int>(i, j);
			int Gy = dy.at<int>(i, j);
			gradients.at<float>(i, j, 0) = (float)std::sqrt(std::pow(Gx, 2) + std::pow(Gy, 2));
			gradients.at<float>(i, j, 1) = (float)std::atan2(Gy, Gx);
		}
	}

	calculate_sGLOH_Descriptor(options.m, options.n, options.psi, options.sigma, gradients, keypoints, _descriptors);
}

void sGLOH::calculate_sGLOH_Descriptor(int m, int n, bool psi, float sigma, Mat& gradients, std::vector<KeyPoint>& keypoints, OutputArray _descriptors)
{
	Mat descriptors;

	// number of floats the descriptor consists of
	int length = m * (m * n + 1 + (m - 1) * (psi ? 0 : 1));

	size_t ksize = keypoints.size();
	_descriptors.create((int)ksize, length, CV_32F);
	descriptors = _descriptors.getMat();
	descriptors.setTo(0);

	for (int keypoint = 0; keypoint < (int)ksize; keypoint++)
	{

		int counter = 0;
		if (psi)
		{
			for (int i = 0; i < m; i++)
			{
				int index = counter + ((i + 0) % m);
				descriptors.at<float>(keypoint, index) = CalculateBin(0, 0, i, m, n, psi, sigma, gradients, keypoints[keypoint]);
			}
			counter += m;
		}
		else
		{
			for (int d = 0; d < m; d++)
			{
				for (int i = 0; i < m; i++)
				{
					int index = counter + ((i + d) % m);
					descriptors.at<float>(keypoint, index) = CalculateBin(0, d, i, m, n, psi, sigma, gradients, keypoints[keypoint]);
				}
				counter += m;
			}
		}

		for (int r = 1; r <= n; r++)
		{
			for (int d = 0; d < m; d++)
			{
				for (int i = 0; i < m; i++)
				{
					int index = counter + ((i + d) % m);
					descriptors.at<float>(keypoint, index) = CalculateBin(r, d, i, m, n, psi, sigma, gradients, keypoints[keypoint]);
				}
				counter += m;
			}
		}

		// reduce descriptor vector to unit length
		float sum = 0;
		for (int i = 0; i < length; i++)
		{
			sum += std::pow(descriptors.at<float>(keypoint, i), 2);
		}
		float norm = std::sqrt(sum);
		for (int i = 0; i < length; i++)
		{
			descriptors.at<float>(keypoint, i) = descriptors.at<float>(keypoint, i) / norm;
		}
	}
}
sGLOH::sGLOH()
{
}

sGLOH::~sGLOH()
{
}
//Ptr<sGLOH> sGLOH::create(int _nfeatures, int _nOctaveLayers,
//	float _contrastThreshold, float _edgeThreshold, float _sigma)
//{
//	return makePtr<sGLOH>(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma);
//}
