#include "sgloh.h"
#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>
double sGLOH::GetM(double q, double x)
{
	if (x < q / 2)
		return x;
	return q - x;
}

double sGLOH::CalculateBin(int r, int d, int i, int m, int n, bool psi, double sigma, Mat& gradients, KeyPoint origin)
{
	// get the x and y ranges for the image patch around the keypoint
	// floor the range borders if they exceed the bounds of the image
	int xRange[] = { (int)std::floorf(origin.pt.x) - (int)(origin.size / 2), (int)std::floorf(origin.pt.x) + (int)(origin.size / 2) + 1 };
	int yRange[] = { (int)std::floorf(origin.pt.y) - (int)(origin.size / 2), (int)std::floorf(origin.pt.y) + (int)(origin.size / 2) + 1 };


	// limit operations to the current ring and slice
	double ringStart = r * (origin.size / 2) / (n + 1);
	double ringEnd = (r + 1) * (origin.size / 2) / (n + 1);
	double sliceStart = d * 2 * PI / m;
	double sliceEnd = (d + 1) * 2 * PI / m;
	if (psi && r == 0 && d == 0)
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
			if (rho > ringStart&& rho <= ringEnd && theta > sliceStart&& theta <= sliceEnd)
			{
				double power = -std::pow(GetM(2 * PI, gradients.at<Vec2d>(j, k)[1] - (2 * PI * i / m)), 2) / std::pow(2 * sigma, 2);

				result += gradients.at<Vec2d>(j, k)[0] * std::exp(power);
			}
		}
	}

	result *= (1 / std::sqrt(2 * PI) * sigma);
	return result;
}

//non-functional
double sGLOH::CalculateBinPlus(int r, int d, int i, int m, int n, int v, bool psi, double sigma, Mat& gradients, KeyPoint origin)
{
	// get the x and y ranges for the image patch around the keypoint
	int xRange[] = { (int)std::floorf(origin.pt.x) - (int)(origin.size / 2), (int)std::floorf(origin.pt.x) + (int)(origin.size / 2) + 1 };
	int yRange[] = { (int)std::floorf(origin.pt.y) - (int)(origin.size / 2), (int)std::floorf(origin.pt.y) + (int)(origin.size / 2) + 1 };

	int count = 0;
	double z = 2 * PI / m;
	double sigmaC = sigma / (2 * PI / m);
	double sigmaLine = (z / v) * sigmaC;
	// limit operations to the current ring and slice
	double ringStart = r * (origin.size / 2) / (n + 1);
	double ringEnd = (r + 1) * (origin.size / 2) / (n + 1);
	double sliceStart = d * 2 * PI / m;
	double sliceEnd = (d + 1) * 2 * PI / m;
	if (psi && r == 0 && d == 0)
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
			if (rho > ringStart&& rho <= ringEnd && theta > sliceStart&& theta <= sliceEnd)
			{

				double x = gradients.at<Vec2d>(j, k)[1];
				double xzfloor = (double)std::floorf((float)x / (float)z);
				double Sm = x - z * xzfloor;
				double mLine = z * i / v;
				double Mz = GetM(z, Sm - mLine);
				double power = -std::pow(Mz, 2) / std::pow(2 * sigmaLine, 2);

				result += gradients.at<Vec2d>(j, k)[0] * std::exp(power);
				count++;
			}
		}
	}
	result *= (1 / std::sqrt(2 * PI) * sigmaLine);
	return result;
}

void sGLOH::detectAndCompute(InputArray _image, std::vector<KeyPoint>& keypoints, OutputArray _descriptors, sGLOH_Options options)
{

	size_t ksize = keypoints.size();
	if ((int) ksize < 1)
	{
		// get points first
	}
	else
	{
		Mat gradients;
		// get gradients from image
		calculate_sGLOH_Descriptor(options.m, options.n, options.psi, options.sigma, gradients, keypoints, _descriptors);
	}
}

void sGLOH::calculate_sGLOH_Descriptor(int m, int n, bool psi, double sigma, Mat& gradients, std::vector<KeyPoint>& keypoints, OutputArray _descriptors)
{
	Mat descriptors;

	// number of doubles the descriptor consists of
	int length = m * (m * n + 1 + (m - 1) * (psi ? 0 : 1));

	size_t ksize = keypoints.size();
	_descriptors.create((int)ksize, length, CV_64F);
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
				descriptors.at<double>(keypoint, index) = CalculateBin(0, 0, i, m, n, psi, sigma, gradients, keypoints[keypoint]);
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
					descriptors.at<double>(keypoint, index) = CalculateBin(0, d, i, m, n, psi, sigma, gradients, keypoints[keypoint]);
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
					descriptors.at<double>(keypoint, index) = CalculateBin(r, d, i, m, n, psi, sigma, gradients, keypoints[keypoint]);
				}
				counter += m;
			}
		}

		// reduce descriptor vector to unit length
		double sum = 0;
		for (int i = 0; i < length; i++)
		{
			sum += std::pow(descriptors.at<double>(keypoint, i), 2);
		}
		double norm = std::sqrt(sum);
		for (int i = 0; i < length; i++)
		{
			descriptors.at<double>(keypoint, i) = descriptors.at<double>(keypoint, i) / norm;
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
//	double _contrastThreshold, double _edgeThreshold, double _sigma)
//{
//	return makePtr<sGLOH>(_nfeatures, _nOctaveLayers, _contrastThreshold, _edgeThreshold, _sigma);
//}
