#include "sgloh.h"

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>

namespace SGloh
{

/**
 *  A function which calculates the value of the descriptor at the given bin.
 *  after the calculations are finished, the value of the final results of the
 *  current bin (At origin) will be returned.
 */
float calculateBin(int r, int d, int i, int m, int n, bool psi, float sigma,
    cv::Mat& gradients, cv::KeyPoint origin)
{
	// get the x and y ranges for the image patch around the keypoint
	// floor the range borders if they exceed the bounds of the image
	int xRange[] = { (int)std::floor(origin.pt.x) - (int)(origin.size / 2),
        (int)std::floor(origin.pt.x) + (int)(origin.size / 2) + 1 };

	int yRange[] = { (int)std::floor(origin.pt.y) - (int)(origin.size / 2),
        (int)std::floor(origin.pt.y) + (int)(origin.size / 2) + 1 };


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
			float rho = std::log10(std::sqrt((pow((float)k -
                origin.pt.x, 2) + pow((float)j - origin.pt.y, 2))));

			float theta = std::atan2((float)j - origin.pt.y, (float)k - origin.pt.x) + PI;
			if (rho >= ringStart&& rho < ringEnd && theta >= sliceStart&& theta < sliceEnd)
			{
				float power = -std::pow(getM(2 * PI, gradients.at<float>(j, k, 1)
                    - ((2 * PI * i) / m)), 2) / std::pow(2 * sigma, 2);

				float foo = gradients.at<float>(j, k, 1);
				float bar = gradients.at<float>(j, k, 0);
				result += gradients.at<float>(j, k, 0) * std::exp(power);
			}
		}
	}

	result *= (1 / (std::sqrt(2 * PI) * sigma));
	return result;
}


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
    cv::Mat& _descriptors, Options options)
{
    //If no keypoints were passed, calculate the keypoints.
	size_t ksize = keypoints.size();
	if ((int) ksize < 1)
	{
        //If sift was requested.
        if(options.type == KpType::SIFT)
        {
            // get points first
    		cv::Ptr<cv::xfeatures2d::SIFT> sift =
                cv::xfeatures2d::SIFT::create(0, 3, 0.04, 10.0, options.sigma);

    		cv::Mat empty, unused;
    		sift->detectAndCompute(_image, empty, keypoints, unused);
        }

        //If our implmented kp extractor was requested.
        if(options.type == KpType::SGLOH)
        {
            detect(_image, keypoints);
        }

        //If SURF was requested.
        if(options.type == KpType::SURF)
        {
            cv::Ptr<cv::xfeatures2d::SURF> surf =
                cv::xfeatures2d::SURF::create();

        	surf->detect(_image, keypoints);		  //Match using descriptors.
        }
	}

    //Calculate teh gradient for the image, and then the descriptor.
	cv::Mat gradients;
	findGradient(_image, gradients);

	calculate_sGLOH_Descriptor(options.m, options.n, options.psi, options.sigma,
        gradients, keypoints, _descriptors);
}


/**
 *  The base function which will calculate the sGLOH descriptor based on the
 *  passed parameters. It will populate the descriptors matrix with the proper
 *  values for the descriptor. Prior to calling this functions, the keypoints
 *  should be calculated.
 */
void calculate_sGLOH_Descriptor(int m, int n, bool psi, float sigma,
    cv::Mat& gradients, std::vector<cv::KeyPoint>& keypoints, cv::Mat& _descriptors)
{

	// number of floats the descriptor consists of
	int length = m * (m * n + 1 + (m - 1) * (psi ? 0 : 1));

	size_t ksize = keypoints.size();
	_descriptors.create((int)ksize, length, CV_32F);

	_descriptors.setTo(0);

	// loop through all keypoints
	for (int keypoint = 0; keypoint < (int)ksize; keypoint++)
	{
		// counter is used to calculate the initial histogram rotation
		int counter = 0;
		if (psi)
		{
			// center ring is a single region
			for (int i = 0; i < m; i++)
			{
				int index = counter + ((i + 0) % m);
				_descriptors.at<float>(keypoint, index) =
                    calculateBin(0, 0, i, m, n, psi, sigma, gradients, keypoints[keypoint]);
			}
			counter += m;
		}
		else
		{
			// center ring is multiple regions
			for (int d = 0; d < m; d++)
			{
				for (int i = 0; i < m; i++)
				{
					int index = counter + ((i + d) % m);
					_descriptors.at<float>(keypoint, index) =
                        calculateBin(0, d, i, m, n, psi, sigma, gradients, keypoints[keypoint]);
				}
				counter += m;
			}
		}
		// work on outer rings
		for (int r = 1; r <= n; r++)
		{
			for (int d = 0; d < m; d++)
			{
				for (int i = 0; i < m; i++)
				{
					int index = counter + ((i + d) % m);
					_descriptors.at<float>(keypoint, index) =
                        calculateBin(r, d, i, m, n, psi, sigma, gradients, keypoints[keypoint]);
				}
				counter += m;
			}
		}

		// reduce descriptor vector to unit length
		float sum = 0;
		for (int i = 0; i < length; i++)
		{
			sum += std::pow(_descriptors.at<float>(keypoint, i), 2);
		}
		float norm = std::sqrt(sum);
		for (int i = 0; i < length; i++)
		{
			_descriptors.at<float>(keypoint, i) =
                _descriptors.at<float>(keypoint, i) / norm;
		}
	}
}

/**
 *  A function which Discretely rotates a descriptor vector by one increment of 2*PI/m
 *  The original options should be passed to it so it can access the m value.
 *  The rotated descriptor will be written to rotated matrix.
 */
void rotateDescriptors(cv::Mat descriptors, cv::Mat& rotated, Options options)
{
	rotated = cv::Mat(descriptors.rows, descriptors.cols, CV_32F);
	// check that the descriptor can actually be rotated
	if (options.m > 1)
	{
		for (int i = 0; i < descriptors.rows; i++)
		{
			int length = options.m * (options.m * options.n + 1 +
                (options.m - 1) * (options.psi ? 0 : 1));

			// centralRegion is used if psi(H) is 0
			float* centralRegion = new float[options.m];
			int rings = options.n;
			if (!options.psi)
			{
				// centralRegion won't be used, so make the outerRegions array larger
				rings++;
			}
			// create a 3d array of floats
			float*** outerRegions = new float** [rings];
			for (int j = 0; j < rings; j++)
			{
				outerRegions[j] = new float* [options.m];
				for (int k = 0; k < options.m; k++)
				{
					outerRegions[j][k] = new float[options.m];
				}
			}
			int start = 0;
			if (options.psi)
			{
				// rotate the central ring as a single region
				for (int j = 0; j < options.m; j++)
				{
					centralRegion[(j + 1) % options.m] = descriptors.at<float>(i, j);
				}
				// assign the rotated values back to the descriptor vector
				for (int j = 0; j < options.m; j++)
				{
					rotated.at<float>(i, j) = centralRegion[j];
				}
				// set the starting point for the outer regions
				start = options.m;
			}
			// rotate each ring as multiple regions
			int tempJ = start;
			for (int j = 0; j < rings; j++)
			{
				for (int k = 0; k < options.m; k++)
				{
					for (int l = 0; l < options.m; l++)
					{
						outerRegions[j][(k + 1) % options.m][l] =
                            descriptors.at<float>(i, tempJ++);
					}
				}
			}
			// assign the rotated values back to the descriptor vector
			tempJ = start;
			for (int j = 0; j < rings; j++)
			{
				for (int k = 0; k < options.m; k++)
				{
					for (int l = 0; l < options.m; l++)
					{
						rotated.at<float>(i, tempJ++) = outerRegions[j][k][l];
					}
				}
			}
			// deallocate the array
			for (int j = 0; j < rings; j++)
			{
				for (int k = 0; k < options.m; k++)
				{
					delete[] outerRegions[j][k];
				}
				delete[] outerRegions[j];
			}
			delete[] outerRegions;
			delete[] centralRegion;
		}
	}
}


/**
 *  A function which calculates the M values specified in the original paper.
 *  If the value of x is smaller than half of q, it return x, otherwise it will
 *  return q-x.
 */
float getM(float q, float x)
{
	if (x < q / 2)
		return x;
	return q - x;
}

}
