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
#include <string>

#include "sgloh.h"
#include "kp.h"
#include "options.h"

const std::string TEST_DATA_PATH = "../test/" ;
const std::string TEST_MANUAL_DATA_PATH = "test_manual" ;

void getCorrectMatches() {

}

void keypointTests()
{
    std::cerr << "Starting the KeyPoint Test Program.\n" << std::endl;
    SGloh::test();
}

void customTest(std::string first, std::string second)
{
    std::cerr << "Starting the Custom Test Program.\n" << std::endl;

    //Read the images from the disk.
    cv::Mat imgOne = cv::imread(TEST_DATA_PATH + first);
	cv::Mat imgTwo = cv::imread(TEST_DATA_PATH + second);

    //Calculate Descriptors using SIFT keypoint extraction.
    SGloh::Options options;
	options.m = 8;
	options.n = 2;
	options.psi = false;
	options.v = 0;
	options.sigma = 1.6;
    options.type = SGloh::KpType::SIFT;

    //Calculate Descritors and keypoints for both images.
    cv::Mat descOne, descTwo;
	std::vector<cv::KeyPoint> kpOne, kpTwo;
	SGloh::detectAndCompute(imgOne, kpOne, descOne, options);
    SGloh::detectAndCompute(imgTwo, kpTwo, descTwo, options);

    //Create a bruteforce matcher,and a vector for holding the matches.
    cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();
    std::vector<cv::DMatch> match;
    matcher->match(descOne, descTwo, match);

    //Make a vector for holding the precision and recall values.
    std::vector<std::vector<cv::DMatch>> matches;
    matches.push_back(match);

    //Create a mask and add it to the vector.
    std::vector<std::vector<uchar>> masks;

    //Create a single mask for the calculations (Find all good matches).
    std::vector<uchar> mask;
    for(cv::DMatch m : match)
    {
        if(m.distance < 0f)
            mask.push_back(true);
        else
            mask.push_back(false);
    }

    masks.push_back(mask);

    std::vector<cv::Point2f> recallPrecisionCurve;
    cv::computeRecallPrecisionCurve(matches, masks, recallPrecisionCurve);

    std::cerr << "Size of points in curve is: " << recallPrecisionCurve.size() << std::endl;
}

void oxfordTests()
{
    std::cerr << "Starting the Oxford Database Test Program.\n" << std::endl;
    std::string pathOne = "test_bark/First.ppm";
    std::string pathTwo = "test_bark/Second.ppm";
    customTest(pathOne, pathTwo);
}

void manualTests()
{
	// for each circular feature patch f there is a descriptor H
	// for all pixels in f determine via gaussian the gradient magnitude Gm and orientation Gd
	// steps for creating descriptor H:
	// first define n, m, and psi, where n is the number of descriptor rings, m is the number of regions per ring, and psi is whether or not the
	// innermost ring has one or m regions (0 or 1 respectively)
	// second, each region is broken up into m histogram bins
	// for each region R sub (r, d) where r is defined as the integer set from 1 to n and d is defined as the integer set from 0 to m - 1
	// the ith histogram bin value h sub (r, d) is defined as
	// sum for each pixel p in R sub (r, d) (Gm(p) *
    std::string prefix = TEST_DATA_PATH + TEST_MANUAL_DATA_PATH;
	cv::Mat image = cv::imread(prefix + "foreground.jpg");
	cv::Mat testImage1 = cv::imread(prefix + "OriginalTestImage1.jpg");
	cv::Mat testImage2 = cv::imread(prefix + "OriginalTestImage2.jpg");
	cv::Mat testImage3 = cv::imread(prefix + "OriginalTestImage3.jpg");
	cv::Mat testImage4 = cv::imread(prefix + "OriginalTestImage4.jpg");
	//namedWindow("image");
	//imshow("image", image);
	//cv::waitKey(0);
	//cv::Mat nuTest1, nuTest2, nuTest3, nuTest4;
	//cv::pyrDown(testImage1, nuTest1);
	//cv::pyrDown(testImage2, nuTest2);
	//cv::pyrDown(testImage3, nuTest3);
	//cv::pyrDown(testImage4, nuTest4);

	//cv::imwrite("SixteenthTestImage1.jpg", nuTest1);
	//cv::imwrite("SixteenthTestImage2.jpg", nuTest2);
	//cv::imwrite("SixteenthTestImage3.jpg", nuTest3);
	//cv::imwrite("SixteenthTestImage4.jpg", nuTest4);
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
	std::vector<cv::KeyPoint> points;
	cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create(0, 3, 0.04, 10.0, sigma);
	//sGLOH* essGLOH;// = new sGLOH();
	SGloh::Options options;
	options.m = 8;
	options.n = 2;
	options.psi = false;
	options.v = 0;
	options.sigma = 1.6;
    options.type = SGloh::KpType::SIFT;
	/*sift->detect(image, points);*/
	cv::Mat siftDescriptors;
	cv::Mat emptyMask;
	// detectAndCompute instead of just detect to allow quality comparison between SIFT descriptor and sGLOH descriptor
	time_t startSIFT = std::time(NULL);
	sift->detectAndCompute(testImage1, emptyMask, points, siftDescriptors);
	time_t stopSIFT = std::time(NULL);
	std::cout << "SIFT took this long to detect and compute:\t\t";
	std::cout << (stopSIFT - startSIFT) << std::endl;
	std::srand(std::time(NULL));

	cv::Mat emm1, emm2;
	std::vector<cv::KeyPoint> tP1, tP2;
	time_t start_sGLOH = std::time(NULL);
	SGloh::detectAndCompute(testImage1, tP1, emm1, options);
	SGloh::detectAndCompute(testImage2, tP2, emm2, options);
	//calculate_sGLOH_Descriptor(m, n, psi, sigma, gradients, points, emm);
	time_t stop_sGLOH = std::time(NULL);
	std::cout << "sGLOH took this long to detect and compute:\t\t";
	std::cout << (stop_sGLOH - start_sGLOH) << std::endl;
	time_t start_Match = std::time(NULL);
	//cv::Ptr<BFMatcher> bruteForceMatcher = BFMatcher::create();
	std::vector<std::vector<cv::DMatch>> matches;
	matches.resize((size_t)options.m);
	size_t sizeSum = 0;
	for (int h = 0; h < options.m; h++)
	{
		//bruteForceMatcher->match(emm1, emm2, matches[h]);
		for (int i = 0; i < emm1.rows; i++)
		{
			cv::DMatch curr = cv::DMatch();
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
		SGloh::rotateDescriptors(emm1.clone(), emm1, options);
	}
	std::vector<cv::DMatch> goodMatches;
	std::vector<cv::DMatch> bestMatches;
	//bestMatches.resize(sizeSum);
	int i = 0;
	for (int j = 0; j < options.m; j++)
	{
		for (int k = 0; k < (int)matches[j].size(); k++)
		{
			cv::DMatch curr(matches[j][k]);
			for (int l = j + 1; l < options.m; l++)
			{
				for (int m = 0; m < (int)matches[l].size(); m++)
				{
					if (curr.queryIdx == matches[l][m].queryIdx &&
						matches[l][m].distance < curr.distance)
					{
						curr = matches[l][m];
					}
				}
			}
			if (curr.distance >= 0 &&
				curr.distance < 0.25f)
			{
				goodMatches.push_back(curr);
			}
		}
	}
	for (int i = 0; i < goodMatches.size(); i++)
	{
		cv::DMatch curr = goodMatches[i];
		bool duplicate = false;
		for (int k = 0; k < bestMatches.size(); k++)
		{
			if (bestMatches[k].queryIdx == goodMatches[i].queryIdx &&
				bestMatches[k].trainIdx == goodMatches[i].trainIdx &&
				std::abs(bestMatches[k].distance - goodMatches[i].distance) < 0.1)
			{
				duplicate = true;
			}
		}
		if (!duplicate)
		{
			bestMatches.push_back(curr);
		}
	}
	time_t stop_Match = std::time(NULL);
	std::cout << "Brute force matching took this long:\t\t";
	std::cout << (stop_Match - start_Match) << std::endl;
	std::cout << "end of file";
	//cv::Ptr<BFMatcher> bruteForceMatcher = BFMatcher::create();
	//std::vector<std::vector<cv::DMatch>> matches;
	//bruteForceMatcher->knnMatch(emm1, emm2, matches, 5);
	cv::Mat matchedImage1, matchedImage2, matchedImage3, matchedImage4, matchedImage5;
	cv::drawMatches(testImage1, tP1, testImage2, tP2, bestMatches, matchedImage1);
	cv::imshow("matches1", matchedImage1);
	cv::namedWindow("matches1", cv::WINDOW_NORMAL);
	cv::waitKey(0);
	//drawMatches(testImage1, tP1, testImage2, tP2, matches[1], matchedImage2);
	//imshow("matches2", matchedImage2);
	//namedWindow("matches2", cv::WINDOW_NORMAL);
	//cv::waitKey(0);
	//drawMatches(testImage1, tP1, testImage2, tP2, matches[2], matchedImage3);
	//imshow("matches3", matchedImage3);
	//namedWindow("matches3", cv::WINDOW_NORMAL);
	//cv::waitKey(0);
	//drawMatches(testImage1, tP1, testImage2, tP2, matches[3], matchedImage4);
	//imshow("matches4", matchedImage4);
	//namedWindow("matches4", cv::WINDOW_NORMAL);
	//cv::waitKey(0);
	//drawMatches(testImage1, tP1, testImage2, tP2, matches[4], matchedImage5);
	//imshow("matches5", matchedImage5);
	//namedWindow("matches5", cv::WINDOW_NORMAL);
	//cv::waitKey(0);
	cv::imwrite("result1-2.jpg", matchedImage1);
	//cv::imwrite("result2.jpg", matchedImage2);
	//cv::imwrite("result3.jpg", matchedImage3);
	//cv::imwrite("result4.jpg", matchedImage4);
	//cv::imwrite("result5.jpg", matchedImage5);
    //	delete essGLOH;
	//std::cout << descriptorsFinal.row(59) << std::endl;
}


int main(int argc, char** argv)
{
    //Check if we have arguments.
    if(argc != 2)
    {
        std::cerr << "No parameter specified. Running Oxford Test by Default." << std::endl;
        oxfordTests();
        return 0;
    }

    //Check which test we're running.

    if(argv[1] == "-o")
        oxfordTests();

    else if (argv[1] == "-m")
        manualTests();

    else if (argv[1] == "-k")
        keypointTests();

    else if (argv[1] == "-c")
    {
        if(argc == 4)
            customTest(argv[2], argv[3]);
        else
        {
            std::cerr << "Invalid Parameters. Please Check README.md" << std::endl;
            return -1;
        }
    }

    //If we get here, the parameters were invalid/unsupported.
    else
    {
        std::cerr << "Invalid Parameter. Please Check README.md" << std::endl;
        return -1;
    }

    return 0;
}
