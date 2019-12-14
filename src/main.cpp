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
#include "matcher.h"

/** The path where the test image directories are. */
const std::string TEST_DATA_PATH = "../test/" ;

/** The name of the manual tests directory within the TEST_DATA_PATH directory. */
const std::string TEST_MANUAL_DATA_PATH = "test_manual" ;

/** The method which the sGLOH finds it's extractors. Defined in kp.h. */
const SGloh::KpType KEYPOINT_EXTRACTION_METHOD = SGloh::KpType::SGLOH;


/**
 *  A function which runs a custom test written to test keypoint detection.
 */
void keypointTests()
{
    std::cerr << "Starting the KeyPoint Test Program.\n" << std::endl;
    SGloh::test();
}


/**
 *  A function which calculates recall and percision for the given matches.
 *
 *  @param currMatch The matched vector from the algorithm.
 *  @param maxThreashold The maximum distance for filtering out bad matches.
 *  @param recallPrecisionCurve The output vector which will hold the results
 *                              Index 0 is recall, index 1 is percision.
 */
void calculateRecallPercision(std::vector<cv::DMatch>& currMatch,
                                std::vector<cv::Point2f>& recallPrecisionCurve,
                                float maxThreashold)
{
    std::cerr << "Starting the recall/percision Program.\n" << std::endl;

    //Make a vector for holding the precision and recall values.
    std::vector<std::vector<cv::DMatch>> matches;
    matches.push_back(currMatch);

    //Create a mask and add it to the vector.
    std::vector<std::vector<uchar>> masks;

    //Create a single mask for the calculations (Find all good matches).
    std::vector<uchar> mask;
    for(cv::DMatch m : currMatch)
    {
        if(m.distance > 0 && m.distance < maxThreashold)
            mask.push_back(true);
        else
            mask.push_back(false);
    }

    masks.push_back(mask);

    //Calculate the recall and percision based on the given mask/
    cv::computeRecallPrecisionCurve(matches, masks, recallPrecisionCurve);
}


/**
 *  A function which calculates descriptors using sift, and saves the matches.
 *
 *  @param testImage1 The first image to use sift with.
 *  @param testImage2 The second image to use sift with.
 *  @param outputFile The name which will be appended to output file.
 */
void siftTest(cv::Mat& testImage1, cv::Mat& testImage2, std::string outputFile)
{
    // detectAndCompute to allow quality comparison between SIFT descriptor and sGLOH descriptor
	time_t startSIFT = std::time(NULL);

    cv::Mat descOne, descTwo;
	std::vector<cv::KeyPoint> kpOne, kpTwo;
    cv::Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
	sift->detectAndCompute(testImage1, cv::noArray(), kpOne, descOne);
    sift->detectAndCompute(testImage2, cv::noArray(), kpTwo, descTwo);

	time_t stopSIFT = std::time(NULL);
	std::cout << "SIFT took this long to detect and compute:\t\t" <<
        (stopSIFT - startSIFT) << std::endl;

    std::vector<cv::DMatch> matches;

    //Create a matcher,and a vector for holding the matches.
	cv::Ptr<cv::BFMatcher> matcher = cv::BFMatcher::create();
	matcher->match(descOne, descTwo, matches);		  //Match using descriptors.

    //Draw the matches to the screen.
	cv::Mat matchedImage;
	cv::drawMatches(testImage1, kpOne, testImage2, kpTwo, matches, matchedImage);

    //Also save it to the disk.
	cv::imwrite(outputFile, matchedImage);
}


/**
 *  A function which calculates descriptors using sloh, and saves the matches.
 *
 *  @param testImage1 The first image to use sgloh with.
 *  @param testImage2 The second image to use sgloh with.
 *  @param outputFile The name which will be appended to output file.
 */
void sglohTest(cv::Mat& testImage1, cv::Mat& testImage2, std::string outputFile)
{
    // for each circular feature patch f there is a descriptor H
    // for all pixels in f determine via gaussian the gradient magnitude Gm and orientation Gd
    // steps for creating descriptor H:
    // first define n, m, and psi, where n is the number of descriptor rings, m
    // is the number of regions per ring, and psi is whether or not the
    // innermost ring has one or m regions (0 or 1 respectively)
    // second, each region is broken up into m histogram bins
    // for each region R sub (r, d) where r is defined as the integer set from 1
    // to n and d is defined as the integer set from 0 to m - 1
    // the ith histogram bin value h sub (r, d) is defined as
    // sum for each pixel p in R sub (r, d) (Gm(p) *

    std::vector<cv::KeyPoint> points;

    //Initialize the options.
	SGloh::Options options;
	options.m = 8;
	options.n = 2;
	options.psi = false;
	options.v = 0;
	options.sigma = 1.6;
    options.type = KEYPOINT_EXTRACTION_METHOD;

	std::srand(std::time(NULL));

    //Calculate the descriptors for both images.
	cv::Mat emm1, emm2;
	std::vector<cv::KeyPoint> tP1, tP2;

    //Find keypoints and calculate descriptor.
	time_t start_sGLOH = std::time(NULL);

	SGloh::detectAndCompute(testImage1, tP1, emm1, options);
	SGloh::detectAndCompute(testImage2, tP2, emm2, options);

	time_t stop_sGLOH = std::time(NULL);
	std::cout << "sGLOH took this long to detect and compute:\t\t" <<
        (stop_sGLOH - start_sGLOH) << std::endl;

	time_t start_Match = std::time(NULL);

    //Run the matching algorithm (Custom bruteforce matcher).
    std::vector<cv::DMatch> bestMatches;
    SGloh::match(emm1, emm2, bestMatches, options, true);

	time_t stop_Match = std::time(NULL);

	std::cout << "Custom Brute force matching took this long:\t\t" <<
        (stop_Match - start_Match) << std::endl;

    //Draw the matches to the screen.
	cv::Mat matchedImage;
	cv::drawMatches(testImage1, tP1, testImage2, tP2, bestMatches, matchedImage);

    //Also save it to the disk.
	cv::imwrite(outputFile, matchedImage);
}


/**
 *  A function which runs SIFT and sGLOH, and outputs their results to files.
 *
 *  @param imgOne The first image to run the algorithms with.
 *  @param imgTwo The second image to run the algorithms with.
 */
void customTest(cv::Mat& imgOne, cv::Mat& imgTwo, std::string outputName)
{
    siftTest(imgOne, imgTwo, "Results_" + outputName + "_SIFT.jpg");
    sglohTest(imgOne, imgTwo, "Results_" + outputName + "_SGLOH.jpg");
}


/**
 *  A function which reads the oxford database images saved in the repository,
 *  and runs both SIFT and sGLOH to provide a comparison.
 */
void oxfordTests()
{
    std::cerr << "Starting the Oxford Database Test Program.\n" << std::endl;
    std::string pathOne = "test_bark/First.ppm";
    std::string pathTwo = "test_bark/Second.ppm";

    // Test the bark images ***************************************

    //Read some of the images for testing.
    std::string prefix = TEST_DATA_PATH + "test_bark/";
	cv::Mat imgOne = cv::imread(prefix + "First.ppm");
	cv::Mat imgTwo = cv::imread(prefix + "Second.ppm");

    //Run both sift and sgloh.
    customTest(imgOne, imgTwo, "Bark");

    // Test the boat images **************************************

    //Update the images for testing.
    prefix = TEST_DATA_PATH + "test_boat/";
	imgOne = cv::imread(prefix + "First.ppm");
	imgTwo = cv::imread(prefix + "Second.ppm");

    //Run both sift and sgloh.
    customTest(imgOne, imgTwo, "Boat");

    // Test the graffiti images *********************************

    //Update the images for testing.
    prefix = TEST_DATA_PATH + "test_graf/";
	imgOne = cv::imread(prefix + "First.ppm");
	imgTwo = cv::imread(prefix + "Second.ppm");

    //Run both sift and sgloh.
    customTest(imgOne, imgTwo, "Graf");


    // Test the wall images *************************************

    //Update the images for testing.
    prefix = TEST_DATA_PATH + "test_wall/";
	imgOne = cv::imread(prefix + "First.ppm");
	imgTwo = cv::imread(prefix + "Second.ppm");

    //Run both sift and sgloh.
    customTest(imgOne, imgTwo, "Wall");
}


/**
 *  A function which runs some manaul tests written with our images.
 *  This also calculates sGLOH and SIFT for comparisons.
 */
void manualTests()
{
    std::cerr << "Starting the Manual Tests Program.\n" << std::endl;

    //Read some of the images for testing.
    std::string prefix = TEST_DATA_PATH + TEST_MANUAL_DATA_PATH + "/";
	cv::Mat testImage1 = cv::imread(prefix + "EighthTestImage1.jpg");
	cv::Mat testImage2 = cv::imread(prefix + "EighthTestImage2.jpg");

    customTest(testImage1, testImage2, "Manual");
}


/**
 *  A function which parses the arguments to check which test to run.
 *  If it's run without arguments, it will run the oxford test database test.
 *
 *  @param argc The number of arguments.
 *  @param argv An array of strings (char*) which are the arguments.
 */
int main(int argc, char** argv)
{
    //Check if we have arguments.
    if(argc != 2)
    {
        std::cerr << "No parameter specified. Running Oxford Test by Default." << std::endl;
        oxfordTests();
        return 0;
    }

    std::string cmd(argv[1]);

    //Check which test we're running.
    if(cmd == "-o")            //Oxford tests, on all the images from the paper.
        oxfordTests();

    else if (cmd == "-m")      //Manual tests, written to check different images.
        manualTests();

    else if (cmd == "-k")      //Keypoint demo test. Randomly generated.
        keypointTests();

    //If we get here, the parameters were invalid/unsupported.
    else
    {
        std::cerr << "Invalid Parameter. Please Check README.md" << std::endl;
        return -1;
    }

    return 0;
}
