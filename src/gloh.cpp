// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>

using namespace cv;

/** Name of the window which is displayed when the show method is called.*/
const std::string SHOW_WINDOW_NAME = "Output" ;

/** Number used to divide the image (each octave) when scaling.*/
const int SCALE_FACTOR = 2 ;

/** Factor used for blurring the image (sqrt(2) / 2 used as default). */
const double BLUR_FACTOR = std::sqrt(2) / 2 ;

/** Default level of scaled images (Based on the original SIFT paper). */
const int DEFAULT_SCALE_LEVELS = 4 ;

/** Default number of blurred images (Based on the original SIFT paper). */
const int DEFAULT_BLUR_LEVELS = 5 ;


/**
 *  A function for displaying a vector of matrix objects to the screen.
 *  It's main purpose is for testing the pyramid computation process.
 *
 *  @param images A vector of all the vectors at different scales of image.
 */
void show(const std::vector<std::vector<Mat>>& images)
{
    //Create a window for the output images.
    namedWindow(SHOW_WINDOW_NAME, WINDOW_NORMAL);

    //Iterate through the pyramid.
    for(std::vector<Mat> scales : images)     //Go through all scale levels.
    {
        for(Mat img : scales)                 //Go through blurred images in scales.
        {
            imshow(SHOW_WINDOW_NAME, img);    //Display the current image.
            waitKey(0);                       //Wait for user input to terminate.
        }
    }
}


/**
 *  An overload of the show function which can easily display a vector of images.
 *
 *  @param img A vector of matrix objects which we want to display on the screen.
 */
void show(const std::vector<Mat> images)
{
    std::vector<std::vector<Mat>> scale;
    scale.push_back(images);
    show(scale);
}


/**
 *  An overload of the show function which can easily display a single mat object.
 *
 *  @param img A matrix object which we want to display on the screen.
 */
void show(const Mat& img)
{
    std::vector<Mat> images;
    images.push_back(img);
    show(images);
}


/**
 *  A function which builds a pyramid of the image passed to it in different
 *  scales. It checks the number of scales passed to it for invalid values. The
 *  scaling is performed based on the scale factor passed to it (Optional).
 *
 *  @param src The source image used to build the pyramid.
 *  @param dest An empty vector which will be filled up with the images.
 *  @param scales The number of scaled images in the destination vector.
 *  @param factor Optional parameter for specifying the scale factor.
 */
void scalePyramid(Mat& src, std::vector<Mat>& dest,
    int scales = DEFAULT_SCALE_LEVELS, int factor = SCALE_FACTOR)
{
    //Get the minimum dimention of the source image.
    int maxScales = 0;
    int minSize = (src.rows < src.cols ? src.rows : src.cols);

    //Calculate the maximum number of scales possible for this source image.
    while(minSize > std::pow(factor, maxScales))
        maxScales++;

    //Check for invalid number of scales.
    if(scales < 1 || scales > maxScales)
        throw std::invalid_argument("Number of scales is not in range.");

    //Add the original source image to the base of the pyramid.
    dest.push_back(src);

    //Build the pyramid.
    for(int i = 1; i < scales; i++) {
        //Calculate new size based on the scaling factor.
        Size s(dest[i - 1].cols / factor, dest[i - 1].rows / factor);

        //Calculate the scaled image, and add it the vector.
        Mat scaled;
        pyrDown(dest[i - 1], scaled, s);
        dest.push_back(scaled);
    }
}


/**
 *  A function which blurs the source image given to it and writes it to a
 *  vector in several levels. The levels follow a sqrt(2)/2 function by default.
 *
 *  @param src The source image which is unblurred,
 *  @param dest Empty destination vector which will be filled out with the images.
 *  @param levels Optional parameter for the number of levels in blurring.
 */
void blurPyramid(Mat& src, std::vector<Mat>& dest,
    int levels = DEFAULT_BLUR_LEVELS)
{
    //Add the original source image to the base of the pyramid.
    dest.push_back(src);

    //Build the pyramid.
    for(int i = 1; i < levels; i++) {
        //Calculate the amount of blur, and Create the blurred image.
        Mat blurred;
        GaussianBlur(dest[i - 1], blurred, Size(), i * BLUR_FACTOR);

        //Add the blurred image to the vector.
        dest.push_back(blurred);
    }
}


/**
 *  A function which creates the scaled, and blurred images. It is a helper
 *  function which fills in the vector with the first dimention as scale,
 *  and the second dimention as blur level.
 *
 *  @param src The original image to build the pyramid from.
 *  @param dest The destination 2D vector for the images.
 */
 void buildPyramid(Mat& src, std::vector<std::vector<Mat>>& dest)
 {
     //Create the scaled images.
     std::vector<Mat> scaled;
     scalePyramid(src, scaled);

     //Go through all the scaled images, and create a blurPyramid for them.
     for(Mat scaledImg : scaled)
     {
         std::vector<Mat> blurred ;
         blurPyramid(scaledImg, blurred);
         dest.push_back(blurred) ;
     }
 }


/**
 *  A function which build a Difference of Gaussians (DoG) for all the scales.
 *  It negates the images with subsequent blur levels.
 *
 *  @param src The pyramid with the various scale, and blur levels.
 *  @param dest The output pyramid which will be filed various scales, and DoG.
 */
void buildDoG(std::vector<std::vector<Mat>>& src, std::vector<std::vector<Mat>>& dest)
{
    //Go through all the scales in the source.
    for(std::vector<Mat> scales : src)
    {
        //Calculate the difference of gradients in different levels.
        std::vector<Mat> dog ;

        //Go though the blur levels and subtract the matrixes (DoG instead of LoG).
        for(int i = 0; i < scales.size() - 1; i++)
            dog.push_back(scales[i] - scales[i + 1]);

        //Add the DoG for the current scale to the destination vector.
        dest.push_back(dog);
    }
}


/**
 *  A test function just for making sure the algorithms above work. It is
 *  currently called by the main method.
 */
void test()
{
    //Generate a random image filled with noise.
	Mat img=Mat::zeros(1000,1000,CV_8UC1);
	for (int i = 0; i < img.rows; i++)
	        for (int j = 0; j < img.cols; j++)
		    	img.at<uchar>(i,j)= rand()%255;

    //Build scales, and blurs pyramid (2D).
    std::vector<std::vector<Mat>> pyr ;
    buildPyramid(img, pyr);

    //Calculate the difference of gaussians.
    std::vector<std::vector<Mat>> dog ;
    buildDoG(pyr, dog);

    //Display the pyramids to the screen.
    show(pyr);
    show(dog);
}
