// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
//#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/traits.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#ifndef KeyPoint_H
#define KeyPoint_H

using namespace cv;

class SGlohKp
{
public:
    static constexpr int DEPTH_TYPE = CV_64F;
    typedef double pixType;

    /** Name of the window which is displayed when the show method is called.*/
    static const std::string SHOW_WINDOW_NAME;

    /** Number used to divide the image (each octave) when scaling.*/
    static constexpr int SCALE_FACTOR = 2 ;

    /** Default level of scaled images (Based on the original SIFT paper). */
    static constexpr int DEFAULT_NUM_OCTAVES = 4;

    /** Default number of blurred images (Based on the original SIFT paper). */
    static constexpr int DEFAULT_BLUR_LEVELS = 5;

    /** Default base sigma used for blurring within the scalespaces. */
    static constexpr double DEFAULT_SIGMA = 1.6;

    /** Default threshold used for minimum contrast when finding keypoints. */
    static constexpr pixType DEFAULT_CONTRAST_THRESHOLD = 0.04;

    /** Default minimum contrast for the curret pixel to filter it. */
    static constexpr pixType DEFAULT_CURVE_THRESHOLD = 10.0;

    /** Rotation calculation disbaled by default since it's not required by sGLOH. */
    static constexpr bool CALC_ROTATION = false;

    /** Default number of bins in the orientation histogram. */
    static constexpr int DEFAULT_HIST_BINS = 36;

    /** Default multiplier used for smoothing the orientation histogram. */
    static constexpr int DEFAULT_HIST_SMOOTHING_MULTI = 2;


    /**
     *  A function for displaying a vector of matrix objects to the screen.
     *  It's main purpose is for testing the pyramid computation process.
     *
     *  @param images A vector of all the vectors at different scales of image.
     */
    static void show(const std::vector<std::vector<Mat>>& images);

    /**
     *  An overload of the show function which can easily display a vector of images.
     *
     *  @param img A vector of matrix objects which we want to display on the screen.
     */
    static void show(const std::vector<Mat>& images);

    /**
     *  An overload of the show function which can easily display a single mat object.
     *
     *  @param img A matrix object which we want to display on the screen.
     */
    static void show(const Mat& img);


    /**
     *  A recursive function which returns the sigma for each level of blurring.
     *  The number of levels and the added sigma (At each level) is passed to it,
     *  and the sigmas for each level are saved in the blurSigmas vector.
     *
     *  @param blurSigmas An empty (output) vector which will hold the sigma levels.
     *  @param level The number of blur levels in total.
     *  @param sigma The base sigma which the image is blurred by in each level.
     */
    static void getSigmas(std::vector<pixType>& sigmas,
                    int levels = DEFAULT_BLUR_LEVELS,
                    double sigma = DEFAULT_SIGMA);


    /**
     *  A function which blurs the source image given to it and writes it to a
     *  vector in several levels. The sigmas are determined by a vector which holds
     *  the sigma values for each level (Starting with the base). Please keep in
     *  mind that blurSigmas[0] is not used (Since the base image is kept intact).
     *
     *  @param src The source image which is unblurred,
     *  @param dest Empty destination vector which will be filled out with the images.
     *  @param sigmas The sigmas for each level of the pyramid (Starting with base).
     */
    static void blurLevels(Mat& src, std::vector<Mat>& dest, std::vector<pixType>& sigmas);


    /**
     *  A function which creates the scaled, and blurred images. It is a helper
     *  function which fills in the vector with the first dimention as scale,
     *  and the second dimention as blur level.
     *
     *  @param src The original image to build the pyramid from.
     *  @param dest The destination 2D vector for the images.
     *  @param sigmas The sigma amount used in each octave's scalespace.
     *  @param octaves The number of octaves in the destination vector.
     */
     static void buildPyramid(Mat& src,
                        std::vector<std::vector<Mat>>& dest,
                        std::vector<double>& sigmas,
                        int octaves = DEFAULT_NUM_OCTAVES);


    /**
     *  A function which build a Difference of Gaussians (DoG) for all the scales.
     *  It negates the images with subsequent blur levels.
     *
     *  @param src The pyramid with the various scale, and blur levels.
     *  @param dest The output pyramid which will be filed various scales, and DoG.
     */
    static void buildDoG(std::vector<std::vector<Mat>>& src, std::vector<std::vector<Mat>>& dest);


    /**
     *  A function which determines if the current pixel is an extreme in the image.
     *  It checks 26 neighboring pixels within the given octave to determine if the
     *  current pixel is an extreme.
     *
     *  @param pyr The current octave of image (With all the blur levels)
     *  @param row The row for the pixel we're examining
     *  @param col The column for the pixel we're examining
     *  @param lvl The current blur level for pixel we're examining.
     *  @return True if the current point is an extremum, false otherwise.
     */
    bool isExtreme(std::vector<Mat>& pyr, int r, int c, int lvl);


    /**
     *  A function which calculates the orientation histogram for the current point
     *  in the image with the given radius. The destination is a vector which will be
     *  filled-up with the histogram values in each bin.
     *  Note: The implementation of this method is based on the OpenSIFT's.
     *        OpenSIFT's implementation is based on the section 5 of lowes paper.
     *        Link: https://github.com/robwhess/opensift/blob/master/src/
     *
     *  @param src The source image.
     *  @param dest An empty vector for the orientation histogram.
     *  @param r The row for the current point.
     *  @param c The column for the current point.
     *  @param stdev Standard deviation for weighing gaussians.
     *  @param radius The radius of matrix which the histogram is calculated at.
     *  @param numBins The total number of bins in the orientation histogram.
     */
    static void orientationHist(Mat& src, std::vector<double>& dest,
                            int r, int c, double stdev,
                            int radius,
                            int numBins = DEFAULT_HIST_BINS);


    /**
     *  A function which smoothes a histogram by a gaussian method. It also can do it
     *  multiple times. Which makes the image possibly smoother.
     *
     *  @param histogram The histogram which we're smoothing.
     *  @param multiplier The number of times tha the histogram is smoothed.
     */
    static void histogramGauss(std::vector<double>& histogram, int multiplier=1);


    /**
     *  A function which creates a keypoint object from the parametes passed to it.
     *  it also calculates the sclae(size) for the keypoint based on the data passed.
     *  Additionally it calculates the dominant orientation for the keypoint.
     *  TODO: It would be nice if it did subpixel approximation.
     *
     *  @param pyr Differnce of gaussians pyramid created prevriously.
     *  @param sigma The sigma value used prevriously in this matrix to perform blurring.
     *  @param octave The current octave in the pyramid for this keypoint.
     *  @param lvl The current blur level in the scale space for this keypoint.
     *  @param r The row for this keypoint within the matrix object.
     *  @param c The column for this keypoint within the matrix object.
     *  @return A keypoint object initialized with the correct parameters.
     */
    KeyPoint getKp(std::vector<std::vector<Mat>>& pyr, double sigma,
        int octave, int lvl, int r, int c);


    /**
     *  Checks if a pixel is an edge. It is based on the strong reponse of DOG along
     *  the edges. This check is performed by a 2 by 2 hessian matrix.
     *  Note: The implementation of this method is based on the OpenSIFT's.
     *        OpenSIFT's implementation is based on the section 4.1 of lowes paper.
     *        Link: https://github.com/robwhess/opensift/blob/master/src/
     *
     *  @param dog The difference of gaussians image.
     *  @param r The current row of the pixel in the image.
     *  @param c The current column of the pixel in the image.
     *  @param curve_threshold The max ratio on the curves.
     *  @return True if the keypoint is edge like, false otherwise.
     */
    bool isEdge(Mat& dog, int r, int c, pixType curve_threshold);



    /**
     *  A function which goes through all the pixels, and calculates the maxima/minima
     *  within the surrounding 26 pixels (Through different blur levels). If the current
     *  pixel is min/max of all pixels, it marks it as a keypoint (Not filtered yet).
     *
     *  @param pyr The difference of gaussians pyramid along different scales (octaves).
     *  @param kp An empty vector which will hold the keypoints accorss different octaves.
     *  @param sigmas The sigma amount used in each octave's scalespace.
     *  @param contrast_threashold The min contrast for the curret pixel to filter it.
     *  @param curve_threshold The max ratio on the curves.
     */
    static void findKeypoints(std::vector<std::vector<Mat>>& pyr,
                        std::vector<KeyPoint>& kp,
                        std::vector<pixType>& sigmas,
                        pixType contrast_threshold=DEFAULT_CONTRAST_THRESHOLD,
                        pixType curve_threshold=DEFAULT_CURVE_THRESHOLD);


    /**
     *  A function which calculates the keypoints for the given source image. It
     *  uses a similar approach to SIFT to detect keypoints. However, as it is, it
     *  does not do subpixel approximation.
     *
     *  @param src The source image which we're calculating these for.
     *  @param dest The output vector which the keypoint will be written to.
     */
    static void detect(Mat& src, std::vector<KeyPoint>& dest);


    /**
     *  A test function just for making sure the algorithms above work. It is
     *  currently called by the main method.
     */
    void test();
};



#endif
