// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "kp.h"

namespace SGloh
{

/**
 *  A function for displaying a vector of matrix objects to the screen.
 *  It's main purpose is for testing the pyramid computation process.
 *
 *  @param images A vector of all the vectors at different scales of image.
 */
void show(const std::vector<std::vector<cv::Mat>>& images)
{
    //Create a window for the output images.
    cv::namedWindow(SHOW_WINDOW_NAME, cv::WINDOW_NORMAL);

    //Iterate through the pyramid.
    for(std::vector<cv::Mat> scales : images)     //Go through all scale levels.
    {
        for(cv::Mat img : scales)                 //Go through blurred images in scales.
        {
            cv::imshow(SHOW_WINDOW_NAME, img);    //Display the current image.
            cv::waitKey(0);                       //Wait for user input to terminate.
        }
    }
}


/**
 *  An overload of the show function which can easily display a vector of images.
 *
 *  @param img A vector of matrix objects which we want to display on the screen.
 */
void show(const std::vector<cv::Mat>& images)
{
    std::vector<std::vector<cv::Mat>> scale;
    scale.push_back(images);
    show(scale);
}


/**
 *  An overload of the show function which can easily display a single mat object.
 *
 *  @param img A matrix object which we want to display on the screen.
 */
void show(const cv::Mat& img)
{
    std::vector<cv::Mat> images;
    images.push_back(img);
    show(images);
}


/**
 *  A recursive function which returns the sigma for each level of blurring.
 *  The number of levels and the added sigma (At each level) is passed to it,
 *  and the sigmas for each level are saved in the blurSigmas vector.
 *
 *  @param blurSigmas An empty (output) vector which will hold the sigma levels.
 *  @param level The number of blur levels in total.
 *  @param sigma The base sigma which the image is blurred by in each level.
 */
void getSigmas(std::vector<pixType>& sigmas,
                int levels, double sigma)
{

    //For the base image, just put a sigma level of 0 (not blurred).
    sigmas.push_back(sigma);

    //Calculate the blurring factor used in each level (2 ^ (1/levels))
    double factor = std::pow(2.0, 1.0 / levels);

    //Calculate the sigma for each level. Based on the reference SIFT implementation.
    //Uses the following formula: total^2 = sigma[i]^2 + sigma[i-1]^2
    for(int i = 1; i < levels; i++)
    {
        double previous = std::pow(factor, (double) i - 1.0) * sigma;
        double total = previous * factor;
        sigmas.push_back(std::sqrt(std::pow(total, 2.0) - std::pow(previous, 2.0)));
    }
}


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
void blurLevels(cv::Mat& src, std::vector<cv::Mat>& dest, std::vector<pixType>& sigmas)
{
    //Add the original source image to the base of the pyramid.
    //The first image should not be blurred (blurSigmas[0] is not used).
    dest.push_back(src);

    //Build the pyramid.
    for(int i = 1; i < sigmas.size(); i++) {
        //Calculate the amount of blur, and Create the blurred image.
        cv::Mat blurred;
        cv::GaussianBlur(dest[i - 1], blurred, cv::Size(), sigmas[i], sigmas[i]);

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
 *  @param sigmas The sigma amount used in each octave's scalespace.
 *  @param octaves The number of octaves in the destination vector.
 */
void buildPyramid(cv::Mat& src,
                    std::vector<std::vector<cv::Mat>>& dest,
                    std::vector<double>& sigmas,
                    int octaves)
 {
     //Get the minimum dimention of the source image.
     int maxOctaves = 0;
     int minSize = (src.rows < src.cols ? src.rows : src.cols);

     //Calculate the maximum number of scales possible for this source image.
     while(minSize > std::pow(SCALE_FACTOR, maxOctaves))
         maxOctaves++;

     //Check for invalid number of scales.
     if(octaves < 1 || octaves > maxOctaves)
         throw std::invalid_argument("Number of octaves is not in range.");

     //Add the original source image and it's blur levels to the base of the pyramid.
     std::vector<cv::Mat> octaveZero;
     blurLevels(src, octaveZero, sigmas);
     dest.push_back(octaveZero);

     //Build the pyramid.
     for(int i = 1; i < octaves; i++) {
         //Calculate new size based on the scaling factor.
         cv::Size s(dest[i - 1][0].cols / SCALE_FACTOR,
             dest[i - 1][0].rows / SCALE_FACTOR);

         //Calculate the scaled image (nearest neighbor interpolation)
         cv::Mat scaled;
         cv::pyrDown(dest[i - 1][sigmas.size() - 1], scaled, s);

         //Current Octave.
         std::vector<cv::Mat> octaveCurr;
         blurLevels(scaled, octaveCurr, sigmas);
         dest.push_back(octaveCurr);
     }
 }


/**
 *  A function which build a Difference of Gaussians (DoG) for all the scales.
 *  It negates the images with subsequent blur levels.
 *
 *  @param src The pyramid with the various scale, and blur levels.
 *  @param dest The output pyramid which will be filed various scales, and DoG.
 */
void buildDoG(std::vector<std::vector<cv::Mat>>& src, std::vector<std::vector<cv::Mat>>& dest)
{
    //Go through all the octaves in the source.
    for(std::vector<cv::Mat> octaves : src)
    {
        //Calculate the difference of gradients in different levels.
        std::vector<cv::Mat> dog ;

        //Go though the blur levels and subtract the matrixes (DoG instead of LoG).
        for(int i = 0; i < octaves.size() - 1; i++)
            dog.push_back(octaves[i] - octaves[i + 1]);

        //Add the DoG for the current scale to the destination vector.
        dest.push_back(dog);
    }
}


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
bool isExtreme(std::vector<cv::Mat>& pyr, int r, int c, int lvl)
{
    //Define booleans for determining min/max.
    bool isMax = true;
    bool isMin = true;

    //Get the value of current pixel (Which we'll use for comparison).
    pixType currentPix = pyr[lvl].at<pixType>(r,c);

    //Go through all the 26 neighboring points.
    for(int rOffset = -1 ; rOffset < 2 ; rOffset++)
        for(int cOffset = -1; cOffset < 2 ; cOffset++)
            for(int lOffset = -1; lOffset < 2 ; lOffset++)
            {
                //To improve speed, don't continue if it's done.
                if(!isMin && !isMax) break;

                //Check if it's larger than the neighbor.
                if(currentPix > pyr[lvl + lOffset].at<pixType>(r + rOffset, c + cOffset))
                    isMin = false;      //It's not a minimum point.

                //Check if it's smaller than the neighbor.
                if(currentPix < pyr[lvl + lOffset].at<pixType>(r + rOffset, c + cOffset))
                    isMax = false;      //It's not a maximum point.
            }

    //If isMin XOR isMax is true, then it is a maxima.
    return (isMin || isMax) && !(isMin && isMax);
}


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
void orientationHist(cv::Mat& src, std::vector<double>& dest,
                        int r, int c, double stdev, int radius, int numBins)
{
    //Initialize the destination vector with the bins (Initially empty).
    for(int b = 0; b < numBins; b++)
        dest.push_back(0.0);

    //Calculate the exponential denominator based on the stdev.
    double denominator = 2.0 * std::pow(stdev, 2.0);

    //Go through the surrounding pixels.
    for(int rOffset = -radius; rOffset <= radius; rOffset++)
        for(int cOffset = -radius; cOffset <= radius; cOffset++)
        {
            int currRow = r + rOffset;      //Calculate the row and col with offset.
            int currCol = c + cOffset;

            //If the offset row and column are outside of image, ignore this point.
            if(currRow > 0  &&  currRow < src.rows - 1  &&
                 currCol > 0  &&  currCol < src.cols - 1)
                 continue;

            //Calculate the gradient magnitude and orientation at the current point.
            double dx = src.at<pixType>(currRow, currCol+1) - src.at<pixType>(currRow, currCol-1);
            double dy = src.at<pixType>(currRow-1, currCol) - src.at<pixType>(currRow+1, currCol);
            double gradMagnitude = std::sqrt(std::pow(dx, 2) + std::pow(dy, 2));
            double gradOrientation = std::atan2(dy, dx);

            //Calculate the multiplier for the magnitude.
            double w = std::exp(-(std::pow(rOffset, 2) + std::pow(cOffset, 2)) / denominator);

            //Calculate which bin should the resolution go to.
            int currBin =  std::round(numBins * (gradOrientation + CV_PI) / (CV_PI * 2.0));

            //If the bin is out of range, set it to the first one.
            if(currBin > numBins)
                currBin = 0;

            //Write it to the correct bin.
            dest[currBin] += (gradMagnitude * w);
        }
}


/**
 *  A function which smoothes a histogram by a gaussian method. It also can do it
 *  multiple times. Which makes the image possibly smoother.
 *
 *  @param histogram The histogram which we're smoothing.
 *  @param multiplier The number of times tha the histogram is smoothed.
 */
void histogramGauss(std::vector<double>& histogram, int multiplier)
{
    //If multiplier was reuqested, smooth it a few times.
    for(int multi = 0; multi < multiplier; multi++)
    {
        //Capture the first and last histogram magnitudes.
        double first = histogram[0];
        double last = histogram[histogram.size() - 1];

        //Go through the histogram (Expect the last bin).
        for(int i = 0; i < histogram.size() - 1; i++)
        {
            //Smooth the histogram using a gradient approach.
            double curr = histogram[i];
            histogram[i] = (last + histogram[i] + ((histogram[i+1] / 2.0))) / 2.0;
            last = curr;
        }

        //For the last bin, smooth it using the value previously saved from first bin.
        int lastIndex = histogram.size() - 1;
        histogram[lastIndex] = (last + histogram[lastIndex] + ((first / 2.0))) / 2.0;
    }
}


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
cv::KeyPoint getKp(std::vector<std::vector<cv::Mat>>& pyr, double sigma,
    int octave, int lvl, int r, int c)
{
    //Calculate the scale for this keypoint based on sigma and blur leve.
    double scale = sigma * pow(2.0, octave + (lvl / pyr[octave].size()));

    //Not needed(used) in sGLOH, so disabled by default to improve performance.
    if(CALC_ROTATION)
    {
        //Calculate the dominant orientation.
        std::vector<double> hist;

        //Calculate the std deviation for orientation histogram.
        double stdev = std::round(4.5 * octave);
        double radius = 1.5 * octave;
        orientationHist(pyr[octave][lvl], hist, r, c, stdev, radius);
        //TODO: To finish this part, we need to calculate dominant orientation, and add
        //      Multiple keypoints per rotation of the orientation.
    }

    //Create a keypoint and detect it.
    return cv::KeyPoint(cv::Point2f(r,c), scale, -1, 0, octave);
}


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
bool isEdge(cv::Mat& dog, int r, int c, pixType curve_threshold)
{
    //Get the pixel values and calculate the hessian matrix.
    pixType d =  dog.at<pixType>(r, c);
    pixType dxx = dog.at<pixType>(r, c+1) + dog.at<pixType>(r, c-1) - (2*d);
    pixType dyy = dog.at<pixType>(r+1, c) + dog.at<pixType>(r-1, c) - (2*d);
    pixType dxy = (dog.at<pixType>(r+1, c+1) - dog.at<pixType>(r+1, c-1)
                    - dog.at<pixType>(r-1, c+1) + dog.at<pixType>(r-1, c-1)) / 4.0 ;

    //Calculate sum of eigenvalues, and then their product.
    pixType tr = dxx + dyy;                         //Sum of eigenvalues.
    pixType det = (dxx * dyy) - (dxy * dxy);        //Calculate determenant.

    //Check if it's a positive determenant, and if it meets the threshold.
    if(det > 0 && std::pow(tr, 2.0) / det < std::pow(curve_threshold + 1.0, 2.0) / curve_threshold)
        return false;

    //If we get here, then the keypoint is on an edge.
    return true;
}



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
void findKeypoints(std::vector<std::vector<cv::Mat>>& pyr,
                     std::vector<cv::KeyPoint>& kp,
                     std::vector<pixType>& sigmas,
                     pixType contrast_threshold,
                     pixType curve_threshold)
{
     //Calculate the threshold based on the levels in each octave.
     pixType threshold = (contrast_threshold / 2.0) / sigmas.size();

     //For keeping track of the current octave
     //Go through all the octaves in the pyramid.
     for(int octave = 0; octave < pyr.size(); octave++)
     {
         //Go through all the images (Except first and last) in all blur levels.
         for(int lvl = 1; lvl < pyr[octave].size() - 1; lvl++)
         {
             //Go through every pixel in the image (Except first and last).
             for(int r = 1; r < pyr[octave][lvl].rows - 1; r++)
                for(int c = 1; c < pyr[octave][lvl].cols - 1; c++)
                {
                    //Check if the pixel meets the contrast threashold.
                    if(std::abs(pyr[octave][lvl].at<pixType>(r,c)) < threshold)
                        continue;

                    //Check if the current pixel is an extremum.
                    if(isExtreme(pyr[octave], r, c, lvl))
                    {
                        //Check if this keypoint is on an edge, reject if it is.
                        //if(isEdge(pyr[octave][lvl], r, c, curve_threshold))
                        //    continue;

                        //Create a keypoint with at this position with the sigma.
                        //The angle and response are set to default (To be calculated).
                        std::cerr << "Keypoint added at: " << r << " " << c << " " << octave << std::endl;
                        kp.push_back(getKp(pyr, sigmas[lvl], octave, lvl, r, c));
                    }
                }
        }
     }
 }


/**
 *  A function which calculates the keypoints for the given source image. It
 *  uses a similar approach to SIFT to detect keypoints. However, as it is, it
 *  does not do subpixel approximation.
 *
 *  @param src The source image which we're calculating these for.
 *  @param dest The output vector which the keypoint will be written to.
 */
void detect(cv::Mat& src, std::vector<cv::KeyPoint>& dest)
{
    //Convert to grayscale if needed.
    cv::Mat orig;
    if( src.channels() >= 3 )
        cvtColor(src, orig, cv::COLOR_BGR2GRAY);
    else
        orig = src;

    //Change color space to the right depth.
    cv::Mat img;
    orig.convertTo(img, DEPTH_TYPE, 1.0/255.0);

    //Get the sigmas for each level of scalespace.
    std::vector<pixType> sigmas;
    getSigmas(sigmas);

    //Show sigmas at differnet levels (For testing).
    //for(double sig : sigmas) {
    //    std::cerr << "Sigma: " << sig << std::endl;
    //
    //}

    //Build scales, and blurs pyramid (2D).
    std::vector<std::vector<cv::Mat>> pyr ;
    buildPyramid(img, pyr, sigmas);


    //Calculate the difference of gaussians.
    std::vector<std::vector<cv::Mat>> dog ;
    buildDoG(pyr, dog);

    //Display the pyramids to the screen (For testing).
    //show(pyr);
    //show(dog);

    //Calculate the maxima to find keypoints (And filter them).
    findKeypoints(dog, dest, sigmas);
}


/**
 *  A test function just for making sure the algorithms above work. It is
 *  currently called by the main method.
 */
void test()
{
    //Generate a test image
    cv::Mat orig=cv::Mat::zeros(1000,1000,CV_8U);
    //Fill the image with noise.
    for (int i = 0; i < orig.rows; i++)
	        for (int j = 0; j < orig.cols; j++)
		    	orig.at<uchar>(i,j)= rand() % 255;

    //Add with two blobs (For testing detection).
    for(int i = 100; i < 120; i++)
        for(int j = 100; j < 120; j++)
            orig.at<uchar>(i,j) = 0;

    for(int i = 800; i < 820; i++)
        for(int j = 800; j < 820; j++)
            orig.at<uchar>(i,j) = 0;

    //Sift keypoint detection for comparison.
    //Ptr<xfeatures2d::SIFT> detector = xfeatures2d::SIFT::create(0, 5, 0.04, 10, 1.6);
    //std::vector<cv::KeyPoint> keypoints;
    //detector->detect(orig, keypoints);
    //std::cerr << keypoints.size() << " KeyPoints found by sift." << std::endl;

    //Calculate the keypoints using the implemented method.
    std::vector<cv::KeyPoint> kp;
    detect(orig, kp);
    std::cerr << kp.size() << " KeyPoints found by sgloh." << std::endl;
}

}
