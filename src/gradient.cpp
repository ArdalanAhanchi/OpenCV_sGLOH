// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

using namespace cv;

/** The type which is used when accessing the matrices. */
typedef double GR_PIX_TYPE;

/** The depth type corresponding to GR_PRIX_TYPE. */
const int GR_DEPTH_TYPE = CV_64F;

/**
 *  A function which calculates gradient magnitudes and angles for the image,
 *  and writes them in a 3 dimentional matrix object. The gradients matrix will
 *  include a Vec2d object in each row and column. The Vec2d Object will include
 *  the magnitude as it's 0th index, and angle as it's 1st.
 *
 *  @param src The image which we're calculating the gradients for.
 *  @param gradients The mat object where we'll write the gradient information.
 */
void findGradient(Mat& src, Mat& gradients)
{
    //Use the sobel operator to calculate gradients at x and y dirs.
    Mat x, y;
    Sobel(src, gx, GR_DEPTH_TYPE, 1, 0, 1);
    Sobel(src, gy, GR_DEPTH_TYPE, 0, 1, 1);

    //Find gradient magnitude (by gx, and gy). And find it's angles.
    Mat magnitudes, angles;
    cartToPolar(gx, gy, magnitudes, angles, 1);

    //Go through the row and columns and populate the vec2d's.
    for(int r = 0; r < src.rows; r++)
        for(int c = 0; c < src.cols; c++)
            gradients.at<Vec2d>(r,c) = Vec2d(magnitudes.at<GR_PIX_TYPE>(r,c),
                                                angles.at<GR_PIX_TYPE>(r,c));
}
