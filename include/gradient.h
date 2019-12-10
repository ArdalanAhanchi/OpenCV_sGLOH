// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>

#ifndef SGLOH_GRADIENT_H
#define SGLOH_GRADIENT_H

class SGlohGradient 
{
    /** The type which is used when accessing the matrices. */
    typedef double GR_PIX_TYPE;

    /** The depth type corresponding to GR_PRIX_TYPE. */
    static const int GR_DEPTH_TYPE = CV_64F;
public:
    /**
     *  A function which calculates gradient magnitudes and angles for the image,
     *  and writes them in a 3 dimentional matrix object. The gradients matrix will
     *  include a Vec2d object in each row and column. The Vec2d Object will include
     *  the magnitude as it's 0th index, and angle as it's 1st.
     *
     *  @param src The image which we're calculating the gradients for.
     *  @param gradients The mat object where we'll write the gradient information.
     */
    static void findGradient(cv::Mat& src, cv::Mat& gradients);
};

#endif
