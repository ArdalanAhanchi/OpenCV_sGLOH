#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/core/traits.hpp>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <stdexcept>

#include "options.h"
#include "sgloh.h"

#ifndef SGLOH_MATCHER_H
#define SGLOH_MATCHER_H

namespace SGloh
{

/**
 *  A function which implements a custom bruteforce matcher for sGLOH. This is
 *  used due to the limitations/compatibility issues with the OpenCV's
 *  implementation of Bruteforce matcher.
 *
 *  @param descOne The descriptor for the image one.
 *  @param descTwo The descriptor for the image two.
 *  @param bestMatches The output which will hold the matches between descriptors.
 *  @param options The options which were used in sGLOH.
 *  @param filter If it's set to true, it will filter-out far matches.
 */
void match(cv::Mat descOne, cv::Mat descTwo, std::vector<cv::DMatch>& bestMatches,
                SGloh::Options options, bool filter);

}

#endif
