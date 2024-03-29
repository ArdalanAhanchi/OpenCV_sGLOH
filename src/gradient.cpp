#include "gradient.h"

namespace SGloh
{

/**
 *  A function which calculates gradient magnitudes and angles for the image,
 *  and writes them in a 3 dimentional matrix object. The gradients matrix will
 *  include a Vec2d object in each row and column. The Vec2d Object will include
 *  the magnitude as it's 0th index, and angle as it's 1st.
 *
 *  @param src The image which we're calculating the gradients for.
 *  @param gradients The mat object where we'll write the gradient information.
 */
void findGradient(cv::Mat& src, cv::Mat& gradients)
{
    //Use the sobel operator to calculate gradients at x and y dirs.
    cv::Mat x, y;
    cv::Sobel(src, x, GR_DEPTH_TYPE, 1, 0, 1);
    cv::Sobel(src, y, GR_DEPTH_TYPE, 0, 1, 1);

    //Find gradient magnitude (by x, and y). And find it's angles.
    cv::Mat magnitudes, angles;
    cv::cartToPolar(x, y, magnitudes, angles, false);
	int dimensions[] = { src.rows, src.cols, 2 };
	gradients = cv::Mat(3, dimensions, CV_32F, cv::Scalar::all(0));
    //Go through the row and columns and populate the vec2d's.
	for (int r = 0; r < src.rows; r++)
	{
		for (int c = 0; c < src.cols; c++)
		{
			gradients.at<float>(r, c, 0) = (float)magnitudes.at<GR_PIX_TYPE>(r, c);
			gradients.at<float>(r, c, 1) = (float)angles.at<GR_PIX_TYPE>(r, c);

		}
	}
}

}
