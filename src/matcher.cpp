#include "matcher.h"

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
                SGloh::Options options, bool filter)
{
    //Hold all the matches through the runs.
    std::vector<std::vector<cv::DMatch>> matches;
	matches.resize((size_t)options.m);
	size_t sizeSum = 0;

    //Do multiple runs while rotating the descriptors.
	for (int h = 0; h < options.m; h++)
	{
        //Go through the descriptors.
		for (int i = 0; i < descOne.rows; i++)
		{
            //Create a new match.
			cv::DMatch curr = cv::DMatch();
			curr.distance = 1000000;
			for (int j = 0; j < descTwo.rows; j++)
			{
				//Get distance
				float sumSquares = 0;
				for (int k = 0; k < descOne.cols; k++)
				{
					float testF1 = descOne.at<float>(i, k);
					float testF2 = descTwo.at<float>(j, k);
					sumSquares += std::pow(descOne.at<float>(i, k) - descTwo.at<float>(j, k), 2);
				}

                //Populate the DMatch object with the results.
				float tempDistance = std::sqrt(sumSquares);
				if (tempDistance < curr.distance)
				{
					curr.distance = tempDistance;
					curr.queryIdx = i;
					curr.trainIdx = j;
				}
			}
            //If it's set, just add it to matches.
			if (curr.distance < 1000000)
			{
				matches[h].push_back(curr);
			}
		}

        //Rotate the descriptors to allow rotation invariance.
		SGloh::rotateDescriptors(descOne.clone(), descOne, options);
	}
	std::vector<cv::DMatch> goodMatches;

    //Go through the orientations, and filter out matches if required.
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

            //If filtering was requested check the thresholds.
            if(filter)
                if (curr.distance >= 0 && curr.distance < 0.25f)
                    goodMatches.push_back(curr);
            else
                goodMatches.push_back(curr);
		}
	}

    //Remove the matched duplicates, to achieve best matches.
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
}

}
