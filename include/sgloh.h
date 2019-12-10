#ifndef SGLOH_H
#define SGLOH_H
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <cmath>
using namespace cv;
class sGLOH : public Feature2D
{
private:
	const float PI = 3.141592653589793;
	float GetM(float q, float x);
	float CalculateBin(int r, int d, int i, int m, int n, bool psi, float sigma, Mat& gradients, KeyPoint origin);
	float CalculateBinPlus(int r, int d, int i, int m, int n, int v, bool psi, float sigma, Mat& gradients, KeyPoint origin);
	
public:
	struct sGLOH_Options
	{
		int m; // number of angular segmentations
		int n; // number of radial segmentations
		int v; // number of bins for sGLOH+
		bool psi; // unique center bin or not
		float sigma; // sigma value to force
	};

	void detectAndCompute(InputArray _image, std::vector<KeyPoint>& keypoints, OutputArray _descriptors, sGLOH_Options options);
	void calculate_sGLOH_Descriptor(int m, int n, bool psi, float sigma, Mat& gradients, std::vector<KeyPoint>& keypoints, OutputArray _descriptors);
	void calculate_sGLOH_Plus_Descriptor(int m, int n, int v, bool psi, float sigma, Mat& gradients, std::vector<KeyPoint>& keypoints, OutputArray _descriptors);

	/*static Ptr<sGLOH> create(int _nfeatures = 0, int _nOctaveLayers = 3,
		float _contrastThreshold = 0.04, float _edgeThreshold = 10.0, float _sigma = 0.7);*/

	sGLOH();
	~sGLOH();
};
#endif