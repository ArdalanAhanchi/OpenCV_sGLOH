#ifndef SGLOH_OPTIONS_H
#define SGLOH_OPTIONS_H

namespace SGloh
{

/**
 * An enum which determines the keypoint extractor type to use in sGLOH.
 * SGloh type is our SIFT based extractor which doesn't perform rotations.
 * it is significantly faster, but also less accurate.
 */
enum KpType {SGLOH, SIFT, SURF};

/**
 *  A class which represents options to use for the SGLOH algorithm.
 *  The parameters are based on the original paper.
 */
class Options
{
public:
	int m;                     //Used in the paper.
	int n;
	int v;
	bool psi;
	float sigma;
    KpType type;               //The type of extractor being used.

    /**
     *  A function which sets the class variables to the passed ones. It is used
     *  to initialize the options easily (Like a constructor).
     */
	void setOptions(int _m, int _n, int _v,
         bool _psi, float _sigma, KpType _type);
};

}

#endif
