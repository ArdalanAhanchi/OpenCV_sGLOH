#ifndef SGLOH_OPTIONS_H
#define SGLOH_OPTIONS_H

namespace SGloh
{

enum KpType {SGLOH, SIFT, SURF};

class Options
{
public:
	int m;
	int n;
	int v;
	bool psi;
	float sigma;
    KpType type;  //The type of extractor being used.

	void setOptions(int _m, int _n, int _v,
         bool _psi, float _sigma, KpType _type);
};

}

#endif
