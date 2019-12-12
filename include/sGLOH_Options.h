#ifndef SGLOH_OPTIONS_H
#define SGLOH_OPTIONS_H

class sGLOH_Options
{
public:
	int m;
	int n;
	int v;
	bool psi;
	float sigma;

	void setOptions(int _m, int _n, int _v, bool _psi, float _sigma);
	sGLOH_Options();
};
#endif
