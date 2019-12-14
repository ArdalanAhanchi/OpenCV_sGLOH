#include "options.h"

namespace SGloh
{

/**
 *  A function which sets the class variables to the passed ones. It is used
 *  to initialize the options easily (Like a constructor).
 */
void Options::setOptions(int _m, int _n, int _v,
    bool _psi, float _sigma, KpType _type)
{
	this->m = _m;
	this->n = _n;
	this->v = _v;
	this->psi = _psi;
	this->sigma = _sigma;
    this->type = _type;
}

}
