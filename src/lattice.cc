#include "lattice.h"
#include <iostream>


void colorCrossProduct(const LatticeColorVector& q1,
		       const LatticeColorVector& q2,
		       LatticeColorVector& ret) {

  int nSites = Layout::sitesOnNode();

#pragma omp parallel for
  for (int iX=0; iX<nSites; ++iX) {
    ret.xPtr[iX].c0  = q1.xPtr[iX].c1 * q2.xPtr[iX].c2;
    ret.xPtr[iX].c0 -= q1.xPtr[iX].c2 * q2.xPtr[iX].c1;

    ret.xPtr[iX].c1  = q1.xPtr[iX].c2 * q2.xPtr[iX].c0;
    ret.xPtr[iX].c1 -= q1.xPtr[iX].c0 * q2.xPtr[iX].c2;

    ret.xPtr[iX].c2  = q1.xPtr[iX].c0 * q2.xPtr[iX].c1;
    ret.xPtr[iX].c2 -= q1.xPtr[iX].c1 * q2.xPtr[iX].c0;
  }
}


void colorVectorContract(const LatticeColorVector& diq,
			 const LatticeColorVector& q3,
			 LatticeComplex& ret) {

  int nSites = Layout::sitesOnNode();

#pragma omp parallel for
  for (int iX=0; iX<nSites; ++iX) {
    ret.xPtr[iX]  = diq.xPtr[iX].c0 * q3.xPtr[iX].c0;
    ret.xPtr[iX] += diq.xPtr[iX].c1 * q3.xPtr[iX].c1;
    ret.xPtr[iX] += diq.xPtr[iX].c2 * q3.xPtr[iX].c2;
  }
}


// ***************************************************************************************

std::tuple<int, int, int> Layout::dimTup = std::make_tuple(0, 0, 0);
