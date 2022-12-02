#ifndef LATTICE_H
#define LATTICE_H

#include <tuple>
#include <complex>
#include "assert.h"

#ifdef DP
using real = double;
#else
using real = float;
#endif
using cmplx = std::complex<real>;
using Momentum = std::tuple<int, int, int>;

// ***************************************************************

struct Layout {

  static void init(const std::tuple<int, int, int>& _dimTup) {
    dimTup = _dimTup;
  }

  static int sitesOnNode() { return std::get<0>(dimTup) * std::get<1>(dimTup) * std::get<2>(dimTup); };

  static int getLexicoIndex(const std::tuple<int, int, int>& coord) {
    return std::get<2>(coord)+std::get<2>(dimTup)*(std::get<1>(coord)+std::get<1>(dimTup)*std::get<0>(coord));
  }

  static std::tuple<int, int, int> dimTup;

};

// ***************************************************************

struct ColorVector {
  cmplx c0;
  cmplx c1;
  cmplx c2;
};


template<typename Inner>
struct LatticeContainer {
  bool ownMem;
  Inner* xPtr;
  

  LatticeContainer() : ownMem(true) {
    assert(Layout::sitesOnNode() > 0);
    xPtr = new Inner[Layout::sitesOnNode()];
  }

  LatticeContainer(Inner* _xPtr, float dummy) : ownMem(false), xPtr(_xPtr) {
    assert(Layout::sitesOnNode() > 0);
  }

  LatticeContainer(const LatticeContainer& rhs) = delete;

  LatticeContainer(LatticeContainer&& rhs) {
    xPtr = rhs.xPtr; rhs.xPtr = nullptr;
    ownMem = rhs.ownMem; rhs.ownMem = false;
  }


  ~LatticeContainer() {
    if (ownMem) delete[] xPtr;
    xPtr = nullptr;
  }

  LatticeContainer& operator=(const LatticeContainer& rhs) {
    int nX = Layout::sitesOnNode();
    for (int iX=0; iX<nX; ++iX) xPtr[iX] = rhs.xPtr[iX];
    return *this;
  }
};



// ***************************************************************


using LatticeColorVector = LatticeContainer<ColorVector>;
using LatticeComplex = LatticeContainer<cmplx>;


// ***************************************************************


  // helper functions
  // computes ret_a = eps_abc q1_b q2_c
void colorCrossProduct(const LatticeColorVector& q1,
		       const LatticeColorVector& q2,
		       LatticeColorVector& ret);


  // computes ret = diq_a q3_a
void colorVectorContract(const LatticeColorVector& diq,
			 const LatticeColorVector& q3,
			 LatticeComplex& ret);

// ***************************************************************

#endif
