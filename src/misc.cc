#include "mics.h"

//  sets  exp(-I*p*x) on each lattice site (slowly)
void makeMomPhaseField(const Momentum& p,
    LatticeComplex& ret)
{
  int Lx, Ly, Lz;
  std::tie(Lx, Ly, Lz) = Layout::dimTup;

  real p0 = 6.28318530718/Lx * std::get<0>(p);
  real p1 = 6.28318530718/Ly * std::get<1>(p);
  real p2 = 6.28318530718/Lz * std::get<2>(p);

  for (int cX=0; cX<Lx; ++cX) {
    for (int cY=0; cY<Ly; ++cY) {
      real px = p0 * cX;
      real py = p1 * cY;
      for (int cZ=0; cZ<Lz; ++cZ) {
        real pdotx = p2 * cZ + px + py;
        int linInd = Layout::getLexicoIndex(std::make_tuple(cX, cY, cZ));
        ret.xPtr[linInd] = {cos(pdotx), -sin(pdotx)};
      }
    }
  }
}

void init_col_vec(LatticeColorVector& colVec) {
  int nSites = Layout::sitesOnNode();
  for (int iX=0; iX<nSites; ++iX) {
    colVec.xPtr[iX].c0 = {real(4.0) * real(std::rand()) / real(RAND_MAX) - real(2.0),
      real(4.0) * real(std::rand()) / real(RAND_MAX) - real(2.0)};

    colVec.xPtr[iX].c1 = {real(4.0) * real(std::rand()) / real(RAND_MAX) - real(2.0),
      real(4.0) * real(std::rand()) / real(RAND_MAX) - real(2.0)};

    colVec.xPtr[iX].c2 = {real(4.0) * real(std::rand()) / real(RAND_MAX) - real(2.0),
      real(4.0) * real(std::rand()) / real(RAND_MAX) - real(2.0)};
  }
}

void init_coeff(multi2d<cmplx>& coeff) {
  int nDil = coeff.nrows();
  int nEv = coeff.ncols();
  std::srand(2);
  for (int iDil=0; iDil<nDil; ++iDil) {
    for (int iEv=0; iEv<nEv; ++iEv) {
      coeff(iDil, iEv) = {real(4.0) * real(std::rand()) / real(RAND_MAX) - real(2.0),
        real(4.0) * real(std::rand()) / real(RAND_MAX) - real(2.0)};
    }
  }
}

bool check_results(const multi4d<cmplx>& res, const multi4d<cmplx>& ref) {
  if (res.size() != ref.size()) {
    std::cout << "Size mismatch\n";
    return false;
  }

  for (unsigned iEl=0; iEl<res.size(); ++iEl) {
    if (abs(res.getElem(iEl) - ref.getElem(iEl)) / abs(ref.getElem(iEl)) > 1e-3 \
        && abs(ref.getElem(iEl)) > 1e-4) {
      std::cout << "Result mismatch at iEl = "<<iEl<<", eps = " << \
        abs(res.getElem(iEl) - ref.getElem(iEl))/abs(ref.getElem(iEl)) \
        <<": "<<res.getElem(iEl) << " " << ref.getElem(iEl) << "\n";
      return false;
    }
  }
  return true;
}
