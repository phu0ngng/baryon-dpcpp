#include "qdp_multi.h"
#include "lattice.h"

void makeMomPhaseField(const Momentum& p, LatticeComplex& ret);

void init_col_vec(LatticeColorVector& colVec);

void init_coeff(multi2d<cmplx>& coeff);

bool check_results(const multi4d<cmplx>& res, const multi4d<cmplx>& ref);

template <typename T>
void init_coeff_dpc(T* coeff, unsigned int nEv, unsigned int nBlockD, unsigned int Blocksize) {
  std::srand(2);
  for (int iBlockD=0; iBlockD < nBlockD; iBlockD++)
    for (int iD=0; iD < Blocksize; ++iD)
      for (int iEv=0; iEv < nEv; ++iEv)
        coeff[iBlockD * nEv + iEv].d[iD] = {real(4.0) * real(std::rand()) / real(RAND_MAX) - real(2.0),
          real(4.0) * real(std::rand()) / real(RAND_MAX) - real(2.0)};
}
