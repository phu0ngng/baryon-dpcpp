#include <algorithm>
#include <iostream>
#include "kernels_qdp3.h"
#include "swatch.h"
#include <omp.h>


BaryonKernelCpuRef::BaryonKernelCpuRef(unsigned int blockSizeMomProj)
  : blockSizeMomProj(blockSizeMomProj), momBuf(nullptr), nMom(0), evBuf(nullptr) { resetTimers(); };

BaryonKernelCpuRef::~BaryonKernelCpuRef() {
  delete[] momBuf; momBuf = nullptr;
  delete[] evBuf; evBuf = nullptr;
}


void BaryonKernelCpuRef::resetTimers() {
  timeReconst = 0.;
  timeComp = 0.;
}

void BaryonKernelCpuRef::setMomentumSet(const std::set<Momentum>& momSet) {
  // regenerate momentum fields if necessary
  unsigned int nSites = Layout::sitesOnNode();

  if (momSet.size() != size_t(nMom)) {
    nMom = momSet.size();
    delete[] momBuf;
    momBuf = new cmplx[nMom*nSites];
  }

  int iMom = 0;
  for (const auto& aM : momSet) {
    LatticeComplex aC(&momBuf[iMom*nSites], 0.);
    makeMomPhaseField(aM, aC);
    iMom++;
  }
}


void BaryonKernelCpuRef::readEV(int _nEv,
    const std::function<const LatticeColorVector&(int)>& readFunc) {
  unsigned int nSites = Layout::sitesOnNode();

  // reserve buffer for eigenvectors
  if (evBuf == nullptr) {
    nEv = _nEv;
    evBuf = new ColorVector[nEv*nSites];
  }

  if (nEv != _nEv) {
    std::cerr << "readEV: Number of eigenvectors "
      << _nEv
      << " doesn't match buffer size " << nEv << std::endl;
    exit(1);
  }

  for (int iEv=0; iEv<nEv; ++iEv) {
    LatticeColorVector aRef(&evBuf[iEv*nSites], 0.);
    aRef = readFunc(iEv);
  }
}

multi4d<cmplx> BaryonKernelCpuRef::apply(
    const multi2d<cmplx>& coeffs1,
    const multi2d<cmplx>& coeffs2,
    const multi2d<cmplx>& coeffs3) {

  if (coeffs1.ncols() != nEv || coeffs2.ncols() != nEv || coeffs3.ncols() != nEv) {
    std::cerr << "BaryonKernelCpuRef: coefficient length doesn't match number of "
      << "eigenvectors, nEv = "<<nEv<<", q1 = "<<coeffs1.ncols()
      << ", q2 = "<<coeffs2.ncols() << ", q3 = "<<coeffs3.ncols()<< std::endl;
    exit(1);
  }

  unsigned int nSites = Layout::sitesOnNode();
  unsigned int n1 = coeffs1.nrows();
  unsigned int n2 = coeffs2.nrows();
  unsigned int n3 = coeffs3.nrows();

  cmplx alpha(1.0,0.0);
  cmplx beta(0.0,0.0); // P : Just calculate C = A*B

#ifdef OMP
  auto time_start = omp_get_wtime();
#endif
  cmplx* evP = (cmplx*)&evBuf[0].c0;
  cmplx* coeff1P = (cmplx*)&coeffs1(0,0);
  cmplx* coeff2P = (cmplx*)&coeffs2(0,0);
  cmplx* coeff3P = (cmplx*)&coeffs3(0,0);

  ColorVector* q1Buf = new ColorVector[n1*nSites];
  ColorVector* q2Buf = new ColorVector[n2*nSites];
  ColorVector* q3Buf = new ColorVector[n3*nSites];

  cmplx* q1P = (cmplx*)&q1Buf[0].c0;
  cmplx* q2P = (cmplx*)&q2Buf[0].c0;
  cmplx* q3P = (cmplx*)&q3Buf[0].c0;

  // reconstruct q1, q2, q3 fields
#ifdef DP
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, 3*nSites, nEv, &alpha, coeff1P, nEv, evP, 3*nSites, &beta, q1P, 3*nSites);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n2, 3*nSites, nEv, &alpha, coeff2P, nEv, evP, 3*nSites, &beta, q2P, 3*nSites);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n3, 3*nSites, nEv, &alpha, coeff3P, nEv, evP, 3*nSites, &beta, q3P, 3*nSites);
#else
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, 3*nSites, nEv, &alpha, coeff1P, nEv, evP, 3*nSites, &beta, q1P, 3*nSites);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n2, 3*nSites, nEv, &alpha, coeff2P, nEv, evP, 3*nSites, &beta, q2P, 3*nSites);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n3, 3*nSites, nEv, &alpha, coeff3P, nEv, evP, 3*nSites, &beta, q3P, 3*nSites);
#endif

#ifdef OMP
  timeReconst = omp_get_wtime() - time_start;
  time_start = omp_get_wtime();
#endif

  // buffer for blocked momentum projection
  cmplx* complBuf = new cmplx[blockSizeMomProj*nSites];
  unsigned int nInBlock = 0; // block count

  multi4d<cmplx> retArr(nMom,n1,n2,n3);
#ifdef ADVISOR
    __itt_resume();
#endif

  for (unsigned int dil1=0; dil1<n1; ++dil1) {
    LatticeColorVector q1(&q1Buf[dil1*nSites], 0.);
    for (unsigned int dil2=0; dil2<n2; ++dil2) {

      LatticeColorVector q2(&q2Buf[dil2*nSites], 0.);
      LatticeColorVector diq;
      colorCrossProduct(q1, q2, diq);

      for (unsigned int dil3=0; dil3<n3; ++dil3) {
        LatticeColorVector q3(&q3Buf[dil3*nSites], 0.);
        LatticeComplex singlet(&complBuf[nSites*nInBlock], 0.);
        colorVectorContract(diq, q3, singlet);
        nInBlock++;

        // blocked momentum projection
        if (nInBlock == blockSizeMomProj || (dil1+1 == n1 && dil2+1 == n2 && dil3+1 == n3)) {
          cmplx* resP = (&retArr(0,dil1,dil2,dil3))-nInBlock+1;
          evaluateMomentumSums(complBuf, resP, nInBlock, n1*n2*n3);
          nInBlock = 0;
        }
      }
    }
  }
#ifdef OMP
  timeComp = omp_get_wtime() - time_start;
#endif
#ifdef ADVISOR
    __itt_pause();
#endif

  delete[] q1Buf;
  delete[] q2Buf;
  delete[] q3Buf;
  delete[] complBuf;

  return retArr;
}

std::vector<std::pair<std::string, double>> BaryonKernelCpuRef::getTimings() const {
  return {
    {"reconstruct fields", timeReconst},
      {"parallel comp time", timeComp}
  };
}
