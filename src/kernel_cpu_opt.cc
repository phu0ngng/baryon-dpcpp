#include <algorithm>
#include <iostream>
#include "kernels_qdp3.h"
#include "swatch.h"
#include <omp.h>

BaryonKernelCpuOpt::BaryonKernelCpuOpt()
  : momBuf(nullptr), nMom(0), evBuf(nullptr){ 
    resetTimers(); 
  };

BaryonKernelCpuOpt::~BaryonKernelCpuOpt() {
  delete[] momBuf; momBuf = nullptr;
  delete[] evBuf; evBuf = nullptr;
  mkl_jit_destroy(jitter);
}

void BaryonKernelCpuOpt::resetTimers() {
  timeReconst = 0.;
  timeComp = 0.;
}

void BaryonKernelCpuOpt::setMomentumSet(const std::set<Momentum>& momSet) {
  // regenerate momentum fields if necessary
  unsigned int nSites = Layout::sitesOnNode();
  const unsigned int nBlockX = nSites / CPU_BLOCK_X;

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

  // Initialize 
  cmplx alpha = 1.0;
  cmplx beta = 1.0;
#ifdef DP
  mkl_jit_status_t status = mkl_jit_create_zgemm(&jitter, 
      MKL_ROW_MAJOR, MKL_NOTRANS, MKL_TRANS,
      CPU_BLOCK_D3 * CPU_BLOCK_D2 * CPU_BLOCK_D1, nMom, CPU_BLOCK_X,
      &alpha, CPU_BLOCK_X, 
      nSites, 
      &beta, nMom);
  if (MKL_JIT_ERROR == status) {
    std::cout << "Error: insufficient memory to JIT and store the ZGEMM kernel." << std::endl;
  }
  jit_gemm = mkl_jit_get_zgemm_ptr(jitter);
#else
  mkl_jit_status_t status = mkl_jit_create_cgemm(&jitter, 
      MKL_ROW_MAJOR, MKL_NOTRANS, MKL_TRANS,
      CPU_BLOCK_D3 * CPU_BLOCK_D2 * CPU_BLOCK_D1, nMom, CPU_BLOCK_X,
      &alpha, CPU_BLOCK_X, 
      nSites, 
      &beta, nMom);
  if (MKL_JIT_ERROR == status) {
    std::cout << "Error: insufficient memory to JIT and store the ZGEMM kernel." << std::endl;
  }
  jit_gemm = mkl_jit_get_cgemm_ptr(jitter);
#endif
}


void BaryonKernelCpuOpt::readEV(int _nEv,
    const std::function<const LatticeColorVector&(int)>& readFunc) {
  const unsigned int nSites = Layout::sitesOnNode();
  const unsigned int nBlockX = nSites / CPU_BLOCK_X;

  // reserve buffer for eigenvectors
  if (evBuf == nullptr) {
    nEv = _nEv;
    evBuf = new ColorVectorChunkCpu[nEv * nBlockX];
  }

  if (nEv != _nEv) {
    std::cerr << "readEV: Number of eigenvectors "
      << _nEv
      << " doesn't match buffer size " << nEv << std::endl;
    exit(1);
  }

  auto evBufNew = new ColorVector[nEv*nSites];
#pragma ivdep
  for (int iEv=0; iEv<nEv; ++iEv) {
    LatticeColorVector aRef(&evBufNew[iEv*nSites], 0.);
    aRef = readFunc(iEv);
  }
  // packing evBuf to new layout [iEv,c,x]
#pragma omp parallel for
  for (int iEv=0; iEv<nEv; ++iEv) {
    for (int iBlockX=0; iBlockX < nBlockX; iBlockX++){
      for (int iX=0; iX < CPU_BLOCK_X; ++iX){
        evBuf[iEv * nBlockX + iBlockX].c0[iX] = evBufNew[iEv*nSites +  iBlockX * CPU_BLOCK_X + iX].c0;
        evBuf[iEv * nBlockX + iBlockX].c1[iX] = evBufNew[iEv*nSites +  iBlockX * CPU_BLOCK_X + iX].c1;
        evBuf[iEv * nBlockX + iBlockX].c2[iX] = evBufNew[iEv*nSites +  iBlockX * CPU_BLOCK_X + iX].c2;
      }
    }
  }
  delete[] evBufNew;
}


multi4d<cmplx> BaryonKernelCpuOpt::apply(
    const multi2d<cmplx>& coeffs1,
    const multi2d<cmplx>& coeffs2,
    const multi2d<cmplx>& coeffs3) {

  if (coeffs1.ncols() != nEv || coeffs2.ncols() != nEv || coeffs3.ncols() != nEv) {
    std::cerr << "BaryonKernelCpuOpt: coefficient length doesn't match number of "
      << "eigenvectors, nEv = "<<nEv<<", q1 = "<<coeffs1.ncols()
      << ", q2 = "<<coeffs2.ncols() << ", q3 = "<<coeffs3.ncols()<< std::endl;
    exit(1);
  }

  unsigned int nSites = Layout::sitesOnNode();
  unsigned int n1 = coeffs1.nrows();
  unsigned int n2 = coeffs2.nrows();
  unsigned int n3 = coeffs3.nrows();
  const unsigned int nBlockX = nSites / CPU_BLOCK_X;

  cmplx alpha(1.0,0.0);
  cmplx beta(0.0,0.0); // P : Just calculate C = A*B

#ifdef OMP
  auto time_start = omp_get_wtime();
#endif

  cmplx* evP = evBuf[0].c0;

  cmplx* coeff1P = (cmplx*)&coeffs1(0,0);
  cmplx* coeff2P = (cmplx*)&coeffs2(0,0);
  cmplx* coeff3P = (cmplx*)&coeffs3(0,0);

  auto q1 = new ColorVectorChunkCpu[n1 * nBlockX];
  auto q2 = new ColorVectorChunkCpu[n2 * nBlockX];
  auto q3 = new ColorVectorChunkCpu[n3 * nBlockX];

  // reconstruct q1, q2, q3 fields
#ifdef DP
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, 3*nSites, nEv, &alpha, coeff1P, nEv, evP, 3*nSites, &beta, &q1[0].c0[0], 3*nSites);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n2, 3*nSites, nEv, &alpha, coeff2P, nEv, evP, 3*nSites, &beta, &q2[0].c0[0], 3*nSites);
  cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n3, 3*nSites, nEv, &alpha, coeff3P, nEv, evP, 3*nSites, &beta, &q3[0].c0[0], 3*nSites);
#else
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, 3*nSites, nEv, &alpha, coeff1P, nEv, evP, 3*nSites, &beta, &q1[0].c0[0], 3*nSites);
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n2, 3*nSites, nEv, &alpha, coeff2P, nEv, evP, 3*nSites, &beta, &q2[0].c0[0], 3*nSites);
  cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n3, 3*nSites, nEv, &alpha, coeff3P, nEv, evP, 3*nSites, &beta, &q3[0].c0[0], 3*nSites);
#endif

#ifdef OMP
  timeReconst = omp_get_wtime() - time_start;
  time_start = omp_get_wtime();
#endif

  multi4d<cmplx> retArr(nMom,n1,n2,n3);
  auto ptrMomBuf = momBuf;

#ifdef ADVISOR
    __itt_resume();
#endif

#pragma omp parallel firstprivate(q1,q2,q3)
  {
    cmplx singlet[CPU_BLOCK_D3][CPU_BLOCK_D2*CPU_BLOCK_D1][CPU_BLOCK_X];
    cmplx diq[CPU_BLOCK_D1*CPU_BLOCK_D2][3][CPU_BLOCK_X];
    cmplx retTmp[n3][CPU_BLOCK_D2*CPU_BLOCK_D1][nMom];
    mkl_set_num_threads(1);
#pragma omp for collapse(2) 
    for (unsigned int d1BlockId = 0; d1BlockId < n1; d1BlockId += CPU_BLOCK_D1){
      for (unsigned int d2BlockId = 0; d2BlockId < n2; d2BlockId += CPU_BLOCK_D2){
        memset(retTmp, 0., sizeof(cmplx) * (nMom*CPU_BLOCK_D2*CPU_BLOCK_D1 * n3));

        for (int xBlockId = 0; xBlockId < nBlockX; xBlockId++){
          for (unsigned int dil1 = d1BlockId; dil1 < d1BlockId + CPU_BLOCK_D1; dil1++) {
            for (unsigned int dil2 = d2BlockId; dil2 < d2BlockId + CPU_BLOCK_D2; dil2++) {
              unsigned int linInd_d1d2 = (dil1 % CPU_BLOCK_D1) * CPU_BLOCK_D2 + (dil2 % CPU_BLOCK_D2);
              auto q1ptr = q1 + dil1 * nBlockX + xBlockId;
              auto q2ptr = q2 + dil2 * nBlockX + xBlockId;
              for (int iX = 0; iX < CPU_BLOCK_X; iX++){
                diq[linInd_d1d2][0][iX]  = q1ptr->c1[iX] * q2ptr->c2[iX];
                diq[linInd_d1d2][0][iX] -= q1ptr->c2[iX] * q2ptr->c1[iX];

                diq[linInd_d1d2][1][iX]  = q1ptr->c2[iX] * q2ptr->c0[iX];
                diq[linInd_d1d2][1][iX] -= q1ptr->c0[iX] * q2ptr->c2[iX];

                diq[linInd_d1d2][2][iX]  = q1ptr->c0[iX] * q2ptr->c1[iX];
                diq[linInd_d1d2][2][iX] -= q1ptr->c1[iX] * q2ptr->c0[iX];
              }
            }
          }
          for (unsigned int d3BlockId = 0; d3BlockId < n3; d3BlockId += CPU_BLOCK_D3){
            for (unsigned int dil3 = d3BlockId; dil3 < d3BlockId + CPU_BLOCK_D3 ; dil3++) {
              for (unsigned int linInd_d1d2 = 0; linInd_d1d2 < CPU_BLOCK_D1*CPU_BLOCK_D2; linInd_d1d2++){
              auto q3ptr = q3 + dil3 * nBlockX + xBlockId;
                for (int iX = 0; iX < CPU_BLOCK_X; iX++){
                  singlet[dil3 % CPU_BLOCK_D3][linInd_d1d2][iX]  = diq[linInd_d1d2][0][iX] * q3ptr->c0[iX];
                  singlet[dil3 % CPU_BLOCK_D3][linInd_d1d2][iX] += diq[linInd_d1d2][1][iX] * q3ptr->c1[iX];
                  singlet[dil3 % CPU_BLOCK_D3][linInd_d1d2][iX] += diq[linInd_d1d2][2][iX] * q3ptr->c2[iX];
                }
              }
            }
            jit_gemm(jitter, &singlet[0][0][0], ptrMomBuf + xBlockId * CPU_BLOCK_X, &retTmp[d3BlockId][0][0]);
          }
        }
        // Unpack the retTmp, retArr += retTmp
        cmplx *resP;
        for (int iMom = 0; iMom < nMom; iMom++)
          for (unsigned int dil1 = 0; dil1 < CPU_BLOCK_D1; dil1++) 
            for (unsigned int dil2 = 0; dil2 < CPU_BLOCK_D2; dil2++)
              for (unsigned int dil3 = 0; dil3 < n3; dil3++) 
                retArr(iMom, dil1 + d1BlockId, dil2 + d2BlockId, dil3) = retTmp[dil3][dil1 * CPU_BLOCK_D2 + dil2][iMom];

      }
    }
  }
#ifdef ADVISOR
    __itt_pause();
#endif
#ifdef OMP
  timeComp = omp_get_wtime() - time_start;
#endif
  delete[] q1;
  delete[] q2;
  delete[] q3;

  return retArr;
}


std::vector<std::pair<std::string, double>> BaryonKernelCpuOpt::getTimings() const {
  return {
    {"reconstruct fields", timeReconst},
      {"parallel comp time", timeComp}
  };
}
