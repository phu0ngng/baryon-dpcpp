#include <vector>
#include <set>
#include <functional>
#include "misc.h"
#include "qdp_multi.h"

// For CPU OPT Kernel
#define MKL_Complex16 cmplx
#define MKL_Complex8 cmplx
#include "mkl.h"

#include <CL/sycl.hpp>
#include "oneapi/mkl.hpp"

#ifdef ADVISOR
#include <ittnotify.h>
#endif

// Parameters for the CPU OPT kernel
#ifndef CPU_BLOCK_X
#define CPU_BLOCK_X 64
#define CPU_BLOCK_D1 4
#define CPU_BLOCK_D2 4
#define CPU_BLOCK_D3 8
#endif

// Parameters for the DPC kernel
#ifndef DPC_BLOCK_D1
#define DPC_BLOCK_D1 2
#define DPC_BLOCK_D2 2
#define DPC_BLOCK_D3 16
#endif

#ifndef REPEAT
#define REPEAT 1
#endif

template <unsigned int BlockSize>
struct ColorVectorChunk{
  cmplx c0[BlockSize];
  cmplx c1[BlockSize];
  cmplx c2[BlockSize];
};


template<unsigned int BLOCK_D>
struct VectorChunk {
  cmplx d[BLOCK_D];
};

using ColorVectorChunkCpu = ColorVectorChunk<CPU_BLOCK_X>;
using ColorVectorChunkDpcD1 = ColorVectorChunk<DPC_BLOCK_D1>;
using ColorVectorChunkDpcD2 = ColorVectorChunk<DPC_BLOCK_D2>;
using ColorVectorChunkDpcD3 = ColorVectorChunk<DPC_BLOCK_D3>;

using VectorChunkDpcD1 = VectorChunk<DPC_BLOCK_D1>;
using VectorChunkDpcD2 = VectorChunk<DPC_BLOCK_D2>;
using VectorChunkDpcD3 = VectorChunk<DPC_BLOCK_D3>;


struct BaryonKernelCpuOpt {

  // physics parameters
  int nEv, nMom;

  // data buffers
  cmplx* momBuf;
  ColorVectorChunkCpu* evBuf;

  double timeComp, timeReconst;


  BaryonKernelCpuOpt();

  ~BaryonKernelCpuOpt();

  void resetTimers();

  void setMomentumSet(const std::set<Momentum>& _momSet);

  void readEV(int _nEv,
      const std::function<const LatticeColorVector&(int)>& readFunc);

  multi4d<cmplx> apply(const multi2d<cmplx>& coeffs1,
      const multi2d<cmplx>& coeffs2,
      const multi2d<cmplx>& coeffs3);

  std::vector<std::pair<std::string, double>> getTimings() const;
#ifdef DP
  zgemm_jit_kernel_t jit_gemm;
#else
  cgemm_jit_kernel_t jit_gemm;
#endif
  void* jitter;

};


struct BaryonKernelCpuRef {

  // tunable parameters
  unsigned int blockSizeMomProj;

  // physics parameters
  int nEv, nMom;

  // data buffers
  cmplx* momBuf;
  ColorVector* evBuf;

  double timeReconst, timeComp;


  BaryonKernelCpuRef(unsigned int blockSizeMomProj);

  ~BaryonKernelCpuRef();

  void resetTimers();

  void setMomentumSet(const std::set<Momentum>& _momSet);

  void readEV(int _nEv,
      const std::function<const LatticeColorVector&(int)>& readFunc);

  multi4d<cmplx> apply(const multi2d<cmplx>& coeffs1,
      const multi2d<cmplx>& coeffs2,
      const multi2d<cmplx>& coeffs3);

  std::vector<std::pair<std::string, double>> getTimings() const;

  // computes momBuf[iMom, iX] * singlet[iRHS, iX] using a zgemm
  void evaluateMomentumSums(const cmplx* const cP, cmplx* resP, unsigned int nRHS, unsigned int ldc) {
    unsigned int nSites = Layout::sitesOnNode();

    cmplx alpha(1.0,0.0);
    cmplx beta(0.0,0.0);
#ifdef DP
    cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nMom, nRHS, nSites, &alpha, momBuf, nSites, cP, nSites, &beta, resP, ldc);
#else
    cblas_cgemm(CblasRowMajor, CblasNoTrans, CblasTrans, nMom, nRHS, nSites, &alpha, momBuf, nSites, cP, nSites, &beta, resP, ldc);
#endif
  }
};


struct BaryonKernelDpcVecD {

  // physics parameters
  int nEv, nMom;

  // data buffers
  cmplx* d_momBuf;
  cmplx* d_evBuf;

  double timeReconst, timeComp;

  BaryonKernelDpcVecD();

  ~BaryonKernelDpcVecD();

  void resetTimers();

  void setMomentumSet(const std::set<Momentum>& _momSet);

  void readEV(int _nEv,
      const std::function<const LatticeColorVector&(int)>& readFunc);

  multi4d <cmplx> apply(
      VectorChunkDpcD1* coeffs1,
      VectorChunkDpcD2* coeffs2,
      VectorChunkDpcD3* coeffs3,
      unsigned int n1, unsigned int n2, unsigned int n3);

  std::vector<std::pair<std::string, double>> getTimings() const;

  // DPCPP
  sycl::queue q;
};

struct BaryonKernelDpcVecX {

    // physics parameters
    int nEv, nMom;

    // data buffers
    cmplx* d_momBuf;
    cmplx* d_evBuf;

    double timeReconst, timeComp;

    BaryonKernelDpcVecX();

    ~BaryonKernelDpcVecX();

    void resetTimers();

    void setMomentumSet(const std::set<Momentum>& _momSet);

    void readEV(int _nEv,
            const std::function<const LatticeColorVector&(int)>& readFunc);

    multi4d<cmplx> apply(const multi2d<cmplx>& coeffs1,
            const multi2d<cmplx>& coeffs2,
            const multi2d<cmplx>& coeffs3);

    std::vector<std::pair<std::string, double>> getTimings() const;

    // DPCPP
    sycl::queue q;
};
