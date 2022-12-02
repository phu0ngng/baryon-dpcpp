#include <iostream>
#include <vector>

#include "kernels_qdp3.h"
#include "swatch.h"
#include <stdlib.h>
#include <CL/sycl.hpp>

#ifndef REPEAT
#define REPEAT 1
#endif

int main(int argc, char* argv[]) {

  // physics parameters
  unsigned int spatL = 16;
  unsigned int nEv = 32;
  unsigned int nDil = 32;

  unsigned int dimx = 64;
  unsigned int dimd1 = 16;
  unsigned int dimd2 = 16;
  unsigned int dimd3 = 16;

  // tunable parameter in BaryonConstrRef
  unsigned int momProjBlockSize = 128;

  const std::set<Momentum> momSet = {
    std::make_tuple(0,0,0),
    std::make_tuple(0,0,1),
    std::make_tuple(0,0,-1),
    std::make_tuple(0,1,0),
    std::make_tuple(0,-1,0),
    std::make_tuple(1,0,0),
    std::make_tuple(-1,0,0),
    std::make_tuple(0,1,1),

    std::make_tuple(0,1,-1),
    std::make_tuple(0,-1,1),
    std::make_tuple(0,-1,-1),
    std::make_tuple(1,0,1),
    std::make_tuple(1,0,-1),
    std::make_tuple(-1,0,1),
    std::make_tuple(-1,0,-1),
    std::make_tuple(1,1,0)

      ,std::make_tuple(1,-1,0),
    std::make_tuple(-1,1,0),
    std::make_tuple(-1,-1,0),
    std::make_tuple(1,1,1),
    std::make_tuple(1,1,-1),
    std::make_tuple(1,-1,1),
    std::make_tuple(-1,1,1),
    std::make_tuple(-1,-1,1),

    std::make_tuple(-1,1,-1),
    std::make_tuple(1,-1,-1),
    std::make_tuple(-1,-1,-1),
    std::make_tuple(0,0,2),
    std::make_tuple(0,0,-2),
    std::make_tuple(0,2,0),
    std::make_tuple(0,-2,0),
    std::make_tuple(2,0,0)

      ,std::make_tuple(-2,0,0)
  };

  const unsigned int nMom = momSet.size();

  // initialize some lattice problem size
  Layout::init(std::make_tuple(spatL, spatL, spatL));

  // set up test eigenvectors
  Timer swatch;
  std::vector<LatticeColorVector> evList;
  evList.reserve(nEv);
  for (unsigned int iEv=0; iEv<nEv; ++iEv) {
    //init_col_vec(evList[iEv]);
    evList.emplace_back();
    init_col_vec(evList.back());
  }
  std::cout << "init colVec time = "<<swatch.time()<<"\n";

  std::cout << "\n";

  double applyTime, sumTime;
  std::vector<std::pair<std::string, double>> timeDetail;

#if defined(KERNEL_CPU_OPT) || defined(KERNEL_CPU_REF)
  multi2d<cmplx> coeff1(nDil, nEv);
  multi2d<cmplx> coeff2(nDil, nEv);
  multi2d<cmplx> coeff3(nDil, nEv);

  init_coeff(coeff1);
  init_coeff(coeff2);
  init_coeff(coeff3);
#endif 

#ifdef KERNEL_CPU_REF
  swatch.reset();
  BaryonKernelCpuRef kernelCpuRef(momProjBlockSize);
  kernelCpuRef.setMomentumSet(momSet);
  kernelCpuRef.readEV(nEv, [&evList](int iEv) -> const LatticeColorVector& { return evList[iEv]; });
  std::cout << "BaryonKernelCpuRef setup time = "<<swatch.time()<<"\n";

  swatch.reset();
  multi4d<cmplx> resultCpuRef(nMom, nDil, nDil, nDil);
  resultCpuRef = kernelCpuRef.apply(coeff1, coeff2, coeff3);
  applyTime = swatch.time(); 
  std::cout << "BaryonConstr Time (warm up) = "<< applyTime <<"\n";

  sumTime = 0.0;
  for (int i = 1; i < REPEAT; i++)
  {
    kernelCpuRef.resetTimers();
    resultCpuRef = kernelCpuRef.apply(coeff1, coeff2, coeff3);
    auto timeDetail = kernelCpuRef.getTimings()[1].second;
    std::cout << "BaryonConstr Time = "<< timeDetail <<"\n";
    sumTime += timeDetail;
  }
  std::cout << "Average Time = " << sumTime/(REPEAT-1) << "\n" << std::endl;
#endif

#ifdef KERNEL_CPU_OPT
  swatch.reset();
  BaryonKernelCpuOpt kernelCpuOpt;
  kernelCpuOpt.setMomentumSet(momSet);

  kernelCpuOpt.readEV(nEv, [&evList](int iEv) -> const LatticeColorVector& { return evList[iEv]; });
  std::cout << "BaryonKernelCpuOpt setup time = "<<swatch.time()<<"\n";

  swatch.reset();
  multi4d<cmplx> resultCpuOpt(nMom, nDil, nDil, nDil);
  resultCpuOpt = kernelCpuOpt.apply(coeff1, coeff2, coeff3);
  applyTime = swatch.time(); 
  std::cout << "BaryonConstr Time (warm up) = "<< applyTime <<"\n";

  sumTime = 0.0;
  for (int i = 1; i < REPEAT; i++)
  {
    kernelCpuOpt.resetTimers();
    resultCpuOpt = kernelCpuOpt.apply(coeff1, coeff2, coeff3);
    auto timeDetail = kernelCpuOpt.getTimings()[1].second;
    std::cout << "BaryonConstr Time = "<< timeDetail <<"\n";
    sumTime += timeDetail;
  }
  std::cout << "Average Time = " << sumTime/(REPEAT-1) << "\n" << std::endl;
#endif

#if defined(RESULT_CHECK) && defined(KERNEL_CPU_OPT) && defined(KERNEL_CPU_REF)
  std::cout << "Result check: ";
  if (check_results(resultCpuOpt, resultCpuRef)) std::cout << "PASS" << std::endl;
  else std::cout << "FAIL" << std::endl;
#endif

#ifdef KERNEL_DPC
  unsigned int nBlockD1 = nDil / DPC_BLOCK_D1;
  unsigned int nBlockD2 = nDil / DPC_BLOCK_D2;
  unsigned int nBlockD3 = nDil / DPC_BLOCK_D3;

  auto coeff1_dpc = new VectorChunkDpcD1[nBlockD1 * nEv];
  auto coeff2_dpc = new VectorChunkDpcD2[nBlockD2 * nEv];
  auto coeff3_dpc = new VectorChunkDpcD3[nBlockD3 * nEv];

  init_coeff_dpc<VectorChunkDpcD1>(coeff1_dpc, nEv, nBlockD1, DPC_BLOCK_D1);
  init_coeff_dpc<VectorChunkDpcD2>(coeff2_dpc, nEv, nBlockD2, DPC_BLOCK_D2);
  init_coeff_dpc<VectorChunkDpcD3>(coeff3_dpc, nEv, nBlockD3, DPC_BLOCK_D3);

  swatch.reset();
  BaryonKernelDPC kernelDPC;
  kernelDPC.setMomentumSet(momSet);

  kernelDPC.readEV(nEv, [&evList](int iEv) -> const LatticeColorVector& { return evList[iEv]; });
  std::cout << "BaryonKernelDPC++ setup time = "<<swatch.time()<<"\n";

  swatch.reset();
  multi4d<cmplx> resultDPC(nMom, nDil, nDil, nDil);
  resultDPC = kernelDPC.apply(coeff1_dpc, coeff2_dpc, coeff3_dpc, nDil, nDil, nDil);
  applyTime = swatch.time(); 
  std::cout << "BaryonConstr Time (warm up) = "<< applyTime <<"\n" << std::endl;
  sumTime = 0.0;
  for (int i = 1; i < REPEAT; i++)
  {
    kernelDPC.resetTimers();
    resultDPC = kernelDPC.apply(coeff1_dpc, coeff2_dpc, coeff3_dpc, nDil, nDil, nDil);
    auto timeDetail = kernelDPC.getTimings()[1].second;
    std::cout << "BaryonConstr Time = "<< timeDetail <<"\n";
    sumTime += timeDetail;
  }
  std::cout << "Average Time = " << sumTime/(REPEAT-1) << "\n" << std::endl;

  delete[] coeff1_dpc;
  delete[] coeff2_dpc;
  delete[] coeff3_dpc;
#endif

#if defined(RESULT_CHECK) && defined(KERNEL_DPC) && defined(KERNEL_CPU_REF)
  std::cout << "Result check: ";
  if (check_results(resultDPC, resultCpuRef)) std::cout << "PASS \n \n" << std::endl;
  else std::cout << "FAIL \n \n" << std::endl;
#endif

}
