#include <algorithm>
#include <iostream>
#include "kernels_qdp3.h"
#include "swatch.h"

#include <exception>

BaryonKernelDPC::BaryonKernelDPC()
  : nMom(0), d_momBuf(nullptr), d_evBuf(nullptr){ 

    resetTimers(); 

    // DPCPP - Selecting device
    try{
      sycl::queue q{sycl::gpu_selector{}, {sycl::property::queue::in_order(), sycl::property::queue::enable_profiling()}};
    }
    catch (sycl::exception const &e) {
      std::cout << "NO requested device is FOUND\n";
      std::terminate();
    }
  // Check if the queue is still working
  std::cout << "Device: " << q.get_device().get_info< sycl::info::device::name >() << "\n";
  auto maxWorkGroupSize = q.get_device().get_info< sycl::info::device::max_work_group_size >();

  std::cout << "Device Info: " << std::endl;
  std::cout << "--- Max Workgroup Size: " << maxWorkGroupSize << "\n";

  auto subgroup_sizes = q.get_device().get_info< sycl::info::device::sub_group_sizes >();
  std::cout << "--- Possible Subgroup Sizes: " ;
  for (auto item: subgroup_sizes)
    std::cout << item << " ";
  std::cout << std::endl;
 // auto maxSubGroupSize = subgroup_sizes.back();

  auto local_mem_size = q.get_device().get_info< sycl::info::device::local_mem_size >();
  std::cout << "--- Local Memory Size: " << local_mem_size << "\n";
  
  }

BaryonKernelDPC::~BaryonKernelDPC() {
  sycl::free(d_evBuf, q);
  sycl::free(d_momBuf, q);
}


void BaryonKernelDPC::resetTimers() {
  timeReconst = 0.;
  timeComp = 0.;
}

void BaryonKernelDPC::setMomentumSet(const std::set<Momentum>& momSet) {
  // regenerate momentum fields if necessary
  unsigned int nSites = Layout::sitesOnNode();

  if (momSet.size() != size_t(nMom)) {
    nMom = momSet.size(); 
    sycl::free(d_momBuf, q);
    d_momBuf  = sycl::malloc_device<cmplx>(nMom * nSites, q);
  }
  auto momBuf = new cmplx[nMom*nSites];
  int iMom = 0;
  for (const auto& aM : momSet) {
    LatticeComplex aC(&momBuf[iMom*nSites], 0.);
    makeMomPhaseField(aM, aC);
    iMom++;
  }

  q.memcpy(d_momBuf,  &momBuf[0]   , nMom * nSites * sizeof(cmplx));
  delete[] momBuf;
}


void BaryonKernelDPC::readEV(int _nEv,
    const std::function<const LatticeColorVector&(int)>& readFunc) {
  unsigned int nSites = Layout::sitesOnNode();

  // reserve buffer for eigenvectors
  if (d_evBuf == nullptr) {
    nEv = _nEv;
    d_evBuf   = sycl::malloc_device<cmplx>(nEv * nSites * 3, q);
  }

  if (nEv != _nEv) {
    std::cerr << "readEV: Number of eigenvectors "
      << _nEv
      << " doesn't match buffer size " << nEv << std::endl;
    exit(1);
  }
  auto evBuf = new ColorVector[nEv * nSites];

//#pragma ivdep
  for (int iEv=0; iEv<nEv; ++iEv) {
    LatticeColorVector aRef(&evBuf[iEv*nSites], 0.);
    aRef = readFunc(iEv);
  }
  q.memcpy(d_evBuf, &evBuf[0].c0 , nEv * nSites * 3 * sizeof(cmplx));
  q.wait();
  delete[] evBuf;
}

multi4d <cmplx> BaryonKernelDPC::apply(
    VectorChunkDpcD1* coeffs1,
    VectorChunkDpcD2* coeffs2,
    VectorChunkDpcD3* coeffs3, 
    unsigned int n1, unsigned int n2, unsigned int n3){

  unsigned int nSites = Layout::sitesOnNode();
  auto nBlockD1 = n1 / DPC_BLOCK_D1;
  auto nBlockD2 = n2 / DPC_BLOCK_D2;
  auto nBlockD3 = n3 / DPC_BLOCK_D3;

  resetTimers(); 
  Timer bulova;

  auto d_coeffs1 = sycl::malloc_device<VectorChunkDpcD1>(nBlockD1 * nEv, q);
  auto d_coeffs2 = sycl::malloc_device<VectorChunkDpcD2>(nBlockD2 * nEv, q);
  auto d_coeffs3 = sycl::malloc_device<VectorChunkDpcD3>(nBlockD3 * nEv, q);

  auto d_q1 = sycl::malloc_device<ColorVectorChunkDpcD1>(nSites * nBlockD1, q);
  auto d_q2 = sycl::malloc_device<ColorVectorChunkDpcD2>(nSites * nBlockD2, q);
  auto d_q3 = sycl::malloc_device<ColorVectorChunkDpcD3>(nSites * nBlockD3, q);

  q.memcpy(&d_coeffs1[0].d[0], &coeffs1[0].d[0], n1 * nEv * sizeof(cmplx));
  q.memcpy(&d_coeffs2[0].d[0], &coeffs2[0].d[0], n2 * nEv * sizeof(cmplx));
  q.memcpy(&d_coeffs3[0].d[0], &coeffs3[0].d[0], n3 * nEv * sizeof(cmplx));
  q.wait();

  cmplx alpha(1.0,0.0);
  cmplx beta(0.0,0.0); // P : Just calculate C = A*B

  /* ***********************************************************************************
   * Eq 3: Calculate q1, q2, q3 from sum_l(coeffsx*ev.Buf), q1 is stored in q1Buf
   *
   *        q1p             = coeff1P * evP
   * or     q1Buf[0].c0     = coeffs1[0,0] * evBuf[0].c0
   * size   n1 x (3*nSites)   n1 x nEv      nEv x (3*nSites) => n1 is number of d1 here
   */
  bulova.reset();
  auto transA = oneapi::mkl::transpose::trans;
  auto transB = oneapi::mkl::transpose::nontrans;
  using oneapi::mkl::blas::row_major::gemm;
  for (unsigned int iBlockD = 0; iBlockD < nBlockD1; iBlockD++)
      gemm(q, transA, transB, nSites * 3, DPC_BLOCK_D1,  nEv, alpha, d_evBuf, 3 * nSites, &d_coeffs1[iBlockD * nEv].d[0], DPC_BLOCK_D1, beta, &d_q1[iBlockD * nSites].c0[0], DPC_BLOCK_D1);
  for (unsigned int iBlockD = 0; iBlockD < nBlockD2; iBlockD++)
      gemm(q, transA, transB, nSites * 3, DPC_BLOCK_D2,  nEv, alpha, d_evBuf, 3 * nSites, &d_coeffs2[iBlockD * nEv].d[0], DPC_BLOCK_D2, beta, &d_q2[iBlockD * nSites].c0[0], DPC_BLOCK_D2);
  for (unsigned int iBlockD = 0; iBlockD < nBlockD3; iBlockD++)
      gemm(q, transA, transB, nSites * 3, DPC_BLOCK_D3,  nEv, alpha, d_evBuf, 3 * nSites, &d_coeffs3[iBlockD * nEv].d[0], DPC_BLOCK_D3, beta, &d_q3[iBlockD * nSites].c0[0], DPC_BLOCK_D3);

  q.wait();
  timeReconst = bulova.time();

  multi4d<cmplx> retArr(nMom,n1,n2,n3);
  auto d_retArr = sycl::malloc_device<cmplx>(nMom * n1 * n2 * n3, q);

  sycl::range<3> global{n1, n2, n3};
  sycl::range<3> local{DPC_BLOCK_D1, DPC_BLOCK_D2, DPC_BLOCK_D3}; 

  q.wait();
  bulova.reset();
  auto ev = q.submit([&](auto &h){
    constexpr unsigned int nMom = 33; // TODO: having nMom compiler time known
    auto d_momBuf = this->d_momBuf;

    h.template parallel_for<class Baryon>(sycl::nd_range<3>{global, local}, [=] (sycl::nd_item<3> it)
        [[intel::reqd_sub_group_size(DPC_BLOCK_D3)]]
        {

      unsigned int local_d1 = it.get_local_id(0); 
      unsigned int local_d2 = it.get_local_id(1);
      unsigned int local_d3 = it.get_local_id(2);

      unsigned int iBlockD1 = it.get_global_id(0)/DPC_BLOCK_D1;
      unsigned int iBlockD2 = it.get_global_id(1)/DPC_BLOCK_D2;
      unsigned int iBlockD3 = it.get_global_id(2)/DPC_BLOCK_D3;

      unsigned int global_linear_d = it.get_global_linear_id();

      cmplx tmpRes[nMom];
        for (unsigned int iM = 0; iM < nMom; ++iM)
          tmpRes[iM] = 0.;

        cmplx q1c0, q1c1, q1c2, q2c0, q2c1, q2c2, q3c0, q3c1, q3c2;
        cmplx diq0, diq1, diq2, singlet;

        auto q1ptr = d_q1 + iBlockD1 * nSites;
        auto q2ptr = d_q2 + iBlockD2 * nSites;
        auto q3ptr = d_q3 + iBlockD3 * nSites;
        for (unsigned int iX = 0; iX < nSites; ++iX){
          q1c0 = q1ptr->c0[local_d1];
          q1c1 = q1ptr->c1[local_d1];
          q1c2 = q1ptr->c2[local_d1];

          q2c0 = q2ptr->c0[local_d2];
          q2c1 = q2ptr->c1[local_d2];
          q2c2 = q2ptr->c2[local_d2];

          q3c0 = q3ptr->c0[local_d3];
          q3c1 = q3ptr->c1[local_d3];
          q3c2 = q3ptr->c2[local_d3];

          diq0  = q1c1 * q2c2 ;
          diq0 -= q1c2 * q2c1 ;

          diq1  = q1c2 * q2c0 ;
          diq1 -= q1c0 * q2c2 ;

          diq2  = q1c0 * q2c1 ;
          diq2 -= q1c1 * q2c0 ;

          singlet  = diq0 * q3c0;
          singlet += diq1 * q3c1;
          singlet += diq2 * q3c2;

          // GEMM : retArr = momBuf * singlet
          for (unsigned int iM = 0; iM < nMom; ++iM)
            tmpRes[iM] += d_momBuf[iM * nSites + iX] * singlet;

          q1ptr++; q2ptr++; q3ptr++;
        }
        for (unsigned int iM = 0; iM < nMom; ++iM)
          d_retArr[global_linear_d + iM * n1 *n2 * n3] = tmpRes[iM];
    });
  });
  q.wait();
  timeComp = bulova.time();

  q.memcpy(&retArr(0,0,0,0), d_retArr, nMom * n1 * n2 * n3 * sizeof(cmplx));

  sycl::free(d_q1, q);
  sycl::free(d_q2, q);
  sycl::free(d_q3, q);
  sycl::free(d_retArr, q);
  sycl::free(d_coeffs1, q);
  sycl::free(d_coeffs2, q);
  sycl::free(d_coeffs3, q);
  q.wait();

  return retArr;
}

std::vector<std::pair<std::string, double>> BaryonKernelDPC::getTimings() const {
  return {
    {"reconstruct fields", timeReconst},
      {"computation", timeComp}
  };
}
