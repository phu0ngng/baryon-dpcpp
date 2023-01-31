#include <algorithm>
#include <iostream>
#include "kernels_qdp3.h"
#include "swatch.h"

#include <exception>

BaryonKernelDpcVecX::BaryonKernelDpcVecX()
  : nMom(0), d_momBuf(nullptr), d_evBuf(nullptr){
    resetTimers();
    // DPCPP - Selecting device
    try{
      sycl::queue q{sycl::gpu_selector_v};
    }
    catch (sycl::exception const &e) {
      std::cout << "NO requested device is FOUND\n";
      std::terminate();
    }
    // Check if the queue is still working
    std::cout << "Device: " << q.get_device().get_info< sycl::info::device::name >() << "\n";
    auto maxWorkGroupSize = q.get_device().get_info< sycl::info::device::max_work_group_size >();
  }

BaryonKernelDpcVecX::~BaryonKernelDpcVecX() {
  sycl::free(d_evBuf, q);
  sycl::free(d_momBuf, q);
}


void BaryonKernelDpcVecX::resetTimers() {
  timeReconst = 0.;
  timeComp = 0.;
}

void BaryonKernelDpcVecX::setMomentumSet(const std::set<Momentum>& momSet) {
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
  q.wait();
  delete[] momBuf;
}

void BaryonKernelDpcVecX::readEV(int _nEv,
    const std::function<const LatticeColorVector&(int)>& readFunc) {
  unsigned int nSites = Layout::sitesOnNode();

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

  for (int iEv=0; iEv<nEv; ++iEv) {
	  LatticeColorVector aRef(&evBuf[iEv*nSites], 0.);
	  aRef = readFunc(iEv);
  }
  q.memcpy(d_evBuf, &evBuf[0].c0 , nEv * nSites * 3 * sizeof(cmplx));
  delete[] evBuf;
}

multi4d<cmplx> BaryonKernelDpcVecX::apply(
    const multi2d<cmplx>& coeffs1,
    const multi2d<cmplx>& coeffs2,
    const multi2d<cmplx>& coeffs3) {

  size_t nSites = Layout::sitesOnNode();
  size_t n1 = coeffs1.nrows();
  size_t n2 = coeffs2.nrows();
  size_t n3 = coeffs3.nrows();

  resetTimers();
  Timer bulova;

  auto d_coeffs1 = sycl::malloc_device<cmplx>(n1 * nEv, q);
  auto d_coeffs2 = sycl::malloc_device<cmplx>(n2 * nEv, q);
  auto d_coeffs3 = sycl::malloc_device<cmplx>(n3 * nEv, q);

  auto d_q1 = sycl::malloc_device<ColorVector>(n1 * nSites, q);
  auto d_q2 = sycl::malloc_device<ColorVector>(n2 * nSites, q);
  auto d_q3 = sycl::malloc_device<ColorVector>(n3 * nSites, q);

  q.memcpy(d_coeffs1, &coeffs1(0,0), n1 * nEv * sizeof(cmplx));
  q.memcpy(d_coeffs2, &coeffs2(0,0), n2 * nEv * sizeof(cmplx));
  q.memcpy(d_coeffs3, &coeffs3(0,0), n3 * nEv * sizeof(cmplx));
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
  auto transA = oneapi::mkl::transpose::nontrans;
  auto transB = oneapi::mkl::transpose::nontrans;
  using oneapi::mkl::blas::row_major::gemm;
  gemm(q, transA, transB, n1, 3 * nSites, nEv, alpha, d_coeffs1, nEv, d_evBuf, 3 * nSites, beta, &d_q1[0].c0, 3 * nSites);
  gemm(q, transA, transB, n2, 3 * nSites, nEv, alpha, d_coeffs2, nEv, d_evBuf, 3 * nSites, beta, &d_q2[0].c0, 3 * nSites);
  gemm(q, transA, transB, n3, 3 * nSites, nEv, alpha, d_coeffs3, nEv, d_evBuf, 3 * nSites, beta, &d_q3[0].c0, 3 * nSites);
  q.wait();
  timeReconst = bulova.time();
  bulova.reset();

  size_t nM = nMom; // TODO: having nMom compiler time known

  multi4d<cmplx> retArr(nMom,n1,n2,n3);
  auto d_retArr   = sycl::malloc_device<cmplx>(nMom * n1 * n2 * n3, q);

  const size_t BLOCK_X = 32;
  const size_t BLOCK_D = 32;

  sycl::range<3> global{n1 * n2, n3, BLOCK_X};
  sycl::range<3> local{1, BLOCK_D, BLOCK_X};

  bulova.reset();
  q.submit([&](auto &h){
      //sycl::local_accessor<cmplx, 2> la_retArr{sycl::range{nM, BLOCK_D}, h};
      auto d_momBuf = this-> d_momBuf;
    h.template parallel_for<class Baryon>(sycl::nd_range<3>{global, local}, [=] (sycl::nd_item<3> it)
        [[intel::reqd_sub_group_size(BLOCK_X)]]
        {
        sycl::group<3>  g  = it.get_group();
        auto sg = it.get_sub_group();

        size_t global_d1 = it.get_global_id(0) / n2; // global id with the kernel's execution range
        size_t global_d2 = it.get_global_id(0) % n2;
        size_t global_d3 = it.get_global_id(1);
        size_t g_linearId = it.get_global_linear_id()/BLOCK_X;

        // Each workgroup work on BLOCK_D of d3, each sub_group is responsible for 1 d3
        size_t local_x = it.get_local_id(2);
        size_t local_d3 = it.get_local_id(1);

        auto nMom_per_item = nM / BLOCK_X;
        cmplx tmp[1];

        for (int i = 0; i < nMom_per_item; i++)
          tmp[i] = 0.0;

        cmplx q1c0, q1c1, q1c2, q2c0, q2c1, q2c2, q3c0, q3c1, q3c2;
        cmplx diq0, diq1, diq2, singlet;

      for (int xBlock = 0; xBlock < nSites; xBlock += BLOCK_X){
        // Each thread calculates each all diq, no broadcast
        q1c0 = d_q1[global_d1 * nSites + xBlock + local_x].c0;
        q1c1 = d_q1[global_d1 * nSites + xBlock + local_x].c1;
        q1c2 = d_q1[global_d1 * nSites + xBlock + local_x].c2;
        q2c0 = d_q2[global_d2 * nSites + xBlock + local_x].c0;
        q2c1 = d_q2[global_d2 * nSites + xBlock + local_x].c1;
        q2c2 = d_q2[global_d2 * nSites + xBlock + local_x].c2;

        diq0  = q1c1 * q2c2;
        diq0 -= q1c2 * q2c1;

        diq1  = q1c2 * q2c0;
        diq1 -= q1c0 * q2c2;

        diq2  = q1c0 * q2c1;
        diq2 -= q1c1 * q2c0;

        // ColorVectorContract
        q3c0 = d_q3[global_d3 * nSites + xBlock + local_x].c0;
        q3c1 = d_q3[global_d3 * nSites + xBlock + local_x].c1;
        q3c2 = d_q3[global_d3 * nSites + xBlock + local_x].c2;

        singlet  = diq0 * q3c0; 
        singlet += diq1 * q3c1;
        singlet += diq2 * q3c2;

        // GEMM : retArr = momBuf * singlet
        cmplx temp_sum;
        for (int iM = 0; iM < nM; iM++)
        {
          temp_sum = d_momBuf[iM * nSites + xBlock + local_x] * singlet;
          temp_sum =  sycl::reduce_over_group(sg, temp_sum, sycl::plus<>());

          if (local_x == (iM % BLOCK_X)) tmp[static_cast<int>(iM / BLOCK_X)] += temp_sum;
        }
      }

      for (int iM = 0; iM < nM; iM+= BLOCK_X)
      {
        d_retArr[(iM + local_x) * n1* n2 * n3 + g_linearId] = tmp[iM % BLOCK_X];
      }
      sg.barrier();
      //sycl::group_barrier(g);

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
  return retArr;
}


std::vector<std::pair<std::string, double>> BaryonKernelDpcVecX::getTimings() const {
  return {
    {"reconstruct fields", timeReconst},
      {"computation", timeComp}
  };
}
