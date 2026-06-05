/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: tests/Test_chromo_magnetic_operator_field.cc

Copyright (C) 2015-2025

Author: Masaaki Tomii <masaaki.tomii@uconn.edu> (original Hadrons kernels)

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License along
with this program; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

See the full license in the file "LICENSE" in the top level distribution directory
*************************************************************************************/
#include "disable_tests_without_instantiations.h"
#ifdef ENABLE_FERMION_INSTANTIATIONS

#include <Grid/Grid.h>
#include <Grid/qcd/utils/A2Autils.h>

using namespace Grid;

typedef WilsonImplD FImpl;
typedef typename FImpl::FermionField FermionField;
typedef typename FImpl::SiteSpinor   vobj;
typedef typename vobj::scalar_type   scalar_type;
typedef typename vobj::vector_type   vector_type;
typedef iSpinColourMatrix<vector_type> SpinColourMatrix_v;
typedef iSpinColourVector<vector_type> SpinColourVector_v;
typedef iColourMatrix<vector_type>     ColourMatrix_v;
typedef iSpinMatrix<vector_type>       SpinMatrix_v;
typedef iSinglet<vector_type>          Scalar_v;
typedef iSinglet<scalar_type>          Scalar_s;
typedef Lattice<SpinColourMatrix_v>    PropagatorField;
typedef LatticeColourMatrix            GaugeMat;
typedef LatticeGaugeField              GaugeField;

// CPU reference, ported from Hadrons/Modules/MContraction/A2AChromoMagneticOperatorField.hpp
// (M. Tomii). Hadrons infrastructure removed; ifOrthog and parity taken as direct parameters.
// use_blas=true replaces the scalar spatial sum with A2ASpatialSum.
class A2AChromoMagneticOperatorFieldRef
{
public:
  static void compute(
      Eigen::Tensor<ComplexD, 3> &result,
      const std::vector<FermionField> &left,
      const std::vector<FermionField> &right,
      const GaugeField &U,
      int ifOrthog,
      int parity,
      bool use_blas = false)
  {
    GridBase *grid = left[0].Grid();

    const int orthogdim = 3;
    int rd    = grid->_rdimensions[orthogdim];
    int ld    = grid->_ldimensions[orthogdim];
    int Nd    = grid->_ndimension;
    int Nsimd = grid->Nsimd();

    int nt  = result.dimension(0);
    int N_i = (int)left.size();
    int N_j = (int)right.size();

    std::string tag = std::string(use_blas ? "[blas" : "[ref ") + " ifOrthog=" + std::to_string(ifOrthog) + " parity=" + std::to_string(parity) + "]";
    auto Tms = [](double us) { return us * 1e-3; };
    double t0;

    // ------------------------------------------------------------------
    // Pack conjugated left vectors
    // ------------------------------------------------------------------
    t0 = usecond();
    std::vector<FermionField> leftv(N_i, grid);
    for (int i = 0; i < N_i; i++)
      leftv[i] = conjugate(left[i]);
    std::cout << GridLogMessage << tag << " pack_left:       " << Tms(usecond()-t0) << " ms\n";

    // ------------------------------------------------------------------
    // Build anti-Hermitian field strength tensors G[munu] and Sigma[munu]
    // ------------------------------------------------------------------
    t0 = usecond();
    int ng = 0;
    std::vector<GaugeMat> G;
    std::vector<Gamma::Algebra> Sigma;
    for (int mu = 0; mu < 3; mu++) {
      if ( mu == orthogdim && ifOrthog == 1 ) continue;
      GaugeMat Umu(U.Grid());
      Umu = peekLorentz(U, mu);
      for (int nu = mu+1; nu < 4; nu++) {
        if ( nu == orthogdim && ifOrthog == 1 ) continue;
        if ( ifOrthog == 0 ) {
          if ( mu != orthogdim && nu != orthogdim ) continue;
        }
        GaugeMat Unu(U.Grid());
        Unu = peekLorentz(U, nu);
        GaugeMat Gmunu(U.Grid());
        WilsonLoops<PeriodicGimplD>::CloverleafMxN(Gmunu, Umu, Unu, mu, nu, 1, 1);
        G.push_back(Gmunu - adj(Gmunu));
        if      (mu == 0 && nu == 1) Sigma.push_back(Gamma::Algebra::SigmaXY);
        else if (mu == 0 && nu == 2) Sigma.push_back(Gamma::Algebra::SigmaXZ);
        else if (mu == 0 && nu == 3) Sigma.push_back(Gamma::Algebra::SigmaXT);
        else if (mu == 1 && nu == 2) Sigma.push_back(Gamma::Algebra::SigmaYZ);
        else if (mu == 1 && nu == 3) Sigma.push_back(Gamma::Algebra::SigmaYT);
        else if (mu == 2 && nu == 3) Sigma.push_back(Gamma::Algebra::SigmaZT);
      }
    }
    assert(Sigma.size() == 3);
    ng = (int)G.size();
    std::cout << GridLogMessage << tag << " field_strength:  " << Tms(usecond()-t0) << " ms\n";

    // ------------------------------------------------------------------
    // Pack modified right vectors:
    //   rightv[j] = sum_munu G[munu] * (Gamma5?) * Gamma(Sigma[munu]) * right[j]
    // All G views opened once; full munu sum done in a single thread_for per j.
    // ------------------------------------------------------------------
    t0 = usecond();
    std::vector<FermionField> rightv(N_j, grid);
    {
      using GView = decltype(G[0].View(CpuRead));
      std::vector<GView> gviews;
      gviews.reserve(ng);
      for (int munu = 0; munu < ng; ++munu)
        gviews.push_back(G[munu].View(CpuRead));

      for (int j = 0; j < N_j; j++) {
        rightv[j] = Zero();
        autoView(lRv, rightv[j], CpuWrite);
        autoView(rv,  right[j],  CpuRead);
        int lNg      = ng;
        int lParity  = parity;
        thread_for(ss, grid->oSites(), {
          SpinColourVector_v lR; lR = Zero();
          for (int munu = 0; munu < lNg; ++munu) {
            SpinColourVector_v tmp2 = Gamma(Sigma[munu]) * rv[ss];
            if (lParity == 1) tmp2 = Gamma(Gamma::Algebra::Gamma5) * tmp2;
            for (int s = 0; s < Ns; ++s)
            for (int c = 0; c < Nc; ++c)
              lR()(s)(c) = lR()(s)(c)
                + gviews[munu][ss]()()(c,0) * tmp2()(s)(0)
                + gviews[munu][ss]()()(c,1) * tmp2()(s)(1)
                + gviews[munu][ss]()()(c,2) * tmp2()(s)(2);
          }
          lRv[ss] = lR;
        });
      }
      for (int munu = 0; munu < ng; ++munu)
        gviews[munu].ViewClose();
    }
    std::cout << GridLogMessage << tag << " pack_right:      " << Tms(usecond()-t0) << " ms\n";

    if (use_blas) {
      // ------------------------------------------------------------------
      // BLAS path: A2ASpatialSum (Allocate + PackLeft + PackRight + Sum)
      // ------------------------------------------------------------------
      A2ASpatialSum<SpinColourVector_v> spatial_sum;
      double t_blas_start = usecond();

      t0 = usecond();
      spatial_sum.Allocate(N_i, N_j, grid);
      std::cout << GridLogMessage << tag << " Allocate:        " << Tms(usecond()-t0) << " ms\n";

      t0 = usecond();
      spatial_sum.PackLeft(leftv);
      std::cout << GridLogMessage << tag << " PackLeft:        " << Tms(usecond()-t0) << " ms\n";

      t0 = usecond();
      spatial_sum.PackRight(rightv);
      std::cout << GridLogMessage << tag << " PackRight:       " << Tms(usecond()-t0) << " ms\n";

      t0 = usecond();
      spatial_sum.Sum(result);
      std::cout << GridLogMessage << tag << " Sum (GEMM+MPI):  " << Tms(usecond()-t0) << " ms\n";

      std::cout << GridLogMessage << tag << " A2ASpatialSum:   " << Tms(usecond()-t_blas_start) << " ms  [TOTAL]\n";

    } else {
      // ------------------------------------------------------------------
      // Reference path: SIMD spatial sum + scalar extraction
      // ------------------------------------------------------------------
      int MFrvol = rd * N_i * N_j;
      int MFlvol = ld * N_i * N_j;

      Vector<Scalar_v> lvSum(MFrvol);
      thread_for(r, MFrvol, { lvSum[r] = Zero(); });

      t0 = usecond();
      {
        int e1     = grid->_slice_nblock[orthogdim];
        int e2     = grid->_slice_block [orthogdim];
        int stride = grid->_slice_stride[orthogdim];
        using LView = decltype(leftv[0].View(CpuRead));
        using RView = decltype(rightv[0].View(CpuRead));
        std::vector<LView> lv_views;
        std::vector<RView> rv_views;
        lv_views.reserve(N_i);
        rv_views.reserve(N_j);
        for (int i = 0; i < N_i; i++) lv_views.push_back(leftv[i].View(CpuRead));
        for (int j = 0; j < N_j; j++) rv_views.push_back(rightv[j].View(CpuRead));
        thread_for(r, rd, {
          int so   = r * grid->_ostride[orthogdim];
          int base = N_i * N_j * r;
          for (int n = 0; n < e1; n++)
          for (int b = 0; b < e2; b++) {
            int ss = so + n * stride + b;
            for (int ii = 0; ii < N_i; ii++)
            for (int jj = 0; jj < N_j; jj++) {
              int idx = jj + N_j * ii + base;
              for (int s = 0; s < Ns; ++s)
              for (int c = 0; c < Nc; ++c)
                lvSum[idx]()()() += lv_views[ii][ss]()(s)(c) * rv_views[jj][ss]()(s)(c);
            }
          }
        });
        for (auto &v : lv_views) v.ViewClose();
        for (auto &v : rv_views) v.ViewClose();
      }
      std::cout << GridLogMessage << tag << " spatial_sum:     " << Tms(usecond()-t0) << " ms\n";

      Vector<Scalar_s> lsSum(MFlvol);
      thread_for(r, MFlvol, { lsSum[r] = scalar_type(0.0); });

      t0 = usecond();
      thread_for(rt, rd, {
        Coordinate icoor(Nd);
        ExtractBuffer<Scalar_s> extracted(Nsimd);
        for (int ii = 0; ii < N_i; ii++)
        for (int jj = 0; jj < N_j; jj++) {
          int ij_rdx = jj + N_j * (ii + N_i * rt);
          extract(lvSum[ij_rdx], extracted);
          for (int idx = 0; idx < Nsimd; idx++) {
            grid->iCoorFromIindex(icoor, idx);
            int ldx    = rt + icoor[orthogdim] * rd;
            int ij_ldx = jj + N_j * (ii + N_i * ldx);
            lsSum[ij_ldx] = lsSum[ij_ldx] + extracted[idx];
          }
        }
      });
      std::cout << GridLogMessage << tag << " simd_extract:    " << Tms(usecond()-t0) << " ms\n";

      int pd = grid->_processors[orthogdim];
      int pc = grid->_processor_coor[orthogdim];

      t0 = usecond();
      Vector<ComplexD> cache(nt * N_i * N_j, ComplexD(0.0));
      for (int lt = 0; lt < ld; lt++)
      for (int pt = 0; pt < pd; pt++) {
        int t = lt + pt * ld;
        for (int ii = 0; ii < N_i; ii++)
        for (int jj = 0; jj < N_j; jj++) {
          if (pt == pc) {
            int ij_ldx = jj + N_j * (ii + N_i * lt);
            cache[jj + N_j * (ii + N_i * t)] = lsSum[ij_ldx]()()();
          }
        }
      }
      grid->GlobalSumVector(cache.data(), nt * N_i * N_j);
      std::cout << GridLogMessage << tag << " globalsum:       " << Tms(usecond()-t0) << " ms\n";

      for (int t  = 0; t  < nt;  t++)
      for (int ii = 0; ii < N_i; ii++)
      for (int jj = 0; jj < N_j; jj++)
        result(t, ii, jj) = cache[jj + N_j * (ii + N_i * t)];
    }
  }
};

// ================================================================
// GPU wrapper functions: build field strength tensors per ifOrthog.
// Not accelerator_for kernels themselves; CloverleafMxN provides
// the GPU dispatch internally.
// ================================================================

// ifOrthog=0: spatial-temporal pairs (x,t),(y,t),(z,t) → SigmaXT,SigmaYT,SigmaZT
void A2ACMOContraction0(std::vector<GaugeMat> &G,
                         Vector<Gamma::Algebra> &Sigma,
                         const GaugeField &U)
{
  static const int            mus[3]  = {0, 1, 2};
  static const int            nus[3]  = {3, 3, 3};
  static const Gamma::Algebra sigs[3] = {
    Gamma::Algebra::SigmaXT,
    Gamma::Algebra::SigmaYT,
    Gamma::Algebra::SigmaZT
  };
  G.clear(); G.reserve(3);
  Sigma.resize(3);
  for (int idx = 0; idx < 3; ++idx) {
    GaugeMat Umu(U.Grid()), Unu(U.Grid()), Gmunu(U.Grid());
    Umu = peekLorentz(U, mus[idx]);
    Unu = peekLorentz(U, nus[idx]);
    WilsonLoops<PeriodicGimplD>::CloverleafMxN(Gmunu, Umu, Unu, mus[idx], nus[idx], 1, 1);
    G.push_back(Gmunu - adj(Gmunu));
    Sigma[idx] = sigs[idx];
  }
}

// ifOrthog=1: spatial-spatial pairs (x,y),(x,z),(y,z) → SigmaXY,SigmaXZ,SigmaYZ
void A2ACMOContraction1(std::vector<GaugeMat> &G,
                         Vector<Gamma::Algebra> &Sigma,
                         const GaugeField &U)
{
  static const int            mus[3]  = {0, 0, 1};
  static const int            nus[3]  = {1, 2, 2};
  static const Gamma::Algebra sigs[3] = {
    Gamma::Algebra::SigmaXY,
    Gamma::Algebra::SigmaXZ,
    Gamma::Algebra::SigmaYZ
  };
  G.clear(); G.reserve(3);
  Sigma.resize(3);
  for (int idx = 0; idx < 3; ++idx) {
    GaugeMat Umu(U.Grid()), Unu(U.Grid()), Gmunu(U.Grid());
    Umu = peekLorentz(U, mus[idx]);
    Unu = peekLorentz(U, nus[idx]);
    WilsonLoops<PeriodicGimplD>::CloverleafMxN(Gmunu, Umu, Unu, mus[idx], nus[idx], 1, 1);
    G.push_back(Gmunu - adj(Gmunu));
    Sigma[idx] = sigs[idx];
  }
}

// ================================================================
// GPU kernel: for each right vector, accumulate
//   loopRight[j](x) = sum_munu G[munu](x) * (parity?Gamma5:1) * Gamma(Sigma[munu]) * right[j](x)
// G (colour) and Sigma (spin) remain separate; no PropagatorField
// intermediate — all temporaries are per-thread register SpinColourVectors.
// ================================================================
void A2ACMOContractRight(FermionField &loopRight,
                          const std::vector<GaugeMat> &G,
                          const Vector<Gamma::Algebra> &Sigma,
                          const FermionField &right,
                          int parity)
{
  int ng = (int)G.size();

  typedef decltype(G[0].View(AcceleratorRead)) GView;
  std::vector<GView> gviews;
  gviews.reserve(ng);
  for (int munu = 0; munu < ng; ++munu)
    gviews.push_back(G[munu].View(AcceleratorRead));

  deviceVector<ColourMatrix_v *> gptrs(ng);
  for (int munu = 0; munu < ng; ++munu)
    acceleratorPut(gptrs[munu], &gviews[munu][0]);

  autoView(lRv, loopRight, AcceleratorWrite);
  autoView(rv,  right,     AcceleratorRead);

  const Gamma::Algebra *SigmaPtr = Sigma.data();
  ColourMatrix_v      **Gptr     = &gptrs[0];
  int lNg     = ng;
  int lParity = parity;

  uint64_t oSites = right.Grid()->oSites();
  int      Nsimd  = SpinColourVector_v::Nsimd();

  accelerator_for(ss, oSites, Nsimd, {
    typedef decltype(coalescedRead(rv[0])) calcSCVector;
    auto rightv = rv(ss);
    calcSCVector lR; lR = Zero();
    for (int munu = 0; munu < lNg; ++munu) {
      auto tmp2  = Gamma(SigmaPtr[munu]) * rightv;
      if (lParity == 1) tmp2 = Gamma(Gamma::Algebra::Gamma5) * tmp2;
      auto Gmunu = coalescedRead(Gptr[munu][ss]);
      for (int s = 0; s < Ns; ++s)
      for (int c = 0; c < Nc; ++c)
        lR()(s)(c) = lR()(s)(c)
                   + Gmunu()()(c,0) * tmp2()(s)(0)
                   + Gmunu()()(c,1) * tmp2()(s)(1)
                   + Gmunu()()(c,2) * tmp2()(s)(2);
    }
    coalescedWrite(lRv[ss], lR);
  });

  for (int munu = 0; munu < ng; ++munu)
    gviews[munu].ViewClose();
}

// ================================================================
// GPU-offloaded CMO field: accelerator_for right contraction
// + A2ASpatialSum GEMM spatial reduction.
// ================================================================
class A2AChromoMagneticOperatorFieldGPU
{
public:
  static void compute(
      Eigen::Tensor<ComplexD, 3> &result,
      const std::vector<FermionField> &left,
      const std::vector<FermionField> &right,
      const GaugeField &U,
      int ifOrthog,
      int parity)
  {
    GridBase *grid = left[0].Grid();
    int N_i = (int)left.size();
    int N_j = (int)right.size();

    std::string tag = std::string("[gpu  ifOrthog=") + std::to_string(ifOrthog) + " parity=" + std::to_string(parity) + "]";
    auto Tms = [](double us) { return us * 1e-3; };
    double t0;

    t0 = usecond();
    std::vector<GaugeMat>  G;
    Vector<Gamma::Algebra> Sigma;
    if (ifOrthog == 0) A2ACMOContraction0(G, Sigma, U);
    else               A2ACMOContraction1(G, Sigma, U);
    std::cout << GridLogMessage << tag << " field_strength:  " << Tms(usecond()-t0) << " ms\n";

    t0 = usecond();
    for (int munu = 0; munu < (int)G.size(); ++munu) { autoView(gv, G[munu], AcceleratorRead); }
    for (int j    = 0; j    < N_j;            ++j)   { autoView(rv, right[j], AcceleratorRead); }
    std::cout << GridLogMessage << tag << " view_open_right: " << Tms(usecond()-t0) << " ms\n";

    t0 = usecond();
    std::vector<FermionField> loopRight(N_j, grid);
    for (int j = 0; j < N_j; j++)
      A2ACMOContractRight(loopRight[j], G, Sigma, right[j], parity);
    std::cout << GridLogMessage << tag << " pack_right:      " << Tms(usecond()-t0) << " ms\n";

    A2ASpatialSum<SpinColourVector_v> spatial_sum;
    double t_blas = usecond();

    t0 = usecond();
    spatial_sum.Allocate(N_i, N_j, grid);
    std::cout << GridLogMessage << tag << " Allocate:        " << Tms(usecond()-t0) << " ms\n";

    t0 = usecond();
    spatial_sum.PackLeftConj(left);
    std::cout << GridLogMessage << tag << " PackLeftConj:    " << Tms(usecond()-t0) << " ms\n";

    t0 = usecond();
    spatial_sum.PackRight(loopRight);
    std::cout << GridLogMessage << tag << " PackRight:       " << Tms(usecond()-t0) << " ms\n";

    t0 = usecond();
    spatial_sum.Sum(result);
    std::cout << GridLogMessage << tag << " Sum (GEMM+MPI):  " << Tms(usecond()-t0) << " ms\n";

    std::cout << GridLogMessage << tag << " A2ASpatialSum:   " << Tms(usecond()-t_blas) << " ms  [TOTAL]\n";
  }
};

int main(int argc, char *argv[])
{
  Grid_init(&argc, &argv);

  Coordinate latt_size   = GridDefaultLatt();
  Coordinate simd_layout = GridDefaultSimd(4, vComplexD::Nsimd());
  Coordinate mpi_layout  = GridDefaultMpi();
  GridCartesian grid(latt_size, simd_layout, mpi_layout);

  int Nt  = latt_size[Tp];
  int N_i = 8;
  int N_j = 8;

  if (GridCmdOptionExists(argv, argv+argc, "--Ni"))
    N_i = std::stoi(GridCmdOptionPayload(argv, argv+argc, "--Ni"));
  if (GridCmdOptionExists(argv, argv+argc, "--Nj"))
    N_j = std::stoi(GridCmdOptionPayload(argv, argv+argc, "--Nj"));

  GridParallelRNG pRNG(&grid);
  pRNG.SeedFixedIntegers({1, 2, 3, 4});

  std::vector<FermionField> left(N_i,  &grid);
  std::vector<FermionField> right(N_j, &grid);
  GaugeField U(&grid);

  for (auto &f : left)  random(pRNG, f);
  for (auto &f : right) random(pRNG, f);
  SU<Nc>::HotConfiguration(pRNG, U);

  Eigen::Tensor<ComplexD, 3> result_ref(Nt, N_i, N_j);
  Eigen::Tensor<ComplexD, 3> result_blas(Nt, N_i, N_j);
  Eigen::Tensor<ComplexD, 3> result_gpu(Nt, N_i, N_j);
  double t_ref = 0, t_blas = 0, t_gpu = 0, start, stop;

  // Force GPU initialisation before any timed section
  { int dummy = 0; accelerator_for(i, 1, 1, { (void)i; }); }

  for (int ifOrthog : {0, 1})
  for (int parity   : {0, 1}) {

    result_ref.setZero();
    start = usecond();
    A2AChromoMagneticOperatorFieldRef::compute(result_ref,  left, right, U, ifOrthog, parity, false);
    stop = usecond(); t_ref = stop - start;

    result_blas.setZero();
    start = usecond();
    A2AChromoMagneticOperatorFieldRef::compute(result_blas, left, right, U, ifOrthog, parity, true);
    stop = usecond(); t_blas = stop - start;

    result_gpu.setZero();
    start = usecond();
    A2AChromoMagneticOperatorFieldGPU::compute(result_gpu,  left, right, U, ifOrthog, parity);
    stop = usecond(); t_gpu = stop - start;

    double norm2_ref = 0.0, norm2_blas = 0.0, norm2_gpu = 0.0;
    double norm2_diff_blas = 0.0, norm2_diff_gpu = 0.0;
    for (int t  = 0; t  < Nt;  t++)
    for (int ii = 0; ii < N_i; ii++)
    for (int jj = 0; jj < N_j; jj++) {
      norm2_ref  += norm2(result_ref(t, ii, jj));
      norm2_blas += norm2(result_blas(t, ii, jj));
      norm2_gpu  += norm2(result_gpu(t, ii, jj));
      ComplexD diff_blas = result_ref(t, ii, jj) - result_blas(t, ii, jj);
      ComplexD diff_gpu  = result_ref(t, ii, jj) - result_gpu(t, ii, jj);
      norm2_diff_blas += norm2(diff_blas);
      norm2_diff_gpu  += norm2(diff_gpu);
    }

    double rel_blas = (norm2_ref > 0) ? std::sqrt(norm2_diff_blas / norm2_ref) : 0.0;
    double rel_gpu  = (norm2_ref > 0) ? std::sqrt(norm2_diff_gpu  / norm2_ref) : 0.0;
    bool   pass_blas = (rel_blas < 1e-10);

    std::cout << GridLogMessage
              << "ifOrthog=" << ifOrthog << "  parity=" << parity
              << "  norm2_ref="  << norm2_ref
              << "  norm2_blas=" << norm2_blas
              << "  norm2_gpu="  << norm2_gpu
              << "  rel_blas="   << rel_blas
              << "  rel_gpu="    << rel_gpu
              << "  t_ref="  << t_ref  * 1e-6 << "s"
              << "  t_blas=" << t_blas * 1e-6 << "s"
              << "  t_gpu="  << t_gpu  * 1e-6 << "s"
              << "  " << (pass_blas ? "PASS" : "FAIL")
              << std::endl;

    GRID_ASSERT(rel_gpu < 1e-10);
  }

  std::cout << GridLogMessage << "All (ifOrthog,parity) pairs passed GPU regression." << std::endl;

  Grid_finalize();
  return EXIT_SUCCESS;
}

#endif
