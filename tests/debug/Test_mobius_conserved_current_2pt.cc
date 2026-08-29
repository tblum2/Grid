/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/debug/Test_mobius_conserved_current_2pt.cc

Copyright (C) 2015

Author: Peter Boyle <paboyle@ph.ed.ac.uk>

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

See the full license in the file "LICENSE" in the top level distribution
directory
*************************************************************************************/
/*  END LEGAL */
//
// Conserved-vector / local-vector current two-point function for Mobius
// DWF, following Eq. (18) of the "Mobius conserved currents" notes
// (T. Blum, L. Jin), via CayleyFermion5D::ContractMobiusConservedCurrent.
//
// G(x,s) = (D_M D_-)^{-1}(x,s;y) is built from a SINGLE MDminus solve,
// sourced at the physical point y via ImportUnphysicalFermion (no Dminus
// pre-multiplication -- MDminus's source convention). The full 5D
// propagator G is passed as BOTH arguments to
// ContractMobiusConservedCurrent, which internally reflects one copy in s
// (s -> Ls-1-s) and reuses WilsonFermion::ContractConservedCurrent's own
// gamma5(.)^dag gamma5 machinery to realize Eq. (18)'s
// gamma5 G^dag(x,Ls-1-s) gamma5 term -- no second linear solve.
//
// The current is computed in all four directions (X, Y, Z, T). Only the
// mu=T correlator is constant in t (the spatial sum makes it a conserved
// charge); the spatial ones carry genuine t dependence. The SV parity
// check should vanish to machine precision in every direction.
//
// Each direction is cross-checked against the existing (unfactored
// action) CayleyFermion5D::ContractConservedCurrent, comparing the full
// spin-colour propagator field rather than a traced correlator, so a
// discrepancy that cancelled under the trace would still be caught. The
// two sides are computed from genuinely different propagators -- G via
// MDminus, the reference via a Schur-preconditioned M solve -- so
// agreement independently validates Eq. (7)-(18), the gamma5 R5 relation
// and the (b+c) normalisation. Observed agreement is ~4e-15 (round-off).
//
// Verified MPI-decomposition and SIMD-layout independent: 4^3x8, Ls=10
// gives bitwise identical VV on all 32 (mu,t) points for
//   --mpi 1.1.1.1                            (default SIMD 1.1.2.2)
//   --mpi 1.1.1.1 --simd 1.2.1.2
//   --mpi 1.1.2.1 --simd 1.2.1.2  (2 ranks, Z split across ranks)
//
#include <Grid/Grid.h>

using namespace std;
using namespace Grid;

// Normal-equations wrapper around MDminus/MDminusDag (same as
// tests/solver/Test_mobius_cg_MDminus.cc), needed because MdagMLinearOperator
// is hardwired to call M()/Mdag().
template<class Matrix,class Field>
class MDminusHermOp : public LinearOperatorBase<Field> {
  Matrix &_Mat;
public:
  MDminusHermOp(Matrix &Mat): _Mat(Mat){};

  void OpDiag (const Field &in, Field &out)                    { assert(0); }
  void OpDir  (const Field &in, Field &out,int dir,int disp)   { assert(0); }
  void OpDirAll (const Field &in, std::vector<Field> &out)     { assert(0); }

  void Op    (const Field &in, Field &out){ _Mat.MDminus(in,out);    }
  void AdjOp (const Field &in, Field &out){ _Mat.MDminusDag(in,out); }

  void HermOpAndNorm(const Field &in, Field &out,RealD &n1,RealD &n2){
    Field tmp(in.Grid());
    _Mat.MDminus(in,tmp);
    _Mat.MDminusDag(tmp,out);
    ComplexD dot = innerProduct(in,out);
    n1 = real(dot);
    n2 = norm2(out);
  }
  void HermOp(const Field &in, Field &out){
    Field tmp(in.Grid());
    _Mat.MDminus(in,tmp);
    _Mat.MDminusDag(tmp,out);
  }
};

int main(int argc, char **argv)
{
  Grid_init(&argc, &argv);

  const int Ls = 10;

  // Optional --simd override, e.g. "--simd 1.2.1.2". The default layout
  // (1.1.2.2 for Nsimd=4) vectorises Z and T; combined with an MPI split
  // of the same direction on a small lattice the reduced dimension can
  // become odd, which red/black checkerboarding forbids. Moving the SIMD
  // vectorisation off the MPI-split direction fixes that without needing
  // a larger volume.
  Coordinate simd_layout = GridDefaultSimd(Nd, vComplex::Nsimd());
  if (GridCmdOptionExists(argv, argv + argc, "--simd")) {
    std::string arg = GridCmdOptionPayload(argv, argv + argc, "--simd");
    GridCmdOptionIntVector(arg, simd_layout);
  }
  std::cout << GridLogMessage << "SIMD layout : " << simd_layout
            << "   MPI layout : " << GridDefaultMpi() << std::endl;

  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), simd_layout, GridDefaultMpi());
  GridRedBlackCartesian  *UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian          *FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls, UGrid);
  GridRedBlackCartesian  *FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls, UGrid);

  std::vector<int> seeds4({1, 2, 3, 4});
  GridParallelRNG RNG4(UGrid);
  RNG4.SeedFixedIntegers(seeds4);

  LatticeGaugeField Umu(UGrid);
  SU<Nc>::HotConfiguration(RNG4, Umu);

  RealD mass = 0.3;
  RealD M5   = 1.0;
  RealD b    = 1.5;   // b+c = 2, matching the 48I/64I convention
  RealD c    = 0.5;

  MobiusFermionD Ddwf(Umu, *FGrid, *FrbGrid, *UGrid, *UrbGrid, mass, M5, b, c);

  LatticePropagator phys_src(UGrid);
  LatticePropagator G(FGrid);

  SpinColourMatrix kronecker; kronecker = 1.0;
  Coordinate coor({0, 0, 0, 0});
  phys_src = Zero();
  pokeSite(kronecker, phys_src, coor);

  MDminusHermOp<MobiusFermionD,LatticeFermion> HermOp(Ddwf);
  ConjugateGradient<LatticeFermion> CG(1.0e-16, 100000);

  for (int s = 0; s < Nd; s++) {
    for (int c_idx = 0; c_idx < Nc; c_idx++) {
      LatticeFermion src4(UGrid);
      PropToFerm<MobiusFermionD>(src4, phys_src, s, c_idx);

      LatticeFermion src5(FGrid);
      Ddwf.ImportUnphysicalFermion(src4, src5);   // no Dminus pre-multiplication

      LatticeFermion src_MdagM(FGrid);
      Ddwf.MDminusDag(src5, src_MdagM);

      LatticeFermion result5(FGrid); result5 = Zero();
      CG(HermOp, src_MdagM, result5);

      FermToProp<MobiusFermionD>(G, result5, s, c_idx);
    }
  }

  // Reference propagator for the cross-check below: the ORDINARY 5D
  // propagator M^{-1} = (D_- D_M)^{-1}, built the standard way with
  // ImportPhysicalFermionSource (which does pre-multiply by D_-) and a
  // Schur-preconditioned solve of M. This is what the existing
  // ContractConservedCurrent consumes; note it is a different object from
  // G above, which is (D_M D_-)^{-1}.
  LatticePropagator prop5(FGrid);
  {
    SchurRedBlackDiagTwoSolve<LatticeFermion> schur(CG);
    ZeroGuesser<LatticeFermion> zpg;
    for (int s = 0; s < Nd; s++) {
      for (int c_idx = 0; c_idx < Nc; c_idx++) {
        LatticeFermion src4(UGrid);
        PropToFerm<MobiusFermionD>(src4, phys_src, s, c_idx);

        LatticeFermion src5(FGrid);
        Ddwf.ImportPhysicalFermionSource(src4, src5);

        LatticeFermion result5(FGrid); result5 = Zero();
        schur(Ddwf, src5, result5, zpg);

        FermToProp<MobiusFermionD>(prop5, result5, s, c_idx);
      }
    }
  }

  // Conserved vector current in all four directions: the three spatial
  // ones (Xdir, Ydir, Zdir) as well as time (Tdir). The local vertex
  // gamma_mu at the source is matched to the conserved current's own
  // direction in each case.
  Gamma::Algebra Gmu_alg[] = {
    Gamma::Algebra::GammaX,
    Gamma::Algebra::GammaY,
    Gamma::Algebra::GammaZ,
    Gamma::Algebra::GammaT,
  };
  const char *mu_name[] = {"X", "Y", "Z", "T"};

  for (int mu_J = 0; mu_J < Nd; mu_J++) {

    Gamma gmu(Gmu_alg[mu_J]);

    // New (Eq. 18) current, from the single MDminus propagator G.
    LatticePropagator C_mu(UGrid);
    Ddwf.ContractMobiusConservedCurrent(G, G, C_mu, mu_J);

    // Reference: the existing unfactored-action conserved current, built
    // from the ordinary M^{-1} propagator. For plain Mobius the "reversed"
    // propagator it wants coincides with prop5 itself (Test_cayley_mres.cc
    // only builds a distinct prop5rev for the ZMobius case, where the
    // omega_s are s-reversed).
    LatticePropagator C_mu_ref(UGrid);
    Ddwf.ContractConservedCurrent(prop5, prop5, C_mu_ref, phys_src,
                                  Current::Vector, mu_J);

    // Full-field comparison: much stronger than comparing traced
    // correlators, since it cannot hide a discrepancy that cancels in the
    // trace.
    LatticePropagator C_diff(UGrid);
    C_diff = C_mu - C_mu_ref;
    RealD n_new = norm2(C_mu);
    RealD n_ref = norm2(C_mu_ref);
    RealD n_dif = norm2(C_diff);
    RealD rel   = (n_ref > 0.0) ? std::sqrt(n_dif / n_ref) : std::sqrt(n_dif);

    LatticeComplex SV(UGrid), VV(UGrid);
    SV = trace(C_mu);          // scalar-vector: vanishes by parity
    VV = trace(gmu * C_mu);    // local-vector x conserved-vector

    LatticeComplex VV_ref(UGrid);
    VV_ref = trace(gmu * C_mu_ref);

    std::vector<TComplex> sumSV, sumVV, sumVVref;
    sliceSum(SV, sumSV, Tdir);
    sliceSum(VV, sumVV, Tdir);
    sliceSum(VV_ref, sumVVref, Tdir);

    const int Nt = static_cast<int>(sumSV.size());
    std::cout << GridLogMessage << "=======================================" << std::endl;
    std::cout << GridLogMessage << "Conserved-local vector current 2pt function, mu = "
              << mu_name[mu_J] << " (" << mu_J << ")" << std::endl;
    std::cout << GridLogMessage << "Parity check: SV should be ~0 at every timeslice" << std::endl;
    for (int t = 0; t < Nt; t++) {
      std::cout << GridLogMessage << " t " << t
                << " SV " << real(TensorRemove(sumSV[t]))
                << " VV " << real(TensorRemove(sumVV[t]))
                << " VV(ref) " << real(TensorRemove(sumVVref[t])) << std::endl;
    }
    std::cout << GridLogMessage << " norm2 new " << n_new
              << "  norm2 ref " << n_ref << std::endl;
    std::cout << GridLogMessage << " || new - ref || / || ref || = " << rel
              << "   (Eq.18 current vs ContractConservedCurrent)" << std::endl;
  }

  Grid_finalize();
}
