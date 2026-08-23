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

  GridCartesian         *UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd, vComplex::Nsimd()), GridDefaultMpi());
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

  const int mu_J = Tdir;
  Gamma gT(Gamma::Algebra::GammaT);

  LatticePropagator C_mu(UGrid);
  Ddwf.ContractMobiusConservedCurrent(G, G, C_mu, mu_J);

  LatticeComplex SV(UGrid), VV(UGrid);
  SV = trace(C_mu);
  VV = trace(gT * C_mu);

  std::vector<TComplex> sumSV, sumVV;
  sliceSum(SV, sumSV, Tdir);
  sliceSum(VV, sumVV, Tdir);

  const int Nt = static_cast<int>(sumSV.size());
  std::cout << GridLogMessage << "Conserved-local vector current 2pt function (mu=" << mu_J << ")" << std::endl;
  std::cout << GridLogMessage << "Parity check: SV should be ~0 at every timeslice" << std::endl;
  for (int t = 0; t < Nt; t++) {
    std::cout << GridLogMessage << " t " << t
              << " SV " << real(TensorRemove(sumSV[t]))
              << " VV " << real(TensorRemove(sumVV[t])) << std::endl;
  }

  Grid_finalize();
}
