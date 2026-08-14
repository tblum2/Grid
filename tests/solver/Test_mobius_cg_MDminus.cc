/*************************************************************************************

Grid physics library, www.github.com/paboyle/Grid

Source file: ./tests/solver/Test_mobius_cg_MDminus.cc

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
// Test/benchmark for CayleyFermion5D::MDminus and MDminusDag, i.e. the
// operator D^5_GDW D_-  (D_- applied on the right/source side, instead of
// the usual M() = D_- D^5_GDW).
//
// Unlike the usual physical-propagator solve, the source here is NOT run
// through ImportPhysicalFermionSource -- it is not chirally inserted at
// the s-boundaries and it is not pre-multiplied by D_-. We want the
// inverse of D^5_GDW D_- applied directly to an arbitrary 5D source.
//
// Neither Test_dwf_cg_prec.cc (raw preconditioned-operator CG on a single
// checkerboard) nor Test_dwf_cg_schur.cc (full production Schur solve) is
// directly reusable here: both are wired to the class's M()/Mdag()
// checkerboard machinery (Meooe/MooeeInv), and we have not built an
// even-odd-preconditioned counterpart for MDminus. So this follows
// Test_dwf_cg_prec.cc's spirit -- an explicit HermOp wrapper handed
// directly to ConjugateGradient, no physical-source machinery -- but
// unpreconditioned and on the full 5D field, matching the scope of
// Test_dwf_cg_schur.cc's source/result fields.
//
#include <Grid/Grid.h>

using namespace std;
using namespace Grid;

// Normal-equations wrapper around MDminus/MDminusDag, mirroring
// MdagMLinearOperator but pointed at the new pair of methods instead of
// M()/Mdag().
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

int main (int argc, char ** argv)
{
  Grid_init(&argc,&argv);

  const int Ls=12;

  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(), GridDefaultSimd(Nd,vComplex::Nsimd()),GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});
  GridParallelRNG          RNG5(FGrid);  RNG5.SeedFixedIntegers(seeds5);
  GridParallelRNG          RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);

  LatticeFermion    src(FGrid); random(RNG5,src);
  LatticeFermion    chi(FGrid); chi=Zero();
  LatticeGaugeField Umu(UGrid); SU<Nc>::HotConfiguration(RNG4,Umu);

  RealD mass = 0.01;
  RealD M5   = 1.8;
  RealD b    = 1.5;
  RealD c    = 0.5;    // b+c=2, as used for the 48I/64I Mobius ensembles
  MobiusFermionD Ddwf(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5,b,c);

  MDminusHermOp<MobiusFermionD,LatticeFermion> HermOp(Ddwf);
  ConjugateGradient<LatticeFermion> CG(1.0e-8,10000);

  // Normal-equations RHS: b = MDminusDag(src). No Dminus pre-multiplication
  // and no boundary insertion -- src is used exactly as generated.
  LatticeFermion src_MdagM(FGrid);
  Ddwf.MDminusDag(src,src_MdagM);

  GridStopWatch CGTimer;
  CGTimer.Start();
  CG(HermOp,src_MdagM,chi);
  CGTimer.Stop();

  std::cout << GridLogMessage << "Total CG time : " << CGTimer.Elapsed() << std::endl;

  // Independent correctness check: re-apply MDminus and compare to src.
  LatticeFermion residual(FGrid);
  Ddwf.MDminus(chi,residual);
  residual = residual - src;

  RealD true_res = std::sqrt(norm2(residual)/norm2(src));
  std::cout << GridLogMessage << "|| MDminus(chi) - src || / || src || = " << true_res << std::endl;

  Grid_finalize();
}
