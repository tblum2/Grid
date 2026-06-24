#pragma once
//#include <Grid/Hadrons/Global.hpp>
#include <Grid/Grid_Eigen_Tensor.h>
#include <Grid/qcd/action/gauge/GaugeImplementations.h>
#include <Grid/qcd/utils/WilsonLoops.h>

NAMESPACE_BEGIN(Grid);

#undef DELTA_F_EQ_2

///////////////////////////////////////////////////////////////////
//Meson 
// Interested in
//
//      sum_x,y Trace[ G S(x,tx,y,ty) G S(y,ty,x,tx) ]
//
// Conventional meson field:
//                 
//    = sum_x,y Trace[ sum_j G |v_j(y,ty)> <w_j(x,tx)|  G sum_i |v_i(x,tx) ><w_i(y,ty)| ]
//    = sum_ij sum_x,y < w_j(x,tx)| G |v_i(x,tx) > <w_i(y,ty) (x)|G| v_j(y,ty) >
//    = sum_ij PI_ji(tx) PI_ij(ty)
//
// G5-Hermiticity
//
//      sum_x,y Trace[ G S(x,tx,y,ty) G S(y,ty,x,tx) ]
//    = sum_x,y Trace[ G S(x,tx,y,ty) G g5 S^dag(x,tx,y,ty) g5 ]
//    = sum_x,y Trace[ g5 G sum_j |v_j(y,ty)> <w_j(x,tx)|  G g5 sum_i   (|v_j(y,ty)> <w_i(x,tx)|)^dag ]      --  (*)
//
// NB:  Dag applies to internal indices spin,colour,complex
//
//    = sum_ij sum_x,y Trace[ g5 G |v_j(y,ty)> <w_j(x,tx)|  G g5  |w_i(x,tx)> <v_i(y,ty)| ]
//    = sum_ij sum_x,y <v_i(y,ty)|g5 G |v_j(y,ty)> <w_j(x,tx)|  G g5 |w_i(x,tx)> 
//    = sum_ij  PionVV(ty) PionWW(tx)
//
// (*) is only correct estimator if w_i and w_j come from distinct noise sets to preserve the kronecker
//     expectation value. Otherwise biased.
////////////////////////////////////////////////////////////////////

template <typename FImpl>
class A2Autils 
{
public:
  typedef typename FImpl::ComplexField ComplexField;
  typedef typename FImpl::FermionField FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;

  typedef typename FImpl::SiteSpinor vobj;
  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  typedef iSpinMatrix<vector_type> SpinMatrix_v;
  typedef iSpinMatrix<scalar_type> SpinMatrix_s;
  typedef iSinglet<vector_type> Scalar_v;
  typedef iSinglet<scalar_type> Scalar_s;

  typedef iSpinColourMatrix<vector_type> SpinColourMatrix_v;

  
  // output: rank 5 tensor, e.g. Eigen::Tensor<ComplexD, 5>
  template <typename TensorType>
  static void MesonField(TensorType &mat,
			 const FermionField *lhs_wi,
			 const FermionField *rhs_vj,
			 std::vector<Gamma::Algebra> gammas,
			 const std::vector<ComplexField > &mom,
			 int orthogdim);
  template <typename TensorType>
  static void MesonField(TensorType &mat,
			 const FermionField *lhs_wi,
			 const FermionField *rhs_vj,
			 std::vector<Gamma::Algebra> gammas,
			 const std::vector<ComplexField > &mom,
			 int orthogdim, double *timer)
  {
    MesonField(mat, lhs_wi, rhs_vj, gammas, mom, orthogdim);
  }

  // Applies phase(x) * Gamma(g) * in(x) -> out(x) site-wise.
  // Used by A2ANewMesonField to pre-contract right vectors before A2ASpatialSum.
  static void PhaseContractRight(FermionField       &out,
                                  const ComplexField &phase,
                                  Gamma::Algebra      g,
                                  const FermionField &in);

  // Applies Gamma(g) * in(x) -> out(x) site-wise, without phase.
  // Used by A2ANewMesonField (phase-trick path) as the zero-momentum base pack.
  static void GammaRight(FermionField       &out,
                          Gamma::Algebra      g,
                          const FermionField &in);

  // Phase 1: SC inner product with on-the-fly gamma application, no phase.
  // W_buf: conj(left) packed as [nt][Nii][nxyz*Nsc] by PackLeftConj.
  // LR_buf: right (no gamma) packed as [nt][Njj][nxyz*Nsc] by PackRight.
  // q_buf output layout: [(t*nxyz+x)*Nii*Njj + i*Njj + j].
  static void SpinColorTrace(
      deviceVector<scalar_type>       &q_buf,
      const deviceVector<scalar_type> &W_buf,
      const deviceVector<scalar_type> &LR_buf,
      Gamma::Algebra ga,
      int Nii, int Njj, int nt, int nxyz, int Nsc);

  // Phase 2: spatial reduction with all momenta simultaneously.
  // q_buf from SpinColorTrace; phase_all packed as [m*nxyz + x].
  // result: rank-4 tensor [nt_global][Nii][Njj][nmom].
  static void MomMesonField(
      Eigen::Tensor<ComplexD, 4>      &result,
      const deviceVector<scalar_type> &q_buf,
      const deviceVector<scalar_type> &phase_all,
      int Nii, int Njj, int nt_local, int nxyz, int nmom,
      GridBase *grid);

  // Phase 2 BLAS path: gemmBatched(OP_N, OP_N, Nii*Njj, nmom, nxyz) over nt_local slices.
  // Same q_buf and phase_all layout as MomMesonField; no custom kernel, no repacking.
  static void MomMesonFieldBLAS(
      Eigen::Tensor<ComplexD, 4>      &result,
      const deviceVector<scalar_type> &q_buf,
      const deviceVector<scalar_type> &phase_all,
      int Nii, int Njj, int nt_local, int nxyz, int nmom,
      GridBase *grid);

  // Pack all momenta phase fields into a single flat buffer: [m*nxyz + x].
  static void PackAllPhases(
      GridBase *grid,
      const std::vector<ComplexField> &ph,
      deviceVector<scalar_type> &phase_all);

  template <typename TensorType> // output: rank 5 tensor, e.g. Eigen::Tensor<ComplexD, 5>
  static void AslashField(TensorType &mat,
			  const FermionField *lhs_wi,
			  const FermionField *rhs_vj,
			  const std::vector<ComplexField> &emB0,
			  const std::vector<ComplexField> &emB1,
			  int orthogdim);

  template <typename TensorType> // output: rank 5 tensor, e.g. Eigen::Tensor<ComplexD, 5>
  static void AslashField(TensorType &mat, 
			  const FermionField *lhs_wi,
			  const FermionField *rhs_vj,
			  const std::vector<ComplexField> &emB0,
			  const std::vector<ComplexField> &emB1,
			  int orthogdim,double *timer)
  {
    AslashField(mat,lhs_wi,rhs_vj,emB0,emB1,orthogdim);
  }
  
  template <typename TensorType>
  typename std::enable_if<(std::is_same<Eigen::Tensor<ComplexD,3>, TensorType>::value ||
                           std::is_same<Eigen::TensorMap<Eigen::Tensor<Complex, 3, Eigen::RowMajor>>, TensorType>::value),
                           void>::type
  static ContractWWVV(std::vector<PropagatorField> &WWVV,
			   const TensorType &WW_sd,
			   const FermionField *vs,
			   const FermionField *vd);

  template <typename TensorType>
  typename std::enable_if<!(std::is_same<Eigen::Tensor<ComplexD,3>, TensorType>::value ||
                            std::is_same<Eigen::TensorMap<Eigen::Tensor<Complex, 3, Eigen::RowMajor>>, TensorType>::value),
                            void>::type
  static ContractWWVV(std::vector<PropagatorField> &WWVV,
			   const TensorType &WW_sd,
			   const FermionField *vs,
			   const FermionField *vd);

  static void ContractFourQuarkColourDiagonal(const PropagatorField &WWVV0,
					      const PropagatorField &WWVV1,
					      const std::vector<Gamma> &gamma0,
					      const std::vector<Gamma> &gamma1,
					      ComplexField &O_trtr,
					      ComplexField &O_fig8);

  static void ContractFourQuarkColourMix(const PropagatorField &WWVV0,
					 const PropagatorField &WWVV1,
					 const std::vector<Gamma> &gamma0,
					 const std::vector<Gamma> &gamma1,
					 ComplexField &O_trtr,
					 ComplexField &O_fig8);
#ifdef DELTA_F_EQ_2
  static void DeltaFeq2(int dt_min,int dt_max,
			Eigen::Tensor<ComplexD,2> &dF2_fig8,
			Eigen::Tensor<ComplexD,2> &dF2_trtr,
			Eigen::Tensor<ComplexD,2> &dF2_fig8_mix,
			Eigen::Tensor<ComplexD,2> &dF2_trtr_mix,
			Eigen::Tensor<ComplexD,1> &denom_A0,
			Eigen::Tensor<ComplexD,1> &denom_P,
			Eigen::Tensor<ComplexD,3> &WW_sd, 
			const FermionField *vs,
			const FermionField *vd,
			int orthogdim);
#endif
private:
  inline static void OuterProductWWVV(PropagatorField &WWVV,
                               const vobj &lhs,
                               const vobj &rhs,
                               const int Ns, const int ss);
};

const int A2Ablocking=96;

template<typename vtype> using iVecSpinMatrix = iVector<iMatrix<iScalar<vtype>, Ns>, A2Ablocking>;
typedef iVecSpinMatrix<Complex  >             VecSpinMatrix;
typedef iVecSpinMatrix<vComplex >             vVecSpinMatrix;
typedef Lattice<vVecSpinMatrix>               LatticeVecSpinMatrix;

template<typename vtype> using iVecComplex = iVector<iScalar<iScalar<vtype> >, A2Ablocking>;
typedef iVecComplex<Complex  >             VecComplex;
typedef iVecComplex<vComplex >             vVecComplex;
typedef Lattice<vVecComplex>               LatticeVecComplex;

#define A2A_GPU_KERNELS

template <class FImpl>
template <typename TensorType>
void A2Autils<FImpl>::MesonField(TensorType &mat,
				 const FermionField *lhs_wi,
				 const FermionField *rhs_vj,
				 std::vector<Gamma::Algebra> gammas,
				 const std::vector<ComplexField > &mom,
				 int orthogdim)
{
  const int block = A2Ablocking;
  typedef typename FImpl::SiteSpinor vobj;

  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  int Lblock = mat.dimension(3); 
  int Rblock = mat.dimension(4);

  //  GRID_ASSERT(Lblock % block==0);
  //  GRID_ASSERT(Rblock % block==0);
  
  GridBase *grid = lhs_wi[0].Grid();
  
  //  const int    Nd = grid->_ndimension;
  const int Nsimd = grid->Nsimd();

  int Nt     = grid->GlobalDimensions()[orthogdim];
  int Ngamma = gammas.size();
  int Nmom   = mom.size();

  LatticeVecSpinMatrix SpinMat(grid);
  LatticeVecSpinMatrix MomSpinMat(grid);

  std::cout <<GridLogMessage<< "A2A Meson Field"<<std::endl;
  MomentumProject<LatticeVecSpinMatrix,ComplexField> MP;
  MP.Allocate(Nmom,grid);
  MP.ImportMomenta(mom);
  std::cout <<GridLogMessage<< "Momentum project momenta imported"<<std::endl;

  double t_view, t_gamma, t_kernel, t_momproj;
  t_view=0;
  t_gamma=0;
  t_kernel=0;
  t_momproj=0;
  
  
  std::vector<VecSpinMatrix> sliced;
  for(int i=0;i<Lblock;i++){
    t_view -= usecond();
    autoView(SpinMat_v,SpinMat,AcceleratorWrite);
    autoView(lhs_v,lhs_wi[i],AcceleratorRead);
    t_view += usecond();
    for(int jo=0;jo<Rblock;jo+=block){
      for(int j=jo;j<MIN(Rblock,jo+block);j++){
	int jj=j%block;
	t_view -= usecond();
	autoView(rhs_v,rhs_vj[j],AcceleratorRead); // Create a vector of views
	t_view += usecond();
	//////////////////////////////////////////
	// Should write a SpinOuterColorTrace
	//////////////////////////////////////////

	t_kernel -= usecond();
	accelerator_for(ss,grid->oSites(),(size_t)Nsimd,{
	    auto left = conjugate(lhs_v(ss));
	    auto right = rhs_v(ss);
	    std::remove_reference_t<decltype(coalescedRead(SpinMat_v[ss])(jj))> spinmat;
	    for(int s1=0;s1<Ns;s1++){
	      for(int s2=0;s2<Ns;s2++){
		spinmat(s1,s2)() = left()(s2)(0) * right()(s1)(0)
		  +                left()(s2)(1) * right()(s1)(1)
		  +                left()(s2)(2) * right()(s1)(2);
	      }}
	    coalescedWrite(SpinMat_v[ss](jj),spinmat);
	  });
	t_kernel += usecond();

      }// j within block
      // After getting the sitewise product do the mom phase loop

      assert(orthogdim==Nd-1);
      t_momproj -= usecond();
      MP.Project(SpinMat,sliced);
      t_momproj +=  usecond();

      t_gamma -=  usecond();
      thread_for2d( m, Nmom,t,Nt,{
	  //      for(int m=0;m<Nmom;m++)
	  //	for(int t=0;t<Nt;t++)
	  int idx = t+m*Nt;
	  for(int j=jo;j<MIN(Rblock,jo+block);j++){
	    int jj=j%block;
	    auto tmp = peekIndex<LorentzIndex>(sliced[idx],jj);
	    for(int mu=0;mu<Ngamma;mu++){
	      auto trSG = trace(tmp*Gamma(gammas[mu]));
	      mat((long)m,mu,(long)t,i,j) = trSG()();
	    }
	  }
      }); 
      t_gamma +=  usecond();
    }//jo
  }
  std::cout << GridLogMessage<<" A2A::MesonField t_view    "<<t_view/1e6<<"s"<<std::endl;
  std::cout << GridLogMessage<<" A2A::MesonField t_momproj "<<t_momproj/1e6<<"s"<<std::endl;
  std::cout << GridLogMessage<<" A2A::MesonField t_kernel  "<<t_kernel/1e6<<"s"<<std::endl;
  std::cout << GridLogMessage<<" A2A::MesonField t_gamma   "<<t_gamma/1e6<<"s"<<std::endl;

}

template <class FImpl>
void A2Autils<FImpl>::PhaseContractRight(FermionField       &out,
                                          const ComplexField &phase,
                                          Gamma::Algebra      g,
                                          const FermionField &in)
{
  GridBase     *grid   = in.Grid();
  int           Nsimd  = grid->Nsimd();
  uint64_t      oSites = grid->oSites();
  Gamma::Algebra ga    = g;
  autoView(out_v, out,   AcceleratorWrite);
  autoView(in_v,  in,    AcceleratorRead);
  autoView(ph_v,  phase, AcceleratorRead);
  accelerator_for(ss, oSites, (size_t)Nsimd, {
    coalescedWrite(out_v[ss], ph_v(ss) * (Gamma(ga) * in_v(ss)));
  });
}

template <class FImpl>
void A2Autils<FImpl>::GammaRight(FermionField       &out,
                                   Gamma::Algebra      g,
                                   const FermionField &in)
{
  GridBase     *grid   = in.Grid();
  int           Nsimd  = grid->Nsimd();
  uint64_t      oSites = grid->oSites();
  Gamma::Algebra ga    = g;
  autoView(out_v, out, AcceleratorWrite);
  autoView(in_v,  in,  AcceleratorRead);
  accelerator_for(ss, oSites, (size_t)Nsimd, {
    coalescedWrite(out_v[ss], Gamma(ga) * in_v(ss));
  });
}

// SpinColorTrace: Phase 1 of the momentum-factored meson field.
// Each thread computes q[(t*nxyz+x)*Nii*Njj + i*Njj + j]
//   = sum_sc conj(left)[t][i][x,sc] * (Gamma * right)[t][j][x,sc].
// W_buf holds conj(left) via PackLeftConj; LR_buf holds right with no gamma via PackRight.
// Gamma is applied to LR on-the-fly using Grid's accelerator_inline Gamma operator,
// following the same cast pattern as PackVectors (scalar_type* alias of sobj).
template <class FImpl>
void A2Autils<FImpl>::SpinColorTrace(
    deviceVector<scalar_type>       &q_buf,
    const deviceVector<scalar_type> &W_buf,
    const deviceVector<scalar_type> &LR_buf,
    Gamma::Algebra ga,
    int Nii, int Njj, int nt, int nxyz, int Nsc)
{
  q_buf.resize((int64_t)nt * nxyz * Nii * Njj);

  const scalar_type *W  = &W_buf[0];
  const scalar_type *LR = &LR_buf[0];
  scalar_type       *Q  = &q_buf[0];
  int lNii = Nii, lNjj = Njj, lnt = nt, lnxyz = nxyz, lNsc = Nsc;
  Gamma::Algebra lga = ga;

  // iter2 = ti = (t*nxyz+x)*Nii + i, one block per (t,x,i) triple.
  // iter1 = j, one thread per j; consecutive j -> stride-1 write to Q (coalesced).
  accelerator_for2d(j, (size_t)lNjj, ti, (size_t)lnt * lnxyz * lNii, 1, {
    int lj  = (int)j;
    int t   = (int)(ti / ((size_t)lnxyz * lNii));
    int rem = (int)(ti % ((size_t)lnxyz * lNii));
    int x   = rem / lNii;
    int i   = rem % lNii;

    int64_t w_base  = (int64_t)t * lNii * lnxyz * lNsc + i * lnxyz * lNsc + x * lNsc;
    int64_t lr_base = (int64_t)t * lNjj * lnxyz * lNsc + lj * lnxyz * lNsc + x * lNsc;

    sobj lr_site;
    scalar_type *lr_s = (scalar_type *)&lr_site;
    for (int sc = 0; sc < lNsc; sc++) lr_s[sc] = LR[lr_base + sc];

    sobj gamma_lr      = Gamma(lga) * lr_site;
    scalar_type *glr_s = (scalar_type *)&gamma_lr;

    scalar_type result = 0;
    for (int sc = 0; sc < lNsc; sc++)
      result += W[w_base + sc] * glr_s[sc];

    Q[(int64_t)ti * lNjj + lj] = result;
  });
}

// PackAllPhases: packs nmom ComplexField phase arrays into phase_all[m*nxyz + x].
// Mirrors A2ASpatialSum::PackPhase but writes all momenta into one contiguous buffer.
template <class FImpl>
void A2Autils<FImpl>::PackAllPhases(
    GridBase *grid,
    const std::vector<ComplexField> &ph,
    deviceVector<scalar_type> &phase_all)
{
  int nd     = grid->_ndimension;
  int lnt    = grid->LocalDimensions()[nd - 1];
  int lnxyz  = grid->lSites() / lnt;
  int nmom   = (int)ph.size();
  int osites = grid->oSites();
  int lNsimd = grid->Nsimd();

  phase_all.resize((int64_t)nmom * lnxyz);
  scalar_type *ph_all = &phase_all[0];

  Coordinate rdimensions = grid->_rdimensions;
  Coordinate ldims       = grid->LocalDimensions();
  Coordinate simd_layout = grid->_simd_layout;

  int lNxyz = lnxyz;
  for (int m = 0; m < nmom; m++) {
    autoView(ph_v, ph[m], AcceleratorRead);
    int lm = m;
    accelerator_for(sf, (size_t)osites, lNsimd, {
#ifdef GRID_SIMT
      {
        int lane = acceleratorSIMTlane(lNsimd);
#else
        for (int lane = 0; lane < lNsimd; lane++) {
#endif
        Coordinate icoor(nd), ocoor(nd), lcoor(nd);
        Lexicographic::CoorFromIndex(icoor, lane, simd_layout);
        Lexicographic::CoorFromIndex(ocoor, sf,   rdimensions);
        for (int d = 0; d < nd; d++)
          lcoor[d] = rdimensions[d] * icoor[d] + ocoor[d];

        Coordinate xyz_coor = lcoor;
        xyz_coor[nd - 1]    = 0;
        int64_t l_xyz;
        Lexicographic::IndexFromCoor(xyz_coor, l_xyz, ldims);

        auto         ph_site = extractLane(lane, ph_v[sf]);
        scalar_type *ph_s    = (scalar_type *)&ph_site;
        ph_all[(int64_t)lm * lNxyz + l_xyz] = ph_s[0];
      }
    });
  }
}

// MomMesonField: Phase 2 of the momentum-factored meson field.
// Each thread accumulates mf[t][i][j][m] = sum_x q[(t*nxyz+x)*Nii*Njj + ij] * phase[m*nxyz+x].
// Followed by MPI GlobalSumVector to combine contributions across spatial ranks.
// result(gt, i, j, m) is filled for all nt_global timeslices and all nmom momenta.
template <class FImpl>
void A2Autils<FImpl>::MomMesonField(
    Eigen::Tensor<ComplexD, 4>      &result,
    const deviceVector<scalar_type> &q_buf,
    const deviceVector<scalar_type> &phase_all,
    int Nii, int Njj, int nt_local, int nxyz, int nmom,
    GridBase *grid)
{
  deviceVector<scalar_type> mf_buf((int64_t)nmom * nt_local * Nii * Njj);

  const scalar_type *Q  = &q_buf[0];
  const scalar_type *Ph = &phase_all[0];
  scalar_type       *MF = &mf_buf[0];
  int lNii = Nii, lNjj = Njj, lnt = nt_local, lnxyz = nxyz, lnmom = nmom;
  int lNiNj = Nii * Njj;

  // iter2 = mt = m*nt_local + t, one block per (m,t).
  // iter1 = ij = i*Njj + j, one thread per mode pair.
  // Consecutive ij -> stride-1 reads of Q and writes to MF (coalesced across warp).
  // Ph[m*nxyz + x] is identical for all ij threads at the same (m,t,x) -> broadcast.
  accelerator_for2d(ij, (size_t)lNiNj, mt, (size_t)lnmom * lnt, 1, {
    int lij = (int)ij;
    int m   = (int)(mt / lnt);
    int t   = (int)(mt % lnt);

    int64_t q_t_base = (int64_t)t * lnxyz * lNiNj;
    int     ph_base  = m * lnxyz;

    scalar_type acc = Zero();
    for (int x = 0; x < lnxyz; x++)
      acc += Q[q_t_base + x * lNiNj + lij] * Ph[ph_base + x];

    MF[(int64_t)mt * lNiNj + lij] = acc;
  });

  int nt_global = result.dimension(0);
  int nd        = grid->Nd();
  int lt_start  = grid->LocalStarts()[nd - 1];

  std::vector<scalar_type> host_mf((int64_t)nmom * nt_local * Nii * Njj);
  acceleratorCopyFromDevice(MF, host_mf.data(),
                            host_mf.size() * sizeof(scalar_type));

  std::vector<scalar_type> global_mf((int64_t)nt_global * Nii * Njj * nmom,
                                     scalar_type(0));
  for (int lt = 0; lt < nt_local; lt++) {
    int gt = lt + lt_start;
    for (int i = 0; i < Nii; i++)
    for (int j = 0; j < Njj; j++)
    for (int m = 0; m < nmom; m++)
      global_mf[((int64_t)gt * Nii * Njj + i * Njj + j) * nmom + m]
          = host_mf[((int64_t)m * nt_local + lt) * Nii * Njj + i * Njj + j];
  }
  grid->GlobalSumVector(global_mf.data(), (int64_t)nt_global * Nii * Njj * nmom);

  for (int gt = 0; gt < nt_global; gt++)
  for (int i  = 0; i  < Nii;      i++)
  for (int j  = 0; j  < Njj;      j++)
  for (int m  = 0; m  < nmom;     m++)
    result(gt, i, j, m) =
        global_mf[((int64_t)gt * Nii * Njj + i * Njj + j) * nmom + m];
}

// MomMesonFieldBLAS: Phase 2 using gemmBatched instead of the custom kernel.
// gemmBatched(OP_N, OP_N, M=Nii*Njj, N=nmom, K=nxyz) over nt_local slices.
// q_buf layout [(t*nxyz+x)*NiNj+ij] is A_Fortran[ij][x] with lda=NiNj (OP_N -> lda=M).
// phase_all layout [m*nxyz+x] is B_Fortran[x][m] with ldb=nxyz (OP_N on B -> ldb=K).
// Output C_Fortran[ij][m] with ldc=NiNj; read back as mf[(t*nmom+m)*NiNj+ij].
template <class FImpl>
void A2Autils<FImpl>::MomMesonFieldBLAS(
    Eigen::Tensor<ComplexD, 4>      &result,
    const deviceVector<scalar_type> &q_buf,
    const deviceVector<scalar_type> &phase_all,
    int Nii, int Njj, int nt_local, int nxyz, int nmom,
    GridBase *grid)
{
  int lNiNj = Nii * Njj;
  deviceVector<scalar_type> mf_buf((int64_t)nt_local * lNiNj * nmom);

  scalar_type *Q  = const_cast<scalar_type*>(&q_buf[0]);
  scalar_type *Ph = const_cast<scalar_type*>(&phase_all[0]);
  scalar_type *MF = &mf_buf[0];

  deviceVector<scalar_type*> q_ptrs(nt_local);
  deviceVector<scalar_type*> ph_ptrs(nt_local);
  deviceVector<scalar_type*> mf_ptrs(nt_local);
  int lnxyz = nxyz, lnmom = nmom;
  for (int t = 0; t < nt_local; t++) {
    acceleratorPut(q_ptrs[t],  Q  + (int64_t)t * lnxyz * lNiNj);
    acceleratorPut(ph_ptrs[t], Ph);
    acceleratorPut(mf_ptrs[t], MF + (int64_t)t * lNiNj * lnmom);
  }

  GridBLAS BLAS;
  BLAS.gemmBatched(GridBLAS_OP_N, GridBLAS_OP_N,
                   lNiNj, nmom, nxyz,
                   scalar_type(1.0),
                   q_ptrs,
                   ph_ptrs,
                   scalar_type(0.0),
                   mf_ptrs);
  BLAS.synchronise();

  int nt_global = result.dimension(0);
  int nd        = grid->Nd();
  int lt_start  = grid->LocalStarts()[nd - 1];

  std::vector<scalar_type> host_mf((int64_t)nt_local * lNiNj * nmom);
  acceleratorCopyFromDevice(MF, host_mf.data(),
                            host_mf.size() * sizeof(scalar_type));

  std::vector<scalar_type> global_mf((int64_t)nt_global * Nii * Njj * nmom,
                                     scalar_type(0));
  for (int lt = 0; lt < nt_local; lt++) {
    int gt = lt + lt_start;
    for (int i = 0; i < Nii; i++)
    for (int j = 0; j < Njj; j++)
    for (int m = 0; m < nmom; m++)
      global_mf[((int64_t)gt * Nii * Njj + i * Njj + j) * nmom + m]
          = host_mf[((int64_t)lt * nmom + m) * Nii * Njj + i * Njj + j];
  }
  grid->GlobalSumVector(global_mf.data(), (int64_t)nt_global * Nii * Njj * nmom);

  for (int gt = 0; gt < nt_global; gt++)
  for (int i  = 0; i  < Nii;      i++)
  for (int j  = 0; j  < Njj;      j++)
  for (int m  = 0; m  < nmom;     m++)
    result(gt, i, j, m) =
        global_mf[((int64_t)gt * Nii * Njj + i * Njj + j) * nmom + m];
}

// "A-slash" field w_i(x)^dag * i * A_mu * gamma_mu * v_j(x)
//
// With:
//
// B_0 = A_0 + i A_1
// B_1 = A_2 + i A_3
// 
// then in spin space
// 
//                 ( 0          0          -conj(B_1) -B_0 )
// i * A_mu g_mu = ( 0          0          -conj(B_0)  B_1 )
//                 ( B_1        B_0        0          0    )
//                 ( conj(B_0)  -conj(B_1) 0          0    )

template <class FImpl>
template <typename TensorType>
void A2Autils<FImpl>::AslashField(TensorType &mat, 
				  const FermionField *lhs_wi,
				  const FermionField *rhs_vj,
				  const std::vector<ComplexField> &emB0,
				  const std::vector<ComplexField> &emB1,
				  int orthogdim) 
{
  const int block=A2Ablocking;
  typedef typename FImpl::SiteSpinor vobj;

  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  int Lblock = mat.dimension(3); 
  int Rblock = mat.dimension(4);

  int Nem = emB0.size();
  GRID_ASSERT(emB1.size() == Nem);

  //  GRID_ASSERT(Lblock % block==0);
  //  GRID_ASSERT(Rblock % block==0);
  
  GridBase *grid = lhs_wi[0].Grid();
  
  const int    Nd = grid->_ndimension;
  const int Nsimd = grid->Nsimd();

  int Nt     = grid->GlobalDimensions()[orthogdim];

  LatticeVecSpinMatrix SpinMat(grid);
  LatticeVecComplex Aslash(grid);
  std::vector<VecComplex> sliced;

  for(int i=0;i<Lblock;i++){
    autoView(SpinMat_v,SpinMat,AcceleratorWrite);
    autoView(Aslash_v,Aslash,AcceleratorWrite);
    autoView(lhs_v,lhs_wi[i],AcceleratorRead);
    for(int jo=0;jo<Rblock;jo+=block){
      for(int j=jo;j<MIN(Rblock,jo+block);j++){
	int jj=j%block;
	autoView(rhs_v,rhs_vj[j],AcceleratorRead); // Create a vector of views
	//////////////////////////////////////////
	// Should write a SpinOuterColorTrace
	//////////////////////////////////////////
	accelerator_for(ss,grid->oSites(),(size_t)Nsimd,{
	    auto left = conjugate(lhs_v(ss));
	    auto right = rhs_v(ss);
	    auto vv =  SpinMat_v(ss);
	    for(int s1=0;s1<Ns;s1++){
	      for(int s2=0;s2<Ns;s2++){
		vv(jj)(s1,s2)() = left()(s2)(0) * right()(s1)(0)
		  +             left()(s2)(1) * right()(s1)(1)
		  +             left()(s2)(2) * right()(s1)(2);
	      }}
	    coalescedWrite(SpinMat_v[ss],vv);
	  });
      }
      for(int m=0;m<Nem;m++){
	autoView(emB0_v,emB0[m],AcceleratorRead);
	autoView(emB1_v,emB1[m],AcceleratorRead);
        accelerator_for(ss,grid->oSites(),(size_t)Nsimd,{
	  auto vv  = SpinMat_v(ss);
	  auto b0  = emB0_v(ss);
	  auto b1  = emB1_v(ss);
	  auto cb0 = conjugate(b0);
	  auto cb1 = conjugate(b1);
	  auto asl  = Aslash_v(ss);
   	  for(int j=jo;j<MIN(Rblock,jo+block);j++){
	    int jj=j%block;
	    asl(jj)()()=- vv(jj)(3,0)()*b0()()()  - vv(jj)(2,0)()*cb1()()()
	               + vv(jj)(3,1)()*b1()()()  - vv(jj)(2,1)()*cb0()()()
     	               + vv(jj)(0,2)()*b1()()()  + vv(jj)(1,2)()*b0()()()
	               + vv(jj)(0,3)()*cb0()()() - vv(jj)(1,3)()*cb1()()();
	    
	  }// j within block
	  coalescedWrite(Aslash_v[ss],asl);
	});

	sliceSum(Aslash,sliced,orthogdim);

	for(int t=0;t<sliced.size();t++){
	  for(int j=jo;j<MIN(Rblock,jo+block);j++){
	    int jj=j%block;
	    mat(m,0,t,i,j) = sliced[t](jj)()();
	  }
	}
      }
    }
  }
}

////////////////////////////////////////////
// Schematic thoughts about more generalised four quark insertion
//
// --  Pupil or fig8 type topology (depending on flavour structure) done below
// --  Have  Bag style   --  WWVV  VVWW
//            _  _      
//           / \/ \
//           \_/\_/   
//                                                                                                                                                                       
//  -   Kpipi style (two pion insertions) 
//                                K     
// *********
// Type 1                    
// *********
//                x
//            ___  _____ pi(b)  
//        K  /   \/____/          
//           \    \
//            \____\pi(a)   
//
//                        
//        W^W_sd V_s(x)  V^_d(xa) Wa(xa) Va^(x)    WW_bb' Vb Vb'(x)
//
//        (Kww.PIvw) VV ;   pi_ww.VV
// 
//  But Norman and Chris say g5 hermiticity not used, except on K
//
//        Kww    PIvw     PIvw       
//
//        (W^W_sd PIa_dd')   PIb_uu'  (v_s v_d' w_u v_u')(x)
// 
//  - Can form (Nmode^3)
//
//   (Kww . PIvw)_sd and then V_sV_d  tensor product contracting this
// 
//  - Can form 
//
//   (PIvw)_uu' W_uV_u'  tensor product.
//
// Both are lattice propagators.
//
// Contract with the same four quark type contraction as BK.
//
// *********
// Type 2
// *********
//            _  _____ pi 
//        K  / \/     |   
//           \_/\_____|pi 
//
// Norman/Chris would say
//
//  (Kww VV)(x) . (PIwv Piwv) VW(x)
//
// Full four quark via g5 hermiticity
//
//  (Kww VV)(x) . (PIww Pivw) VV(x)
//
// *********
// Type 3
// *********
//            ___  _____ pi 
//        K  /   \/     |   
//           \   /\     |
//            \  \/     |
//             \________|pi 
// 
// 
//   (V(x) . Kww . pi_vw . pivw . V(x)). WV(x)
//
// No difference possible to Norman, Chris, Diaqian
//
// *********
// Type 4
// *********
//            ___        pi 
//        K  /   \/\     |\
//           \___/\/     |/
//                        pi
//            
//  (Kww VV ) WV        x   Tr(PIwv PIwv)
//        
// Could use alternate PI_ww PIvv for disco loop interesting comparison
//
// *********
// Primitives / Utilities for assembly
// *********
// 
// i)   Basic type for meson field - mode indexed object, whether WW, VW, VV etc..
// ii)  Multiply two meson fields (modes^3) ; use BLAS MKL via Eigen
// iii) Multiply and trace two meson fields ; use BLAS MKL via Eigen. The element wise product trick
// iv)  Contract a meson field (whether WW,VW, VV WV) with W and V fields to form LatticePropagator
// v)   L^3 sum a four quark contaction with two LatticePropagators.
//
// Use lambda functions to enable flexibility in v)
// Use lambda functions to unify the pion field / meson field contraction codes.
////////////////////////////////////////////



////////////////////////////////////////////////////////////////////////
// DeltaF=2 contraction ; use exernal WW field for Kaon, anti Kaon sink
////////////////////////////////////////////////////////////////////////
//
// WW -- i vectors have adjoint, and j vectors not. 
//    -- Think of "i" as the strange quark, forward prop from 0
//    -- Think of "j" as the anti-down quark.
//
// WW_sd are w^dag_s w_d
//
// Hence VV vectors correspondingly are  v^dag_d,  v_s    from t=0
//                                  and  v^dag_d,  v_s    from t=dT
//
// There is an implicit g5 associated with each v^dag_d from use of g5 Hermiticity.
// The other gamma_5 lies in the WW external meson operator.
//
// From UKhadron wallbag.cc:
//
//   LatticePropagator anti_d0 =  adj( Gamma(G5) * Qd0 * Gamma(G5));
//   LatticePropagator anti_d1 =  adj( Gamma(G5) * Qd1 * Gamma(G5));
//
//   PR1 = Qs0 * Gamma(G5) * anti_d0;
//   PR2 = Qs1 * Gamma(G5) * anti_d1;
//
//   TR1 = trace( PR1 * G1 );
//   TR2 = trace( PR2 * G2 );
//   Wick1 = TR1 * TR2;
//
//   Wick2 = trace( PR1* G1 * PR2 * G2 );
//   // was      Wick2 = trace( Qs0 * Gamma(G5) * anti_d0 * G1 * Qs1 * Gamma(G5) * anti_d1 * G2 );
//
// TR TR(tx) = Wick1 = sum_x WW[t0]_sd < v^_d |g5 G| v_s>   WW[t1]_s'd' < v^_d' |g5 G| v_s'> |_{x,tx)
//           = sum_x [ Trace(WW[t0] VgV(t,x) )  x Trace( WW_[t1] VgV(t,x) ) ]
//           
//
// Calc all Nt Trace(WW VV) products at once, take Nt^2 products of these.
//
// Fig8(tx)  = Wick2 = sum_x WW[t0]_sd WW[t1]_s'd'  < v^_d |g5 G| v_s'> < v^_d' |g5 G| v_s> |_{x,tx}
//
//                   = sum_x Trace( WW[t0] VV[t,x] WW[t1] VV[t,x] )
// 
///////////////////////////////////////////////////////////////////////////////
//
// WW is w_s^dag (x) w_d       (G5 implicitly absorbed)
//
// WWVV will have spin-col (x) spin-col tensor.
//
// Want this to look like a strange fwd prop, anti-d prop.
// 
// Take WW_sd v^dag_d (x) v_s
// 

template <class FImpl>
template <typename TensorType>
typename std::enable_if<(std::is_same<Eigen::Tensor<ComplexD,3>, TensorType>::value ||
                         std::is_same<Eigen::TensorMap<Eigen::Tensor<Complex, 3, Eigen::RowMajor>>, TensorType>::value),
                         void>::type
A2Autils<FImpl>::ContractWWVV(std::vector<PropagatorField> &WWVV,
			      const TensorType &WW_sd,
			      const FermionField *vs,
			      const FermionField *vd)
{
  GridBase *grid = vs[0].Grid();

  //  int nd    = grid->_ndimension;
  int Nsimd = grid->Nsimd();
  int N_t   = WW_sd.dimension(0);
  int N_s = WW_sd.dimension(1); 
  int N_d = WW_sd.dimension(2);

  int d_unroll = 32;// Empirical optimisation

  for(int t=0;t<N_t;t++){
    WWVV[t] = Zero();
  }

  thread_for(ss,grid->oSites(),{
    for(int d_o=0;d_o<N_d;d_o+=d_unroll){
      for(int t=0;t<N_t;t++){
      for(int s=0;s<N_s;s++){
	autoView(vs_v,vs[s],CpuRead);
	auto tmp1 = vs_v[ss];
	vobj tmp2 = Zero();
	vobj tmp3 = Zero();
	for(int d=d_o;d<MIN(d_o+d_unroll,N_d);d++){
	  autoView(vd_v,vd[d],CpuRead);
	  Scalar_v coeff = WW_sd(t,s,d);
	  tmp3 = conjugate(vd_v[ss]);
	  mac(&tmp2, &coeff, &tmp3);
	}

	//////////////////////////
	// Fast outer product of tmp1 with a sum of terms suppressed by d_unroll
	//////////////////////////
	OuterProductWWVV(WWVV[t], tmp1, tmp2, Ns, ss);

      }}
    }
  });
}

template <class FImpl>
template <typename TensorType>
typename std::enable_if<!(std::is_same<Eigen::Tensor<ComplexD, 3>, TensorType>::value ||
                          std::is_same<Eigen::TensorMap<Eigen::Tensor<Complex, 3, Eigen::RowMajor>>, TensorType>::value),
                          void>::type
A2Autils<FImpl>::ContractWWVV(std::vector<PropagatorField> &WWVV,
                              const TensorType &WW_sd,
                              const FermionField *vs,
                              const FermionField *vd)
{
  GridBase *grid = vs[0].Grid();

  int nd    = grid->_ndimension;
  int Nsimd = grid->Nsimd();
  int N_t = WW_sd.dimensions()[0];
  int N_s = WW_sd.dimensions()[1];
  int N_d = WW_sd.dimensions()[2];

  int d_unroll = 32;// Empirical optimisation

  Eigen::Matrix<Complex, -1, -1, Eigen::RowMajor> buf;

  for(int t=0;t<N_t;t++){
    WWVV[t] = Zero();
  }

  for (int t = 0; t < N_t; t++){
    buf = WW_sd[t];
    thread_for(ss,grid->oSites(),{
      for(int d_o=0;d_o<N_d;d_o+=d_unroll){
        for(int s=0;s<N_s;s++){
	  autoView(vs_v,vs[s],CpuRead);
	  auto tmp1 = vs_v[ss];
	  vobj tmp2 = Zero();
	  vobj tmp3 = Zero();
	  for(int d=d_o;d<MIN(d_o+d_unroll,N_d);d++){
	    autoView(vd_v,vd[d],CpuRead);
	    Scalar_v coeff = buf(s,d);
	    tmp3 = conjugate(vd_v[ss]);
	    mac(&tmp2, &coeff, &tmp3);
	  }
	  //////////////////////////
	  // Fast outer product of tmp1 with a sum of terms suppressed by d_unroll
	  //////////////////////////
	  OuterProductWWVV(WWVV[t], tmp1, tmp2, Ns, ss);
      }}
    });
  }
}

template <class FImpl>
inline void A2Autils<FImpl>::OuterProductWWVV(PropagatorField &WWVV,
                                             const vobj &lhs,
                                             const vobj &rhs,
                                             const int Ns, const int ss)
{
  autoView(WWVV_v,WWVV,CpuWrite);
  for (int s1 = 0; s1 < Ns; s1++){
    for (int s2 = 0; s2 < Ns; s2++){
      WWVV_v[ss]()(s1,s2)(0, 0) += lhs()(s1)(0) * rhs()(s2)(0);
      WWVV_v[ss]()(s1,s2)(0, 1) += lhs()(s1)(0) * rhs()(s2)(1);
      WWVV_v[ss]()(s1,s2)(0, 2) += lhs()(s1)(0) * rhs()(s2)(2);
      WWVV_v[ss]()(s1,s2)(1, 0) += lhs()(s1)(1) * rhs()(s2)(0);
      WWVV_v[ss]()(s1,s2)(1, 1) += lhs()(s1)(1) * rhs()(s2)(1);
      WWVV_v[ss]()(s1,s2)(1, 2) += lhs()(s1)(1) * rhs()(s2)(2);
      WWVV_v[ss]()(s1,s2)(2, 0) += lhs()(s1)(2) * rhs()(s2)(0);
      WWVV_v[ss]()(s1,s2)(2, 1) += lhs()(s1)(2) * rhs()(s2)(1);
      WWVV_v[ss]()(s1,s2)(2, 2) += lhs()(s1)(2) * rhs()(s2)(2);
    }
  }
}

template<class FImpl>
void A2Autils<FImpl>::ContractFourQuarkColourDiagonal(const PropagatorField &WWVV0,
						      const PropagatorField &WWVV1,
						      const std::vector<Gamma> &gamma0,
						      const std::vector<Gamma> &gamma1,
						      ComplexField &O_trtr,
						      ComplexField &O_fig8)
{
  GRID_ASSERT(gamma0.size()==gamma1.size());
  int Ng = gamma0.size();

  GridBase *grid = WWVV0.Grid();

  autoView(WWVV0_v , WWVV0,CpuRead);
  autoView(WWVV1_v , WWVV1,CpuRead);
  autoView(O_trtr_v, O_trtr,CpuWrite);
  autoView(O_fig8_v, O_fig8,CpuWrite);
  thread_for(ss,grid->oSites(),{

    typedef typename ComplexField::vector_object vobj;

    vobj v_trtr;
    vobj v_fig8;

    auto VV0 = WWVV0_v[ss];
    auto VV1 = WWVV1_v[ss];
    
    for(int g=0;g<Ng;g++){

      v_trtr = trace(VV0 * gamma0[g])* trace(VV1*gamma1[g]);
      v_fig8 = trace(VV0 * gamma0[g] * VV1 * gamma1[g]);

      if ( g==0 ) {
	O_trtr_v[ss] = v_trtr; 
	O_fig8_v[ss] = v_fig8;
      } else { 
	O_trtr_v[ss]+= v_trtr; 
	O_fig8_v[ss]+= v_fig8;
      }
      
    }
  });
}

template<class FImpl>
void A2Autils<FImpl>::ContractFourQuarkColourMix(const PropagatorField &WWVV0,
						 const PropagatorField &WWVV1,
						 const std::vector<Gamma> &gamma0,
						 const std::vector<Gamma> &gamma1,
						 ComplexField &O_trtr,
						 ComplexField &O_fig8)
{
  GRID_ASSERT(gamma0.size()==gamma1.size());
  int Ng = gamma0.size();

  GridBase *grid = WWVV0.Grid();

  autoView( WWVV0_v , WWVV0,CpuRead);
  autoView( WWVV1_v , WWVV1,CpuRead);
  autoView( O_trtr_v, O_trtr,CpuWrite);
  autoView( O_fig8_v, O_fig8,CpuWrite);

  thread_for(ss,grid->oSites(),{

    typedef typename ComplexField::vector_object vobj;

    auto VV0 = WWVV0_v[ss];
    auto VV1 = WWVV1_v[ss];
    
    for(int g=0;g<Ng;g++){

      auto VV0G = VV0 * gamma0[g];  // Spin multiply
      auto VV1G = VV1 * gamma1[g];

      vobj v_trtr=Zero();
      vobj v_fig8=Zero();

      /////////////////////////////////////////
      // Colour mixed
      /////////////////////////////////////////
      // _                 _
      // s_sa G_st d_tb    s_s'b  G_s't' d_t'a
      // 
      //                                         
      // Contracted with prop factor (VV0)_sd,ab (VV1)_s'd',ba
      //
      // Wick1 [ spin TR TR ]
      //
      //    (VV0*G0)_ss,ba .  (VV1*G1)_tt,ab
       //
      // Wick2 [ spin fig8 ]
      //
      //    (VV0*G0)_st,aa (VV1*G1)_ts,bb
      // 
      /////////////////////////////////////////

      for(int a=0;a<Nc;a++){
      for(int b=0;b<Nc;b++){
      for(int s=0;s<Ns;s++){
      for(int t=0;t<Ns;t++){
	// Mixed traces
	v_trtr()()() += VV0G()(s,s)(a,b)*VV1G()(t,t)(b,a); // Was the fig8 before Fierzing
	v_fig8()()() += VV0G()(s,t)(a,a)*VV1G()(t,s)(b,b); // Was the trtr before Fierzing

	/*
	 * CHECKS -- use Fierz identities as a strong test, 4 Oct 2018.
	 * 
BagMix [8,0]  fig8 (21.5596,-3.83908e-17) trtr (0.064326,2.51001e-17) // Fierz        -1 0   0 0 0    
BagMix [8,1]  fig8 (-1346.99,1.2481e-16) trtr (34.2501,-3.36935e-17)  //               0 0   2 0 0 
BagMix [8,2]  fig8 (13.7536,-6.04625e-19) trtr (-215.542,3.24326e-17) //               0 1/2 0 0 0
BagMix [8,3]  fig8 (555.878,-7.39942e-17) trtr (463.82,-4.73909e-17)  //               0 0   0 1/2 -1/2  
BagMix [8,4]  fig8 (-1602.48,9.08511e-17) trtr (-936.302,1.14156e-16) //               0 0   0 -3/2 -1/2

Bag [8,0]  fig8 (-0.064326,1.06281e-17) trtr (-21.5596,1.06051e-17)   
Bag [8,2]  fig8 (17.125,-3.40959e-17) trtr (-673.493,7.68134e-17)   
Bag [8,1]  fig8 (-431.084,2.76423e-17) trtr (27.5073,-5.76967e-18)    /////////// TR TR                           FIG8
Bag [8,3]  fig8 (700.061,-1.14925e-16) trtr (1079.18,-1.35476e-16)    //    555.878 = 0.5(1079.18+32.5776) ;   463.82     =0.5(700.061+227.58)
Bag [8,4]  fig8 (-227.58,3.58808e-17) trtr (-32.5776,1.83286e-17)     //  - 1602.48 = - 1.5*1079.18 + .5* 32.5776; 936.302=-1.5* 700+0.5*227
	*/
	
	//Unmixed debug check consistency
	//	v_trtr()()() += VV0G()(s,s)(a,a)*VV1G()(t,t)(b,b);
	//	v_fig8()()() += VV0G()(s,t)(a,b)*VV1G()(t,s)(b,a);
      }}}}

      if ( g==0 ) {
	O_trtr_v[ss] = v_trtr; 
	O_fig8_v[ss] = v_fig8;
      } else { 
	O_trtr_v[ss]+= v_trtr; 
	O_fig8_v[ss]+= v_fig8;
      }
      
    }
  });
}



#ifdef DELTA_F_EQ_2
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Perhaps this should move out of the utils and into Hadrons module
// Now makes use of the primitives above and doesn't touch inside 
// the lattice structures.
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<class FImpl>
void A2Autils<FImpl>::DeltaFeq2(int dt_min,int dt_max,
				Eigen::Tensor<ComplexD,2> &dF2_fig8,
				Eigen::Tensor<ComplexD,2> &dF2_trtr,
				Eigen::Tensor<ComplexD,2> &dF2_fig8_mix,
				Eigen::Tensor<ComplexD,2> &dF2_trtr_mix,
				Eigen::Tensor<ComplexD,1> &denom_A0,
				Eigen::Tensor<ComplexD,1> &denom_P,
				Eigen::Tensor<ComplexD,3> &WW_sd, 
				const FermionField *vs,
				const FermionField *vd,
				int orthogdim)
{
  GridBase *grid = vs[0].Grid();

  LOG(Message) << "Computing A2A DeltaF=2 graph" << std::endl;

  auto G5 = Gamma(Gamma::Algebra::Gamma5);
  
  double nodes = grid->NodeCount();
  int nd    = grid->_ndimension;
  int Nsimd = grid->Nsimd();
  int N_t = WW_sd.dimension(0); 
  int N_s = WW_sd.dimension(1); 
  int N_d = WW_sd.dimension(2);

  GRID_ASSERT(grid->GlobalDimensions()[orthogdim] == N_t);
  double vol         = 1.0;
  for(int dim=0;dim<nd;dim++){
    vol = vol * grid->GlobalDimensions()[dim];
  }

  double t_tot  = -usecond();
  std::vector<PropagatorField> WWVV (N_t,grid);

  double t_outer= -usecond();
  ContractWWVV(WWVV,WW_sd,&vs[0],&vd[0]);
  t_outer+=usecond();

  //////////////////////////////
  // Implicit gamma-5 
  //////////////////////////////
  for(int t=0;t<N_t;t++){
    WWVV[t] = WWVV[t]* G5 ; 
  }

  ////////////////////////////////////////////////////////
  // Contraction
  ////////////////////////////////////////////////////////
  int Ng=5;
  dF2_trtr.resize(N_t,Ng);
  dF2_fig8.resize(N_t,Ng);
  dF2_trtr_mix.resize(N_t,Ng);
  dF2_fig8_mix.resize(N_t,Ng);
  denom_A0.resize(N_t);
  denom_P.resize(N_t);
  for(int t=0;t<N_t;t++){
    for(int g=0;g<Ng;g++) dF2_trtr(t,g)= ComplexD(0.0);
    for(int g=0;g<Ng;g++) dF2_fig8(t,g)= ComplexD(0.0);
    for(int g=0;g<Ng;g++) dF2_trtr_mix(t,g)= ComplexD(0.0);
    for(int g=0;g<Ng;g++) dF2_fig8_mix(t,g)= ComplexD(0.0);
    denom_A0(t) =ComplexD(0.0);
    denom_P(t) =ComplexD(0.0);
  }

  ComplexField D0(grid);   D0 = Zero(); // <P|A0> correlator from each wall
  ComplexField D1(grid);   D1 = Zero();

  ComplexField O1_trtr(grid);  O1_trtr = Zero();
  ComplexField O2_trtr(grid);  O2_trtr = Zero();
  ComplexField O3_trtr(grid);  O3_trtr = Zero();
  ComplexField O4_trtr(grid);  O4_trtr = Zero();
  ComplexField O5_trtr(grid);  O5_trtr = Zero();

  ComplexField O1_fig8(grid);  O1_fig8 = Zero();
  ComplexField O2_fig8(grid);  O2_fig8 = Zero();
  ComplexField O3_fig8(grid);  O3_fig8 = Zero();
  ComplexField O4_fig8(grid);  O4_fig8 = Zero();
  ComplexField O5_fig8(grid);  O5_fig8 = Zero();

  ComplexField VV_trtr(grid);  VV_trtr = Zero();
  ComplexField AA_trtr(grid);  AA_trtr = Zero();
  ComplexField SS_trtr(grid);  SS_trtr = Zero();
  ComplexField PP_trtr(grid);  PP_trtr = Zero();
  ComplexField TT_trtr(grid);  TT_trtr = Zero();

  ComplexField VV_fig8(grid);  VV_fig8 = Zero();
  ComplexField AA_fig8(grid);  AA_fig8 = Zero();
  ComplexField SS_fig8(grid);  SS_fig8 = Zero();
  ComplexField PP_fig8(grid);  PP_fig8 = Zero();
  ComplexField TT_fig8(grid);  TT_fig8 = Zero();

  //////////////////////////////////////////////////
  // Used to store appropriate correlation funcs
  //////////////////////////////////////////////////
  std::vector<TComplex>  C1;
  std::vector<TComplex>  C2;
  std::vector<TComplex>  C3;
  std::vector<TComplex>  C4;
  std::vector<TComplex>  C5;
  
  //////////////////////////////////////////////////////////
  // Could do AA, VV, SS, PP, TT and form linear combinations later.
  // Almost 2x. but for many modes, the above loop dominates.
  //////////////////////////////////////////////////////////
  double t_contr= -usecond();

  // Tr Tr Wick contraction
  auto VX = Gamma(Gamma::Algebra::GammaX);
  auto VY = Gamma(Gamma::Algebra::GammaY);
  auto VZ = Gamma(Gamma::Algebra::GammaZ);
  auto VT = Gamma(Gamma::Algebra::GammaT);
  
  auto AX = Gamma(Gamma::Algebra::GammaXGamma5);
  auto AY = Gamma(Gamma::Algebra::GammaYGamma5);
  auto AZ = Gamma(Gamma::Algebra::GammaZGamma5);
  auto AT = Gamma(Gamma::Algebra::GammaTGamma5);
  
  auto S  = Gamma(Gamma::Algebra::Identity);
  auto P  = Gamma(Gamma::Algebra::Gamma5);
  
  auto T0 = Gamma(Gamma::Algebra::SigmaXY);
  auto T1 = Gamma(Gamma::Algebra::SigmaXZ);
  auto T2 = Gamma(Gamma::Algebra::SigmaXT);
  auto T3 = Gamma(Gamma::Algebra::SigmaYZ);
  auto T4 = Gamma(Gamma::Algebra::SigmaYT);
  auto T5 = Gamma(Gamma::Algebra::SigmaZT);

  std::cout <<GridLogMessage << " dt " <<dt_min<<"..." <<dt_max<<std::endl;

  for(int t0=0;t0<N_t;t0++){
    std::cout <<GridLogMessage << " t0 " <<t0<<std::endl;
    //    for(int dt=dt_min;dt<dt_max;dt++){
    {     
      int dt  = dt_min;
      int t1 = (t0+dt)%N_t;

      std::cout <<GridLogMessage << " t1 " <<t1<<std::endl;
      std::vector<Gamma> VV({VX,VY,VZ,VT});
      std::vector<Gamma> AA({AX,AY,AZ,AT});
      std::vector<Gamma> SS({S});
      std::vector<Gamma> PP({P});
      std::vector<Gamma> TT({T0,T1,T2,T3,T4,T5});
      std::vector<Gamma> A0({AT});

      ContractFourQuarkColourDiagonal(WWVV[t0],	WWVV[t1],VV,VV,VV_trtr,VV_fig8); // VV
      ContractFourQuarkColourDiagonal(WWVV[t0],	WWVV[t1],AA,AA,AA_trtr,AA_fig8); // AA
      ContractFourQuarkColourDiagonal(WWVV[t0],	WWVV[t1],SS,SS,SS_trtr,SS_fig8); // SS
      ContractFourQuarkColourDiagonal(WWVV[t0],	WWVV[t1],PP,PP,PP_trtr,PP_fig8); // PP
      ContractFourQuarkColourDiagonal(WWVV[t0],	WWVV[t1],TT,TT,TT_trtr,TT_fig8); // TT

      O1_trtr = VV_trtr+AA_trtr;      O2_trtr = VV_trtr-AA_trtr;  // VV+AA,VV-AA
      O1_fig8 = VV_fig8+AA_fig8;      O2_fig8 = VV_fig8-AA_fig8;  

      O3_trtr = SS_trtr-PP_trtr;      O4_trtr = SS_trtr+PP_trtr;  // SS+PP,SS-PP
      O3_fig8 = SS_fig8-PP_fig8;      O4_fig8 = SS_fig8+PP_fig8;  

      O5_trtr = TT_trtr;
      O5_fig8 = TT_fig8;

      sliceSum(O1_trtr,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_trtr(t,0)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O2_trtr,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_trtr(t,1)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O3_trtr,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_trtr(t,2)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O4_trtr,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_trtr(t,3)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O5_trtr,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_trtr(t,4)+= 2.0*C1[(t+t0)%N_t]()()()/vol;

      sliceSum(O1_fig8,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_fig8(t,0)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O2_fig8,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_fig8(t,1)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O3_fig8,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_fig8(t,2)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O4_fig8,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_fig8(t,3)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O5_fig8,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_fig8(t,4)+= 2.0*C1[(t+t0)%N_t]()()()/vol;

      ContractFourQuarkColourDiagonal(WWVV[t0],	WWVV[t1],A0,A0,AA_trtr,AA_fig8); // A0 insertion

      sliceSum(AA_trtr,C1, orthogdim);
      sliceSum(PP_trtr,C2, orthogdim);

      for(int t=0;t<N_t;t++){
	denom_A0(t)+=C1[(t+t0)%N_t]()()()/vol;
	denom_P(t) +=C2[(t+t0)%N_t]()()()/vol;
      }

      ///////////////////////////////////////////////////////////////////////////
      // Colour mixed contractions
      ///////////////////////////////////////////////////////////////////////////

      ContractFourQuarkColourMix(WWVV[t0],	WWVV[t1],VV,VV,VV_trtr,VV_fig8); // VV
      ContractFourQuarkColourMix(WWVV[t0],	WWVV[t1],AA,AA,AA_trtr,AA_fig8); // AA
      ContractFourQuarkColourMix(WWVV[t0],	WWVV[t1],SS,SS,SS_trtr,SS_fig8); // SS
      ContractFourQuarkColourMix(WWVV[t0],	WWVV[t1],PP,PP,PP_trtr,PP_fig8); // PP
      ContractFourQuarkColourMix(WWVV[t0],	WWVV[t1],TT,TT,TT_trtr,TT_fig8); // TT

      O1_trtr = VV_trtr+AA_trtr;      O2_trtr = VV_trtr-AA_trtr;  // VV+AA,VV-AA
      O1_fig8 = VV_fig8+AA_fig8;      O2_fig8 = VV_fig8-AA_fig8;  

      O3_trtr = SS_trtr-PP_trtr;      O4_trtr = SS_trtr+PP_trtr;  // SS+PP,SS-PP
      O3_fig8 = SS_fig8-PP_fig8;      O4_fig8 = SS_fig8+PP_fig8;  

      O5_trtr = TT_trtr;
      O5_fig8 = TT_fig8;

      sliceSum(O1_trtr,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_trtr_mix(t,0)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O2_trtr,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_trtr_mix(t,1)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O3_trtr,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_trtr_mix(t,2)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O4_trtr,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_trtr_mix(t,3)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O5_trtr,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_trtr_mix(t,4)+= 2.0*C1[(t+t0)%N_t]()()()/vol;

      sliceSum(O1_fig8,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_fig8_mix(t,0)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O2_fig8,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_fig8_mix(t,1)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O3_fig8,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_fig8_mix(t,2)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O4_fig8,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_fig8_mix(t,3)+= 2.0*C1[(t+t0)%N_t]()()()/vol;
      sliceSum(O5_fig8,C1, orthogdim); for(int t=0;t<N_t;t++) dF2_fig8_mix(t,4)+= 2.0*C1[(t+t0)%N_t]()()()/vol;

    }
  }
  t_contr +=usecond();
  
  t_tot+=usecond();
  double million=1.0e6;
  LOG(Message) << "Computing A2A DeltaF=2 graph t_tot      " << t_tot      /million << " s "<< std::endl;
  LOG(Message) << "Computing A2A DeltaF=2 graph t_outer    " << t_outer    /million << " s "<< std::endl;
  LOG(Message) << "Computing A2A DeltaF=2 graph t_contr    " << t_contr    /million << " s "<< std::endl;
}
#endif 

  /*
  static void PionFieldWVmom(Eigen::Tensor<ComplexD,4> &mat, 
			     const FermionField *wi,
			     const FermionField *vj,
			     const std::vector<ComplexField > &mom,
			     int orthogdim);

  static void PionFieldXX(Eigen::Tensor<ComplexD,3> &mat, 
			  const FermionField *wi,
			  const FermionField *vj,
			  int orthogdim,
			  int g5);
  
  static void PionFieldWV(Eigen::Tensor<ComplexD,3> &mat, 
			  const FermionField *wi,
			  const FermionField *vj,
			  int orthogdim);
  static void PionFieldWW(Eigen::Tensor<ComplexD,3> &mat, 
			  const FermionField *wi,
			  const FermionField *wj,
			  int orthogdim);
  static void PionFieldVV(Eigen::Tensor<ComplexD,3> &mat, 
			  const FermionField *vi,
			  const FermionField *vj,
			  int orthogdim);
  */

/*

template <class FImpl>
template <typename TensorType>
void A2Autils<FImpl>::MesonField(TensorType &mat, 
				 const FermionField *lhs_wi,
				 const FermionField *rhs_vj,
				 std::vector<Gamma::Algebra> gammas,
				 const std::vector<ComplexField > &mom,
				 int orthogdim, double *t_kernel, double *t_gsum) 
{
  typedef typename FImpl::SiteSpinor vobj;

  typedef typename vobj::scalar_object sobj;
  typedef typename vobj::scalar_type scalar_type;
  typedef typename vobj::vector_type vector_type;

  typedef iSpinMatrix<vector_type> SpinMatrix_v;
  typedef iSpinMatrix<scalar_type> SpinMatrix_s;
  
  int Lblock = mat.dimension(3); 
  int Rblock = mat.dimension(4);

  GridBase *grid = lhs_wi[0].Grid();
  
  const int    Nd = grid->_ndimension;
  const int Nsimd = grid->Nsimd();

  int Nt     = grid->GlobalDimensions()[orthogdim];
  int Ngamma = gammas.size();
  int Nmom   = mom.size();

  int fd=grid->_fdimensions[orthogdim];
  int ld=grid->_ldimensions[orthogdim];
  int rd=grid->_rdimensions[orthogdim];

  // will locally sum vectors first
  // sum across these down to scalars
  // splitting the SIMD
  int MFrvol = rd*Lblock*Rblock*Nmom;
  int MFlvol = ld*Lblock*Rblock*Nmom;

  std::vector<SpinMatrix_v > lvSum(MFrvol);
  for(int r=0;r<MFrvol;r++){
    lvSum[r] = Zero();
  }

  std::vector<SpinMatrix_s > lsSum(MFlvol);             
  for(int r=0;r<MFlvol;r++){
    lsSum[r]=scalar_type(0.0);
  }

  int e1=    grid->_slice_nblock[orthogdim];
  int e2=    grid->_slice_block [orthogdim];
  int stride=grid->_slice_stride[orthogdim];

  // potentially wasting cores here if local time extent too small
  if (t_kernel) *t_kernel = -usecond();
  for(int r=0;r<rd;r++) {

    int so=r*grid->_ostride[orthogdim]; // base offset for start of plane 

    for(int n=0;n<e1;n++){
      for(int b=0;b<e2;b++){

	int ss= so+n*stride+b;

	for(int i=0;i<Lblock;i++){

	  // Recreate view potentially expensive outside fo UVM mode
	  autoView(lhs_v,lhs_wi[i],CpuRead);
	  auto left = conjugate(lhs_v[ss]);
	  for(int j=0;j<Rblock;j++){

	    SpinMatrix_v vv;
	    // Recreate view potentially expensive outside fo UVM mode
	    autoView(rhs_v,rhs_vj[j],CpuRead);
	    auto right = rhs_v[ss];
	    for(int s1=0;s1<Ns;s1++){
	    for(int s2=0;s2<Ns;s2++){
	      vv()(s1,s2)() = left()(s2)(0) * right()(s1)(0)
		+             left()(s2)(1) * right()(s1)(1)
		+             left()(s2)(2) * right()(s1)(2);
	    }}
	    
	    // After getting the sitewise product do the mom phase loop
	    int base = Nmom*i+Nmom*Lblock*j+Nmom*Lblock*Rblock*r;
	    for ( int m=0;m<Nmom;m++){
	      int idx = m+base;
	      autoView(mom_v,mom[m],CpuRead);
	      auto phase = mom_v[ss];
	      mac(&lvSum[idx],&vv,&phase);
	    }
	  }
	}
      }
    }
  };

  // Sum across simd lanes in the plane, breaking out orthog dir.
  for(int rt=0;rt<rd;rt++){

    Coordinate icoor(Nd);
    ExtractBuffer<SpinMatrix_s> extracted(Nsimd);               

    for(int i=0;i<Lblock;i++){
    for(int j=0;j<Rblock;j++){
    for(int m=0;m<Nmom;m++){

      int ij_rdx = m+Nmom*i+Nmom*Lblock*j+Nmom*Lblock*Rblock*rt;

      extract(lvSum[ij_rdx],extracted);

      for(int idx=0;idx<Nsimd;idx++){

	grid->iCoorFromIindex(icoor,idx);

	int ldx    = rt+icoor[orthogdim]*rd;

	int ij_ldx = m+Nmom*i+Nmom*Lblock*j+Nmom*Lblock*Rblock*ldx;

	lsSum[ij_ldx]=lsSum[ij_ldx]+extracted[idx];

      }
    }}}
  }
  if (t_kernel) *t_kernel += usecond();
  GRID_ASSERT(mat.dimension(0) == Nmom);
  GRID_ASSERT(mat.dimension(1) == Ngamma);
  GRID_ASSERT(mat.dimension(2) == Nt);

  // ld loop and local only??
  int pd = grid->_processors[orthogdim];
  int pc = grid->_processor_coor[orthogdim];
  thread_for_collapse(2,lt,ld,{
    for(int pt=0;pt<pd;pt++){
      int t = lt + pt*ld;
      if (pt == pc){
	for(int i=0;i<Lblock;i++){
	  for(int j=0;j<Rblock;j++){
	    for(int m=0;m<Nmom;m++){
	      int ij_dx = m+Nmom*i + Nmom*Lblock * j + Nmom*Lblock * Rblock * lt;
	      for(int mu=0;mu<Ngamma;mu++){
		// this is a bit slow
		mat(m,mu,t,i,j) = trace(lsSum[ij_dx]*Gamma(gammas[mu]))()()();
	      }
	    }
	  }
	}
      } else { 
	const scalar_type zz(0.0);
	for(int i=0;i<Lblock;i++){
	  for(int j=0;j<Rblock;j++){
	    for(int mu=0;mu<Ngamma;mu++){
	      for(int m=0;m<Nmom;m++){
		mat(m,mu,t,i,j) =zz;
	      }
	    }
	  }
	}
      }
    }
  });

  ////////////////////////////////////////////////////////////////////
  // This global sum is taking as much as 50% of time on 16 nodes
  // Vector size is 7 x 16 x 32 x 16 x 16 x sizeof(complex) = 2MB - 60MB depending on volume
  // Healthy size that should suffice
  ////////////////////////////////////////////////////////////////////
  if (t_gsum) *t_gsum = -usecond();
  grid->GlobalSumVector(&mat(0,0,0,0,0),Nmom*Ngamma*Nt*Lblock*Rblock);
  if (t_gsum) *t_gsum += usecond();
}

template<class FImpl>
void A2Autils<FImpl>::PionFieldXX(Eigen::Tensor<ComplexD,3> &mat, 
				  const FermionField *wi,
				  const FermionField *vj,
				  int orthogdim,
				  int g5) 
{
  int Lblock = mat.dimension(1); 
  int Rblock = mat.dimension(2);

  GridBase *grid = wi[0].Grid();
  
  const int    nd = grid->_ndimension;
  const int Nsimd = grid->Nsimd();

  int Nt     = grid->GlobalDimensions()[orthogdim];

  int fd=grid->_fdimensions[orthogdim];
  int ld=grid->_ldimensions[orthogdim];
  int rd=grid->_rdimensions[orthogdim];

  // will locally sum vectors first
  // sum across these down to scalars
  // splitting the SIMD
  int MFrvol = rd*Lblock*Rblock;
  int MFlvol = ld*Lblock*Rblock;

  std::vector<vector_type > lvSum(MFrvol);
  thread_for(r,MFrvol,{
    lvSum[r] = Zero();
  });

  std::vector<scalar_type > lsSum(MFlvol);             
  thread_for(r,MFlvol,{
    lsSum[r]=scalar_type(0.0);
  });

  int e1=    grid->_slice_nblock[orthogdim];
  int e2=    grid->_slice_block [orthogdim];
  int stride=grid->_slice_stride[orthogdim];

  thread_for(r,rd,{

    int so=r*grid->_ostride[orthogdim]; // base offset for start of plane 

    for(int n=0;n<e1;n++){
      for(int b=0;b<e2;b++){

	int ss= so+n*stride+b;

	for(int i=0;i<Lblock;i++){

	  autoView(wi_v,wi[i],CpuRead);
	  auto w = conjugate(wi_v[ss]);
	  if (g5) {
	    w()(2)(0) = - w()(2)(0);
	    w()(2)(1) = - w()(2)(1);
	    w()(2)(2) = - w()(2)(2);
	    w()(3)(0) = - w()(3)(0);
	    w()(3)(1) = - w()(3)(1);
	    w()(3)(2) = - w()(3)(2);
	  }
	  for(int j=0;j<Rblock;j++){
	    
	    autoView(vj_v,vj[j],CpuRead);
	    auto v  = vj_v[ss];
	    auto vv = v()(0)(0);

	    vv =      w()(0)(0) * v()(0)(0)// Gamma5 Dirac basis explicitly written out
	      +       w()(0)(1) * v()(0)(1)
	      +       w()(0)(2) * v()(0)(2)
	      +       w()(1)(0) * v()(1)(0)
	      +       w()(1)(1) * v()(1)(1)
	      +       w()(1)(2) * v()(1)(2)
	      +       w()(2)(0) * v()(2)(0)
	      +       w()(2)(1) * v()(2)(1)
	      +       w()(2)(2) * v()(2)(2)
	      +       w()(3)(0) * v()(3)(0)
	      +       w()(3)(1) * v()(3)(1)
	      +       w()(3)(2) * v()(3)(2);
	    
	    int idx = i+Lblock*j+Lblock*Rblock*r;
	    lvSum[idx] = lvSum[idx]+vv;
	  }
	}
      }
    }
  });

  // Sum across simd lanes in the plane, breaking out orthog dir.
  thread_for(rt,rd,{

      Coordinate icoor(nd);
    iScalar<vector_type> temp; 
    ExtractBuffer<iScalar<scalar_type> > extracted(Nsimd);               

    for(int i=0;i<Lblock;i++){
    for(int j=0;j<Rblock;j++){

      int ij_rdx = i+Lblock*j+Lblock*Rblock*rt;

      temp._internal =lvSum[ij_rdx];
      extract(temp,extracted);

      for(int idx=0;idx<Nsimd;idx++){

	grid->iCoorFromIindex(icoor,idx);

	int ldx    = rt+icoor[orthogdim]*rd;

	int ij_ldx =i+Lblock*j+Lblock*Rblock*ldx;

	lsSum[ij_ldx]=lsSum[ij_ldx]+extracted[idx]._internal;

      }
    }}
  });

  GRID_ASSERT(mat.dimension(0) == Nt);
  // ld loop and local only??
  int pd = grid->_processors[orthogdim];
  int pc = grid->_processor_coor[orthogdim];
  thread_for_collapse(2,lt,ld,{
    for(int pt=0;pt<pd;pt++){
      int t = lt + pt*ld;
      if (pt == pc){
	for(int i=0;i<Lblock;i++){
	  for(int j=0;j<Rblock;j++){
	    int ij_dx = i + Lblock * j + Lblock * Rblock * lt;
	    mat(t,i,j) = lsSum[ij_dx];
	  }
	}
      } else { 
	const scalar_type zz(0.0);
	for(int i=0;i<Lblock;i++){
	  for(int j=0;j<Rblock;j++){
	    mat(t,i,j) =zz;
	  }
	}
      }
    }
  });

  grid->GlobalSumVector(&mat(0,0,0),Nt*Lblock*Rblock);
}

template<class FImpl>
void A2Autils<FImpl>::PionFieldWVmom(Eigen::Tensor<ComplexD,4> &mat, 
				     const FermionField *wi,
				     const FermionField *vj,
				     const std::vector<ComplexField > &mom,
				     int orthogdim) 
{
  int Lblock = mat.dimension(2); 
  int Rblock = mat.dimension(3);

  GridBase *grid = wi[0].Grid();
  
  const int    nd = grid->_ndimension;
  const int Nsimd = grid->Nsimd();

  int Nt     = grid->GlobalDimensions()[orthogdim];
  int Nmom   = mom.size();

  int fd=grid->_fdimensions[orthogdim];
  int ld=grid->_ldimensions[orthogdim];
  int rd=grid->_rdimensions[orthogdim];

  // will locally sum vectors first
  // sum across these down to scalars
  // splitting the SIMD
  int MFrvol = rd*Lblock*Rblock*Nmom;
  int MFlvol = ld*Lblock*Rblock*Nmom;

  std::vector<vector_type > lvSum(MFrvol);
  thread_for(r,MFrvol,{
    lvSum[r] = Zero();
  });

  std::vector<scalar_type > lsSum(MFlvol);             
  thread_for(r,MFlvol,{
    lsSum[r]=scalar_type(0.0);
  });

  int e1=    grid->_slice_nblock[orthogdim];
  int e2=    grid->_slice_block [orthogdim];
  int stride=grid->_slice_stride[orthogdim];

  thread_for(r,rd,{

    int so=r*grid->_ostride[orthogdim]; // base offset for start of plane 

    for(int n=0;n<e1;n++){
      for(int b=0;b<e2;b++){

	int ss= so+n*stride+b;

	for(int i=0;i<Lblock;i++){

	  autoView(wi_v,wi[i],CpuRead);
	  auto w = conjugate(wi_v[ss]);

	  for(int j=0;j<Rblock;j++){

	    autoView(vj_v,vj[j],CpuRead);
	    auto v = vj_v[ss];

	    auto vv = w()(0)(0) * v()(0)(0)// Gamma5 Dirac basis explicitly written out
	      +       w()(0)(1) * v()(0)(1)
	      +       w()(0)(2) * v()(0)(2)
	      +       w()(1)(0) * v()(1)(0)
	      +       w()(1)(1) * v()(1)(1)
	      +       w()(1)(2) * v()(1)(2)
	      -       w()(2)(0) * v()(2)(0)
	      -       w()(2)(1) * v()(2)(1)
	      -       w()(2)(2) * v()(2)(2)
	      -       w()(3)(0) * v()(3)(0)
	      -       w()(3)(1) * v()(3)(1)
	      -       w()(3)(2) * v()(3)(2);

	    
	    // After getting the sitewise product do the mom phase loop
	    int base = Nmom*i+Nmom*Lblock*j+Nmom*Lblock*Rblock*r;
	    for ( int m=0;m<Nmom;m++){
	      int idx = m+base;
	      autoView(mom_v,mom[m],CpuRead);
	      auto phase = mom_v[ss];
	      mac(&lvSum[idx],&vv,&phase()()());
	    }
	  }
	}
      }
    }
  });


  // Sum across simd lanes in the plane, breaking out orthog dir.
  thread_for(rt,rd,{

    Coordinate icoor(nd);
    iScalar<vector_type> temp; 
    ExtractBuffer<iScalar<scalar_type> > extracted(Nsimd);               

    for(int i=0;i<Lblock;i++){
    for(int j=0;j<Rblock;j++){
    for(int m=0;m<Nmom;m++){

      int ij_rdx = m+Nmom*i+Nmom*Lblock*j+Nmom*Lblock*Rblock*rt;

      temp._internal = lvSum[ij_rdx];
      extract(temp,extracted);

      for(int idx=0;idx<Nsimd;idx++){

	grid->iCoorFromIindex(icoor,idx);

	int ldx    = rt+icoor[orthogdim]*rd;

	int ij_ldx = m+Nmom*i+Nmom*Lblock*j+Nmom*Lblock*Rblock*ldx;

	lsSum[ij_ldx]=lsSum[ij_ldx]+extracted[idx]._internal;

      }
    }}}
  });

  GRID_ASSERT(mat.dimension(0) == Nmom);
  GRID_ASSERT(mat.dimension(1) == Nt);
 
  int pd = grid->_processors[orthogdim];
  int pc = grid->_processor_coor[orthogdim];
  thread_for_collapse(2,lt,ld,{
    for(int pt=0;pt<pd;pt++){
      int t = lt + pt*ld;
      if (pt == pc){
	for(int i=0;i<Lblock;i++){
	  for(int j=0;j<Rblock;j++){
	    for(int m=0;m<Nmom;m++){
	      int ij_dx = m+Nmom*i + Nmom*Lblock * j + Nmom*Lblock * Rblock * lt;
	      mat(m,t,i,j) = lsSum[ij_dx];
	    }
	  }
	}
      } else { 
	const scalar_type zz(0.0);
	for(int i=0;i<Lblock;i++){
	  for(int j=0;j<Rblock;j++){
	    for(int m=0;m<Nmom;m++){
	      mat(m,t,i,j) =zz;
	    }
	  }
	}
      }
    }
  });

  grid->GlobalSumVector(&mat(0,0,0,0),Nmom*Nt*Lblock*Rblock);
}

template<class FImpl>
void A2Autils<FImpl>::PionFieldWV(Eigen::Tensor<ComplexD,3> &mat, 
				  const FermionField *wi,
				  const FermionField *vj,
				  int orthogdim) 
{
  const int g5=1;
  PionFieldXX(mat,wi,vj,orthogdim,g5);
}
template<class FImpl>
void A2Autils<FImpl>::PionFieldWW(Eigen::Tensor<ComplexD,3> &mat, 
				  const FermionField *wi,
				  const FermionField *wj,
				  int orthogdim) 
{
  const int nog5=0;
  PionFieldXX(mat,wi,wj,orthogdim,nog5);
}
template<class FImpl>
void A2Autils<FImpl>::PionFieldVV(Eigen::Tensor<ComplexD,3> &mat, 
				  const FermionField *vi,
				  const FermionField *vj,
				  int orthogdim) 
{
  const int nog5=0;
  PionFieldXX(mat,vi,vj,orthogdim,nog5);
}
*/


// ============================================================
// GPU-accelerated extended meson field contraction.
//
// All kernels run via accelerator_for / coalescedRead/Write.
// A2ASpatialSum (batched GEMM + MPI reduction) handles the
// spatial sum; include <Grid/algorithms/blas/A2ASpatialSum.h>
// (pulled in automatically via <Grid/Grid.h>).
//
// Usage:
//   A2AExtendedMesonFieldGPU<FImpl>::compute(result,
//       left, right, loop1, loop2, gamma1, gamma2, type);
// ============================================================
template <typename FImpl>
class A2AExtendedMesonField
{
public:
  typedef typename FImpl::FermionField    FermionField;
  typedef typename FImpl::PropagatorField PropagatorField;
  typedef typename FImpl::SiteSpinor      vobj;
  typedef typename vobj::scalar_type      scalar_type;
  typedef typename vobj::vector_type      vector_type;

  typedef iSpinColourMatrix<vector_type> SpinColourMatrix_v;
  typedef iSpinColourVector<vector_type> SpinColourVector_v;
  typedef iSpinMatrix<vector_type>       SpinMatrix_v;

  // ----------------------------------------------------------
  // Kernel: loop propagator = sum_k outerProduct(loop1[k], loop2[k])
  // ----------------------------------------------------------
  static void LoopPropagator(PropagatorField &loop,
                              const std::vector<FermionField> &loop1,
                              const std::vector<FermionField> &loop2)
  {
    int Nk          = (int)loop1.size();
    uint64_t oSites = loop.Grid()->oSites();
    int Nsimd       = SpinColourVector_v::Nsimd();

    typedef decltype(loop1[0].View(AcceleratorRead)) View;
    std::vector<View> v1, v2;
    v1.reserve(Nk); v2.reserve(Nk);
    for (int k = 0; k < Nk; k++) {
      v1.push_back(loop1[k].View(AcceleratorRead));
      v2.push_back(loop2[k].View(AcceleratorRead));
    }

    deviceVector<SpinColourVector_v *> l1p(Nk), l2p(Nk);
    for (int k = 0; k < Nk; k++) {
      acceleratorPut(l1p[k], &v1[k][0]);
      acceleratorPut(l2p[k], &v2[k][0]);
    }

    autoView(loopv, loop, AcceleratorWrite);
    SpinColourVector_v **l1 = &l1p[0];
    SpinColourVector_v **l2 = &l2p[0];
    int lNk = Nk;
    accelerator_for(ss, oSites, Nsimd, {
      auto res = outerProduct(coalescedRead(l1[0][ss]), coalescedRead(l2[0][ss]));
      for (int k = 1; k < lNk; k++)
        res = res + outerProduct(coalescedRead(l1[k][ss]), coalescedRead(l2[k][ss]));
      coalescedWrite(loopv[ss], res);
    });
    for (int k = 0; k < Nk; k++) {
      v1[k].ViewClose();
      v2[k].ViewClose();
    }
  }

  // ----------------------------------------------------------
  // Type 0 loop contraction: tloop(s1,s2)(0,0) = Tr_colour[loop(s1,s2)]
  // ----------------------------------------------------------
  static void LoopContractionType0(PropagatorField &tloop,
                                        const PropagatorField &loop)
  {
    autoView(tloopv, tloop, AcceleratorWrite);
    autoView(loopv,  loop,  AcceleratorRead);
    uint64_t Osites = loop.Grid()->oSites();
    int Nsimd = SpinColourMatrix_v::Nsimd();
    accelerator_for(ss, Osites, Nsimd, {
      auto l = loopv(ss);
      auto tmp = l; tmp = Zero();
      for (int s1 = 0; s1 < Ns; ++s1)
      for (int s2 = 0; s2 < Ns; ++s2)
        tmp()(s1,s2)(0,0) = l()(s1,s2)(0,0) + l()(s1,s2)(1,1) + l()(s1,s2)(2,2);
      coalescedWrite(tloopv[ss], tmp);
    });
  }

  // ----------------------------------------------------------
  // Type 1 loop contraction: tloop = sum_mu G(g1[mu]) * loop * G(g2[mu])
  // ----------------------------------------------------------
  static void LoopContractionType1(PropagatorField &tloop,
                                        const PropagatorField &loop,
                                        const Vector<Gamma::Algebra> &gamma1,
                                        const Vector<Gamma::Algebra> &gamma2)
  {
    int ng = (int)gamma1.size();
    const Gamma::Algebra *g1 = gamma1.data();
    const Gamma::Algebra *g2 = gamma2.data();
    autoView(tloopv, tloop, AcceleratorWrite);
    autoView(loopv,  loop,  AcceleratorRead);
    uint64_t Osites = loop.Grid()->oSites();
    int Nsimd = SpinColourMatrix_v::Nsimd();
    accelerator_for(ss, Osites, Nsimd, {
      auto l = loopv(ss);
      auto tmp = l; tmp = Zero();
      for (int mu = 0; mu < ng; ++mu)
        tmp = tmp + Gamma(g1[mu]) * l * Gamma(g2[mu]);
      coalescedWrite(tloopv[ss], tmp);
    });
  }

  // ----------------------------------------------------------
  // Type 2 loop contraction:
  //   mu = (s1*Ns + s2); tloop(s1,s2)(c1,c2) = Tr_spin[G(g2[mu])*loop](c1,c2)
  // ----------------------------------------------------------
  static void LoopContractionType2(PropagatorField &tloop,
                                        const PropagatorField &loop,
                                        const Vector<Gamma::Algebra> &gamma2)
  {
    int ng = (int)gamma2.size();
    const Gamma::Algebra *g2 = gamma2.data();
    autoView(tloopv, tloop, AcceleratorWrite);
    autoView(loopv,  loop,  AcceleratorRead);
    uint64_t Osites = loop.Grid()->oSites();
    int Nsimd = SpinColourMatrix_v::Nsimd();
    accelerator_for(ss, Osites, Nsimd, {
      auto l = loopv(ss);
      auto tmp = l; tmp = Zero();
      for (int mu = 0; mu < ng; ++mu) {
        auto gtmp = Gamma(g2[mu]) * l;
        int s1 = mu / Ns;
        int s2 = mu % Ns;
        for (int c1 = 0; c1 < Nc; ++c1)
        for (int c2 = 0; c2 < Nc; ++c2)
          tmp()(s1,s2)(c1,c2) = gtmp()(0,0)(c1,c2) + gtmp()(1,1)(c1,c2)
                              + gtmp()(2,2)(c1,c2) + gtmp()(3,3)(c1,c2);
      }
      coalescedWrite(tloopv[ss], tmp);
    });
  }

  // ----------------------------------------------------------
  // Type 3 loop contraction:
  //   spinLoop(s1,s2) = Tr_colour[loop(s1,s2)]
  //   tloop(s1,s2)(0,0) = sum_mu [G(g1[mu])*spinLoop*G(g2[mu])](s1,s2)
  // ----------------------------------------------------------
  static void LoopContractionType3(PropagatorField &tloop,
                                        const PropagatorField &loop,
                                        const Vector<Gamma::Algebra> &gamma1,
                                        const Vector<Gamma::Algebra> &gamma2)
  {
    int ng = (int)gamma1.size();
    const Gamma::Algebra *g1 = gamma1.data();
    const Gamma::Algebra *g2 = gamma2.data();
    autoView(tloopv, tloop, AcceleratorWrite);
    autoView(loopv,  loop,  AcceleratorRead);
    uint64_t Osites = loop.Grid()->oSites();
    int Nsimd = SpinColourMatrix_v::Nsimd();
    accelerator_for(ss, Osites, Nsimd, {
      typedef decltype(coalescedRead(loopv[0])) calcSCMatrix;
      typedef iSpinMatrix<typename calcSCMatrix::vector_type> calcSpinMatrix;
      auto l = loopv(ss);
      calcSpinMatrix spinLoop; spinLoop = Zero();
      for (int s1 = 0; s1 < Ns; ++s1)
      for (int s2 = 0; s2 < Ns; ++s2)
        spinLoop()(s1,s2)() = l()(s1,s2)(0,0) + l()(s1,s2)(1,1) + l()(s1,s2)(2,2);
      auto tmp = l; tmp = Zero();
      for (int mu = 0; mu < ng; ++mu) {
        calcSpinMatrix tmp2 = Gamma(g1[mu]) * spinLoop * Gamma(g2[mu]);
        for (int s1 = 0; s1 < Ns; ++s1)
        for (int s2 = 0; s2 < Ns; ++s2)
          tmp()(s1,s2)(0,0) = tmp()(s1,s2)(0,0) + tmp2()(s1,s2)();
      }
      coalescedWrite(tloopv[ss], tmp);
    });
  }

  // ----------------------------------------------------------
  // Type 0 right contraction:
  //   spinLoop(s1,s2) = tloop(s1,s2)(0,0)  [already colour-traced]
  //   loopRight += sum_mu Tr[G(g2[mu])*spinLoop] * G(g1[mu])*right
  // ----------------------------------------------------------
  static void LoopRightContractionType0(FermionField &loopRight,
                                         const PropagatorField &tloop,
                                         const FermionField &right,
                                         const Vector<Gamma::Algebra> &gamma1,
                                         const Vector<Gamma::Algebra> &gamma2)
  {
    int ng = (int)gamma1.size();
    const Gamma::Algebra *g1 = gamma1.data();
    const Gamma::Algebra *g2 = gamma2.data();
    autoView(lRv, loopRight, AcceleratorWrite);
    autoView(tlv, tloop,     AcceleratorRead);
    autoView(rv,  right,     AcceleratorRead);
    uint64_t Osites = right.Grid()->oSites();
    int Nsimd = SpinColourVector_v::Nsimd();
    accelerator_for(ss, Osites, Nsimd, {
      typedef decltype(coalescedRead(rv[0]))  calcSCVector;
      typedef decltype(coalescedRead(tlv[0])) calcSCMatrix;
      typedef iSpinMatrix<typename calcSCMatrix::vector_type> calcSpinMatrix;
      auto loopm  = tlv(ss);
      auto rightv = rv(ss);
      calcSpinMatrix spinLoop; spinLoop = Zero();
      for (int s1 = 0; s1 < Ns; ++s1)
      for (int s2 = 0; s2 < Ns; ++s2)
        spinLoop()(s1,s2)() = loopm()(s1,s2)(0,0);
      calcSCVector lR; lR = Zero();
      for (int mu = 0; mu < ng; ++mu) {
        auto GLoop   = Gamma(g2[mu]) * spinLoop;
        auto trGLoop = GLoop()(0,0)() + GLoop()(1,1)() + GLoop()(2,2)() + GLoop()(3,3)();
        auto Grightv = Gamma(g1[mu]) * rightv;
        for (int s = 0; s < Ns; ++s)
        for (int c = 0; c < Nc; ++c)
          lR()(s)(c) = lR()(s)(c) + Grightv()(s)(c) * trGLoop;
      }
      coalescedWrite(lRv[ss], lR);
    });
  }

  // ----------------------------------------------------------
  // Type 1 right contraction: loopRight = tloop * right
  // ----------------------------------------------------------
  static void LoopRightContractionType1(FermionField &loopRight,
                                         const PropagatorField &tloop,
                                         const FermionField &right)
  {
    autoView(lRv, loopRight, AcceleratorWrite);
    autoView(tlv, tloop,     AcceleratorRead);
    autoView(rv,  right,     AcceleratorRead);
    uint64_t Osites = right.Grid()->oSites();
    int Nsimd = SpinColourVector_v::Nsimd();
    accelerator_for(ss, Osites, Nsimd, {
      coalescedWrite(lRv[ss], tlv(ss) * rv(ss));
    });
  }

  // ----------------------------------------------------------
  // Type 2 right contraction:
  //   mu = (s1*Ns+s2); loopRight(s)(c) += sum_{mu,c'} tloop(s1,s2)(c,c') * [G(g1[mu])*right](s)(c')
  // ----------------------------------------------------------
  static void LoopRightContractionType2(FermionField &loopRight,
                                         const PropagatorField &tloop,
                                         const FermionField &right,
                                         const Vector<Gamma::Algebra> &gamma1)
  {
    int ng = (int)gamma1.size();
    const Gamma::Algebra *g1 = gamma1.data();
    autoView(lRv, loopRight, AcceleratorWrite);
    autoView(tlv, tloop,     AcceleratorRead);
    autoView(rv,  right,     AcceleratorRead);
    uint64_t Osites = right.Grid()->oSites();
    int Nsimd = SpinColourVector_v::Nsimd();
    accelerator_for(ss, Osites, Nsimd, {
      typedef decltype(coalescedRead(rv[0])) calcSCVector;
      auto loopm  = tlv(ss);
      auto rightv = rv(ss);
      calcSCVector lR; lR = Zero();
      for (int mu = 0; mu < ng; ++mu) {
        int s1 = mu / Ns;
        int s2 = mu % Ns;
        auto Grightv = Gamma(g1[mu]) * rightv;
        for (int s = 0; s < Ns; ++s)
        for (int c = 0; c < Nc; ++c)
          lR()(s)(c) = lR()(s)(c)
                     + loopm()(s1,s2)(c,0) * Grightv()(s)(0)
                     + loopm()(s1,s2)(c,1) * Grightv()(s)(1)
                     + loopm()(s1,s2)(c,2) * Grightv()(s)(2);
      }
      coalescedWrite(lRv[ss], lR);
    });
  }

  // ----------------------------------------------------------
  // Type 3 right contraction:
  //   loopRight(s)(c) = sum_{s'} tloop(s,s')(0,0) * right(s')(c)
  // ----------------------------------------------------------
  static void LoopRightContractionType3(FermionField &loopRight,
                                         const PropagatorField &tloop,
                                         const FermionField &right)
  {
    autoView(lRv, loopRight, AcceleratorWrite);
    autoView(tlv, tloop,     AcceleratorRead);
    autoView(rv,  right,     AcceleratorRead);
    uint64_t Osites = right.Grid()->oSites();
    int Nsimd = SpinColourVector_v::Nsimd();
    accelerator_for(ss, Osites, Nsimd, {
      typedef decltype(coalescedRead(rv[0])) calcSCVector;
      auto loopm  = tlv(ss);
      auto rightv = rv(ss);
      calcSCVector lR; lR = Zero();
      for (int s = 0; s < Ns; ++s)
      for (int c = 0; c < Nc; ++c)
        lR()(s)(c) = loopm()(s,0)(0,0) * rightv()(0)(c)
                   + loopm()(s,1)(0,0) * rightv()(1)(c)
                   + loopm()(s,2)(0,0) * rightv()(2)(c)
                   + loopm()(s,3)(0,0) * rightv()(3)(c);
      coalescedWrite(lRv[ss], lR);
    });
  }

  // ----------------------------------------------------------
  // compute: GPU extended meson field for one (type, gamma pair).
  //   left   - original left vectors (conjugated during packing)
  //   loop   - pre-built loop propagator (from LoopPropagator)
  //   result[t][i][j] - rank-3 Eigen tensor (nt x N_i x N_j)
  // ----------------------------------------------------------
  template <typename TensorType>
  static void compute(
      TensorType &result,
      const std::vector<FermionField> &left,
      const std::vector<FermionField> &right,
      const PropagatorField &loop,
      const std::vector<Gamma::Algebra> &gamma1_in,
      const std::vector<Gamma::Algebra> &gamma2_in,
      int type)
  {
    GridBase *grid = loop.Grid();
    int N_i = (int)left.size();
    int N_j = (int)right.size();

    // Copy gamma arrays into UVM-accessible Vector so kernels can read on device.
    Vector<Gamma::Algebra> gamma1(gamma1_in.begin(), gamma1_in.end());
    Vector<Gamma::Algebra> gamma2(gamma2_in.begin(), gamma2_in.end());

    PropagatorField tloop(grid);
    tloop = Zero();
    switch (type) {
    case 0: LoopContractionType0(tloop, loop);                 break;
    case 1: LoopContractionType1(tloop, loop, gamma1, gamma2); break;
    case 2: LoopContractionType2(tloop, loop, gamma2);         break;
    case 3: LoopContractionType3(tloop, loop, gamma1, gamma2); break;
    default: assert(0 && "A2AExtendedMesonFieldGPU: unknown contraction type"); break;
    }

    std::vector<FermionField> loopRight(N_j, grid);
    for (int j = 0; j < N_j; j++) {
      switch (type) {
      case 0: LoopRightContractionType0(loopRight[j], tloop, right[j], gamma1, gamma2); break;
      case 1: LoopRightContractionType1(loopRight[j], tloop, right[j]);                 break;
      case 2: LoopRightContractionType2(loopRight[j], tloop, right[j], gamma1);         break;
      case 3: LoopRightContractionType3(loopRight[j], tloop, right[j]);                 break;
      default: assert(0 && "A2AExtendedMesonFieldGPU: unknown contraction type"); break;
      }
    }

    A2ASpatialSum<SpinColourVector_v> spatial_sum;
    spatial_sum.Allocate(N_i, N_j, grid);
    spatial_sum.PackLeftConj(left);
    spatial_sum.PackRight(loopRight);
    spatial_sum.Sum(result);
  }
};

// ============================================================
// GPU-accelerated chromo-magnetic operator field contraction.
//
// G[munu] (colour space, LatticeColourMatrix) and Sigma[munu]
// (spin space, Gamma::Algebra) are kept separate throughout,
// avoiding a PropagatorField intermediate. CMOContractRight
// accumulates the full sum_munu G*Sigma operator into each
// modified right vector via accelerator_for; A2ASpatialSum
// (batched GEMM + MPI reduction) then handles the spatial sum.
//
// Usage (no blocking):
//   A2AChromoMagneticOperator<GImpl,FImpl>::compute(result,
//       left, right, U, ifOrthog, parity);
// ============================================================
template <typename GImpl, typename FImpl>
class A2AChromoMagneticOperator
{
public:
  typedef typename GImpl::GaugeLinkField GaugeMat;
  typedef typename GImpl::GaugeField     GaugeField;
  typedef typename FImpl::FermionField   FermionField;
  typedef typename FImpl::SiteSpinor     vobj;
  typedef typename vobj::scalar_type     scalar_type;
  typedef typename vobj::vector_type     vector_type;

  typedef iSpinColourVector<vector_type> SpinColourVector_v;
  typedef iColourMatrix<vector_type>     ColourMatrix_v;

  // ----------------------------------------------------------
  // ifOrthog=0: spatial-temporal pairs (x,t),(y,t),(z,t)
  //             -> G[0..2] anti-Hermitian, Sigma = SigmaXT,SigmaYT,SigmaZT
  // ----------------------------------------------------------
  static void CMOContraction0(std::vector<GaugeMat> &G,
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
      WilsonLoops<GImpl>::CloverleafMxN(Gmunu, Umu, Unu, mus[idx], nus[idx], 1, 1);
      G.push_back(Gmunu - adj(Gmunu));
      Sigma[idx] = sigs[idx];
    }
  }

  // ----------------------------------------------------------
  // ifOrthog=1: spatial-spatial pairs (x,y),(x,z),(y,z)
  //             -> G[0..2] anti-Hermitian, Sigma = SigmaXY,SigmaXZ,SigmaYZ
  // ----------------------------------------------------------
  static void CMOContraction1(std::vector<GaugeMat> &G,
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
      WilsonLoops<GImpl>::CloverleafMxN(Gmunu, Umu, Unu, mus[idx], nus[idx], 1, 1);
      G.push_back(Gmunu - adj(Gmunu));
      Sigma[idx] = sigs[idx];
    }
  }

  // ----------------------------------------------------------
  // GPU kernel: for each right vector accumulate
  //   loopRight(x) = sum_munu G[munu](x)*(parity?G5:1)*Gamma(Sigma[munu])*right(x)
  // G (colour) and Sigma (spin) are kept separate; all temporaries
  // are per-thread register SpinColourVectors - no PropagatorField.
  // ----------------------------------------------------------
  static void CMOContractRight(FermionField &loopRight,
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

  // ----------------------------------------------------------
  // compute: GPU CMO field for one (ifOrthog, parity) pair.
  //   No blocking - processes all N_i x N_j vectors at once.
  //   result[t][i][j] - rank-3 Eigen tensor (nt x N_i x N_j)
  // ----------------------------------------------------------
  template <typename TensorType>
  static void compute(
      TensorType &result,
      const std::vector<FermionField> &left,
      const std::vector<FermionField> &right,
      const GaugeField &U,
      int ifOrthog,
      int parity)
  {
    GridBase *grid = left[0].Grid();
    int N_i = (int)left.size();
    int N_j = (int)right.size();

    std::vector<GaugeMat>  G;
    Vector<Gamma::Algebra> Sigma;
    if (ifOrthog == 0) CMOContraction0(G, Sigma, U);
    else               CMOContraction1(G, Sigma, U);

    std::vector<FermionField> loopRight(N_j, grid);
    for (int j = 0; j < N_j; j++)
      CMOContractRight(loopRight[j], G, Sigma, right[j], parity);

    A2ASpatialSum<SpinColourVector_v> spatial_sum;
    spatial_sum.Allocate(N_i, N_j, grid);
    spatial_sum.PackLeftConj(left);
    spatial_sum.PackRight(loopRight);
    spatial_sum.Sum(result);
  }
};

NAMESPACE_END(Grid);

