/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: Grid/algorithms/blas/A2ASpatialSum.h

    Copyright (C) 2025

Author: Peter Boyle <pboyle@bnl.gov>
Author: Jonas Hildebrand <jonas.hildebrand@uconn.edu>

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
/*  END LEGAL */
#pragma once

NAMESPACE_BEGIN(Grid);

/*
  A2ASpatialSum

  Replaces the scalar spatial accumulation loop in A2A extended meson field
  contractions with a batched GEMM over local time slices, enabling GPU offload.

  Given:
    leftv[N_i][osite]    - conjugated left SpinColourVectors (SIMD-packed)
    loopRight[N_j][osite]- type-contracted right SpinColourVectors (SIMD-packed)

  Computes:
    EMF[i,j,t] = sum_{x,s,c} leftv[i][x,t,s,c] * loopRight[j][x,t,s,c]

  via batched GEMM over nt local time slices, then GlobalSumVector across MPI.

  Memory layout (all C row-major):
    W_buf  [nt][N_i][nxyz*Nsc]  - W[t][i][x*Nsc+sc]  = leftv[i] at (x,t)
    LR_buf [nt][N_j][nxyz*Nsc]  - LR[t][j][x*Nsc+sc] = loopRight[j] at (x,t)
    EMF_buf[nt][N_j][N_i]       - column-major result; EMF[i,j,t] = EMF_buf[t][j][i]

  BLAS call (column-major, OP_T on A so A is read as W[i][k]):
    C = A^T * B  where A=W[N_ixK C-row], B=LR[N_jxK C-row], C=[N_jxN_i C-row]
    -> C[i,j] = sum_k W[i][k] * LR[j][k] = EMF[i,j]

  Multi-momentum path (SumAllMomenta): folds a momentum index into the
  GEMM's N dimension instead of redoing the whole GEMM once per momentum.
  ApplyPhasesRight reads the single unphased pack in LR_buf (built by the
  existing PackRight, unchanged) and writes nmom phase-multiplied copies
  into a separate, wider buffer:

    LR_mom_buf [nt][nmom][N_j][nxyz*Nsc] - LRM[t][m][j][x*Nsc+sc]
                                            = LR_buf[t][j][x*Nsc+sc] * phase_m[x]
    EMF_mom_buf[nt][nmom*N_j][N_i]       - same layout as EMF_buf, N widened

  K and M (nxyz*Nsc and N_i) are unchanged; only N widens from N_j to
  nmom*N_j, and m is the slower-varying sub-index within that widened N so
  each timeslice's LR_mom_buf slice is nmom contiguous [N_j][nxyz*Nsc]
  blocks -- exactly what gemmBatched wants, no further repacking.
  LR_buf/EMF_buf and W_buf/W_ptrs are untouched by this path, so existing
  single-momentum callers (EMF, CMF, Sum()) are unaffected.
*/
template<class vobj>
class A2ASpatialSum
{
public:
  typedef typename vobj::scalar_type   scalar;
  typedef typename vobj::scalar_object sobj;

  GridBase *grid;
  int N_i, N_j;
  int nt, nxyz, Nsc;
  int nmom;

  deviceVector<scalar>   W_buf;
  deviceVector<scalar>   LR_buf;
  deviceVector<scalar>   EMF_buf;
  deviceVector<scalar *> W_ptrs;
  deviceVector<scalar *> LR_ptrs;
  deviceVector<scalar *> EMF_ptrs;

  // Multi-momentum path: built alongside, not instead of, the buffers
  // above -- see class comment.
  deviceVector<scalar>   LR_mom_buf;
  deviceVector<scalar>   EMF_mom_buf;
  deviceVector<scalar *> LR_mom_ptrs;
  deviceVector<scalar *> EMF_mom_ptrs;

  A2ASpatialSum() : grid(nullptr), N_i(0), N_j(0), nt(0), nxyz(0), Nsc(0), nmom(1) {}

  void Allocate(int _N_i, int _N_j, GridBase *_grid)
  {
    grid = _grid;
    N_i  = _N_i;
    N_j  = _N_j;
    Coordinate ldims = grid->LocalDimensions();
    nt   = ldims[grid->Nd() - 1];
    nxyz = grid->lSites() / nt;
    Nsc  = sizeof(sobj) / sizeof(scalar);
  
    W_buf.resize(nt * N_i * nxyz * Nsc);
    LR_buf.resize(nt * N_j * nxyz * Nsc);
    EMF_buf.resize(nt * N_j * N_i);
  
    // Build persistent batch pointer arrays
    W_ptrs.resize(nt);
    LR_ptrs.resize(nt);
    EMF_ptrs.resize(nt);
    scalar *Wh   = &W_buf[0];
    scalar *LRh  = &LR_buf[0];
    scalar *EMFh = &EMF_buf[0];
    int lN_i = N_i, lN_j = N_j, lnxyz = nxyz, lNsc = Nsc;
    for (int t = 0; t < nt; t++) {
      acceleratorPut(W_ptrs[t],   Wh   + t * lN_i * lnxyz * lNsc);
      acceleratorPut(LR_ptrs[t],  LRh  + t * lN_j * lnxyz * lNsc);
      acceleratorPut(EMF_ptrs[t], EMFh + t * lN_j * lN_i);
    }
  }

  void AllocateRight(int _N_j, GridBase *_grid)
  {
    grid = _grid;
    N_j  = _N_j;
    Coordinate ldims = grid->LocalDimensions();
    nt   = ldims[grid->Nd() - 1];
    nxyz = grid->lSites() / nt;
    Nsc  = sizeof(sobj) / sizeof(scalar);

    size_t LRsz = (size_t)nt * N_j * nxyz * Nsc;
    if (LR_buf.size()  < LRsz)        LR_buf.resize(LRsz);
    if (LR_ptrs.size() < (size_t)nt)  LR_ptrs.resize(nt);

    scalar *LRh = &LR_buf[0];
    int lN_j = N_j, lnxyz = nxyz, lNsc = Nsc;
    for (int t = 0; t < nt; t++)
      acceleratorPut(LR_ptrs[t], LRh + t * lN_j * lnxyz * lNsc);
  }

  // Multi-momentum overload: delegates to the two-argument AllocateRight
  // above for everything it already does, then unconditionally builds
  // LR_mom_buf/LR_mom_ptrs (SumAllMomenta's right operand). Callers of the
  // two-argument overload never reach this and pay nothing for it.
  void AllocateRight(int _N_j, GridBase *_grid, int _nmom)
  {
    AllocateRight(_N_j, _grid);
    nmom = _nmom;

    size_t LRMsz = (size_t)nt * nmom * N_j * nxyz * Nsc;
    if (LR_mom_buf.size()  < LRMsz)       LR_mom_buf.resize(LRMsz);
    if (LR_mom_ptrs.size() < (size_t)nt)  LR_mom_ptrs.resize(nt);

    scalar *LRMh  = &LR_mom_buf[0];
    int     lN_j  = N_j, lnxyz = nxyz, lNsc = Nsc, lnmom = nmom;
    for (int t = 0; t < nt; t++)
      acceleratorPut(LR_mom_ptrs[t], LRMh + (size_t)t * lnmom * lN_j * lnxyz * lNsc);
  }

  // AllocateRight must be called first: N_j, nt, nxyz, Nsc must be set.
  void AllocateLeft(int _N_i)
  {
    N_i = _N_i;

    size_t Wsz   = (size_t)nt * N_i * nxyz * Nsc;
    size_t EMFsz = (size_t)nt * N_j * N_i;
    if (W_buf.size()    < Wsz)         W_buf.resize(Wsz);
    if (EMF_buf.size()  < EMFsz)       EMF_buf.resize(EMFsz);
    if (W_ptrs.size()   < (size_t)nt)  W_ptrs.resize(nt);
    if (EMF_ptrs.size() < (size_t)nt)  EMF_ptrs.resize(nt);

    scalar *Wh   = &W_buf[0];
    scalar *EMFh = &EMF_buf[0];
    int lN_i = N_i, lN_j = N_j, lnxyz = nxyz, lNsc = Nsc;
    for (int t = 0; t < nt; t++) {
      acceleratorPut(W_ptrs[t],   Wh   + t * lN_i * lnxyz * lNsc);
      acceleratorPut(EMF_ptrs[t], EMFh + t * lN_j * lN_i);
    }
  }

  // Multi-momentum overload: delegates to the one-argument AllocateLeft
  // above, then unconditionally builds EMF_mom_buf/EMF_mom_ptrs. The
  // three-argument AllocateRight must have been called first so nmom is
  // set; _nmom here is checked against it rather than trusted blindly.
  void AllocateLeft(int _N_i, int _nmom)
  {
    AllocateLeft(_N_i);
    GRID_ASSERT(_nmom == nmom);

    size_t EMFMsz = (size_t)nt * nmom * N_j * N_i;
    if (EMF_mom_buf.size()  < EMFMsz)      EMF_mom_buf.resize(EMFMsz);
    if (EMF_mom_ptrs.size() < (size_t)nt)  EMF_mom_ptrs.resize(nt);

    scalar *EMFMh = &EMF_mom_buf[0];
    int     lN_j  = N_j, lnmom = nmom;
    for (int t = 0; t < nt; t++)
      acceleratorPut(EMF_mom_ptrs[t], EMFMh + (size_t)t * lnmom * lN_j * N_i);
  }

  void PackLeft(const std::vector<Lattice<vobj>> &leftv, int start = 0, int count = -1)
  {
    if (count < 0) count = (int)leftv.size();
    GRID_ASSERT(start + count <= (int)leftv.size());
    GRID_ASSERT(count == N_i);
    PackVectors(leftv, &W_buf[0], N_i, start);
  }

  void PackRight(const std::vector<Lattice<vobj>> &loopRight, int start = 0, int count = -1)
  {
    if (count < 0) count = (int)loopRight.size();
    GRID_ASSERT(start + count <= (int)loopRight.size());
    GRID_ASSERT(count == N_j);
    PackVectors(loopRight, &LR_buf[0], N_j, start);
  }

  // Read directly from original (unconjugated) left vectors, conjugating during pack.
  void PackLeftConj(const std::vector<Lattice<vobj>> &left, int start = 0, int count = -1)
  {
    if (count < 0) count = (int)left.size();
    GRID_ASSERT(start + count <= (int)left.size());
    GRID_ASSERT(count == N_i);
    PackVectors<true>(left, &W_buf[0], N_i, start);
  }

public:
  // Pack vecs[start..start+N-1] lattice fields into buf[nt][N][nxyz*Nsc], extracting all SIMD lanes.
  // DoConj=true conjugates each element during extraction (used by PackLeftConj).
  template<bool DoConj = false>
  void PackVectors(const std::vector<Lattice<vobj>> &vecs, scalar *buf, int N, int start = 0)
  {
    int nd     = grid->_ndimension;
    int osites = grid->oSites();
    int Nsimd  = vobj::Nsimd();
    int lN     = N;
    int lNsc   = Nsc;
    int lnxyz  = nxyz;
    Coordinate rdimensions = grid->_rdimensions;
    Coordinate ldims       = grid->LocalDimensions();
    Coordinate simd        = grid->_simd_layout;

    for (int n = 0; n < N; n++) {
      autoView(src_v, vecs[start + n], AcceleratorRead);
      accelerator_for(sf, osites, Nsimd, {
#ifdef GRID_SIMT
        {
          int lane = acceleratorSIMTlane(Nsimd);
#else
          for (int lane = 0; lane < Nsimd; lane++) {
#endif
          Coordinate icoor(nd), ocoor(nd), lcoor(nd);
          Lexicographic::CoorFromIndex(icoor, lane, simd);
          Lexicographic::CoorFromIndex(ocoor, sf, rdimensions);
          for (int d = 0; d < nd; d++)
            lcoor[d] = rdimensions[d] * icoor[d] + ocoor[d];

          int     l_t = lcoor[nd - 1];
          Coordinate xyz_coor = lcoor;
          xyz_coor[nd - 1] = 0;
          int64_t l_xyz;
          Lexicographic::IndexFromCoor(xyz_coor, l_xyz, ldims);

          sobj    data   = extractLane(lane, src_v[sf]);
          if constexpr (DoConj) data = conjugate(data);
          scalar *data_s = (scalar *)&data;

          int64_t base = (int64_t)l_t * lN * lnxyz * lNsc
                       + (int64_t)n   * lnxyz * lNsc
                       + l_xyz * lNsc;
          for (int sc = 0; sc < lNsc; sc++)
            buf[base + sc] = data_s[sc];
        }
      });
    }
  }

public:

  // Batched GEMM + MPI reduction -> result[nt_global][N_i][N_j]
  //
  // BLAS (column-major, OP_T on A):
  //   C[N_jxN_i] = A^T[N_ixK] * B[N_jxK]    with K=nxyz*Nsc
  //   reading A as C row-major [N_i][K] and B as C row-major [N_j][K]
  //   -> C[i,j] = sum_k W[i,k] * LR[j,k] = EMF[i,j]
  //
  // result's layout defaults to ColMajor (Eigen::Tensor's own default) so
  // every existing caller is unaffected; callers whose consumer expects
  // j-fastest (e.g. writing into a RowMajor A2AMatrixSet for HDF5) should
  // pass a RowMajor result -- this loop's writes are then contiguous
  // instead of striding by nt_global*N_i per j, with no code change here.
  // timings[0]: GEMM + synchronise
  // timings[1]: device->host copy
  // timings[2]: transpose (host_emf -> global_emf)
  // timings[3]: GlobalSumVector
  // timings[4]: transpose (global_emf -> result)
  template <int Layout = Eigen::ColMajor>
  void Sum(Eigen::Tensor<ComplexD, 3, Layout> &result,
           std::array<double, 5> *timings = nullptr)
  {
    GridBLAS BLAS;
    double dt;

    int K = nxyz * Nsc;
    dt = -usecond();
    BLAS.gemmBatched(GridBLAS_OP_T, GridBLAS_OP_N,
                     N_i, N_j, K,
                     scalar(1.0),
                     W_ptrs,
                     LR_ptrs,
                     scalar(0.0),
                     EMF_ptrs);
    BLAS.synchronise();
    dt += usecond();
    if (timings) (*timings)[0] += dt;

    int nt_global = result.dimension(0);
    int nd        = grid->Nd();
    int lt_start  = grid->LocalStarts()[nd - 1];

    std::vector<scalar> host_emf(nt * N_j * N_i);
    dt = -usecond();
    acceleratorCopyFromDevice(&EMF_buf[0], host_emf.data(),
                              nt * N_j * N_i * sizeof(scalar));
    dt += usecond();
    if (timings) (*timings)[1] += dt;

    // Both loops are pure host-side CPU work; loop nests are perfectly nested
    // for collapse(3) following the thread_for_collapse precedent at A2Autils.h:1503.
    std::vector<scalar> global_emf(nt_global * N_i * N_j, scalar(0.0));
    dt = -usecond();
    thread_for_collapse(3, lt, nt, {
        for (int i = 0; i < N_i; i++)
        for (int j = 0; j < N_j; j++)
          global_emf[((int)lt + lt_start) * N_i * N_j + i * N_j + j]
              = host_emf[(int)lt * N_j * N_i + j * N_i + i];
    });
    dt += usecond();
    if (timings) (*timings)[2] += dt;

    dt = -usecond();
    grid->GlobalSumVector(global_emf.data(), nt_global * N_i * N_j);
    dt += usecond();
    if (timings) (*timings)[3] += dt;

    dt = -usecond();
    thread_for_collapse(3, gt, nt_global, {
        for (int i = 0; i < N_i; i++)
        for (int j = 0; j < N_j; j++)
          result((int)gt, i, j) = global_emf[(int)gt * N_i * N_j + i * N_j + j];
    });
    dt += usecond();
    if (timings) (*timings)[4] += dt;
  }

  // Unpack a ComplexField phase into a flat array of one scalar per spatial site l_xyz.
  // ph is assumed time-independent; all t-layers write the same value so redundant
  // writes across timeslices are safe.  Mirrors the PackVectors SIMD/SIMT extraction.
  template<class phvobj>
  static void PackPhase(GridBase *_grid, const Lattice<phvobj> &ph,
                        deviceVector<scalar> &phase_buf)
  {
    int nd     = _grid->_ndimension;
    int lnt    = _grid->LocalDimensions()[nd - 1];
    int lnxyz  = _grid->lSites() / lnt;
    int osites = _grid->oSites();
    int lNsimd = _grid->Nsimd();

    phase_buf.resize(lnxyz);
    scalar *phase_data = &phase_buf[0];

    Coordinate rdimensions = _grid->_rdimensions;
    Coordinate ldims       = _grid->LocalDimensions();
    Coordinate simd_layout = _grid->_simd_layout;

    autoView(ph_v, ph, AcceleratorRead);

    accelerator_for(sf, osites, lNsimd, {
#ifdef GRID_SIMT
      {
        int lane = acceleratorSIMTlane(lNsimd);
#else
        for (int lane = 0; lane < lNsimd; lane++) {
#endif
        Coordinate icoor(nd), ocoor(nd), lcoor(nd);
        Lexicographic::CoorFromIndex(icoor, lane, simd_layout);
        Lexicographic::CoorFromIndex(ocoor, sf, rdimensions);
        for (int d = 0; d < nd; d++)
          lcoor[d] = rdimensions[d] * icoor[d] + ocoor[d];

        Coordinate xyz_coor = lcoor;
        xyz_coor[nd - 1]    = 0;
        int64_t l_xyz;
        Lexicographic::IndexFromCoor(xyz_coor, l_xyz, ldims);

        auto    ph_site = extractLane(lane, ph_v[sf]);
        scalar *ph_s    = (scalar *)&ph_site;
        phase_data[l_xyz] = ph_s[0];
      }
    });
  }

  // Multiply LR_buf[t][j][l_xyz*Nsc + sc] by phase_buf[l_xyz] for all (t, j, sc).
  // Nsc lanes per (j, l_xyz) pair: adjacent lanes access consecutive sc values -> stride-1 coalesced.
  void ApplyPhaseRight(const deviceVector<scalar> &phase_buf)
  {
    scalar       *LR  = &LR_buf[0];
    const scalar *ph  = &phase_buf[0];
    int lN_j = N_j, lnxyz = nxyz, lNsc = Nsc, lnt = nt;
    accelerator_for(idx, (size_t)(lN_j * lnxyz), lNsc, {
      int    j     = idx / lnxyz;
      int    l_xyz = idx % lnxyz;
      int    sc    = acceleratorSIMTlane(lNsc);
      scalar ph_val = ph[l_xyz];
      for (int t = 0; t < lnt; t++) {
        int64_t base = (int64_t)t * lN_j * lnxyz * lNsc
                     + (int64_t)j * lnxyz * lNsc
                     + l_xyz * lNsc;
        LR[base + sc] *= ph_val;
      }
    });
  }

  // Read the unphased pack in LR_buf (built by PackRight, untouched by this
  // path) and write nmom phase-multiplied copies into LR_mom_buf[t][m][j][
  // l_xyz*Nsc+sc]. One kernel launch, with m folded into the parallel index
  // space alongside (j, l_xyz) -- not nmom separate launches. Requires the
  // three-argument AllocateRight to have been called first.
  void ApplyPhasesRight(const std::vector<deviceVector<scalar>> &phase_bufs)
  {
    GRID_ASSERT((int)phase_bufs.size() == nmom);

    deviceVector<scalar *> ph_ptrs(nmom);
    for (int m = 0; m < nmom; m++)
      acceleratorPut(ph_ptrs[m], const_cast<scalar *>(&phase_bufs[m][0]));

    const scalar *LR  = &LR_buf[0];
    scalar       *LRM = &LR_mom_buf[0];
    scalar      **ph  = &ph_ptrs[0];
    int lN_j = N_j, lnxyz = nxyz, lNsc = Nsc, lnt = nt, lnmom = nmom;

    accelerator_for(idx, (size_t)(lnmom * lN_j * lnxyz), lNsc, {
      int    m      = idx / (lN_j * lnxyz);
      int    rem    = idx % (lN_j * lnxyz);
      int    j      = rem / lnxyz;
      int    l_xyz  = rem % lnxyz;
      int    sc     = acceleratorSIMTlane(lNsc);
      scalar ph_val = ph[m][l_xyz];
      for (int t = 0; t < lnt; t++) {
        int64_t src = (int64_t)t * lN_j * lnxyz * lNsc
                    + (int64_t)j * lnxyz * lNsc
                    + l_xyz * lNsc;
        int64_t dst = (int64_t)t * lnmom * lN_j * lnxyz * lNsc
                    + (int64_t)m * lN_j * lnxyz * lNsc
                    + (int64_t)j * lnxyz * lNsc
                    + l_xyz * lNsc;
        LRM[dst + sc] = LR[src + sc] * ph_val;
      }
    });
  }

  // Batched GEMM + MPI reduction, folding momentum into N: same K=nxyz*Nsc
  // and M=N_i as Sum(), N widens from N_j to nmom*N_j by reading
  // LR_mom_ptrs/EMF_mom_ptrs (built by ApplyPhasesRight and the
  // three-argument Allocate* overloads) instead of LR_ptrs/EMF_ptrs. One
  // GEMM and one GlobalSumVector for the whole block, covering every
  // momentum, instead of one pair per momentum via Sum().
  //
  // result[nt_global][N_i][N_j][nmom]; timings[] slots match Sum()'s.
  template <int Layout = Eigen::ColMajor>
  void SumAllMomenta(Eigen::Tensor<ComplexD, 4, Layout> &result,
                      std::array<double, 5> *timings = nullptr)
  {
    GridBLAS BLAS;
    double dt;

    int K     = nxyz * Nsc;
    int Nwide = nmom * N_j;

    dt = -usecond();
    BLAS.gemmBatched(GridBLAS_OP_T, GridBLAS_OP_N,
                     N_i, Nwide, K,
                     scalar(1.0),
                     W_ptrs,
                     LR_mom_ptrs,
                     scalar(0.0),
                     EMF_mom_ptrs);
    BLAS.synchronise();
    dt += usecond();
    if (timings) (*timings)[0] += dt;

    int nt_global = result.dimension(0);
    int nd        = grid->Nd();
    int lt_start  = grid->LocalStarts()[nd - 1];

    std::vector<scalar> host_emf((size_t)nt * Nwide * N_i);
    dt = -usecond();
    acceleratorCopyFromDevice(&EMF_mom_buf[0], host_emf.data(),
                              (size_t)nt * Nwide * N_i * sizeof(scalar));
    dt += usecond();
    if (timings) (*timings)[1] += dt;

    std::vector<scalar> global_emf((size_t)nt_global * N_i * Nwide, scalar(0.0));
    dt = -usecond();
    thread_for_collapse(3, lt, nt, {
        for (int i = 0; i < N_i; i++)
        for (int col = 0; col < Nwide; col++)
          global_emf[((int)lt + lt_start) * N_i * Nwide + i * Nwide + col]
              = host_emf[(int)lt * Nwide * N_i + col * N_i + i];
    });
    dt += usecond();
    if (timings) (*timings)[2] += dt;

    dt = -usecond();
    grid->GlobalSumVector(global_emf.data(), (size_t)nt_global * N_i * Nwide);
    dt += usecond();
    if (timings) (*timings)[3] += dt;

    dt = -usecond();
    thread_for_collapse(3, gt, nt_global, {
        for (int i = 0; i < N_i; i++)
        for (int col = 0; col < Nwide; col++) {
          int m = col / N_j;
          int j = col % N_j;
          result((int)gt, i, j, m) = global_emf[(int)gt * N_i * Nwide + i * Nwide + col];
        }
    });
    dt += usecond();
    if (timings) (*timings)[4] += dt;
  }

};

NAMESPACE_END(Grid);
