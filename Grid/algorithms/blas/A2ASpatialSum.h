/*************************************************************************************

    Grid physics library, www.github.com/paboyle/Grid

    Source file: Grid/algorithms/blas/A2ASpatialSum.h

    Copyright (C) 2025

Author: Peter Boyle <pboyle@bnl.gov>

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
    -> C[i,j] = sum_k W[i][k] * LR[j][k] = EMF[i,j]*/
template<class vobj>
class A2ASpatialSum
{
public:
  typedef typename vobj::scalar_type   scalar;
  typedef typename vobj::scalar_object sobj;

  GridBase *grid;
  int N_i, N_j;
  int nt, nxyz, Nsc;

  deviceVector<scalar>   W_buf;
  deviceVector<scalar>   LR_buf;
  deviceVector<scalar>   EMF_buf;
  deviceVector<scalar *> W_ptrs;
  deviceVector<scalar *> LR_ptrs;
  deviceVector<scalar *> EMF_ptrs;

  A2ASpatialSum() : grid(nullptr), N_i(0), N_j(0), nt(0), nxyz(0), Nsc(0) {}

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
  void Sum(Eigen::Tensor<ComplexD, 3> &result)
  {
    GridBLAS BLAS;

    int K = nxyz * Nsc;
    BLAS.gemmBatched(GridBLAS_OP_T, GridBLAS_OP_N,
                     N_i, N_j, K,
                     scalar(1.0),
                     W_ptrs,
                     LR_ptrs,
                     scalar(0.0),
                     EMF_ptrs);
    BLAS.synchronise();

    // Copy from device and distribute into global-t layout
    int nt_global = result.dimension(0);
    int nd        = grid->Nd();
    int lt_start  = grid->LocalStarts()[nd - 1];

    std::vector<scalar> host_emf(nt * N_j * N_i);
    acceleratorCopyFromDevice(&EMF_buf[0], host_emf.data(),
                              nt * N_j * N_i * sizeof(scalar));

    // EMF_buf[t][j*N_i + i] = EMF[i,j] for local t
    std::vector<scalar> global_emf(nt_global * N_i * N_j, scalar(0.0));
    for (int lt = 0; lt < nt; lt++) {
      int gt = lt + lt_start;
      for (int i = 0; i < N_i; i++)
      for (int j = 0; j < N_j; j++)
        global_emf[gt * N_i * N_j + i * N_j + j] = host_emf[lt * N_j * N_i + j * N_i + i];
    }
    grid->GlobalSumVector(global_emf.data(), nt_global * N_i * N_j);

    for (int gt = 0; gt < nt_global; gt++)
    for (int i  = 0; i  < N_i;      i++)
    for (int j  = 0; j  < N_j;      j++)
      result(gt, i, j) = global_emf[gt * N_i * N_j + i * N_j + j];
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
  // One thread per (j, l_xyz); inner loop over (t, sc) is sequential.
  void ApplyPhaseRight(const deviceVector<scalar> &phase_buf)
  {
    scalar       *LR  = &LR_buf[0];
    const scalar *ph  = &phase_buf[0];
    int lN_j = N_j, lnxyz = nxyz, lNsc = Nsc, lnt = nt;
    accelerator_for(idx, (size_t)(lN_j * lnxyz), 1, {
      int    j     = idx / lnxyz;
      int    l_xyz = idx % lnxyz;
      scalar ph_val = ph[l_xyz];
      for (int t = 0; t < lnt; t++) {
        int64_t base = (int64_t)t * lN_j * lnxyz * lNsc
                     + (int64_t)j * lnxyz * lNsc
                     + l_xyz * lNsc;
        for (int sc = 0; sc < lNsc; sc++)
          LR[base + sc] *= ph_val;
      }
    });
  }

  // Multiply LR_buf by conj(phase_buf[l_xyz]) - undoes ApplyPhaseRight exactly.
  void RestorePhaseRight(const deviceVector<scalar> &phase_buf)
  {
    scalar       *LR  = &LR_buf[0];
    const scalar *ph  = &phase_buf[0];
    int lN_j = N_j, lnxyz = nxyz, lNsc = Nsc, lnt = nt;
    accelerator_for(idx, (size_t)(lN_j * lnxyz), 1, {
      int    j     = idx / lnxyz;
      int    l_xyz = idx % lnxyz;
      scalar ph_val = Grid::conjugate(ph[l_xyz]);
      for (int t = 0; t < lnt; t++) {
        int64_t base = (int64_t)t * lN_j * lnxyz * lNsc
                     + (int64_t)j * lnxyz * lNsc
                     + l_xyz * lNsc;
        for (int sc = 0; sc < lNsc; sc++)
          LR[base + sc] *= ph_val;
      }
    });
  }

};

NAMESPACE_END(Grid);
