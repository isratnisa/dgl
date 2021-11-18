/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/sddmm.cuh
 * \brief SDDMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_SDDMM_CUH_
#define DGL_ARRAY_CUDA_SDDMM_CUH_

#include <dgl/bcast.h>
#include "macro.cuh"
#include "atomic.cuh"
#include "functor.cuh"
#include "fp16.cuh"
#include "./utils.h"
#include "../selector.h"
#include "../../runtime/cuda/cuda_common.h"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {

constexpr unsigned int full_mask = 0xffffffff;

/*!
 * \brief CUDA kernel of g-SDDMM on Coo format.
 * \note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different positions
 *       in feature dimension.
 */
template <typename Idx, typename DType, typename BinaryOp,
          bool UseBcast = false, bool UseIdx = false,
          int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCooKernel(
  const DType* __restrict__ lhs,
  const DType* __restrict__ rhs,
  DType* __restrict__ out,
  const Idx* __restrict__ row,
  const Idx* __restrict__ col,
  const Idx* __restrict__ edge_map,
  int64_t N, int64_t M, int64_t E, int64_t reduce_size,
  const int64_t* __restrict__ lhs_off,
  const int64_t* __restrict__ rhs_off,
  int64_t lhs_len, int64_t rhs_len, int64_t out_len) {
  // SDDMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    const DType* lhsoff = BinaryOp::use_lhs ?
      (lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len): nullptr;
    const DType* rhsoff = BinaryOp::use_rhs ?
      (rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len): nullptr;
    DType* outoff = out + eid * out_len;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride_x = blockDim.x * gridDim.x;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? lhs_off[tx] : tx;
      const Idx rhs_add = UseBcast ? rhs_off[tx] : tx;
      DType val = BinaryOp::Call(
          lhsoff + lhs_add * reduce_size,
          rhsoff + rhs_add * reduce_size,
          reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA kernel of SDDMM-dot on Coo format, accelerated with tree reduction.
 * \note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different positions
 *       in feature dimension.
 */
template <typename Idx, typename DType,
          bool UseBcast = false, bool UseIdx = false,
          int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCooTreeReduceKernel(
  const DType* __restrict__ lhs,
  const DType* __restrict__ rhs,
  DType* __restrict__ out,
  const Idx* __restrict__ row,
  const Idx* __restrict__ col,
  const Idx* __restrict__ edge_map,
  int64_t N, int64_t M, int64_t E, int64_t reduce_size,
  const int64_t* __restrict__ lhs_off,
  const int64_t* __restrict__ rhs_off,
  int64_t lhs_len, int64_t rhs_len, int64_t out_len) {
  Idx ty = blockIdx.x * blockDim.y + threadIdx.y;
  if (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    const DType* lhsoff = lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len;
    const DType* rhsoff = rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len;
    DType* outoff = out + eid * out_len;
    int tx = threadIdx.x;  // tx < 32
    for (int i = blockIdx.y; i < out_len; i += gridDim.y) {  // over output feature dimension
      const Idx lhs_add = UseBcast ? __ldg(lhs_off + i) : i;
      const Idx rhs_add = UseBcast ? __ldg(rhs_off + i) : i;
      DType val = reduce::Sum<Idx, DType>::zero();;
      for (int j = tx; j < reduce_size; j += 64) {
        val += lhsoff[lhs_add * reduce_size + j] * rhsoff[rhs_add * reduce_size + j];
        if (j + 32 < reduce_size)
          val += lhsoff[lhs_add * reduce_size + j + 32] * rhsoff[rhs_add * reduce_size + j + 32];
      }
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2)
        val += __shfl_down_sync(full_mask, val, offset);
      if (tx == 0)
        outoff[i] = val;
    }
  }
}

// Binary search the row_offsets to find the source node of the edge id.
template <typename Idx>
__device__ __forceinline__ Idx BinarySearchSrc(const Idx *array, Idx length, Idx eid) {
  Idx lo = 0, hi = length - 1;
  while (lo < hi) {
    Idx mid = (lo + hi) >> 1;
    if (_ldg(array + mid) <= eid) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  // INVARIANT: lo == hi
  if (_ldg(array + hi) == eid) {
    return hi;
  } else {
    return hi - 1;
  }
}

/*!
 * \brief CUDA kernel of g-SDDMM on Csr format.
 * \note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different positions
 *       in feature dimension.
 *       To efficiently find the source node idx and destination node index of an
 *       given edge on Csr format, it uses binary search (time complexity O(log N)).
 */
template <typename Idx, typename DType, typename BinaryOp,
          bool UseBcast = false, bool UseIdx = false,
          int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCsrKernel(
  const DType* __restrict__ lhs,
  const DType* __restrict__ rhs,
  DType* __restrict__ out,
  const Idx* __restrict__ indptr,
  const Idx* __restrict__ indices,
  const Idx* __restrict__ edge_map,
  int64_t N, int64_t M, int64_t E, int64_t reduce_size,
  const int64_t* __restrict__ lhs_off,
  const int64_t* __restrict__ rhs_off,
  int64_t lhs_len, int64_t rhs_len, int64_t out_len) {
  // SDDMM with Csr.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = BinarySearchSrc<Idx>(indptr, N + 1, ty);
    const Idx dst = _ldg(indices + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* lhsoff = BinaryOp::use_lhs ?
      (lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len): nullptr;
    const DType* rhsoff = BinaryOp::use_rhs ?
      (rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len): nullptr;
    DType* outoff = out + eid * out_len;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? lhs_off[tx] : tx;
      const Idx rhs_add = UseBcast ? rhs_off[tx] : tx;
      DType val = BinaryOp::Call(
          lhsoff + lhs_add * reduce_size,
          rhsoff + rhs_add * reduce_size,
          reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA kernel of g-SDDMM on Csr format.
 * \note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different positions
 *       in feature dimension.
 *       To efficiently find the source node idx and destination node index of an
 *       given edge on Csr format, it uses binary search (time complexity O(log N)).
 */
template <typename Idx, typename DType, typename BinaryOp,
          bool UseBcast = false, bool UseIdx = false,
          int LhsTarget = 0, int RhsTarget = 2>
__global__ void SDDMMCsrKernel_MergedEtypes(
  const DType** __restrict__ lhs_ptrs,
  const DType** __restrict__ rhs_ptrs,
  DType** __restrict__ out_ptrs,
  const Idx** __restrict__ indptr_ptrs,
  const Idx** __restrict__ indices_ptrs,
  const Idx** __restrict__ emap_ptrs,
  const int64_t* E_per_etype,
  const int64_t* N_per_etype,
  int64_t reduce_size,
  const int64_t* __restrict__ lhs_off,
  const int64_t* __restrict__ rhs_off,
  int64_t lhs_len, int64_t rhs_len, int64_t out_len,
  int blk_load) {
  // SDDMM with merged relationship on Csr.
  const Idx stride_y = blockDim.y * blk_load;
  const int etype = blockIdx.y / blk_load;
  const Idx* indptr = indptr_ptrs[etype];
  const Idx* indices = indices_ptrs[etype];
  const Idx* edge_map = emap_ptrs[etype];
  const DType* lhs = lhs_ptrs[etype];
  const DType* rhs = rhs_ptrs[etype];
  DType* out = out_ptrs[etype];
  const int64_t E = E_per_etype[etype];
  const int64_t N = N_per_etype[etype];
  const int num_etypes = gridDim.y / blk_load;

  Idx ty = (blockIdx.y % blk_load) * blockDim.y + threadIdx.y;
  while (ty < E) {
    const Idx src = BinarySearchSrc<Idx>(indptr, N + 1, ty);
    const Idx dst = _ldg(indices + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* lhsoff = BinaryOp::use_lhs ?
      (lhs + Selector<LhsTarget>::Call(src, eid, dst) * lhs_len): nullptr;
    const DType* rhsoff = BinaryOp::use_rhs ?
      (rhs + Selector<RhsTarget>::Call(src, eid, dst) * rhs_len): nullptr;
    DType* outoff = out + eid * out_len;
    while (tx < out_len) {
      const Idx lhs_add = UseBcast ? lhs_off[tx] : tx;
      const Idx rhs_add = UseBcast ? rhs_off[tx] : tx;
      DType val = BinaryOp::Call(
          lhsoff + lhs_add * reduce_size,
          rhsoff + rhs_add * reduce_size,
          reduce_size);
      outoff[tx] = val;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA implementation of g-SDDMM on Coo format.
 * \param bcast Broadcast information.
 * \param coo The Coo matrix.
 * \param lhs The left hand side operand feature.
 * \param rhs The right hand size operand feature.
 * \param out The result feature on edges.
 */
template <typename Idx, typename DType, typename Op,
          int LhsTarget = 0, int RhsTarget = 2>
void SDDMMCoo(
    const BcastOff& bcast,
    const COOMatrix& coo,
    NDArray lhs,
    NDArray rhs,
    NDArray out) {
  const Idx *row = coo.row.Ptr<Idx>();
  const Idx *col = coo.col.Ptr<Idx>();
  const Idx *edge_map = coo.data.Ptr<Idx>();
  const DType *lhs_data = lhs.Ptr<DType>();
  const DType *rhs_data = rhs.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  int64_t *lhs_off = nullptr, *rhs_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  int64_t reduce_dim = bcast.reduce_size;

  const int64_t nnz = coo.row->shape[0];
  const bool use_idx = !IsNullArray(coo.data);

  if (std::is_same<Op, binary::Dot<DType> >::value && reduce_dim >= 32) {
    const int ntx = 32;  // on feature dimension
    const int nty = 8;   // on out dimension
    const int nbx = (nnz + nty - 1) / nty;
    const int nby = FindNumBlocks<'y'>(len);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_IDX_CTX_SWITCH(bcast, use_idx, out->ctx, lhs_off, rhs_off, {
      CUDA_KERNEL_CALL((SDDMMCooTreeReduceKernel<Idx, DType, UseBcast, UseIdx, LhsTarget, RhsTarget>),
          nblks, nthrs, 0, thr_entry->stream,
          lhs_data, rhs_data, out_data,
          row, col, edge_map,
          coo.num_rows, coo.num_cols, nnz, reduce_dim,
          lhs_off, rhs_off,
          lhs_len, rhs_len, len);
    });
  } else {
    const int ntx = FindNumThreads(len);
    const int nty = CUDA_MAX_NUM_THREADS / ntx;
    const int nbx = (len + ntx - 1) / ntx;
    const int nby = FindNumBlocks<'y'>((nnz + nty - 1) / nty);
    const dim3 nblks(nbx, nby);
    const dim3 nthrs(ntx, nty);
    BCAST_IDX_CTX_SWITCH(bcast, use_idx, out->ctx, lhs_off, rhs_off, {
      CUDA_KERNEL_CALL((SDDMMCooKernel<Idx, DType, Op, UseBcast, UseIdx, LhsTarget, RhsTarget>),
          nblks, nthrs, 0, thr_entry->stream,
          lhs_data, rhs_data, out_data,
          row, col, edge_map,
          coo.num_rows, coo.num_cols, nnz, reduce_dim,
          lhs_off, rhs_off,
          lhs_len, rhs_len, len);
    });
  }
}

/*!
 * \brief CUDA implementation of g-SDDMM on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param lhs The left hand side operand feature.
 * \param rhs The right hand size operand feature.
 * \param out The result feature on edges.
 */
template <typename Idx, typename DType, typename Op,
          int LhsTarget = 0, int RhsTarget = 2>
void SDDMMCsr(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray lhs,
    NDArray rhs,
    NDArray out) {
  const Idx *indptr = csr.indptr.Ptr<Idx>();
  const Idx *indices = csr.indices.Ptr<Idx>();
  const Idx *edge_map = csr.data.Ptr<Idx>();
  const DType *lhs_data = lhs.Ptr<DType>();
  const DType *rhs_data = rhs.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t N = csr.num_rows, M = csr.num_cols, E = csr.indices->shape[0];

  int64_t *lhs_off = nullptr, *rhs_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  int64_t reduce_dim = bcast.reduce_size;

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, out->ctx, lhs_off, rhs_off, {
    CUDA_KERNEL_CALL((SDDMMCsrKernel<Idx, DType, Op, UseBcast, UseIdx, LhsTarget, RhsTarget>),
        nblks, nthrs, 0, thr_entry->stream,
        lhs_data, rhs_data, out_data,
        indptr, indices, edge_map,
        N, M, E, reduce_dim,
        lhs_off, rhs_off,
        lhs_len, rhs_len, len);
  });
}

/*!
 * \brief CUDA implementation of g-SDDMM on heterograph using Csr format.
 * \param bcast Broadcast information.
 * \param vec_csr Vector of the Csr matrices.
 * \param vec_lhs Vector of the left hand side operand features.
 * \param vec_rhs Vector of the right hand size operand features.
 * \param vec_out Vector of the result features on edges.
 * \param lhs_eid Vector of node type ids for each lhs in vec_lhs.
 * \param rhs_eid Vector of node type ids for each Rhs in vec_rhs.
 * \param strm_id cudaStream id.
 */
template <typename Idx, typename DType, typename Op,
          int LhsTarget = 0, int RhsTarget = 2>
void SDDMMCsrHetero_mergedEtypes(
    const BcastOff& bcast,
    const std::vector<CSRMatrix>& vec_csr,
    const std::vector<NDArray>& vec_lhs,
    const std::vector<NDArray>& vec_rhs,
    std::vector<NDArray> vec_out,
    const std::vector<dgl_type_t>& lhs_eid,
    const std::vector<dgl_type_t>& rhs_eid,
    cudaStream_t strm_id) {

  int num_etypes = vec_csr.size();
  const DLContext ctx = vec_lhs[lhs_eid[0]]->ctx;
  int64_t *lhs_off = nullptr, *rhs_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  int64_t reduce_dim = bcast.reduce_size;

  std::vector<Idx*> indptr_ptrs(num_etypes, NULL);
  std::vector<Idx*> indices_ptrs(num_etypes, NULL);
  std::vector<Idx*> emap_ptrs(num_etypes, NULL);
  std::vector<DType*> lhs_ptrs(num_etypes, NULL);
  std::vector<DType*> rhs_ptrs(num_etypes, NULL);
  std::vector<DType*> out_ptrs(num_etypes, NULL);
  std::vector<int64_t> N_per_etype(num_etypes, 0);
  std::vector<int64_t> E_per_etype(num_etypes, 0);

  for (dgl_type_t etype = 0; etype < num_etypes; ++etype) {
    CSRMatrix csr = vec_csr[etype];
    indptr_ptrs[etype] = csr.indptr.Ptr<Idx>();
    indices_ptrs[etype] = csr.indices.Ptr<Idx>();
    emap_ptrs[etype] = csr.data.Ptr<Idx>();
    lhs_ptrs[etype] = vec_lhs[lhs_eid[etype]].Ptr<DType>();
    rhs_ptrs[etype] = vec_rhs[rhs_eid[etype]].Ptr<DType>();
    out_ptrs[etype] = vec_out[etype].Ptr<DType>();
    E_per_etype[etype] = csr.indices->shape[0];
    N_per_etype[etype] = csr.num_rows;
  }
  DType** d_lhs_ptrs, **d_rhs_ptrs, **d_out_ptrs;
  Idx** d_indptr_ptrs, **d_indices_ptrs, **d_emap_ptrs;
  int64_t *d_E_per_etype, *d_N_per_etype;

  CUDA_CALL(cudaMalloc(&d_lhs_ptrs, num_etypes * sizeof(DType*)));
  CUDA_CALL(cudaMalloc(&d_rhs_ptrs, num_etypes * sizeof(DType*)));
  CUDA_CALL(cudaMalloc(&d_out_ptrs, num_etypes * sizeof(DType*)));
  CUDA_CALL(cudaMalloc(&d_indptr_ptrs, num_etypes * sizeof(Idx*)));
  CUDA_CALL(cudaMalloc(&d_indices_ptrs, num_etypes * sizeof(Idx*)));
  CUDA_CALL(cudaMalloc(&d_emap_ptrs, num_etypes * sizeof(Idx*)));
  CUDA_CALL(cudaMalloc(&d_E_per_etype, num_etypes * sizeof(int64_t)));
  CUDA_CALL(cudaMalloc(&d_N_per_etype, num_etypes * sizeof(int64_t)));

  CUDA_CALL(cudaMemcpy(d_lhs_ptrs, &(lhs_ptrs[0]), num_etypes * sizeof(DType*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_rhs_ptrs, &(rhs_ptrs[0]), num_etypes * sizeof(DType*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_out_ptrs, &(out_ptrs[0]), num_etypes * sizeof(DType*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_indptr_ptrs, &(indptr_ptrs[0]), num_etypes * sizeof(Idx*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_indices_ptrs, &(indices_ptrs[0]), num_etypes * sizeof(Idx*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_emap_ptrs, &(emap_ptrs[0]), num_etypes * sizeof(Idx*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_E_per_etype, &E_per_etype[0], num_etypes * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_N_per_etype, &N_per_etype[0], num_etypes * sizeof(int64_t), cudaMemcpyHostToDevice));

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  // TODO(Israt): Using the same number of blocks to process all etypes is not ideal.
  // Binning can be used to group etypes with similar load and launch each bin seperately.
  const int blk_load = 16; // number of blocks assigned per etype
  // Assinging blk_load number of block to process each etype
  const int nby = FindNumBlocks<'y'>((num_etypes * blk_load));// + nty - 1) / nty);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(vec_csr[0].data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, vec_out[0]->ctx, lhs_off, rhs_off, {
    CUDA_KERNEL_CALL((SDDMMCsrKernel_MergedEtypes<Idx, DType, Op, UseBcast, UseIdx,
      LhsTarget, RhsTarget>),
        nblks, nthrs, 0, strm_id,
        (const DType**)d_lhs_ptrs,
        (const DType**)d_rhs_ptrs, (DType**)d_out_ptrs,
        (const Idx**)d_indptr_ptrs, (const Idx**) d_indices_ptrs,
        (const Idx**)d_emap_ptrs, (const int64_t*)d_E_per_etype,
        (const int64_t*)d_N_per_etype, reduce_dim,
        lhs_off, rhs_off,
        lhs_len, rhs_len, len, blk_load);
  });
  cudaFree(d_lhs_ptrs); cudaFree(d_rhs_ptrs); cudaFree(d_out_ptrs);
  cudaFree(d_indptr_ptrs); cudaFree(d_indices_ptrs); cudaFree(d_emap_ptrs);
}


}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
