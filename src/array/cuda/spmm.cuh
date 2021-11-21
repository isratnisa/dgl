/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/spmm.cuh
 * \brief SPMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_SPMM_CUH_
#define DGL_ARRAY_CUDA_SPMM_CUH_

#include <dgl/bcast.h>
#include "macro.cuh"
#include "fp16.cuh"
#include "atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"
#include <bits/stdc++.h>
#include <omp.h>

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {


/*!
 * \brief CUDA kernel of g-SpMM on Coo format.
 * \note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different positions
 *       in feature dimension.
 *       To avoid possible data hazards, it uses atomic operators for reduction.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCooKernel(
  const DType* __restrict__ ufeat,
  const DType* __restrict__ efeat,
  DType* __restrict__ out,
  Idx* __restrict__ arg_u,
  Idx* __restrict__ arg_e,
  const Idx* __restrict__ row,
  const Idx* __restrict__ col,
  const Idx* __restrict__ edge_map,
  int64_t N, int64_t M, int64_t E,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with COO.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len): nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
    DType* outoff = out + dst * out_len;
    while (tx < out_len) {
      const int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      Idx* arguoff = nullptr;  // arguoff is not used in SpMMCoo.
      Idx* argeoff = nullptr;  // argeoff is not used in SpMMCoo.
      ReduceOp::Call(outoff + tx, arguoff, argeoff, val, src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA kernel to compute argu and arge in g-SpMM on Coo format.
 * \note it uses edge parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different edges. Threadblocks
 *       on the x-axis are responsible for the computation on different positions
 *       in feature dimension.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void ArgSpMMCooKernel(
  const DType* __restrict__ ufeat,
  const DType* __restrict__ efeat,
  DType* __restrict__ out,
  Idx* __restrict__ arg_u,
  Idx* __restrict__ arg_e,
  const Idx* __restrict__ row,
  const Idx* __restrict__ col,
  const Idx* __restrict__ edge_map,
  int64_t N, int64_t M, int64_t E,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with COO arg max/min.
  Idx ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  while (ty < E) {
    const Idx src = _ldg(row + ty);
    const Idx dst = _ldg(col + ty);
    const Idx eid = UseIdx ? _ldg(edge_map + ty) : ty;
    int64_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride_x = blockDim.x * gridDim.x;
    const DType* uoff = BinaryOp::use_lhs ? (ufeat + src * ufeat_len): nullptr;
    const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
    const DType* outoff = out + dst * out_len;
    Idx* arguoff = BinaryOp::use_lhs ? (arg_u + dst * out_len): nullptr;
    Idx* argeoff = BinaryOp::use_rhs ? (arg_e + dst * out_len): nullptr;
    while (tx < out_len) {
      int64_t lhs_add = UseBcast ? ubcast_off[tx] : tx;
      int64_t rhs_add = UseBcast ? ebcast_off[tx] : tx;
      DType val = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
      ReduceOp::CallArg(tx, arguoff, argeoff, val, outoff[tx], src, eid);
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA kernel of g-SpMM on Csr format.
 * \note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */


template <typename Idx, typename DType>
__global__ void find_nonempty_rows(  const Idx* __restrict__ indptr,
   int* non_empty, const int num_rows, int count) {
    int idx = threadIdx.x;
    int row = threadIdx.x;
    int sum = 0;
    for (int row = threadIdx.x; row < num_rows; row += blockDim.x) {
      if ((indptr[row + 1] - indptr[row]) > 0) {
        sum++;
        non_empty[row] = true;
      }
    }
    static const int blockSize = 1024;
    __shared__ int r[blockSize];
    r[idx] = sum;
    __syncthreads();
    for (int size = blockDim.x/2; size>0; size/=2) { //uniform
        if (idx<size)
            r[idx] += r[idx+size];
        __syncthreads();
    }
    if (idx == 0) {
      count = r[0];
      // printf("GPU %d\n", count );
    }
}

/*!
 * \brief CUDA kernel of g-SpMM on Csr format.
 * \note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel_Xdim(
  const DType* __restrict__ ufeat,
  const DType* __restrict__ efeat,
  DType* __restrict__ out,
  Idx* __restrict__ arg_u,
  Idx* __restrict__ arg_e,
  const Idx* __restrict__ indptr,
  const Idx* __restrict__ indices,
  const Idx* __restrict__ edge_map,
  int64_t num_rows, int64_t num_cols,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len,
  const int* __restrict__ row_list, int64_t count) {

  // SPMM with CSR.
  unsigned int tId = threadIdx.x;
  unsigned int laneId = tId & 31;
  unsigned int gId = (blockIdx.x * blockDim.x + threadIdx.x);
  unsigned int warpId = gId >> 5; // gId >> 5
  unsigned int row = warpId; // row_list[warpId];
  // if( row < count) {
  if( row < num_rows) {
    for(unsigned int k = laneId; k < out_len; k += 32) {
      DType local_accum = ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[k] : k;
      const int rhs_add = UseBcast ? ebcast_off[k] : k;
      for (Idx col = indptr[row]; col < indptr[row + 1]; ++col) {
        const Idx eid = UseIdx ? _ldg(edge_map + col) : col;
        const Idx cid = _ldg(indices + col);
        const DType* uoff = BinaryOp::use_lhs ? (ufeat + cid * ufeat_len): nullptr;
        const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }
      out[row * out_len + k] += local_accum;
    }
  }
}

/*!
 * \brief CUDA kernel of g-SpMM on Csr format.
 * \note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel(
  const DType* __restrict__ ufeat,
  const DType* __restrict__ efeat,
  DType* __restrict__ out,
  Idx* __restrict__ arg_u,
  Idx* __restrict__ arg_e,
  const Idx* __restrict__ indptr,
  const Idx* __restrict__ indices,
  const Idx* __restrict__ edge_map,
  int64_t num_rows, int64_t num_cols,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len) {
  // SPMM with CSR.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      DType local_accum = ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff = BinaryOp::use_lhs ? (ufeat + cid * ufeat_len): nullptr;
        const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }

      // TODO(isratnisa, BarclayII)
      // The use of += is a quick hack to compute for cross-type reducing
      //     C = SpMM(SpA, B) + C
      // To make it work on max-reducer and min-reducer, i.e.
      //     C = Max(SpMM<BinaryOp, Max>(SpA, B), C)
      // it requires at least the following:
      // 1. Initialize the output buffer with ReducerOp::zero.
      // 2. Record also which edge type has the maximum/minimum in argmax/argmin.
      //    This requires non-trivial changes in SpMMCsrKernel itself or writing a new kernel.
      //    So we leave it to future PRs.
      out[ty * out_len + tx] += local_accum;
      if (ReduceOp::require_arg && BinaryOp::use_lhs)
        arg_u[ty * out_len + tx] = local_argu;
      if (ReduceOp::require_arg && BinaryOp::use_rhs)
        arg_e[ty * out_len + tx] = local_arge;
      tx += stride_x;
    }
    ty += stride_y;
  }
}

/*!
 * \brief CUDA kernel of g-SpMM on Csr format.
 * \note it uses node parallel strategy, different threadblocks (on y-axis)
 *       is responsible for the computation on different destination nodes.
 *       Threadblocks on the x-axis are responsible for the computation on
 *       different positions in feature dimension.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel_bin(
  const DType* __restrict__ ufeat,
  const DType* __restrict__ efeat,
  DType* __restrict__ out,
  Idx* __restrict__ arg_u,
  Idx* __restrict__ arg_e,
  const Idx* __restrict__ indptr,
  const Idx* __restrict__ indices,
  const Idx* __restrict__ edge_map,
  int64_t num_rows, int64_t num_cols,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len,
  const int *rowGroupPtr, const int numRowsPerGroup) {
  // SPMM with CSR.
  int ty = blockIdx.y * blockDim.y + threadIdx.y;
  const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < numRowsPerGroup) {
    ty = rowGroupPtr[ty];
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    while (tx < out_len) {
      DType local_accum = ReduceOp::zero();
      Idx local_argu = 0, local_arge = 0;
      const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
      const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
      for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
        const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
        const Idx cid = _ldg(indices + i);
        const DType* uoff = BinaryOp::use_lhs ? (ufeat + cid * ufeat_len): nullptr;
        const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
        DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
        ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
      }
      out[ty * out_len + tx] += local_accum;
      if (ReduceOp::require_arg && BinaryOp::use_lhs)
        arg_u[ty * out_len + tx] = local_argu;
      if (ReduceOp::require_arg && BinaryOp::use_rhs)
        arg_e[ty * out_len + tx] = local_arge;
      tx += stride_x;
    }
    ty += stride_y;
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
 * \brief CUDA kernel of g-SpMM on Csr format.
 * \note Pocess all relation types in the same kernel. Uses binning
 * strategy to address inter-relation load_imbalance.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel_mergedEtypes_bin(
  const DType** __restrict__ lhs_ptrs,
  const DType** __restrict__ rhs_ptrs,
  DType** __restrict__ out_ptrs,
  Idx* __restrict__ arg_u,
  Idx* __restrict__ arg_e,
  const Idx** __restrict__ indptr_ptrs,
  const Idx** __restrict__ indices_ptrs,
  const Idx** __restrict__ emap_ptrs,
  const int64_t* E_per_etype,
  const int64_t* N_per_etype,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len,
  const int64_t* blk_load_etype) {
  // SPMM with merged relationship on Csr.
  int blk_load = 16;
  const Idx stride_y = blockDim.y * blk_load;
  const int etype = blockIdx.y / blk_load;
  const Idx* indptr = indptr_ptrs[etype];
  const Idx* indices = indices_ptrs[etype];
  const Idx* edge_map = emap_ptrs[etype];
  const DType* ufeat = lhs_ptrs[etype];
  const DType* efeat = rhs_ptrs[etype];
  DType* out = out_ptrs[etype];
  const int64_t E = E_per_etype[etype];
  const int64_t num_rows = N_per_etype[etype];

  // int ty = blockIdx.y * blockDim.y + threadIdx.y;
  Idx ty = (blockIdx.y % blk_load) * blockDim.y + threadIdx.y;
  // const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    if ((indptr[ty +1 ] - indptr[ty]) > 0) {
      int tx = blockIdx.x * blockDim.x + threadIdx.x;
      while (tx < out_len) {
        DType local_accum = ReduceOp::zero();
        Idx local_argu = 0, local_arge = 0;
        const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
        const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
        // TODO (Israt): avoid accessing pointer for empty rows

        for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
          const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
          const Idx cid = _ldg(indices + i);
          const DType* uoff = BinaryOp::use_lhs ? (ufeat + cid * ufeat_len): nullptr;
          const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
          DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
          ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
        }
        out[ty * out_len + tx] += local_accum;
        if (ReduceOp::require_arg && BinaryOp::use_lhs)
          arg_u[ty * out_len + tx] = local_argu;
        if (ReduceOp::require_arg && BinaryOp::use_rhs)
          arg_e[ty * out_len + tx] = local_arge;
        tx += stride_x;
      }
    }
    ty += stride_y;
  }
}


/*!
 * \brief CUDA kernel of g-SpMM on Csr format.
 * \note Pocess all relation types in the same kernel. Uses prefix-sum
 * to estimate work load for each relation type. That's how it tries
 * to address inter-relation load_imbalance.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp,
          bool UseBcast = false, bool UseIdx = false>
__global__ void SpMMCsrKernel_mergedEtypes(
  const DType** __restrict__ lhs_ptrs,
  const DType** __restrict__ rhs_ptrs,
  DType** __restrict__ out_ptrs,
  Idx* __restrict__ arg_u,
  Idx* __restrict__ arg_e,
  const Idx** __restrict__ indptr_ptrs,
  const Idx** __restrict__ indices_ptrs,
  const Idx** __restrict__ emap_ptrs,
  const int64_t* E_per_etype,
  const int64_t* N_per_etype,
  const int64_t* __restrict__ ubcast_off,
  const int64_t* __restrict__ ebcast_off,
  int64_t ufeat_len, int64_t efeat_len, int64_t out_len,
  const int64_t* blk_load_etype, const int num_etypes) {
  // SPMM with merged relationship on Csr.
  int blk_load = 4;

  // const int etype = BinarySearchSrc<int64_t>(blk_load_etype, num_etypes + 1, blockIdx.y);
  const int etype = blockIdx.y / blk_load;
  const int blk_load_prefix = blk_load * etype;//
  // const int blk_load_prefix = blk_load_etype[etype];
  // const int blk_load = blk_load_etype[ etype + 1] - blk_load_etype[ etype];

  const Idx stride_y = blockDim.y * blk_load;

  const Idx* indptr = indptr_ptrs[etype];
  const Idx* indices = indices_ptrs[etype];
  const Idx* edge_map = emap_ptrs[etype];
  const DType* ufeat = lhs_ptrs[etype];
  const DType* efeat = rhs_ptrs[etype];
  DType* out = out_ptrs[etype];
  const int64_t E = E_per_etype[etype];
  const int64_t num_rows = N_per_etype[etype];

  // // int ty = blockIdx.y * blockDim.y + threadIdx.y;
  Idx ty = blockIdx.y  * blockDim.y + threadIdx.y;
  if(etype)
    ty = (blockIdx.y % blk_load_prefix) * blockDim.y + threadIdx.y;

  // const Idx stride_y = blockDim.y * gridDim.y;
  const int stride_x = blockDim.x * gridDim.x;
  while (ty < num_rows) {
    if ((indptr[ty + 1] - indptr[ty]) > 0) {
      int tx = blockIdx.x * blockDim.x + threadIdx.x;
      while (tx < out_len) {
        DType local_accum = ReduceOp::zero();
        Idx local_argu = 0, local_arge = 0;
        const int lhs_add = UseBcast ? ubcast_off[tx] : tx;
        const int rhs_add = UseBcast ? ebcast_off[tx] : tx;
        for (Idx i = indptr[ty]; i < indptr[ty + 1]; ++i) {
          const Idx eid = UseIdx ? _ldg(edge_map + i) : i;
          const Idx cid = _ldg(indices + i);
          const DType* uoff = BinaryOp::use_lhs ? (ufeat + cid * ufeat_len): nullptr;
          const DType* eoff = BinaryOp::use_rhs ? (efeat + eid * efeat_len): nullptr;
          DType out = BinaryOp::Call(uoff + lhs_add, eoff + rhs_add);
          ReduceOp::Call(&local_accum, &local_argu, &local_arge, out, cid, eid);
        }
        // out[ty * out_len + tx] += local_accum;
        atomicAdd(&out[ty * out_len + tx], local_accum);
        // if (ReduceOp::require_arg && BinaryOp::use_lhs)
        //   arg_u[ty * out_len + tx] = local_argu;
        // if (ReduceOp::require_arg && BinaryOp::use_rhs)
        //   arg_e[ty * out_len + tx] = local_arge;
        tx += stride_x;
      }
    }
    ty += stride_y;
  }
}


/*!
 * \brief CUDA implementation of g-SpMM on Coo format.
 * \param bcast Broadcast information.
 * \param coo The Coo matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCoo(
    const BcastOff& bcast,
    const COOMatrix& coo,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
#if defined(CUDART_VERSION) && CUDART_VERSION <= 10000
  if (std::is_same<DType, half>::value)
    LOG(FATAL) << "SpMMCoo requires atomicCAS, which is not supported "
               << "for float16 in CUDA 10.0. Please upgrade your CUDA "
               << "to later versions.";
#endif
  const Idx *row = coo.row.Ptr<Idx>(),
            *col = coo.col.Ptr<Idx>(),
            *edge_map = coo.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>(),
              *efeat_data = efeat.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();
  Idx *argu_data = argu.Ptr<Idx>(),
      *arge_data = arge.Ptr<Idx>();
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int64_t N = coo.num_rows, M = coo.num_cols, E = coo.row->shape[0];

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;

  int64_t out_size = out.NumElements();
  const int nt = FindNumThreads(out_size);
  const int nb = (out_size + nt - 1) / nt;
  CUDA_KERNEL_CALL(_FillKernel, nb, nt, 0, thr_entry->stream,
      out_data, out_size, ReduceOp::zero());

  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((E + nty - 1) / nty);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(coo.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL((SpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, thr_entry->stream,
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        row, col, edge_map,
        N, M, E,
        ubcast_off, ebcast_off,
        lhs_len, rhs_len, len);
    if (ReduceOp::require_arg) {
      CUDA_KERNEL_CALL((ArgSpMMCooKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
          nblks, nthrs, 0, thr_entry->stream,
          ufeat_data, efeat_data, out_data, argu_data, arge_data,
          row, col, edge_map,
          N, M, E,
          ubcast_off, ebcast_off,
          lhs_len, rhs_len, len);
    }
  });
}


/*!
 * \brief CUDA implementation of g-SpMM on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param ufeat The feature on source nodes.
 * \param efeat The feature on edges.
 * \param out The result feature on destination nodes.
 * \param argu Arg-Min/Max on source nodes, which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 * \param arge Arg-Min/Max on edges. which refers the source node indices
 *        correspond to the minimum/maximum values of reduction result on
 *        destination nodes. It's useful in computing gradients of Min/Max reducer.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCsr(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge) {
  const Idx *indptr = csr.indptr.Ptr<Idx>();
  const Idx *indices = csr.indices.Ptr<Idx>();
  const Idx *edge_map = csr.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>();
  const DType *efeat_data = efeat.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  const int ntx = FindNumThreads(len);
  const int nty = CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  const int nby = FindNumBlocks<'y'>((csr.num_rows + nty - 1) / nty);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL((SpMMCsrKernel<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, thr_entry->stream,
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        indptr, indices, edge_map,
        csr.num_rows, csr.num_cols,
        ubcast_off, ebcast_off,
        lhs_len, rhs_len, len)
  });
}


/* SpMM on CSR for using binning to address inter-row load_imbalance.
 * Binning: Rows with similar degress are put in the same bins. Bins are
 * processed sperately. Each bin uses a number of thread blocks proportional to its
 * nodes' degree for better load balance. All the rows in the same bin uses same
 * number of thread blocks.
 */
float tot_mili_oneDim = 0;
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCsr_oneDim(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge,
    cudaStream_t strm_id) {
    const Idx *indptr = csr.indptr.Ptr<Idx>();
    const Idx *indices = csr.indices.Ptr<Idx>();
    const Idx *edge_map = csr.data.Ptr<Idx>();
    const DType *ufeat_data = ufeat.Ptr<DType>();
    const DType *efeat_data = efeat.Ptr<DType>();
    DType *out_data = out.Ptr<DType>();
    Idx* argu_data = argu.Ptr<Idx>();
    Idx* arge_data = arge.Ptr<Idx>();

    auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

    int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
    int64_t len = bcast.out_len,
            lhs_len = bcast.lhs_len,
            rhs_len = bcast.rhs_len;

   int count = 0;
  int *d_nonempty_row;

  /* helps with perf (58ms to 37 ms on AM dataseet) but high preprocessing */
  /* Preprocessing for maintaing list of non empty rows */
  /*
  NDArray csr_indptr_cpu = csr.indptr.CopyTo(DLContext{kDLCPU, 0});
  Idx* indptr_data = static_cast<Idx*>(csr_indptr_cpu->data);
  std::vector<int> nonempty_rows(csr.num_rows);

  #pragma omp parallel for reduction (+:count)
  for (int row = 0; row < (csr.indptr->shape[0] - 1); row++) {
    if ((indptr_data[row + 1] - indptr_data[row]) > 0) //nonempty_flags[row]) //
      nonempty_rows[count++] = row;
  }

  CUDA_CALL(cudaMalloc((void **)&d_nonempty_row, csr.num_rows * sizeof(int)));
  CUDA_CALL(cudaMemcpy(d_nonempty_row, nonempty_rows.data(), count * sizeof(int), cudaMemcpyHostToDevice));
  */
  /* Preprocessing ends */

  /* Preprocessing in CUDA (?) */
  /*std::vector<bool> nonempty_flags(csr.num_rows);
  bool* nonempty_flags = (bool*) malloc (csr.num_rows * sizeof(bool));
  bool *d_nonempty_flag;
  CUDA_CALL(cudaMalloc((bool **)&d_nonempty_flag, csr.num_rows * sizeof(bool)));
  CUDA_CALL(cudaMemset(d_nonempty_flag, 0, csr.num_rows * sizeof(bool)));
  CUDA_CALL(cudaMemcpy(d_nonempty_row, nonempty_rows.data(), count * sizeof(int), cudaMemcpyHostToDevice));

  const int ntx = 1024;//128;
  int nbx = 1; //((csr.num_rows + ntx - 1) / ntx);// ((csr.num_rows + ntx - 1) / ntx);
   dim3 nblks(nbx);
   dim3 nthrs(ntx);
  CUDA_KERNEL_CALL((find_nonempty_rows<Idx, DType>),
      nblks, nthrs, 0, thr_entry->stream,
      indptr, d_nonempty_flag, csr.num_rows, count);

  std::cout << count << " count\n";
  CUDA_CALL(cudaMemcpy(nonempty_flags, d_nonempty_flag, csr.num_rows * sizeof(bool), cudaMemcpyDeviceToHost));
  cudaDeviceSynchronize(); */

  const int ntx = 128;
  const int nbx =  ((csr.num_rows + ntx - 1) / ntx); //((count + ntx - 1) / ntx);
  const dim3 nblks1(nbx);
  const dim3 nthrs1(ntx);

  const bool use_idx = !IsNullArray(csr.data);
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL((SpMMCsrKernel_Xdim<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks1, nthrs1, 0, thr_entry->stream,
        ufeat_data, efeat_data, out_data, argu_data, arge_data,
        indptr, indices, edge_map,
        csr.num_rows, csr.num_cols,
        ubcast_off, ebcast_off,
        lhs_len, rhs_len, len,
        d_nonempty_row, count);
  });
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  tot_mili_oneDim += milliseconds;
  // std::cout << "SpMM indiv kernel: " << milliseconds << " " <<tot_mili_oneDim << " " << std::endl;

  // CUDA_CALL(cudaFree(d_nonempty_row));
  // CUDA_CALL(cudaFree(d_nonempty_flag));

}

float tot_mili =0;
/* SpMM on CSR for using binning to address inter-row load_imbalance.
 * Binning: Rows with similar degress are put in the same bins. Bins are
 * processed sperately. Each bin uses a number of thread blocks proportional to its
 * nodes' degree for better load balance. All the rows in the same bin uses same
 * number of thread blocks.
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCsr_bin(
    const BcastOff& bcast,
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge,
    cudaStream_t strm_id) {
  const Idx *indptr = csr.indptr.Ptr<Idx>();
  const Idx *indices = csr.indices.Ptr<Idx>();
  const Idx *edge_map = csr.data.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>();
  const DType *efeat_data = efeat.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();
  Idx* argu_data = argu.Ptr<Idx>();
  Idx* arge_data = arge.Ptr<Idx>();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;
  const int ntx = 16; //FindNumThreads(len);
  const int nty = 128/16; //CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  int work = std::max(csr.num_rows, csr.indices->shape[0]);
  const int nby = FindNumBlocks<'y'>((csr.indices->shape[0] + nty - 1) / nty);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(csr.data);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEvent_t start1, stop1;
  cudaEventCreate(&start1);
  cudaEventCreate(&stop1);
  // binning
  // Extract CSR group info on CPU
  // cudaEventRecord(start);
  const int nBin = 10;
  int THREADLOAD = 2;
  int dimRow = csr.num_rows;
  int *count = (int*) malloc (nBin * sizeof(int));
  int *host_rowGroupPtr = (int*) malloc(nBin * dimRow * sizeof(int));
  int LB[nBin], UB[nBin];
  for (int i = 0; i < nBin; i++) {
    count [i] = 0;
    UB[i] = (1 << (i+5)) * THREADLOAD + 1;
    LB[i] = UB[i] >> 1;
    // std::cout << LB[i] << " - " << UB[i] << std::endl;
  }
  LB[0] = 0;
  UB[nBin - 1] = 999999999; // dimCol + 1;

  NDArray csr_indptr_cpu = csr.indptr.CopyTo(DLContext{kDLCPU, 0});
  Idx* indptr_data = static_cast<Idx*>(csr_indptr_cpu->data);
  CHECK_NOTNULL(indptr_data);

  omp_set_num_threads(nBin);  // create as many CPU threads as there are # of bins
  // #pragma omp parallel
  {
    // unsigned int cpu_thread_id = omp_get_thread_num();
    // int i = cpu_thread_id;
    for (int i = 0; i < nBin; ++i)
    {
    for (int row = 0; row < dimRow; row++) {
      int NNZ = (indptr_data[row + 1] - indptr_data[row]);
      if(NNZ > 0) {
        if (NNZ > LB[i] && NNZ < UB[i]) {
          // if (csr.num_rows == 1046 && csr.indices->shape[0] == 566273) {
          //   std::cout << "bin " << i << " row " << row << " nnz " << NNZ << std::endl;
          // }
          host_rowGroupPtr[i * dimRow + count[i]++] = row;
          // break;
        }
      }
    }
    }
  }
  int *rowGroupPtr;
  CUDA_CALL(cudaMalloc((void **)&rowGroupPtr, dimRow * sizeof(int)));

  int sum = 0;
  for (int i = 0; i < nBin; i++) {
    if (count[i] > 0) {
      CUDA_CALL(cudaMemcpy(rowGroupPtr + sum, host_rowGroupPtr + (i * dimRow), count[i] * sizeof(int), cudaMemcpyHostToDevice));
      sum += count[i];
    }
  }
  // cudaEventRecord(stop);
  // cudaEventSynchronize(stop);
  // float milliseconds = 0;
  // cudaEventElapsedTime(&milliseconds, start, stop);
  // std::cout << "Preprocessing: " << csr.num_rows << " indices " << csr.indices->shape[0] << " SpMM kernel: " << milliseconds << " " << std::endl;


  sum = 0;
  // std::cout << std::endl;

  cudaEventRecord(start);
  for (int bin = 0; bin < nBin; ++bin) {
    if(count[bin] > 0) {
      int work =  pow(2, (bin+5)) * count[bin];
      const int nby = FindNumBlocks<'y'>((work + nty - 1) / nty);
      const dim3 nblks(nbx, nby);
      const dim3 nthrs(ntx, nty);
      // std::cout << bin << " #blocks " << pow(2, bin) << ", #rows in bin: " << count[bin] <<
      // " nby " << nby << " nty " << nty << std::endl;
    if (csr.num_rows == 1046 && csr.indices->shape[0] == 566273) {
      BCAST_IDX_CTX_SWITCH(bcast, use_idx, ufeat->ctx, ubcast_off, ebcast_off, {
        CUDA_KERNEL_CALL((SpMMCsrKernel_bin<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
            nblks, nthrs, 0, strm_id,
            ufeat_data, efeat_data, out_data, argu_data, arge_data,
            indptr, indices, edge_map,
            csr.num_rows, csr.num_cols,
            ubcast_off, ebcast_off,
            lhs_len, rhs_len, len, rowGroupPtr+sum, count[bin])
      });
    }
      sum += count[bin];
    }
  }
  CUDA_CALL(cudaFree(rowGroupPtr));

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  tot_mili += milliseconds;
  if(milliseconds > 1.0)
    std::cout << csr.num_rows << " indices " << csr.indices->shape[0] << " SpMM kernel: " << milliseconds << " total: "
      << tot_mili << std::endl;
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void SpMMCsrHetero_mergedEtypes(
  const BcastOff& bcast,
  const std::vector<CSRMatrix>& vec_csr,
  const std::vector<NDArray>& vec_ufeat,
  const std::vector<NDArray>& vec_efeat,
  std::vector<NDArray> vec_out,
  const std::vector<NDArray>& out_aux,
  const std::vector<dgl_type_t>& ufeat_ntids,  // ufeat node type id
  const std::vector<dgl_type_t>& out_ntids){

  int num_etypes = vec_csr.size();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();

  std::vector<Idx*> indptr_ptrs(num_etypes, NULL);
  std::vector<Idx*> indices_ptrs(num_etypes, NULL);
  std::vector<Idx*> emap_ptrs(num_etypes, NULL);
  std::vector<DType*> ufeat_ptrs(num_etypes, NULL);
  std::vector<DType*> efeat_ptrs(num_etypes, NULL);
  std::vector<DType*> out_ptrs(num_etypes, NULL);
  std::vector<int64_t> N_per_etype(num_etypes, 0);
  std::vector<int64_t> E_per_etype(num_etypes, 0);
  std::vector<int64_t> blk_load_etype(num_etypes + 1, 0); // number of blocks assigned per etype
  const int nbin = 15;
  std::vector<int> limit(nbin, 0);
  std::vector<int> bin(nbin * num_etypes);
  std::vector<int> count(nbin, 0);
  int THREADLOAD = 2;
  for (int i = 1; i < nbin; ++i)
  {
    // limit[i] = (1 << i) * THREADLOAD + 1;
    limit[i] =  1 << (i+6);

    // std::cout << "bin limit " << limit[i-1] << " " << limit[i] << " given load "
    // << (1 << (i - 1)) << std::endl;
  }
  limit[nbin - 1] = 999999999; //std::INT_MAX
  for (dgl_type_t etype = 0; etype < num_etypes; ++etype) {
    CSRMatrix csr = vec_csr[etype];
    indptr_ptrs[etype] = csr.indptr.Ptr<Idx>();
    indices_ptrs[etype] = csr.indices.Ptr<Idx>();
    emap_ptrs[etype] = csr.data.Ptr<Idx>();
    ufeat_ptrs[etype] = vec_ufeat[ufeat_ntids[etype]].Ptr<DType>();
    efeat_ptrs[etype] = vec_efeat[etype].Ptr<DType>();
    out_ptrs[etype] = vec_out[out_ntids[etype]].Ptr<DType>();
    E_per_etype[etype] = csr.indices->shape[0];
    N_per_etype[etype] = csr.num_rows;
    blk_load_etype[etype] = 16;

    // blk_load_etype[etype + 1] = blk_load_etype[etype] + 16;
    // for (int i = 0; i < nbin; ++i) {
    //   int work = csr.num_rows; //csr.indices->shape[0];
    //   if(work >= limit[i] && work < limit[i+1]) {
    //     bin[i * num_etypes + count[i]++] = etype;
    //     blk_load_etype[etype + 1] = blk_load_etype[etype] + (1 << (i-1)); // (i+1)*5 + 1;
    //  //   std::cout << etype << ": " << " load: " << blk_load_etype[etype + 1] - blk_load_etype[etype]
    //  // << " bin " << i << " work " << work << std::endl;
    //     break;
    //   }
    // }
  }
  // for (dgl_type_t etype = 0; etype < num_etypes; ++etype) {
  //   std::cout << etype << ": " << " load: " << blk_load_etype[etype + 1] - blk_load_etype[etype]
  //    << " bin " << i << " work " <<vec_csr[etype].indices->shape[0] << std::endl;
  // }
  // for (dgl_type_t b = 0; b < nbin; ++b) {
  //   std::cout << "bin " << b << std::endl;
  //   for (int i = 0; i < num_etypes; ++i) {
  //     std::cout << bin[b * num_etypes + i] << " ";
  //   }
  //   std::cout << std::endl;
  // }

  // TODO(Israt) : hardcoded for sum+generic kernel
  Idx* argu_data = nullptr; //out_aux[0].Ptr<Idx>();
  Idx* arge_data = nullptr; //out_aux[1].Ptr<Idx>();

  int64_t *ubcast_off = nullptr, *ebcast_off = nullptr;
  int64_t len = bcast.out_len,
          lhs_len = bcast.lhs_len,
          rhs_len = bcast.rhs_len;

  DType** d_ufeat_ptrs, **d_efeat_ptrs, **d_out_ptrs;
  Idx** d_indptr_ptrs, **d_indices_ptrs, **d_emap_ptrs;
  int64_t *d_E_per_etype, *d_N_per_etype, *d_blk_load_etype, *d_bin;

  // TODO (Israt) : change to use DGL memory pull
  CUDA_CALL(cudaMalloc(&d_ufeat_ptrs, num_etypes * sizeof(DType*)));
  CUDA_CALL(cudaMalloc(&d_efeat_ptrs, num_etypes * sizeof(DType*)));
  CUDA_CALL(cudaMalloc(&d_out_ptrs, num_etypes * sizeof(DType*)));
  CUDA_CALL(cudaMalloc(&d_indptr_ptrs, num_etypes * sizeof(Idx*)));
  CUDA_CALL(cudaMalloc(&d_indices_ptrs, num_etypes * sizeof(Idx*)));
  CUDA_CALL(cudaMalloc(&d_emap_ptrs, num_etypes * sizeof(Idx*)));
  CUDA_CALL(cudaMalloc(&d_E_per_etype, num_etypes * sizeof(int64_t)));
  CUDA_CALL(cudaMalloc(&d_N_per_etype, num_etypes * sizeof(int64_t)));
  CUDA_CALL(cudaMalloc(&d_blk_load_etype, (num_etypes + 1) * sizeof(int64_t)));

  CUDA_CALL(cudaMemcpy(d_ufeat_ptrs, &(ufeat_ptrs[0]), num_etypes * sizeof(DType*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_efeat_ptrs, &(efeat_ptrs[0]), num_etypes * sizeof(DType*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_out_ptrs, &(out_ptrs[0]), num_etypes * sizeof(DType*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_indptr_ptrs, &(indptr_ptrs[0]), num_etypes * sizeof(Idx*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_indices_ptrs, &(indices_ptrs[0]), num_etypes * sizeof(Idx*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_emap_ptrs, &(emap_ptrs[0]), num_etypes * sizeof(Idx*), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_E_per_etype, &E_per_etype[0], num_etypes * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_N_per_etype, &N_per_etype[0], num_etypes * sizeof(int64_t), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(d_blk_load_etype, &blk_load_etype[0], (num_etypes + 1) * sizeof(int64_t), cudaMemcpyHostToDevice));

  const int ntx = 16;//FindNumThreads(len);
  const int nty = 128/ntx; //CUDA_MAX_NUM_THREADS / ntx;
  const int nbx = (len + ntx - 1) / ntx;
  // TODO(Israt): Using the same number of blocks to process all etypes is not ideal.
  // Binning can be used to group etypes with similar load and launch each bin seperately.

  // Assinging blk_load number of block to process each etype
  const int blk_load = 4;

  const int nby = FindNumBlocks<'y'>((num_etypes * blk_load));
  // const int nby = (num_etypes * blk_load);
  // const int nby = FindNumBlocks<'y'>((blk_load_etype[num_etypes]));
  // // const int nby = FindNumBlocks<'y'>((csr.num_rows + nty - 1) / nty);
  //LOG(INFO) << "nblks=(" << nbx << ", " << nby << ") nthrs=(" << ntx << ", " << nty << ")";
  //   std::cout << "#blocks required: " << (num_etypes * blk_load)
  // << " block lunched: " << nbx << " " << nby
  // << " th " << ntx << " " << nty  << std::endl;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  const bool use_idx = !IsNullArray(vec_csr[0].data);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  BCAST_IDX_CTX_SWITCH(bcast, use_idx, vec_out[0]->ctx, ubcast_off, ebcast_off, {
    CUDA_KERNEL_CALL((SpMMCsrKernel_mergedEtypes<Idx, DType, BinaryOp, ReduceOp, UseBcast, UseIdx>),
        nblks, nthrs, 0, thr_entry->stream,
        (const DType**)d_ufeat_ptrs, (const DType**)d_efeat_ptrs,
        (DType**)d_out_ptrs, (Idx*)argu_data, (Idx*)arge_data,
        (const Idx**)d_indptr_ptrs, (const Idx**) d_indices_ptrs,
        (const Idx**)d_emap_ptrs, (const int64_t*)d_E_per_etype,
        (const int64_t*)d_N_per_etype,
        (const int64_t*)ubcast_off, (const int64_t*)ebcast_off,
        lhs_len, rhs_len, len, (const int64_t*)d_blk_load_etype,
        num_etypes);
  });
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  // std::cout << "SpMM kernel: " << milliseconds << " " << std::endl;
  cudaFree(d_ufeat_ptrs); cudaFree(d_efeat_ptrs); cudaFree(d_out_ptrs);
  cudaFree(d_indptr_ptrs); cudaFree(d_indices_ptrs); cudaFree(d_emap_ptrs);
}


}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
