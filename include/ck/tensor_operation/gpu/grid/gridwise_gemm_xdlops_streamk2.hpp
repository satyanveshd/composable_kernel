// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2022, Advanced Micro Devices, Inc. All rights reserved.
#pragma once

#include "ck/utility/common_header.hpp"
#include "ck/tensor_description/multi_index_transform_helper.hpp"
#include "ck/tensor_description/tensor_descriptor.hpp"
#include "ck/tensor_description/tensor_descriptor_helper.hpp"
#include "ck/tensor_operation/gpu/grid/block_to_ctile_map.hpp"
#include "ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_selector.hpp"
#include "ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v4r1.hpp"
#include "ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v6r1r2.hpp"
#include "ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp"
#include "ck/tensor_operation/gpu/element/element_wise_operation.hpp"
#include "ck/utility/workgroup_barrier.hpp"
#include "ck/utility/reduction_functions_accumulate.hpp"

namespace ck {

template <typename GridwiseGemm>
__global__ void
#if CK_USE_LAUNCH_BOUNDS
    __launch_bounds__(CK_MAX_THREAD_PER_BLOCK, CK_MIN_BLOCK_PER_CU)
#endif
        kernel_gemm_xdlops_streamk(const typename GridwiseGemm::FloatA* p_a_grid,
                                   const typename GridwiseGemm::FloatB* p_b_grid,
                                   typename GridwiseGemm::FloatC* p_c_grid,
                                   void* p_workspace,
                                   index_t M,
                                   index_t N,
                                   index_t K,
                                   index_t StrideA,
                                   index_t StrideB,
                                   index_t StrideC,
                                   typename GridwiseGemm::Block2CTileMap block_mapping,
                                   typename GridwiseGemm::AElementwiseOperation a_element_op,
                                   typename GridwiseGemm::BElementwiseOperation b_element_op,
                                   typename GridwiseGemm::CElementwiseOperation c_element_op)
{
#if(!defined(__HIP_DEVICE_COMPILE__) || defined(__gfx908__) || defined(__gfx90a__) || \
    defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__))
    constexpr index_t shared_size = GridwiseGemm::GetSharedMemoryNumberOfByte();

    __shared__ uint8_t p_shared[shared_size];

    GridwiseGemm::Run(p_a_grid,
                      p_b_grid,
                      p_c_grid,
                      p_workspace,
                      M,
                      N,
                      K,
                      StrideA,
                      StrideB,
                      StrideC,
                      block_mapping,
                      a_element_op,
                      b_element_op,
                      c_element_op,
                      static_cast<void*>(p_shared));
#else
    ignore = p_a_grid;
    ignore = p_b_grid;
    ignore = p_c_grid;
    ignore = p_workspace;
    ignore = M;
    ignore = N;
    ignore = K;
    ignore = StrideA;
    ignore = StrideB;
    ignore = StrideC;
    ignore = block_mapping;
    ignore = a_element_op;
    ignore = b_element_op;
    ignore = c_element_op;
#endif
}

template <index_t BlockSize,
          typename Block2CTileMap_,
          typename FloatA_,
          typename FloatB_,
          typename FloatAcc_,
          typename FloatC_,
          typename ALayout,
          typename BLayout,
          typename CLayout,
          typename AElementwiseOperation,
          typename BElementwiseOperation,
          typename CElementwiseOperation,
          index_t MPerBlock,
          index_t NPerBlock,
          index_t K0PerBlock,
          index_t MPerXDL,
          index_t NPerXDL,
          index_t K1Value,
          index_t MRepeat,
          index_t NRepeat,
          typename ABlockTransferThreadClusterLengths_K0_M_K1,
          typename ABlockTransferThreadClusterArrangeOrder,
          typename ABlockTransferSrcAccessOrder,
          index_t ABlockTransferSrcVectorDim,
          index_t ABlockTransferSrcScalarPerVector,
          index_t ABlockTransferDstScalarPerVector_K1,
          bool AThreadTransferSrcResetCoordinateAfterRun,
          index_t ABlockLdsExtraM,
          typename BBlockTransferThreadClusterLengths_K0_N_K1,
          typename BBlockTransferThreadClusterArrangeOrder,
          typename BBlockTransferSrcAccessOrder,
          index_t BBlockTransferSrcVectorDim,
          index_t BBlockTransferSrcScalarPerVector,
          index_t BBlockTransferDstScalarPerVector_K1,
          bool BThreadTransferSrcResetCoordinateAfterRun,
          index_t BBlockLdsExtraN,
          index_t CShuffleMRepeatPerShuffle,
          index_t CShuffleNRepeatPerShuffle,
          index_t CBlockTransferScalarPerVector_NWaveNPerXDL,
          typename CBlockTransferClusterLengths_MBlock_MPerBlock_NBlock_NPerBlock,
          LoopScheduler LoopSched = LoopScheduler::Default,
          PipelineVersion PipelineVer = PipelineVersion::v3>
struct GridwiseGemm_bk0mk1_bk0nk1_mn_xdlops_streamk
{
    static constexpr auto I0 = Number<0>{};
    static constexpr auto I1 = Number<1>{};
    static constexpr auto I2 = Number<2>{};
    static constexpr auto I3 = Number<3>{};
    static constexpr auto I4 = Number<4>{};
    static constexpr auto I5 = Number<5>{};
    static constexpr auto I6 = Number<6>{};
    static constexpr auto I7 = Number<7>{};

    // K1 should be Number<...>
    static constexpr auto K1        = Number<K1Value>{};
    static constexpr auto M01       = 1;
    static constexpr auto N01       = 1;
    static constexpr auto KPerBlock = K0PerBlock * K1;

    using ThisThreadBlock = ThisThreadBlock<BlockSize>;
    using FloatAcc        = FloatAcc_;
    using FloatCShuffle   = FloatAcc;

    using Block2CTileMap = Block2CTileMap_;
    using FloatA         = FloatA_;
    using FloatB         = FloatB_;
    using FloatC         = FloatC_;

    struct Argument : public ck::tensor_operation::device::BaseArgument
    {
        const FloatA* p_a_grid;
        const FloatB* p_b_grid;
        FloatC* p_c_grid;
        void* p_workspace_;
        index_t M;
        index_t N;
        index_t K;
        index_t StrideA;
        index_t StrideB;
        index_t StrideC;
        index_t num_cu, occupancy;
        Block2CTileMap block_mapping;
        AElementwiseOperation a_element_op;
        BElementwiseOperation b_element_op;
        CElementwiseOperation c_element_op;

        Argument(const FloatA* p_a_grid_,
                 const FloatB* p_b_grid_,
                 FloatC* p_c_grid_,
                 index_t M_,
                 index_t N_,
                 index_t K_,
                 index_t StrideA_,
                 index_t StrideB_,
                 index_t StrideC_,
                 uint32_t num_cu_,
                 uint32_t occupancy_,
                 uint32_t num_sk_blocks_,
                 AElementwiseOperation a_element_op_,
                 BElementwiseOperation b_element_op_,
                 CElementwiseOperation c_element_op_)
            : p_a_grid(p_a_grid_),
              p_b_grid(p_b_grid_),
              p_c_grid(p_c_grid_),
              p_workspace_(nullptr),
              M(M_),
              N(N_),
              K(K_),
              StrideA(StrideA_),
              StrideB(StrideB_),
              StrideC(StrideC_),
              num_cu(num_cu_),
              occupancy(occupancy_),
              block_mapping(M, N, K, num_cu_, occupancy_, num_sk_blocks_),
              a_element_op(a_element_op_),
              b_element_op(b_element_op_),
              c_element_op(c_element_op_)
        {
        }

        void Print() const
        {
            std::cout << "arg {"
                      << "M:" << M << ", "
                      << "N:" << N << ", "
                      << "K:" << K << ", "
                      << "SA:" << StrideA << ", "
                      << "SB:" << StrideB << ", "
                      << "SC:" << StrideC << std::endl;
        }
    };

    __host__ __device__ static auto CalculateGridSize(const Argument& karg)
    {
        return karg.block_mapping.get_grid_dims();
    }

    __host__ __device__ static auto CalculateK0(index_t KPad) { return KPad / K1; }

    __host__ __device__ static auto
    MakeAGridDescriptor_K0_M_K1(index_t M, index_t MPad, index_t K, index_t KPad, index_t StrideA)
    {
        const index_t K0 = CalculateK0(KPad);

        const auto a_grid_desc_m_k = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(StrideA, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, ALayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, K), make_tuple(I1, StrideA));
            }
        }();

        const auto a_grid_desc_m_kpad = transform_tensor_descriptor(
            a_grid_desc_m_k,
            make_tuple(make_pass_through_transform(M), make_right_pad_transform(K, KPad - K)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return transform_tensor_descriptor(a_grid_desc_m_kpad,
                                           make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                                      make_right_pad_transform(M, MPad - M)),
                                           make_tuple(Sequence<1>{}, Sequence<0>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    __host__ __device__ static auto
    MakeBGridDescriptor_K0_N_K1(index_t K, index_t KPad, index_t N, index_t NPad, index_t StrideB)
    {
        const index_t K0 = CalculateK0(KPad);

        const auto b_grid_desc_k_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(StrideB, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, BLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(K, N), make_tuple(I1, StrideB));
            }
        }();

        const auto b_grid_desc_kpad_n = transform_tensor_descriptor(
            b_grid_desc_k_n,
            make_tuple(make_right_pad_transform(K, KPad - K), make_pass_through_transform(N)),
            make_tuple(Sequence<0>{}, Sequence<1>{}),
            make_tuple(Sequence<0>{}, Sequence<1>{}));

        return transform_tensor_descriptor(b_grid_desc_kpad_n,
                                           make_tuple(make_unmerge_transform(make_tuple(K0, K1)),
                                                      make_right_pad_transform(N, NPad - N)),
                                           make_tuple(Sequence<0>{}, Sequence<1>{}),
                                           make_tuple(Sequence<0, 2>{}, Sequence<1>{}));
    }

    __host__ __device__ static auto
    MakeCGridDescriptor_M_N(index_t M, index_t MPad, index_t N, index_t NPad, index_t StrideC)
    {
        const auto c_grid_desc_m_n = [&]() {
            if constexpr(is_same<tensor_layout::gemm::RowMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(StrideC, I1));
            }
            else if constexpr(is_same<tensor_layout::gemm::ColumnMajor, CLayout>::value)
            {
                return make_naive_tensor_descriptor(make_tuple(M, N), make_tuple(I1, StrideC));
            }
        }();

        return transform_tensor_descriptor(c_grid_