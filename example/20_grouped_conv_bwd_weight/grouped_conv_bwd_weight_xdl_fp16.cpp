// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle.hpp"
#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp"
#include "ck/utility/blkgemmpipe_scheduler.hpp"

using InDataType  = BF16;
using WeiDataType = BF16;
using OutDataType = BF16;
using AccDataType = F32;

using InElementOp  = PassThrough;
using WeiElementOp = PassThrough;
using OutElementOp = PassThrough;
//[K,      M]      x [K,              N  ] - >   [M,     N]
//[N * Ho * Wo, K] x [N * Ho * Wo, Y * X *C] -> [K, Y * X * C]
// Col Row
// 1. Change Lds layout
// 2. Improve vector store (padding for workspace?)
// 3. Cache settings for workspace
// M =64
// N = 117
// K = 810000

template <ck::index_t NDimSpatial>
using DeviceConvBwdWeightInstance =
        ck::tensor_operation::device::DeviceGroupedConvBwdWeightTwoStage_Xdl_CShuffle<
            NDimSpatial,
            ck::tensor_layout::convolution::NHWGC,
            ck::tensor_layout::convolution::GKYXC,
            ck::tensor_layout::convolution::NHWGK,
            BF16,    BF16,    BF16,     F32,
            PassThrough, PassThrough, PassThrough,
            ConvBwdWeightDefault,
            256,    64,    64,
            32,   8,
            32,   32,
            1,    1,
            S<4, 64,  1>, S<2, 0, 1>,  S<1, 0, 2>,
            1, 1, 8, false,
            S<4, 64,  1>,  S<2, 0, 1>,  S<1, 0, 2>,
            1, 1, 8, false,
            1, 1, S<1, 32, 1, 8>, 1,
            ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1>;
            // ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v3>;

template <ck::index_t NDimSpatial>
using HostConvBwdWeightInstance = ck::tensor_operation::host::ReferenceConvBwdWeight<NDimSpatial,
                                                                                     InDataType,
                                                                                     WeiDataType,
                                                                                     OutDataType,
                                                                                     InElementOp,
                                                                                     WeiElementOp,
                                                                                     OutElementOp>;

#include "run_grouped_conv_bwd_weight_example.inc"

int main(int argc, char* argv[])
{
    ExecutionConfig config;
    ck::utils::conv::ConvParam conv_param = DefaultConvParam;

    if(!parse_cmd_args(argc, argv, config, conv_param))
    {
        return 1;
    }

    switch(conv_param.num_dim_spatial_)
    {
    // case 1: return !run_grouped_conv_bwd_weight<1>(config, conv_param);
    case 2: return !run_grouped_conv_bwd_weight<2>(config, conv_param);
    // case 3: return !run_grouped_conv_bwd_weight<3>(config, conv_param);
    default: break;
    }

    return 1;
}
