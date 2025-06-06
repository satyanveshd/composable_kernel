// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp"
#include "common.hpp"

using OutDataType      = BF16;
using WeiDataType      = BF16;
using AccDataType      = FP32;
using CShuffleDataType = BF16;
using DsDataType       = ck::Tuple<>;
using InDataType       = BF16;

using OutLayout = ck::tensor_layout::convolution::GNHWK;
using WeiLayout = ck::tensor_layout::convolution::GKYXC;
using DsLayout  = ck::Tuple<>;
using InLayout  = ck::tensor_layout::convolution::GNHWC;

using OutElementOp = PassThrough;
using WeiElementOp = PassThrough;
using InElementOp  = PassThrough;

// n:16, c:128, h:24, w:16, k:192, y:2, x:2, sy:2, sx:2, dy:1, dx:1, py:0, px:0, ho:12, wo:8, group:1
// grid: (240,1,1), block: (256,1,1)
//   fastest: [66]igemm_bwd_gtcx3_nhwc_bf16_
//   bx0_ex1_
//   bt64x64x64_
//   wt16x16x16_
//   ws1x1_
//   wr2x2_
//   ta1x8x2x1_1x8x1x32_
//   tb1x8x1x2_1x8x1x32_
//   mh, 
//  cost:0.005ms, tflops:59.723(59.81%)
// ```
//  M            K     K, N
// [N * Ho * Wo, K] * [K, Y * X * C] -> [N * Ho * Wo, Y * X * C]

// M = 1536
// K = 192
// N = 512

// clang-format off
using DeviceConvInstance = ck::tensor_operation::device::DeviceGroupedConvBwdDataMultipleD_Xdl_CShuffle_v1
// ######| NDimSpatial|   ALayout|   BLayout|   DsLayout|  ELayout|       AData|       BData|     AccData|         CShuffle|       DsData|      EData| AElementwise| BElementwise| CDEElementwise| ConvolutionBackward| DoPad| DoPad|      NumGemmK| Block|  MPer|  NPer|  KPer| AK1| BK1| MPer| NPer|    MXdl|    NXdl|    ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|    BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds| CShuffleMXdl| CShuffleNXdl|   CDEBlockTransfer| CDEBlockTransfer|
// ######|            |          |          |           |         |        Type|        Type|        Type|         DataType|         Type|       Type|    Operation|    Operation|      Operation|  DataSpecialization| GemmM| GemmN| PrefetchStage|  Size| Block| Block| Block|    |    |  XDL|  XDL| PerWave| PerWave|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraM|     ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar|    ExtraN|      PerWave|      PerWave|  _MBlock_MPerBlock|  ScalarPerVector|
// ######|            |          |          |           |         |            |            |            |                 |             |           |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        | Lengths_AK0_M_AK1|   ArrangeOrder|               |               |      PerVector|  PerVector_AK1|          | Lengths_BK0_N_BK1|   ArrangeOrder|               |               |      PerVector|  PerVector_BK1|          |   PerShuffle|   PerShuffle|  _NBlock_NPerBlock|       _NPerBlock|
// ######|            |          |          |           |         |            |            |            |                 |             |           |             |             |               |                    |      |      |              |      |      |      |      |    |    |     |     |        |        |                  |               |               |               |               |               |          |                  |               |               |               |               |               |          |             |             |                   |                 |
        //  < NDimSpatial, OutLayout, WeiLayout,   DsLayout, InLayout, OutDataType, WeiDataType, AccDataType, CShuffleDataType,   DsDataType, InDataType, OutElementOp, WeiElementOp,    InElementOp,  ConvBwdDataDefault,  true,  true,             1,   256,   128,   256,    32,   8,   2,   32,   32,       2,       4,       S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,       S<4, 64, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              4,              2,         0,            1,            1,     S<1, 32, 1, 8>,                8>;
         < NDimSpatial, OutLayout, WeiLayout,   DsLayout, InLayout, OutDataType, WeiDataType, AccDataType, CShuffleDataType,   DsDataType, InDataType, OutElementOp, WeiElementOp,    InElementOp,  ConvBwdDataDefault,  true,  true,             1,   256,   64,    64,    64,   2,   2,   16,   16,       2,       2,       S<32, 8, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              2,              2,         1,       S<32, 8, 1>,     S<0, 2, 1>,     S<0, 2, 1>,              1,              8,              2,         0,            2,            2,     S<1, 32, 1, 8>,                8>;
// clang-format on

#include "run_grouped_conv_bwd_data_example.inc"

int main(int argc, char* argv[]) { return run_grouped_conv_bwd_data_example(argc, argv); }
