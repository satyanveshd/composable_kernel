// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#include "common.hpp"

#include "ck/tensor_operation/gpu/device/impl/device_gemm_xdl_streamk2.hpp"

// FP8 data types
using ADataType = ck::f8_t;       // FP8 E4M3
using BDataType = ck::f8_t;       // FP8 E4M3
using AccDataType = float;        // FP32 accumulation
using CShuffleDataType = float;   // FP32 for shuffle
using CDataType = ck::half_t;     // FP16 output

using ALayout = Row;
using BLayout = Row;
using CLayout = Row;

using AElementOp = PassThrough;
using BElementOp = PassThrough;
using CElementOp = PassThrough;

// clang-format off
using DeviceGemmStreamK = ck::tensor_operation::device::DeviceGemmXdlStreamK
// ######|     AData|     BData|     CData|     AccData| ALayout| BLayout| CLayout|           A|           B|           C|  Block|  MPer|  NPer| K0Per| K1| MPer| NPer| MXdl| NXdl|  ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockTransfer| ABlockLds|  BBlockTransfer| BBlockTransfer| BBlockTransfer| BlockTransfer| BBlockTransfer| BBlockTransfer| BBlockLds|    CShuffle|    CShuffle| CBlockTransferClusterLengths|  CBlockTransfer| LoopSched| PipelineVer|
// ######|      Type|      Type|      Type|        Type|        |        |        | Elementwise| Elementwise| Elementwise|   Size| Block| Block| Block|   |  XDL|  XDL|  Per|  Per|   ThreadCluster|  ThreadCluster| SrcAccessOrder|   SrcVectorDim|      SrcScalar|      DstScalar| AddExtraM|   ThreadCluster|  ThreadCluster| SrcAccessOrder|  SrcVectorDim|      SrcScalar|      DstScalar| AddExtraN| MXdlPerWave| NXdlPerWave|         _MBlock_MWaveMPerXdl| ScalarPerVector|          |            |
// ######|          |          |          |            |        |        |        |   Operation|   Operation|   Operation|       |      |      |      |   |     |     | Wave| Wave| Lengths_K0_M_K1|   ArrangeOrder|               |               |      PerVector|   PerVector_K1|          | Lengths_K0_N_K1|   ArrangeOrder|               |              |      PerVector|   PerVector_K1|          |  PerShuffle|  PerShuffle|         _NBlock_NWaveNPerXdl|   _NWaveNPerXdl|          |            |
// ######|          |          |          |            |        |        |        |            |            |            |       |      |      |      |   |     |     |     |     |                |               |               |               |               |               |          |                |               |               |              |               |               |          |            |            |                             |                |          |            |
    < ADataType, BDataType, CDataType, AccDataType, ALayout, BLayout, CLayout,  AElementOp,  BElementOp,  CElementOp,    256,   128,   128,     4,  8,   32,   32,    2,    2,     S<4, 64, 1>,     S<1, 0, 2>,     S<1, 0, 2>,              2,              8,              8,         1,     S<4, 64, 1>,      S<0, 2, 1>,     S<0, 2, 1>,             1,              2,              8,         1,           1,           1,               S<1, 32, 1, 8>,               8, LoopScheduler::Default, PipelineVersion::v3>;
// clang-format on

using DeviceGemmInstance = DeviceGemmStreamK;

using ReferenceGemmInstance = ck::tensor_operation::host::
    ReferenceGemm<ADataType, BDataType, CDataType, AccDataType, AElementOp, BElementOp, CElementOp>;

#include "run_gemm_example.inc"

int main(int argc, char* argv[]) 
{ 
    // For FP8, you might want to add scaling factors
    // This is a simplified example - you'd need to modify run_gemm_example.inc
    // to support scaling factors for production use
    return !run_gemm_streamk_example(argc, argv); 
}