// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include "ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_exp_gemm_xdl_universal_mk_nk_mn_instance.hpp"
#include "ck/host_utility/device_prop.hpp"

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                                NHWGC,
                                                                GKYXC,
                                                                Empty_Tuple,
                                                                NHWGK,
                                                                F16,
                                                                F16,
                                                                Empty_Tuple,
                                                                F16,
                                                                PassThrough,
                                                                PassThrough,
                                                                PassThrough>>>& instances)
{
   add_exp_grouped_conv_fwd_device_operation_instances<
        2,
        NHWGC,
        GKYXC, Empty_Tuple,
        NHWGK,
        F16,
        F16, Empty_Tuple,
        F16,
        PassThrough,
        PassThrough,
        PassThrough,
        device_gemm_xdl_universal_mk_nk_mn_mem_instances<F16, Interwave, GemmDefault>>(instances);

    if(ck::get_device_name() != "gfx950")
    {
       add_exp_grouped_conv_fwd_device_operation_instances<
            2,
            NHWGC,
            GKYXC, Empty_Tuple,
            NHWGK,
            F16,
            F16, Empty_Tuple,
            F16,
            PassThrough,
            PassThrough,
            PassThrough,
            device_gemm_xdl_universal_mk_nk_mn_mem_instances_part2<F16, Interwave, GemmDefault>>(instances);
    }
}

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances)
{
   add_exp_grouped_conv_fwd_device_operation_instances<
        3,
        NDHWGC,
        GKZYXC, Empty_Tuple,
        NDHWGK,
        F16,
        F16, Empty_Tuple,
        F16,
        PassThrough,
        PassThrough,
        PassThrough,
        device_gemm_xdl_universal_mk_nk_mn_mem_instances<F16, Interwave, GemmDefault>>(instances);

    if(ck::get_device_name() != "gfx950")
    {
       add_exp_grouped_conv_fwd_device_operation_instances<
            3,
            NDHWGC,
            GKZYXC, Empty_Tuple,
            NDHWGK,
            F16,
            F16, Empty_Tuple,
            F16,
            PassThrough,
            PassThrough,
            PassThrough,
            device_gemm_xdl_universal_mk_nk_mn_mem_instances_part2<F16, Interwave, GemmDefault>>(instances);
    }
}

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
