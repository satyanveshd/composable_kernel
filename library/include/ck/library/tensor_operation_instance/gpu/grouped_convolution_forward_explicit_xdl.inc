// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
// 2D
#ifdef CK_ENABLE_BF16

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

#endif
#ifdef CK_ENABLE_FP16

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
                                                           NHWGC,
                                                           GKYXC, Empty_Tuple,
                                                           NHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif
// 3D
#ifdef CK_ENABLE_BF16

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_mem_v2_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_bf16_bf16_bf16_exp_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           BF16,
                                                           BF16, Empty_Tuple,
                                                           BF16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

#endif
#ifdef CK_ENABLE_FP16

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_comp_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_comp_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_mem_v1_default_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_mem_v1_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);

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
                                                           PassThrough>>>& instances);

void add_device_grouped_convnd_fwd_f16_f16_f16_exp_mem_v2_kpadding_instances(
    std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
                                                           NDHWGC,
                                                           GKZYXC, Empty_Tuple,
                                                           NDHWGK,
                                                           F16,
                                                           F16, Empty_Tuple,
                                                           F16,
                                                           PassThrough,
                                                           PassThrough,
                                                           PassThrough>>>& instances);
#endif

} // namespace instance
} // namespace device
} // namespace tensor_operation
} // namespace ck
