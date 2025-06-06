// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once
#ifndef __HIPCC_RTC__
#include <hip/hip_runtime.h>

#include "ck/ck.hpp"
#include "ck/utility/env.hpp"
#include "ck/utility/tuple.hpp"
#include "ck/stream_config.hpp"
#include "ck/host_utility/hip_check_error.hpp"

#include <chrono>

template <typename KernelImpl,
          typename... Args>
auto
make_kernel(KernelImpl f, dim3 grid_dim, dim3 block_dim, std::size_t lds_byte, Args... args)
{
    if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
    {
        std::cout << "Make Kernel grid: {" 
                  << grid_dim.x << "," << grid_dim.y << "," << grid_dim.z << "} "
                  << "block: {"
                  << block_dim.x << "," << block_dim.y << "," << block_dim.z << "} "
                  << std::endl;
    }
    return [=](const StreamConfig& s) {
        f<<<grid_dim, block_dim, lds_byte, s.stream_id_>>>(args...);
    };
}

template <typename Callable>
void launch_and_check(const StreamConfig& sc, Callable&& callable)
{
    if(!(static_cast<void>(callable(sc)), hipPeekAtLastError() == hipSuccess))
    {
        HIP_CHECK_ERROR(hipGetLastError());
    }
}

template <typename... StreamConfigs, typename... Callables>
float launch_and_time_kernels(const ck::Tuple<StreamConfigs...>& s_configs,
                              const ck::Tuple<Callables...>& callables)
{
    std::size_t cold_iters = s_configs[ck::Number<0>{}].cold_niters_;
    std::size_t nrepeat = s_configs[ck::Number<0>{}].nrepeat_;
    
    // warm up
    for(std::size_t c_it = 0; c_it < cold_iters; ++c_it)
    {
        ck::static_for<0, ck::Tuple<StreamConfigs...>::Size(), 1>{}(
            [&](auto i) {
                using SC_t = ck::tuple_element_t<i, ck::Tuple<StreamConfigs...>>;
                using K_t = ck::tuple_element_t<i, ck::Tuple<Callables...>>;
                launch_and_check(std::forward<const SC_t&>(s_configs[ck::Number<i>{}]),
                                 std::forward<const K_t&>(callables[ck::Number<i>{}]));
            });
    }

    std::chrono::time_point<std::chrono::high_resolution_clock> start_tick;
    std::chrono::time_point<std::chrono::high_resolution_clock> stop_tick;

    hip_check_error(hipDeviceSynchronize());
    start_tick = std::chrono::high_resolution_clock::now();
    for(std::size_t c_it = 0; c_it < nrepeat; ++c_it)
    {
        ck::static_for<0, ck::Tuple<StreamConfigs...>::Size(), 1>{}(
            [&](auto i) {
                using SC_t = ck::tuple_element_t<i, ck::Tuple<StreamConfigs...>>;
                using K_t = ck::tuple_element_t<i, ck::Tuple<Callables...>>;
                launch_and_check(std::forward<const SC_t&>(s_configs[ck::Number<i>{}]),
                                 std::forward<const K_t&>(callables[ck::Number<i>{}]));
            });
    }
    hip_check_error(hipDeviceSynchronize());
    stop_tick = std::chrono::high_resolution_clock::now();

    double sec =
            std::chrono::duration_cast<std::chrono::duration<double>>(stop_tick - start_tick)
                .count();
    float total_time = static_cast<float>(sec * 1e3);

    return total_time / nrepeat;
}

template <typename... Args, typename F>
float launch_and_time_kernel(const StreamConfig& stream_config,
                             F kernel,
                             dim3 grid_dim,
                             dim3 block_dim,
                             std::size_t lds_byte,
                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
                   __func__,
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Warm up %d times\n", stream_config.cold_niters_);
        }
        // warm up
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        const int nrepeat = stream_config.nrepeat_;
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("Start running %d times...\n", nrepeat);
        }
        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(start, stream_config.stream_id_));

        for(int i = 0; i < nrepeat; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(stop));

        float total_time = 0;

        hip_check_error(hipEventElapsedTime(&total_time, start, stop));

        hip_check_error(hipEventDestroy(start));
        hip_check_error(hipEventDestroy(stop));

        return total_time / nrepeat;
    }
    else
    {
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}

template <typename... Args, typename F, typename PreProcessFunc>
float launch_and_time_kernel_with_preprocess(const StreamConfig& stream_config,
                                             PreProcessFunc preprocess,
                                             F kernel,
                                             dim3 grid_dim,
                                             dim3 block_dim,
                                             std::size_t lds_byte,
                                             Args... args)
{
#if CK_TIME_KERNEL
    if(stream_config.time_kernel_)
    {
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("%s: grid_dim {%u, %u, %u}, block_dim {%u, %u, %u} \n",
                   __func__,
                   grid_dim.x,
                   grid_dim.y,
                   grid_dim.z,
                   block_dim.x,
                   block_dim.y,
                   block_dim.z);

            printf("Warm up %d times\n", stream_config.cold_niters_);
        }
        // warm up
        preprocess();
        for(int i = 0; i < stream_config.cold_niters_; ++i)
        {
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        const int nrepeat = stream_config.nrepeat_;
        if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
        {
            printf("Start running %d times...\n", nrepeat);
        }
        hipEvent_t start, stop;

        hip_check_error(hipEventCreate(&start));
        hip_check_error(hipEventCreate(&stop));

        hip_check_error(hipDeviceSynchronize());
        hip_check_error(hipEventRecord(start, stream_config.stream_id_));

        for(int i = 0; i < nrepeat; ++i)
        {
            preprocess();
            kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
            hip_check_error(hipGetLastError());
        }

        hip_check_error(hipEventRecord(stop, stream_config.stream_id_));
        hip_check_error(hipEventSynchronize(stop));

        float total_time = 0;

        hip_check_error(hipEventElapsedTime(&total_time, start, stop));

        hip_check_error(hipEventDestroy(start));
        hip_check_error(hipEventDestroy(stop));

        return total_time / nrepeat;
    }
    else
    {
        preprocess();
        kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
        hip_check_error(hipGetLastError());

        return 0;
    }
#else
    kernel<<<grid_dim, block_dim, lds_byte, stream_config.stream_id_>>>(args...);
    hip_check_error(hipGetLastError());

    return 0;
#endif
}
#endif
