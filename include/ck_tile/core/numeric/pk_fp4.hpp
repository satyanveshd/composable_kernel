// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2025, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck_tile/core/numeric/half.hpp"

namespace ck_tile {

using fp4_t = unsigned _BitInt(4);
// TODO: use fp4x2_t to replace pk_fp4_t
using fp32x2_t = float __attribute__((ext_vector_type(2)));

struct pk_float4_e2m1_t
{
    static constexpr int exponent = 2;
    static constexpr int mantissa = 1;
    static constexpr int bias = 1;
    using raw_type = uint8_t;
    using type = raw_type;
    raw_type data;
    // Refer: ONNX 1.19 Documentation
    static constexpr float e2m1_to_fp32_table[16] = 
    {0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6};
	static constexpr fp16_t e2m1_to_fp16_table[16] = { // Need to TEST this encoding.
		bit_cast<fp16_t>(static_cast<uint16_t>(0x0000)), //  0
		bit_cast<fp16_t>(static_cast<uint16_t>(0x3800)), //  0.5
		bit_cast<fp16_t>(static_cast<uint16_t>(0x3C00)), //  1
		bit_cast<fp16_t>(static_cast<uint16_t>(0x3E00)), //  1.5
		bit_cast<fp16_t>(static_cast<uint16_t>(0x4000)), //  2
		bit_cast<fp16_t>(static_cast<uint16_t>(0x4200)), //  3
		bit_cast<fp16_t>(static_cast<uint16_t>(0x4400)), //  4
		bit_cast<fp16_t>(static_cast<uint16_t>(0x4600)), //  6
		bit_cast<fp16_t>(static_cast<uint16_t>(0x8000)), // -0
		bit_cast<fp16_t>(static_cast<uint16_t>(0xB800)), // -0.5
		bit_cast<fp16_t>(static_cast<uint16_t>(0xBC00)), // -1
		bit_cast<fp16_t>(static_cast<uint16_t>(0xBE00)), // -1.5
		bit_cast<fp16_t>(static_cast<uint16_t>(0xC000)), // -2
		bit_cast<fp16_t>(static_cast<uint16_t>(0xC200)), // -3
		bit_cast<fp16_t>(static_cast<uint16_t>(0xC400)), // -4
		bit_cast<fp16_t>(static_cast<uint16_t>(0xC600))  // -6
	};

    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t(): data{type{}} {}
    CK_TILE_HOST_DEVICE constexpr pk_float4_e2m1_t(type init): data{init} {}
    CK_TILE_HOST_DEVICE constexpr operator type() const { return data; }
    CK_TILE_HOST_DEVICE constexpr operator fp32x2_t() const;
    CK_TILE_HOST_DEVICE constexpr operator fp16x2_t() const;
// add init from two float / get / to_fp8x2
};

using pk_fp4_t = pk_float4_e2m1_t;
using pk_fp4_raw_t = typename pk_fp4_t::raw_type;

template <>
struct numeric_traits<pk_fp4_t>
{
    using bitwise_type = pk_fp4_raw_t;

    static constexpr int exp  = 2;
    static constexpr int mant = 1;
    static constexpr int bias = 1;
    static constexpr uint8_t abs_mask = 0b01110111;
    static constexpr int PackedSize   = 2;
};

// limits
template <class T>
struct numeric;

template <>
struct numeric<pk_fp4_t>
{
    // minimum finite value, or minimum positive normalized value for float
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t min()
    {
        constexpr uint8_t val = 0b00010001; // 0.5, 0.5
        return pk_fp4_t(bit_cast<pk_fp4_raw_t>(val));
    }

    // minumum finite value
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t lowest()
    {
        constexpr uint8_t val = 0b11111111; // -6, -6
        return pk_fp4_t(bit_cast<pk_fp4_raw_t>(val));
    }

    // maximum finite value
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t max()
    {
        constexpr uint8_t val = 0b01110111; // 6, 6
        return pk_fp4_t(bit_cast<pk_fp4_raw_t>(val));
    }

    // difference between 1.0 and next value representable by float
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t epsilon()
    {
	constexpr uint8_t val = 0b00010001; // 0.5, 0.5
	return pk_fp4_t(bit_cast<pk_fp4_raw_t>(val));
    }

    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t round_error()
    {
	constexpr uint8_t val = 0b00010001; // 0.5, 0.5
	return pk_fp4_t(bit_cast<pk_fp4_raw_t>(val));
    }

    // positive infinity value
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t infinity()
    {
        return max();
    }
    // quiet NaN
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t quiet_NaN()
    {
        return max();
    }

    // signaling NaN
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t signaling_NaN()
    {
        return max();
    }

    // smallest positive subnormal value
    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t denorm_min()
    {
        return zero(); // not support
    }

    CK_TILE_HOST_DEVICE static constexpr pk_fp4_t zero() {
		constexpr uint8_t val = 0b00000000; // 0.5, 0.5
		return pk_fp4_t(bit_cast<pk_fp4_raw_t>(val));
    }
};

CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator fp32x2_t() const {
    return fp32x2_t{e2m1_to_fp32_table[data & 0xf], e2m1_to_fp32_table[(data >> 4) & 0xf]};
}
CK_TILE_HOST_DEVICE constexpr pk_fp4_t::operator fp16x2_t() const {
    return fp16x2_t{e2m1_to_fp16_table[data & 0xf], e2m1_to_fp16_table[(data >> 4) & 0xf]};
}

CK_TILE_HOST_DEVICE constexpr pk_fp4_raw_t float_to_e2m1(float x) {
    // {0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6}
	pk_fp4_raw_t res = (x<0 ? 0b00001000 : 0);
	x = std::abs(x);
	if(x < 2.25 ) { res |= int((x + 0.25)*2); }
	else if(x < 2.5) { res |= 0b00000100; }
	else if(x < 3.5) { res |= 0b00000101; }
	else if(x < 5) { res |= 0b00000110; }
	else { res |= 0b00000111; }
	return res;
}
CK_TILE_HOST_DEVICE constexpr fp32x2_t pk_fp4_to_fp32x2_t(const pk_fp4_t& x) { return fp32x2_t(x); }
CK_TILE_HOST_DEVICE constexpr pk_fp4_t fp32x2_t_to_pk_fp4(const fp32x2_t& x) 
{
	return pk_fp4_t(float_to_e2m1(x[0]) | (float_to_e2m1(x[1]) << 4));
}


} // namespace ck_tile
