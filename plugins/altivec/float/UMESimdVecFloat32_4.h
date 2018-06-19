// The MIT License (MIT)
//
// Copyright (c) 2015-2017 CERN
//
// Author: Przemyslaw Karpinski
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
//
//
//  This piece of code was developed as part of ICE-DIP project at CERN.
//  "ICE-DIP is a European Industrial Doctorate project funded by the European Community's 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".
//

#ifndef UME_SIMD_VEC_FLOAT32_4_H_
#define UME_SIMD_VEC_FLOAT32_4_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

#define SET_F32(x, a) { alignas(16) const float setf32_array[4] = {a, a, a, a}; \
                             x = (__vector float) vec_ld(0, setf32_array); }
#define MASK_TO_VEC(x, mask) { alignas(16) uint32_t mask_to_vec_array[4] = { (mask.mMask[0] ? 0xFFFFFFFF : 0), (mask.mMask[1] ? 0xFFFFFFFF : 0), (mask.mMask[2] ? 0xFFFFFFFF : 0), (mask.mMask[3] ? 0xFFFFFFFF : 0)}; \
                             x = (__vector uint32_t) vec_ld(0, mask_to_vec_array); }

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_f<float, 4> :
        public SIMDVecFloatInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_u<uint32_t, 4>,
            SIMDVec_i<int32_t, 4>,
            float,
            4,
            uint32_t,
            int32_t,
            SIMDVecMask<4>,
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_f<float, 4>,
            SIMDVec_f<float, 2>>
    {
    private:
        __vector float mVec;

        typedef SIMDVec_u<uint32_t, 4>    VEC_UINT_TYPE;
        typedef SIMDVec_i<int32_t, 4>     VEC_INT_TYPE;
        typedef SIMDVec_f<float, 2>       HALF_LEN_VEC_TYPE;

        friend class SIMDVec_f<float, 8>;

        UME_FORCE_INLINE explicit SIMDVec_f(__vector float const & x) {
            this->mVec = x;
        }
    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_f() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f) {
            SET_F32(mVec, f);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_f(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, float>::value,
                                    void*>::type = nullptr)
        : SIMDVec_f(static_cast<float>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_f(float const *p) {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) float raw[4] = {p[0], p[1], p[2], p[3]};
            mVec = vec_ld(0, raw);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_f(float f0, float f1, float f2, float f3) {
            alignas(16) float raw[4] = {f0, f1, f2, f3};
            mVec = vec_ld(0, raw);
        }

        // EXTRACT
        UME_FORCE_INLINE float extract(uint32_t index) const {
            return vec_extract(mVec, index);
        }
        UME_FORCE_INLINE float operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_f & insert(uint32_t index, float value) {
            ((float*)&mVec)[index] = value;
	    //mVec = vec_insert(value, mVec, index); throws unused parameter for value, and maybe uninitialized
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_f, float> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_f, float>(index, static_cast<SIMDVec_f &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_f, float, SIMDVecMask<4>>(mask, static_cast<SIMDVec_f &>(*this));
        }
#endif
        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVec_f const & src) {
            mVec = src.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (SIMDVec_f const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<4> const & mask, SIMDVec_f const & src) {
            mVec = vec_sel(mVec, src.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(float b) {
            SET_F32(mVec, b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator= (float b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_f & assign(SIMDVecMask<4> const & mask, float b) {
            __vector float t0;
            SET_F32(t0, b);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_f & load(float const *p) {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) float raw[4] = {p[0], p[1], p[2], p[3]};
            mVec = vec_ld(0, raw);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_f & load(SIMDVecMask<4> const & mask, float const *p) {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) float raw[4] = {p[0], p[1], p[2], p[3]};
            __vector float t0 = vec_ld(0, raw);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_f & loada(float const *p) {
            mVec = vec_ld(0, p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_f & loada(SIMDVecMask<4> const & mask, float const *p) {
            __vector float t0 = vec_ld(0, p);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE float* store(float* p) const {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) float raw[4];
            vec_st(mVec, 0, raw);
            p[0] = raw[0];
            p[1] = raw[1];
            p[2] = raw[2];
            p[3] = raw[3];
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE float* store(SIMDVecMask<4> const & mask, float* p) const {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) float raw[4];
            alignas(16) uint32_t raw_mask[4];
            vec_st(mVec, 0, raw);
            vec_st(mask.mMask, 0, raw_mask);
            if(raw_mask[0] != 0) p[0] = raw[0];
            if(raw_mask[1] != 0) p[1] = raw[1];
            if(raw_mask[2] != 0) p[2] = raw[2];
            if(raw_mask[3] != 0) p[3] = raw[3];
            return p;
        }
        // STOREA
        UME_FORCE_INLINE float* storea(float* p) const {
            vec_st(mVec, 0, p);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE float* storea(SIMDVecMask<4> const & mask, float* p) const {
            alignas(16) float raw[4];
            alignas(16) uint32_t raw_mask[4];
            vec_st(mVec, 0, raw);
            vec_st(mask.mMask, 0, raw_mask);
            if(raw_mask[0] != 0) p[0] = raw[0];
            if(raw_mask[1] != 0) p[1] = raw[1];
            if(raw_mask[2] != 0) p[2] = raw[2];
            if(raw_mask[3] != 0) p[3] = raw[3];
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __vector float t1 = vec_sel(mVec, b.mVec, mask.mMask);
            return SIMDVec_f(t1);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_f blend(SIMDVecMask<4> const & mask, float b) const {
            SIMDVec_f t0(b, b, b, b);  
            __vector float t1 = vec_sel(mVec, t0.mVec, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVec_f const & b) const {
            __vector float t0 = vec_add(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (SIMDVec_f const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __vector float t0 = vec_add(mVec, b.mVec);
            __vector float t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_f add(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_add(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator+ (float b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_f add(SIMDVecMask<4> const & mask, float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_add(mVec, t0);
            __vector float t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVec_f const & b) {
            mVec = vec_add(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (SIMDVec_f const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __vector float t0 = vec_add(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(float b) {
            __vector float t0;
            SET_F32(t0, b);
            mVec = vec_add(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator+= (float b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_f & adda(SIMDVecMask<4> const & mask, float b) {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_add(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
/*        // SADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVec_f const & b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            float t1 = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            float t2 = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            float t3 = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            }
            if (mask.mMask[2] == true) {
                t2 = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            }
            if (mask.mMask[3] == true) {
                t3 = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            }
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_f sadd(float b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            float t1 = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            float t2 = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            float t3 = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_f sadd(SIMDVecMask<4> const & mask, float b) const {
            const float MAX_VAL = std::numeric_limits<float>::max();
            float t0 = mVec[0], t1 = mVec[1], t2 = mVec[2], t3 = mVec[3];
            if (mask.mMask[0] == true) {
                t0 = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            }
            if (mask.mMask[1] == true) {
                t1 = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            }
            if (mask.mMask[2] == true) {
                t2 = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            }
            if (mask.mMask[3] == true) {
                t3 = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            }
            return SIMDVec_f(t0, t1, t2, t3);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVec_f const & b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            mVec[1] = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            mVec[2] = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            mVec[3] = (mVec[3] > MAX_VAL - b.mVec[3]) ? MAX_VAL : mVec[3] + b.mVec[3];
            return *this;
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] > MAX_VAL - b.mVec[0]) ? MAX_VAL : mVec[0] + b.mVec[0];
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] > MAX_VAL - b.mVec[1]) ? MAX_VAL : mVec[1] + b.mVec[1];
            }
            if (mask.mMask[2] == true) {
                mVec[2] = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            }
            if (mask.mMask[1] == true) {
                mVec[2] = (mVec[2] > MAX_VAL - b.mVec[2]) ? MAX_VAL : mVec[2] + b.mVec[2];
            }
            return *this;
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(float b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            mVec[0] = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            mVec[1] = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            mVec[2] = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            mVec[3] = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            return *this;
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_f & sadda(SIMDVecMask<4> const & mask, float b) {
            const float MAX_VAL = std::numeric_limits<float>::max();
            if (mask.mMask[0] == true) {
                mVec[0] = (mVec[0] > MAX_VAL - b) ? MAX_VAL : mVec[0] + b;
            }
            if (mask.mMask[1] == true) {
                mVec[1] = (mVec[1] > MAX_VAL - b) ? MAX_VAL : mVec[1] + b;
            }
            if (mask.mMask[2] == true) {
                mVec[2] = (mVec[2] > MAX_VAL - b) ? MAX_VAL : mVec[2] + b;
            }
            if (mask.mMask[3] == true) {
                mVec[3] = (mVec[3] > MAX_VAL - b) ? MAX_VAL : mVec[3] + b;
            }
            return *this;
        } */
        // POSTINC
        UME_FORCE_INLINE SIMDVec_f postinc() {
            __vector float t0;
            SET_F32(t0, 1.0);
            __vector float t1 = mVec;
            mVec = vec_add(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_f postinc(SIMDVecMask<4> const & mask) {
            __vector float t0;
            SET_F32(t0, 1.0);
            __vector float t1 = mVec;
            __vector float t2 = vec_add(mVec, t0);
            mVec = vec_sel(mVec, t2, mask.mMask);
            return SIMDVec_f(t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc() {
            __vector float t0;
            SET_F32(t0, 1.0);
            mVec = vec_add(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_f & prefinc(SIMDVecMask<4> const & mask) {
            __vector float t0;
            SET_F32(t0, 1.0);
            __vector float t1 = vec_add(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVec_f const & b) const {
            __vector float t0 = vec_sub(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (SIMDVec_f const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __vector float t0 = vec_sub(mVec, b.mVec);
            __vector float t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_f sub(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_sub(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- (float b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_f sub(SIMDVecMask<4> const & mask, float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_sub(mVec, t0);
            __vector float t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVec_f const & b) {
            mVec = vec_sub(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (SIMDVec_f const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __vector float tmp = vec_sub(mVec, b.mVec);
            mVec= vec_sel(mVec, tmp, mask.mMask);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(float b) {
            __vector float t0;
            SET_F32(t0, b);
            mVec = vec_sub(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-= (float b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_f & suba(SIMDVecMask<4> const & mask, float b) {
            __vector float tmp;
            SET_F32(tmp, b);
            __vector float tmp2 = vec_sub(mVec, tmp);
            mVec= vec_sel(mVec, tmp2, mask.mMask);
            return *this;
        }
//        // SSUBV
//        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVec_f const & b) const {
//            const float t0 = std::numeric_limits<float>::min();
//            float t1 = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
//            float t2 = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
//            float t3 = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
//            float t4 = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
//            return SIMDVec_f(t1, t2, t3, t4);
//        }
//        // MSSUBV
//        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
//            const float t0 = std::numeric_limits<float>::min();
//            float t1 = mVec[0], t2 = mVec[1], t3 = mVec[2], t4 = mVec[3];
//            if (mask.mMask[0] == true) {
//                t1 = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
//            }
//            if (mask.mMask[1] == true) {
//                t2 = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
//            }
//            if (mask.mMask[2] == true) {
//                t3 = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
//            }
//            if (mask.mMask[3] == true) {
//                t4 = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
//            }
//            return SIMDVec_f(t1, t2, t3, t4);
//        }
//        // SSUBS
//        UME_FORCE_INLINE SIMDVec_f ssub(float b) const {
//            const float t0 = std::numeric_limits<float>::min();
//            float t1 = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
//            float t2 = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
//            float t3 = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
//            float t4 = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
//            return SIMDVec_f(t1, t2, t3, t4);
//        }
//        // MSSUBS
//        UME_FORCE_INLINE SIMDVec_f ssub(SIMDVecMask<4> const & mask, float b) const {
//            const float t0 = std::numeric_limits<float>::min();
//            float t1 = mVec[0], t2 = mVec[1], t3 = mVec[2], t4 = mVec[3];
//            if (mask.mMask[0] == true) {
//                t1 = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
//            }
//            if (mask.mMask[1] == true) {
//                t2 = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
//            }
//            if (mask.mMask[2] == true) {
//                t3 = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
//            }
//            if (mask.mMask[3] == true) {
//                t4 = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
//            }
//            return SIMDVec_f(t1, t2, t3, t4);
//        }
//        // SSUBVA
//        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVec_f const & b) {
//            const float t0 = std::numeric_limits<float>::min();
//            mVec[0] = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
//            mVec[1] = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
//            mVec[2] = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
//            mVec[3] = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
//            return *this;
//        }
//        // MSSUBVA
//        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
//            const float t0 = std::numeric_limits<float>::min();
//            if (mask.mMask[0] == true) {
//                mVec[0] = (mVec[0] < t0 + b.mVec[0]) ? t0 : mVec[0] - b.mVec[0];
//            }
//            if (mask.mMask[1] == true) {
//                mVec[1] = (mVec[1] < t0 + b.mVec[1]) ? t0 : mVec[1] - b.mVec[1];
//            }
//            if (mask.mMask[2] == true) {
//                mVec[2] = (mVec[2] < t0 + b.mVec[2]) ? t0 : mVec[2] - b.mVec[2];
//            }
//            if (mask.mMask[3] == true) {
//                mVec[3] = (mVec[3] < t0 + b.mVec[3]) ? t0 : mVec[3] - b.mVec[3];
//            }
//            return *this;
//        }
//        // SSUBSA
//        UME_FORCE_INLINE SIMDVec_f & ssuba(float b) {
//            const float t0 = std::numeric_limits<float>::min();
//            mVec[0] = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
//            mVec[1] = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
//            mVec[2] = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
//            mVec[3] = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
//            return *this;
//        }
//        // MSSUBSA
//        UME_FORCE_INLINE SIMDVec_f & ssuba(SIMDVecMask<4> const & mask, float b)  {
//            const float t0 = std::numeric_limits<float>::min();
//            if (mask.mMask[0] == true) {
//                mVec[0] = (mVec[0] < t0 + b) ? t0 : mVec[0] - b;
//            }
//            if (mask.mMask[1] == true) {
//                mVec[1] = (mVec[1] < t0 + b) ? t0 : mVec[1] - b;
//            }
//            if (mask.mMask[2] == true) {
//                mVec[2] = (mVec[2] < t0 + b) ? t0 : mVec[2] - b;
//            }
//            if (mask.mMask[3] == true) {
//                mVec[3] = (mVec[3] < t0 + b) ? t0 : mVec[3] - b;
//            }
//            return *this;
//        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVec_f const & b) const {
            __vector float t0 = vec_sub(b.mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __vector float t0 = vec_sub(b.mVec, mVec);
            __vector float t2 = vec_sel(b.mVec, t0, mask.mMask);
            return SIMDVec_f(t2);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_sub(t0, mVec);
            return SIMDVec_f(t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_f subfrom(SIMDVecMask<4> const & mask, float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_sub(t0, mVec);
            __vector float t3 = vec_sel(t0, t1, mask.mMask);
            return SIMDVec_f(t3);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVec_f const & b) {
            mVec = vec_sub(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __vector float tmp = vec_sub(b.mVec, mVec);
            mVec = vec_sel(b.mVec, tmp, mask.mMask);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(float b) {
            __vector float t0;
            SET_F32(t0, b);
            mVec = vec_sub(t0, mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_f & subfroma(SIMDVecMask<4> const & mask, float b) {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_sub(t0, mVec);
            mVec = vec_sel(t0, t1, mask.mMask);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec() {
            __vector float t0;
            SET_F32(t0, 1.0f);
            __vector float t1 = mVec;
            mVec = vec_sub(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_f postdec(SIMDVecMask<4> const & mask) {
            __vector float t0;
            SET_F32(t0, 1.0f);
            __vector float t1 = mVec;
            __vector float t2 = vec_sub(mVec, t0);
            mVec = vec_sel(mVec, t2, mask.mMask);
            return SIMDVec_f(t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec() {
            __vector float t0;
            SET_F32(t0, 1.0f);
            mVec = vec_sub(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_f & prefdec(SIMDVecMask<4> const & mask) {
            __vector float t0;
            SET_F32(t0, 1.0);
            __vector float t1 = vec_sub(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVec_f const & b) const {
            __vector float t0 = vec_mul(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (SIMDVec_f const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __vector float t0 = mVec * b.mVec;
            __vector float t2 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_f mul(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = mVec * t0;
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator* (float b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_f mul(SIMDVecMask<4> const & mask, float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_mul(mVec, t0);
            __vector float t3 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t3);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVec_f const & b) {
            mVec = vec_mul(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (SIMDVec_f const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __vector float t0 = vec_mul(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_f & mula(float b) {
            __vector float t0;
            SET_F32(t0, b);
            mVec = vec_mul(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator*= (float b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_f & mula(SIMDVecMask<4> const & mask, float b) {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_mul(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVec_f const & b) const {
            __vector float t0 = vec_div(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (SIMDVec_f const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __vector float t0 = vec_div(mVec, b.mVec);
            __vector float t2 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_f(t2);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_f div(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_div(mVec, t0);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator/ (float b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_f div(SIMDVecMask<4> const & mask, float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_div(mVec, t0);
            __vector float t3 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t3);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVec_f const & b) {
            mVec = vec_div(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (SIMDVec_f const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __vector float t0 = vec_div(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(float b) {
            __vector float t0;
            SET_F32(t0, b);
            mVec = vec_div(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_f & operator/= (float b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_f & diva(SIMDVecMask<4> const & mask, float b) {
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_div(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_f rcp() const {
            //__vector double t0 = vec_recip(SET_F64(1.0), mVec);
            __vector float t0;
            SET_F32(t0, 1.0);
            __vector float t1 = vec_div(t0, mVec);
            return SIMDVec_f(t1);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<4> const & mask) const {
            //__vector double t0 = vec_recip(SET_F64(1.0), mVec);
            __vector float t0;
            SET_F32(t0, 1.0);
            __vector float t1 = vec_div(t0, mVec);
            __vector float t3 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t3);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_f rcp(float b) const {
            //__vector double t0 = vec_recip(SET_F64(b), mVec);
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_div(t0, mVec);
            return SIMDVec_f(t1);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_f rcp(SIMDVecMask<4> const & mask, float b) const {
            //__vector double t0 = vec_recip(SET_F64(b), mVec);
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_div(t0, mVec);
            __vector float t3 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t3);
        }
       // RCPA
       UME_FORCE_INLINE SIMDVec_f & rcpa() {
           //__vector double t0 = vec_recip(SET_F64(1.0), mVec);
            __vector float t0;
            SET_F32(t0, 1.0);
            mVec = vec_div(t0, mVec);
            return *this;
       }
       // MRCPA
       UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<4> const & mask) {
           //__vector double t0 = vec_recip(SET_F64(1.0), mVec);
            __vector float t0;
            SET_F32(t0, 1.0);
            __vector float t1 = vec_div(t0, mVec);
            mVec t3 = vec_sel(mVec, t1, mask.mMask);
            return *this;
       }
       // RCPSA
       UME_FORCE_INLINE SIMDVec_f & rcpa(float b) {
           //__vector double t0 = vec_recip(SET_F64(b), mVec);
            __vector float t0;
            SET_F32(t0, b);
            mVec = vec_div(t0, mVec);
            return *this;
       }
       // MRCPSA
       UME_FORCE_INLINE SIMDVec_f & rcpa(SIMDVecMask<4> const & mask, float b) {
            //__vector double t0 = vec_recip(SET_F64(b), mVec);
            __vector float t0;
            SET_F32(t0, b);
            __vector float t1 = vec_div(t0, mVec);
            mVec = vec_sel(mVec, t1, mask.mMask);
           return *this;
       }

        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(SIMDVec_f const & b) const {
            __vector __bool int t0 = vec_cmpeq(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (SIMDVec_f const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector __bool int t1 = vec_cmpeq(mVec, t0);
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (float b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(SIMDVec_f const & b) const {
            __vector float t0;
            union {
                    uint32_t l;
                    float d;
            }magic;

            magic.l = SIMDVecMask<4>::TRUE_VAL();
            SET_F32(t0, magic.d);
            __vector float t1 = vec_xor(vec_cmpeq(mVec, b.mVec), t0);
            return SIMDVecMask<4>((__vector __bool int)t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (SIMDVec_f const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(float b) const {
            __vector float t0, t1;

            union {
                    uint32_t l;
                    float d;
            }magic;

            magic.l = SIMDVecMask<4>::TRUE_VAL();
            SET_F32(t0, magic.d);
            SET_F32(t1, b);
            __vector float t2 = vec_xor(vec_cmpeq(mVec, t1), t0);
            return SIMDVecMask<4>((__vector __bool int)t2);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (float b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(SIMDVec_f const & b) const {
            __vector __bool int t0 = vec_cmpgt(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (SIMDVec_f const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector __bool int t1 = vec_cmpgt(mVec, t0);
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (float b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(SIMDVec_f const & b) const {
            __vector __bool int t0 = vec_cmplt(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (SIMDVec_f const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<4> cmplt(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector __bool int t1 = vec_cmplt(mVec, t0);
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator< (float b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(SIMDVec_f const & b) const {
            __vector __bool int t0 = vec_cmpge(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (SIMDVec_f const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector __bool int t1 = vec_cmpge(mVec, t0);
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (float b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<4> cmple(SIMDVec_f const & b) const {
            __vector __bool int t0 = vec_cmple(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (SIMDVec_f const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<4> cmple(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            __vector __bool int t1 = vec_cmple(mVec, t0);
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (float b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_f const & b) const {
            return vec_all_eq(mVec, b.mVec);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(float b) const {
            __vector float t0;
            SET_F32(t0, b);
            return vec_all_eq(mVec, t0);
        }
//        // UNIQUE
//        UME_FORCE_INLINE bool unique() const {
//            bool m0 = mVec[0] != mVec[1];
//            bool m1 = mVec[0] != mVec[2];
//            bool m2 = mVec[0] != mVec[3];
//            bool m3 = mVec[1] != mVec[2];
//            bool m4 = mVec[1] != mVec[3];
//            bool m5 = mVec[2] != mVec[3];
//            return m0 && m1 && m2 && m3 && m4 && m5;
//        }
//        // HADD
//        UME_FORCE_INLINE float hadd() const {
//            return mVec[0] + mVec[1] + mVec[2] + mVec[3];
//        }
//        // MHADD
//        UME_FORCE_INLINE float hadd(SIMDVecMask<4> const & mask) const {
//            float t0 = mask.mMask[0] ? mVec[0] : 0;
//            float t1 = mask.mMask[1] ? mVec[1] : 0;
//            float t2 = mask.mMask[2] ? mVec[2] : 0;
//            float t3 = mask.mMask[3] ? mVec[3] : 0;
//            return t0 + t1 + t2 + t3;
//        }
//        // HADDS
//        UME_FORCE_INLINE float hadd(float b) const {
//            return mVec[0] + mVec[1] + mVec[2] + mVec[3] + b;
//        }
//        // MHADDS
//        UME_FORCE_INLINE float hadd(SIMDVecMask<4> const & mask, float b) const {
//            float t0 = mask.mMask[0] ? mVec[0] + b : b;
//            float t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
//            float t2 = mask.mMask[2] ? mVec[2] + t1 : t1;
//            float t3 = mask.mMask[3] ? mVec[3] + t2 : t2;
//            return t3;
//        }
//        // HMUL
//        UME_FORCE_INLINE float hmul() const {
//            return mVec[0] * mVec[1] * mVec[2] * mVec[3];
//        }
//        // MHMUL
//        UME_FORCE_INLINE float hmul(SIMDVecMask<4> const & mask) const {
//            float t0 = mask.mMask[0] ? mVec[0] : 1;
//            float t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
//            float t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
//            float t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
//            return t3;
//        }
//        // HMULS
//        UME_FORCE_INLINE float hmul(float b) const {
//            return mVec[0] * mVec[1] * mVec[2] * mVec[3] * b;
//        }
//        // MHMULS
//        UME_FORCE_INLINE float hmul(SIMDVecMask<4> const & mask, float b) const {
//            float t0 = mask.mMask[0] ? mVec[0] * b : b;
//            float t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
//            float t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
//            float t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
//            return t3;
//        }
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __vector float t0 = vec_madd(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_f fmuladd(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __vector float t0 = vec_madd(mVec, b.mVec, c.mVec);
            __vector float t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __vector float t0 = vec_msub(mVec, b.mVec, c.mVec);
            return SIMDVec_f(t0);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_f fmulsub(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __vector float t0 = vec_msub(mVec, b.mVec, c.mVec);
            __vector float t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __vector float t0 = vec_add(mVec, b.mVec);
            __vector float t1 = vec_mul(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_f faddmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __vector float t0 = vec_add(mVec, b.mVec);
            __vector float t1 = vec_mul(t0, c.mVec);
            __vector float t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVec_f const & b, SIMDVec_f const & c) const {
            __vector float t0 = vec_sub(mVec, b.mVec);
            __vector float t1 = vec_mul(t0, c.mVec);
            return SIMDVec_f(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_f fsubmul(SIMDVecMask<4> const & mask, SIMDVec_f const & b, SIMDVec_f const & c) const {
            __vector float t0 = vec_sub(mVec, b.mVec);
            __vector float t1 = vec_mul(t0, c.mVec);
            __vector float t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVec_f const & b) const {
            __vector float t0 = vec_max(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __vector float t0 = vec_max(mVec, b.mVec);
            __vector float t1 = vec_sel(mVec, t2, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_f max(float b) const {
            SIMDVec_f t0(b, b, b, b);
            __vector float t1 = vec_max(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_f max(SIMDVecMask<4> const & mask, float b) const {
            SIMDVec_f t0(b, b, b, b);
            __vector float t1 = vec_max(mVec, t0);
            __vector float t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVec_f const & b) {
            mVec = vec_max(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __vector float t0 = vec_max(mVec, b.mVec);
            mVec = vec_sel(mVec, t2, mask.mMask);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(float b) {
            SIMDVec_f t0(b, b, b, b);
            mVec = vec_max(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_f & maxa(SIMDVecMask<4> const & mask, float b) {
            SIMDVec_f t0(b, b, b, b);
            __vector float t1 = vec_max(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVec_f const & b) const {
            __vector float t0 = vec_min(mVec, b.mVec);
            return SIMDVec_f(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __vector float t0 = vec_min(mVec, b.mVec);
            __vector float t1 = vec_sel(mVec, t2, mask.mMask);
            return SIMDVec_f(t1);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_f min(float b) const {
            SIMDVec_f t0(b, b, b, b);
            __vector float t1 = vec_min(mVec, t0);
            return SIMDVec_f(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_f min(SIMDVecMask<4> const & mask, float b) const {
            SIMDVec_f t0(b, b, b, b);
            __vector float t1 = vec_min(mVec, t0);
            __vector float t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVec_f const & b) {
            mVec = vec_min(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<4> const & mask, SIMDVec_f const & b) {
            __vector float t0 = vec_min(mVec, b.mVec);
            mVec = vec_sel(mVec, t2, mask.mMask);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_f & mina(float b) {
            SIMDVec_f t0(b, b, b, b);
            mVec = vec_min(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_f & mina(SIMDVecMask<4> const & mask, float b) {
            SIMDVec_f t0(b, b, b, b);
            __vector float t1 = vec_min(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
//         // HMAX
//         UME_FORCE_INLINE float hmax () const {
//             float t0 = mVec[0] > mVec[1] ? mVec[0] : mVec[1];
//             float t1 = mVec[2] > mVec[3] ? mVec[2] : mVec[3];
//             return t0 > t1 ? t0 : t1;
//         }
//         // MHMAX
//         UME_FORCE_INLINE float hmax(SIMDVecMask<4> const & mask) const {
//             float t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<float>::lowest();
//             float t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
//             float t2 = (mask.mMask[2] && mVec[2] > t1) ? mVec[2] : t1;
//             float t3 = (mask.mMask[3] && mVec[3] > t2) ? mVec[3] : t2;
//             return t3;
//         }
//         // IMAX
//         UME_FORCE_INLINE uint32_t imax() const {
//             uint32_t t0 = mVec[0] > mVec[1] ? 0 : 1;
//             uint32_t t1 = mVec[2] > mVec[3] ? 2 : 3;
//             return mVec[t0] > mVec[t1] ? t0 : t1;
//         }
//         // MIMAX
//         UME_FORCE_INLINE uint32_t imax(SIMDVecMask<4> const & mask) const {
//             uint32_t i0 = 0xFFFFFFFF;
//             float t0 = std::numeric_limits<float>::lowest();
//             if(mask.mMask[0] == true) {
//                 i0 = 0;
//                 t0 = mVec[0];
//             }
//             if(mask.mMask[1] == true && mVec[1] > t0) {
//                 i0 = 1;
//                 t0 = mVec[1];
//             }
//             if (mask.mMask[2] == true && mVec[2] > t0) {
//                 i0 = 2;
//                 t0 = mVec[2];
//             }
//             if (mask.mMask[3] == true && mVec[3] > t0) {
//                 i0 = 3;
//             }
//             return i0;
//         }
//         // HMIN
//         UME_FORCE_INLINE float hmin() const {
//             float t0 = mVec[0] < mVec[1] ? mVec[0] : mVec[1];
//             float t1 = mVec[2] < mVec[3] ? mVec[2] : mVec[3];
//             return t0 < t1 ? t0 : t1;
//         }
//         // MHMIN
//         UME_FORCE_INLINE float hmin(SIMDVecMask<4> const & mask) const {
//             float t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<float>::max();
//             float t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
//             float t2 = (mask.mMask[2] && mVec[2] < t1) ? mVec[2] : t1;
//             float t3 = (mask.mMask[3] && mVec[3] < t2) ? mVec[3] : t2;
//             return t3;
//         }
//         // IMIN
//         UME_FORCE_INLINE uint32_t imin() const {
//             uint32_t t0 = mVec[0] < mVec[1] ? 0 : 1;
//             uint32_t t1 = mVec[2] < mVec[3] ? 2 : 3;
//             return mVec[t0] < mVec[t1] ? t0 : t1;
//         }
//         // MIMIN
//         UME_FORCE_INLINE uint32_t imin(SIMDVecMask<4> const & mask) const {
//             uint32_t i0 = 0xFFFFFFFF;
//             float t0 = std::numeric_limits<float>::max();
//             if (mask.mMask[0] == true) {
//                 i0 = 0;
//                 t0 = mVec[0];
//             }
//             if (mask.mMask[1] == true && mVec[1] < t0) {
//                 i0 = 1;
//                 t0 = mVec[1];
//             }
//             if (mask.mMask[2] == true && mVec[2] < t0) {
//                 i0 = 2;
//                 t0 = mVec[2];
//             }
//             if (mask.mMask[3] == true && mVec[3] < t0) {
//                 i0 = 3;
//             }
//             return i0;
//         }

//         // GATHERS
//         UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, uint32_t const * indices) {
//             mVec[0] = baseAddr[indices[0]];
//             mVec[1] = baseAddr[indices[1]];
//             mVec[2] = baseAddr[indices[2]];
//             mVec[3] = baseAddr[indices[3]];
//             return *this;
//         }
//         // MGATHERS
//         UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<4> const & mask, float const * baseAddr, uint32_t const * indices) {
//             if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
//             if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
//             if (mask.mMask[2] == true) mVec[2] = baseAddr[indices[2]];
//             if (mask.mMask[3] == true) mVec[3] = baseAddr[indices[3]];
//             return *this;
//         }
//         // GATHERV
//         UME_FORCE_INLINE SIMDVec_f & gather(float const * baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
//             mVec[0] = baseAddr[indices.mVec[0]];
//             mVec[1] = baseAddr[indices.mVec[1]];
//             mVec[2] = baseAddr[indices.mVec[2]];
//             mVec[3] = baseAddr[indices.mVec[3]];
//             return *this;
//         }
//         // MGATHERV
//         UME_FORCE_INLINE SIMDVec_f & gather(SIMDVecMask<4> const & mask, float const * baseAddr, SIMDVec_u<uint32_t, 4> const & indices) {
//             if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
//             if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
//             if (mask.mMask[2] == true) mVec[2] = baseAddr[indices.mVec[2]];
//             if (mask.mMask[3] == true) mVec[3] = baseAddr[indices.mVec[3]];
//             return *this;
//         }
//         // SCATTERS
//         UME_FORCE_INLINE float* scatter(float* baseAddr, uint32_t* indices) const {
//             baseAddr[indices[0]] = mVec[0];
//             baseAddr[indices[1]] = mVec[1];
//             baseAddr[indices[2]] = mVec[2];
//             baseAddr[indices[3]] = mVec[3];
//             return baseAddr;
//         }
//         // MSCATTERS
//         UME_FORCE_INLINE float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, uint32_t* indices) const {
//             if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
//             if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
//             if (mask.mMask[2] == true) baseAddr[indices[2]] = mVec[2];
//             if (mask.mMask[3] == true) baseAddr[indices[3]] = mVec[3];
//             return baseAddr;
//         }
//         // SCATTERV
//         UME_FORCE_INLINE float* scatter(float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) const {
//             baseAddr[indices.mVec[0]] = mVec[0];
//             baseAddr[indices.mVec[1]] = mVec[1];
//             baseAddr[indices.mVec[2]] = mVec[2];
//             baseAddr[indices.mVec[3]] = mVec[3];
//             return baseAddr;
//         }
//         // MSCATTERV
//         UME_FORCE_INLINE float* scatter(SIMDVecMask<4> const & mask, float* baseAddr, SIMDVec_u<uint32_t, 4> const & indices) const {
//             if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
//             if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
//             if (mask.mMask[2] == true) baseAddr[indices.mVec[2]] = mVec[2];
//             if (mask.mMask[3] == true) baseAddr[indices.mVec[3]] = mVec[3];
//             return baseAddr;
//         }
        // NEG
        UME_FORCE_INLINE SIMDVec_f neg() const {
            __vector float t0;
            SET_F32(t0, 0.0f);
            __vector float t1 = vec_sub(t0, mVec);
            return SIMDVec_f(t1);
        }
        UME_FORCE_INLINE SIMDVec_f operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_f neg(SIMDVecMask<4> const & mask) const {
            __vector float t0;
            SET_F32(t0, 0.0f);
            __vector float t1 = vec_sub(t0, mVec);
            __vector float t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t2);
        }

       // NEGA
       UME_FORCE_INLINE SIMDVec_f & nega() {
            __vector float t0;
            SET_F32(t0, 0.0f);
            mVec = vec_sub(t0, mVec);
            return *this;
       }
       // MNEGA
       UME_FORCE_INLINE SIMDVec_f & nega(SIMDVecMask<4> const & mask) {
            __vector float t0;
            SET_F32(t0, 0.0f);
            __vector float t1 = vec_sub(t0, mVec);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
       }
        // ABS
        UME_FORCE_INLINE SIMDVec_f abs() const {
            __vector float t0 = vec_abs(mVec);
            return SIMDVec_f(t0);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_f abs(SIMDVecMask<4> const & mask) const {
            __vector float t0 = vec_abs(mVec);
            __vector float t2 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_f(t2);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_f & absa() {
            mVec = vec_abs(mVec);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_f & absa(SIMDVecMask<4> const & mask) {
            __vector float t0 = vec_abs(mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }

        // COPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVec_f const & b) const {
            __vector float t0 = vec_abs(b.mVec);
            __vector float t1 = vec_xor(b.mVec, t0);
            __vector float t2 = vec_abs(mVec);
            __vector float t3 = vec_or(t1, t2);
            return SIMDVec_f(t3);
        }
        // MCOPYSIGN
        UME_FORCE_INLINE SIMDVec_f copysign(SIMDVecMask<4> const & mask, SIMDVec_f const & b) const {
            __vector float t0 = vec_abs(b.mVec);
            __vector float t1 = vec_xor(b.mVec, t0);
            __vector float t2 = vec_abs(mVec);
            __vector float t3 = vec_or(t1, t2);
            __vector float t5 = vec_sel(mVec, t3, mask.mMask);
            return SIMDVec_f(t5);
        }

        // CMPEQRV
        // CMPEQRS

        // SQR
        UME_FORCE_INLINE SIMDVec_f sqr() const {
            __vector float t0 = vec_mul(mVec, mVec);
            return SIMDVec_f(t0);
        }
        // MSQR
        UME_FORCE_INLINE SIMDVec_f sqr(SIMDVecMask<4> const & mask) const {
            __vector float t0 = vec_mul(mVec, mVec);
            __vector float t1 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_f(t1);
        }
        // SQRA
        UME_FORCE_INLINE SIMDVec_f & sqra() {
            mVec = vec_mul(mVec, mVec);
            return *this;
        }
        // MSQRA
        UME_FORCE_INLINE SIMDVec_f & sqra(SIMDVecMask<8> const & mask) {
            __vector float t0 = vec_mul(mVec, mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // SQRT
        UME_FORCE_INLINE SIMDVec_f sqrt() const {
            __vector float tmp = vec_sqrt(mVec);
            return SIMDVec_f(tmp);
        }
        // MSQRT
        UME_FORCE_INLINE SIMDVec_f sqrt(SIMDVecMask<4> const & mask) const {
            __vector float tmp = vec_sqrt(mVec);
            __vector float tmp2 = vec_sel(mVec, tmp, mask.mMask);
            return SIMDVec_f(tmp2);
        }
        // SQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta() {
            mVec = vec_sqrt(mVec);
            return *this;
        }
        // MSQRTA
        UME_FORCE_INLINE SIMDVec_f & sqrta(SIMDVecMask<4> const & mask) {
            __vector float tmp = vec_sqrt(mVec);
            mVec = vec_sel(mVec, tmp, mask.mMask);
            return *this;
        }
        // POWV
        // MPOWV
        // POWS
        // MPOWS
        // ROUND
        UME_FORCE_INLINE SIMDVec_f round() const {
            __vector float t0 = vec_round(mVec);
            return SIMDVec_f(t0);
        }
        // MROUND
        UME_FORCE_INLINE SIMDVec_f round(SIMDVecMask<4> const & mask) const {
            __vector float t0 = vec_round(mVec);
            __vector float t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // TRUNC
       UME_FORCE_INLINE SIMDVec_i<int32_t, 4> trunc() const {
           __vector float t0 = vec_trunc(mVec);
           __vector int32_t t1 = vec_cts(t0);
           return SIMDVec_i<int32_t, 4>(t1);
       }
       // MTRUNC
       UME_FORCE_INLINE SIMDVec_i<int32_t, 4> trunc(SIMDVecMask<4> const & mask) const {
           SIMDVec_f allZero(0, 0, 0, 0);
           __vector float t0 = vec_trunc(mVec);
           __vector float t1 = vec_sel(allZero.mVec, t0, mask.mMask);
           __vector int32_t t2 = vec_cts(t1);
           return SIMDVec_i<int32_t, 4>(t2);
       }
        // FLOOR
        UME_FORCE_INLINE SIMDVec_f floor() const {
            __vector float t0 = vec_floor(mVec);
            return SIMDVec_f(t0);
        }
        // MFLOOR
        UME_FORCE_INLINE SIMDVec_f floor(SIMDVecMask<4> const & mask) const {
            __vector float t0 = vec_floor(mVec);
            __vector float t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // CEIL
        UME_FORCE_INLINE SIMDVec_f ceil() const {
            __vector float t0 = vec_ceil(mVec);
            return SIMDVec_f(t0);
        }
        // MCEIL
        UME_FORCE_INLINE SIMDVec_f ceil(SIMDVecMask<4> const & mask) const {
            __vector float t0 = vec_ceil(mVec);
            __vector float t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_f(t1);
        }
        // ISFIN
        // ISINF
        // ISAN
        // ISNAN
        // ISSUB
        // ISZERO
        // ISZEROSUB
        // SIN
        // MSIN
        // COS
        // MCOS
        // TAN
        // MTAN
        // CTAN
        // MCTAN

        // PACK
//        UME_FORCE_INLINE SIMDVec_f & pack(SIMDVec_f<float, 2> const & a, SIMDVec_f<float, 2> const & b) {
//            mVec[0] = a[0];
//            mVec[1] = a[1];
//            mVec[2] = b[0];
//            mVec[3] = b[1];
//            return *this;
//        }
//        // PACKLO
//        UME_FORCE_INLINE SIMDVec_f & packlo(SIMDVec_f<float, 2> const & a) {
//            mVec[0] = a[0];
//            mVec[1] = a[1];
//            return *this;
//        }
//        // PACKHI
//        UME_FORCE_INLINE SIMDVec_f & packhi(SIMDVec_f<float, 2> const & b) {
//            mVec[2] = b[0];
//            mVec[3] = b[1];
//            return *this;
//        }
//        // UNPACK
//        void unpack(SIMDVec_f<float, 2> & a, SIMDVec_f<float, 2> & b) const {
//            a.insert(0, mVec[0]);
//            a.insert(1, mVec[1]);
//            b.insert(0, mVec[2]);
//            b.insert(1, mVec[3]);
//        }
//        // UNPACKLO
//        UME_FORCE_INLINE SIMDVec_f<float, 2> unpacklo() const {
//            return SIMDVec_f<float, 2>(mVec[0], mVec[1]);
//        }
//        // UNPACKHI
//        UME_FORCE_INLINE SIMDVec_f<float, 2> unpackhi() const {
//            return SIMDVec_f<float, 2>(mVec[2], mVec[3]);
//        }
        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_f<double, 4>() const;
        // DEGRADE
        // -

        // FTOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 4>() const;
        // FTOI
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 4>() const;
    };

}
}

#undef BLEND
#undef SET_F32
#undef MASK_TO_VEC

#endif
