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

#ifndef UME_SIMD_VEC_INT32_4_H_
#define UME_SIMD_VEC_INT32_4_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

#define SET_I32(x, a) { alignas(16) int32_t seti32_array[4] = {a, a, a, a}; \
                      x = vec_ld(0, seti32_array); }
#define SET_UI32(x, a) { alignas(16) uint32_t setui32_array[4] = {a, a, a, a}; \
                      x = vec_ld(0, setui32_array); }

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 4> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int32_t, 4>,
            SIMDVec_u<uint32_t, 4>,
            int32_t,
            4,
            uint32_t,
            SIMDVecMask<4>,
            SIMDSwizzle<4>> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int32_t, 4>,
            SIMDVec_i<int32_t, 2>>
    {
    private:
        __vector signed int mVec;

        friend class SIMDVec_u<uint32_t, 4>;
        friend class SIMDVec_f<float, 4>;
        friend class SIMDVec_f<double, 4>;

        friend class SIMDVec_i<int32_t, 8>;
        
        UME_FORCE_INLINE explicit SIMDVec_i(__vector signed int const & x) {
            this->mVec = x;
        }
    public:
        constexpr static uint32_t length() { return 4; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i) {
            SET_I32(mVec, i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, int32_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<int32_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(int32_t const *p) {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) int32_t raw[4] = {p[0], p[1], p[2], p[3]};
            mVec = vec_ld(0, raw);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3) {
            alignas(16) int32_t raw[4] = {i0, i1, i2, i3};
            mVec = vec_ld(0, raw);
        }

        // EXTRACT
        UME_FORCE_INLINE int32_t extract(uint32_t index) const {
            return ((unsigned int*)&mVec)[index];
        }
        UME_FORCE_INLINE int32_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int32_t value) {
            ((signed int*)&mVec)[index] = value;
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int32_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int32_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>> operator() (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>> operator[] (SIMDVecMask<4> const & mask) {
            return IntermediateMask<SIMDVec_i, int32_t, SIMDVecMask<4>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & src) {
            mVec = src.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<4> const & mask, SIMDVec_i const & src) {
            mVec = vec_sel(mVec, src.mVec, mask.mMask);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(int32_t b) {
            SET_I32(mVec, b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (int32_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<4> const & mask, int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(int32_t const *p) {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) int32_t raw[4] = {p[0], p[1], p[2], p[3]};
            mVec = vec_ld(0, raw);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<4> const & mask, int32_t const *p) {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) int32_t raw[4] = {p[0], p[1], p[2], p[3]};
            __vector int32_t t0 = vec_ld(0, raw);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int32_t const *p) {
            mVec = vec_ld(0, p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<4> const & mask, int32_t const *p) {
            __vector int32_t t0 = vec_ld(0, p);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int32_t* store(int32_t* p) const {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) int32_t raw[4];
            vec_st(mVec, 0, raw);
            p[0] = raw[0];
            p[1] = raw[1];
            p[2] = raw[2];
            p[3] = raw[3];
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int32_t* store(SIMDVecMask<4> const & mask, int32_t* p) const {
            // From PIM:
            // "In the AltiVec architecture, an unaligned load/store does not cause an 
            // alignment exception that might lead to (slow) loading of the bytes at the 
            // given address. Instead, the low-order bits of the address are quietly ignored."
            
            // The data needs to be re-aligned so that we don't loose bits.
            alignas(16) int32_t raw[4];
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
        UME_FORCE_INLINE int32_t* storea(int32_t* p) const {
            vec_st(mVec, 0, p);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE int32_t* storea(SIMDVecMask<4> const & mask, int32_t* p) const {
            __vector int32_t t0 = vec_ld(0, p);
            __vector int32_t t1 = vec_sel(t0, mVec, mask.mMask);
            vec_st(t1, 0, p);
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_sel(mVec, b.mVec, mask.mMask);
            return SIMDVec_i(t0);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<4> const & mask, int32_t b) const {
            __vector int32_t t0, t1;
            SET_I32(t0, b);
            t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_add(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_add(mVec, b.mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_add(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (int32_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<4> const & mask, int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_add(mVec, t0);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = vec_add(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_add(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            mVec = vec_add(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (int32_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<4> const & mask, int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_add(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // SADDV
        UME_FORCE_INLINE SIMDVec_i sadd(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_adds(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MSADDV
        UME_FORCE_INLINE SIMDVec_i sadd(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_adds(mVec, b.mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // SADDS
        UME_FORCE_INLINE SIMDVec_i sadd(int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_adds(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MSADDS
        UME_FORCE_INLINE SIMDVec_i sadd(SIMDVecMask<4> const & mask, int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_adds(mVec, t0);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // SADDVA
        UME_FORCE_INLINE SIMDVec_i & sadda(SIMDVec_i const & b) {
            mVec = vec_adds(mVec, b.mVec);

            return *this;
        }
        // MSADDVA
        UME_FORCE_INLINE SIMDVec_i & sadda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_adds(mVec, b.mVec);
            mVec= vec_sel(mVec, t0, mask.mMask);

            return *this;
        }
        // SADDSA
        UME_FORCE_INLINE SIMDVec_i & sadda(int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            mVec = vec_adds(mVec, t0);

            return *this;
        }
        // MSADDSA
        UME_FORCE_INLINE SIMDVec_i & sadda(SIMDVecMask<4> const & mask, int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_adds(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);

            return *this;
        }
        // POSTINC
        UME_FORCE_INLINE SIMDVec_i postinc() {
            __vector int32_t t0;
            SET_I32(t0, 1);
            __vector int32_t t1 = mVec;
            mVec = vec_add(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<4> const & mask) {
            __vector int32_t t0;
            SET_I32(t0, 1);
            __vector int32_t t1 = mVec;
            __vector int32_t t2 = vec_add(mVec, t0);
            mVec = vec_sel(mVec, t2, mask.mMask);
            return SIMDVec_i(t1);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            __vector int32_t t0;
            SET_I32(t0, 1);
            mVec = vec_add(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<4> const & mask) {
            __vector int32_t t0;
            SET_I32(t0, 1);
            __vector int32_t t1 = vec_add(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_sub(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_sub(mVec, b.mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_sub(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (int32_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<4> const & mask, int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_sub(mVec, t0);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = vec_sub(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_sub(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            mVec = vec_sub(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (int32_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<4> const & mask, int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_sub(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // SSUBV
        UME_FORCE_INLINE SIMDVec_i ssub(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_subs(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MSSUBV
        UME_FORCE_INLINE SIMDVec_i ssub(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_subs(mVec, b.mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // SSUBS
        UME_FORCE_INLINE SIMDVec_i ssub(int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_subs(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MSSUBS
        UME_FORCE_INLINE SIMDVec_i ssub(SIMDVecMask<4> const & mask, int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_subs(mVec, t0);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // SSUBVA
        UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVec_i const & b) {
            mVec = vec_subs(mVec, b.mVec);
            return *this;
        }
        // MSSUBVA
        UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_subs(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // SSUBSA
        UME_FORCE_INLINE SIMDVec_i & ssuba(int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            mVec = vec_subs(mVec, t0);
            return *this;
        }
        // MSSUBSA
        UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVecMask<4> const & mask, int32_t b)  {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_subs(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_sub(b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_sub(b.mVec, mVec);
            __vector int32_t t1 = vec_sel(b.mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b)
            __vector int32_t t1 = vec_sub(t0, mVec);
            return SIMDVec_i(t1);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<4> const & mask, int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b)
            __vector int32_t t1 = vec_sub(t0, mVec);
            __vector int32_t t2 = vec_sel(t0, t1, mask.mMask);
            return SIMDVec_i(t2);
        } 
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = vec_sub(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t tmp = vec_sub(b.mVec, mVec);
            mVec = vec_sel(b.mVec, tmp, mask.mMask);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            mVec = vec_sub(t0, mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<4> const & mask, int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_sub(t0, mVec);
            mVec = vec_sel(t0, t1, mask.mMask);
            return *this;
        } 
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec() {
            __vector int32_t t0;
            SET_I32(t0, 1);
            __vector int32_t t1 = mVec;
            mVec = vec_sub(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec(SIMDVecMask<4> const & mask) {
            __vector int32_t t0;
            SET_I32(t0, 1);
            __vector int32_t t1 = mVec;
            __vector int32_t t2 = vec_sub(mVec, t0);
            mVec = vec_sel(mVec, t2, mask.mMask);
            return SIMDVec_i(t1);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec() {
            __vector int32_t t0;
            SET_I32(t0, 1);
            mVec = vec_sub(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec(SIMDVecMask<4> const & mask) {
            __vector int32_t t0;
            SET_I32(t0, 1);
            __vector int32_t t1 = vec_sub(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        } 
        // MULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_mul(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_mul(mVec, b.mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_i mul(int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b)
            __vector int32_t t1 = vec_mul(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (int32_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<4> const & mask, int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b)
            __vector int32_t t1 = vec_mul(mVec, t0);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec = vec_mul(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_mul(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_i & mula(int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            mVec = vec_mul(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (int32_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<4> const & mask, int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_mul(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_i div(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_div(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_div(mVec, b.mVec);
            __vector int32_t t2 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t2);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_i div(int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_div(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (int32_t b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<4> const & mask, int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_div(mVec, t0);
            __vector int32_t t3 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t3);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVec_i const & b) {
            mVec = vec_div(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (SIMDVec_i const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_div(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_i & diva(int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            mVec = vec_div(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (int32_t b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<4> const & mask, int32_t b) {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_div(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_i rcp() const {
            //__vector double t0 = vec_recip(SET_F64(1.0), mVec);
            __vector int32_t t0;
            SET_I32(t0, 1.0);
            __vector int32_t t1 = vec_div(t0, mVec);
            return SIMDVec_i(t1);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_i rcp(SIMDVecMask<4> const & mask) const {
            //__vector double t0 = vec_recip(SET_F64(1.0), mVec);
            __vector int32_t t0;
            SET_I32(t0, 1.0);
            __vector int32_t t1 = vec_div(t0, mVec);
            __vector int32_t t3 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t3);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_i rcp(int32_t b) const {
            //__vector double t0 = vec_recip(SET_F64(b), mVec);
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_div(t0, mVec);
            return SIMDVec_i(t1);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_i rcp(SIMDVecMask<4> const & mask, int32_t b) const {
            //__vector double t0 = vec_recip(SET_F64(b), mVec);
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_div(t0, mVec);
            __vector int32_t t3 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t3);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_i & rcpa() {
           //__vector double t0 = vec_recip(SET_F64(1.0), mVec);
            __vector int32_t t0;
            SET_I32(t0, 1.0);
            mVec = vec_div(t0, mVec);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_i & rcpa(SIMDVecMask<4> const & mask) {
           //__vector double t0 = vec_recip(SET_F64(1.0), mVec);
            __vector int32_t t0;
            SET_I32(t0, 1.0);
            __vector int32_t t1 = vec_div(t0, mVec);
            mVec t3 = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_i & rcpa(int32_t b) {
           //__vector double t0 = vec_recip(SET_F64(b), mVec);
            __vector int32_t t0;
            SET_I32(t0, b);
            mVec = vec_div(t0, mVec);
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_i & rcpa(SIMDVecMask<4> const & mask, int32_t b) {
            //__vector double t0 = vec_recip(SET_F64(b), mVec);
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector int32_t t1 = vec_div(t0, mVec);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
       
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(SIMDVec_i const & b) const {
            // __vector __bool int32_t and __vector int32_t does not work
            __vector __bool int t0 = vec_cmpeq(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<4> cmpeq(int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector __bool int t1 = vec_cmpeq(mVec, t0);
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator== (int32_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(SIMDVec_i const & b) const {
            __vector int32_t t0;
            SET_I32(t0, (int32_t) SIMDVecMask<4>::TRUE_VAL());
            __vector int32_t t1 = vec_xor(vec_cmpeq(mVec, b.mVec), t0);
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<4> cmpne(int32_t b) const {
            __vector int32_t t0, t1;
            SET_I32(t0, (int32_t) SIMDVecMask<4>::TRUE_VAL());
            SET_I32(t1, b);
            __vector int32_t t2 = vec_xor(vec_cmpeq(mVec, t1), t0);
            return SIMDVecMask<4>(t2);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator!= (int32_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(SIMDVec_i const & b) const {
            __vector __bool int t0 = vec_cmpgt(mVec, b.mVec);
            return SIMDVecMask<4>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<4> cmpgt(int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            __vector __bool int t1 = vec_cmpgt(mVec, t0);
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator> (int32_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
         UME_FORCE_INLINE SIMDVecMask<4> cmplt(SIMDVec_i const & b) const {
             __vector __bool int t0 = vec_cmplt(mVec, b.mVec);
             return SIMDVecMask<4>(t0);
         }
         UME_FORCE_INLINE SIMDVecMask<4> operator< (SIMDVec_i const & b) const {
             return cmplt(b);
         }
        // CMPLTS
         UME_FORCE_INLINE SIMDVecMask<4> cmplt(int32_t b) const {
             __vector int32_t t0;
             SET_I32(t0, b);
             __vector __bool int t1 = vec_cmplt(mVec, t0);
             return SIMDVecMask<4>(t1);
         }
         UME_FORCE_INLINE SIMDVecMask<4> operator< (int32_t b) const {
             return cmplt(b);
         }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(SIMDVec_i const & b) const {
            __vector __bool int t0 = vec_or(vec_cmpgt(mVec, b.mVec), vec_cmpeq(mVec, b.mVec));
            return SIMDVecMask<4>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<4> cmpge(int32_t b) const {
            __vector int32_t t0;
                SET_I32(t0, b);
            __vector __bool int t1 = vec_or(vec_cmpgt(mVec, t0), vec_cmpeq(mVec, t0));
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator>= (int32_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<4> cmple(SIMDVec_i const & b) const {
            __vector __bool int t0 = vec_or(vec_cmplt(mVec, b.mVec), vec_cmpeq(mVec, b.mVec));
            return SIMDVecMask<4>(t0);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<4> cmple(int32_t b) const {
            __vector int32_t t0;
                SET_I32(t0, b);
                __vector __bool int t1 = vec_or(vec_cmplt(mVec, t0), vec_cmpeq(mVec, t0));
            return SIMDVecMask<4>(t1);
        }
        UME_FORCE_INLINE SIMDVecMask<4> operator<= (int32_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe(SIMDVec_i const & b) const {
            return vec_all_eq(mVec, b.mVec);
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, b);
            return vec_all_eq(mVec, t0);
        }
        // UNIQUE
        UME_FORCE_INLINE bool unique() const {
            __vector uint32_t t1, t2, t3;

            SET_UI32(t1, sizeof(int32_t) * 8);
            __vector int32_t m1 = vec_rl(mVec, t1);
            int32_t res1 = vec_all_ne(mVec, m1);

            SET_UI32(t2, 2 * sizeof(int32_t) * 8);
            __vector int32_t m2 = vec_rl(mVec, t2);
            int32_t res2 = vec_all_ne(mVec, m2);


            SET_UI32(t3, 3 * sizeof(int32_t) * 8);
            __vector int32_t m3 = vec_rl(mVec, t3);
            int32_t res3 = vec_all_ne(mVec, m3);

            return res1 && res2 && res3;

//            probably is this one faster (must change store)
//            alignas(16) int32_t raw[4];
//            _mm_store_si128((__m128i*)raw, mVec);
//            for (unsigned int i = 0; i < 3; i++) {
//                for (unsigned int j = i + 1; j < 4; j++) {
//                    if (raw[i] == raw[j]) return false;
//                }
//            }
//            return true;
        }
        // HADD
        UME_FORCE_INLINE int32_t hadd() const {
            // test if this is faster:
            // 1) a,b,c,d
            // 2) rotate left: b,c,d,a
            // 3) add 1) + 2) = a+b, b+c, c+d, d+a
            // 4) permute 3) = c+d, d+a, a+b, b+c
            // 5) add 3) + 4) = 4x  sum
            // 6) return single element

            alignas(16) int32_t raw[4];
            vec_st(mVec, 0, raw);

            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // MHADD
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<4> const & mask) const {
            // we know: mask element either 0xFFF.. or 0x000..
            // and() = write mVec[i] where mask[i] = 0xFFF.., else write 0x000..
            __vector int32_t t0 = vec_and(mVec, (__vector int32_t) mask.mMask);

            alignas(16) int32_t raw[4];
            vec_st(t0, 0, raw);

            return raw[0] + raw[1] + raw[2] + raw[3];
        }
        // HADDS
        UME_FORCE_INLINE int32_t hadd(int32_t b) const {
            // see HADD for "maybe" improvement

            alignas(16) int32_t raw[4];
            vec_st(mVec, 0, raw);

            return raw[0] + raw[1] + raw[2] + raw[3] + b;
        }
        // MHADDS
        UME_FORCE_INLINE int32_t hadd(SIMDVecMask<4> const & mask, int32_t b) const {
            __vector int32_t t0;
            SET_I32(t0, 0);
            __vector int32_t t1 = vec_sel(t0, mVec, mask.mMask);

            alignas(16) int32_t raw[4];
            vec_st(t1, 0, raw);

            return raw[0] + raw[1] + raw[2] + raw[3] + b;
        } 
//         // HMUL
//         UME_FORCE_INLINE int32_t hmul() const {
//             alignas(16) int32_t raw[4];
//             vec_st(mVec, 0, raw);
// 
//             return raw[0] * raw[1] * raw[2] * raw[3];
//         }
//         // MHMUL
//         UME_FORCE_INLINE int32_t hmul(SIMDVecMask<4> const & mask) const {
//             int32_t t0 = mask.mMask[0] ? mVec[0] : 1;
//             int32_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
//             int32_t t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
//             int32_t t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
//             return t3;
//         }
//         // HMULS
//         UME_FORCE_INLINE int32_t hmul(int32_t b) const {
//             return mVec[0] * mVec[1] * mVec[2] * mVec[3] * b;
//         }
//         // MHMULS
//         UME_FORCE_INLINE int32_t hmul(SIMDVecMask<4> const & mask, int32_t b) const {
//             int32_t t0 = mask.mMask[0] ? mVec[0] * b : b;
//             int32_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
//             int32_t t2 = mask.mMask[2] ? mVec[2] * t1 : t1;
//             int32_t t3 = mask.mMask[3] ? mVec[3] * t2 : t2;
//             return t3;
//         } 
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __vector int32_t t0 = vec_madd(mVec, b.mVec, c.mVec);
            return SIMDVec_i(t0);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __vector int32_t t0 = vec_madd(mVec, b.mVec, c.mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __vector int32_t t0 = vec_msub(mVec, b.mVec, c.mVec);
            return SIMDVec_i(t0);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __vector int32_t t0 = vec_msub(mVec, b.mVec, c.mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __vector int32_t t0 = vec_add(mVec, b.mVec);
            __vector int32_t t1 = vec_mul(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __vector int32_t t0 = vec_add(mVec, b.mVec);
            __vector int32_t t1 = vec_mul(t0, c.mVec);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __vector int32_t t0 = vec_sub(mVec, b.mVec);
            __vector int32_t t1 = vec_mul(t0, c.mVec);
            return SIMDVec_i(t1);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVecMask<4> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __vector int32_t t0 = vec_sub(mVec, b.mVec);
            __vector int32_t t1 = vec_mul(t0, c.mVec);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_max(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_max(mVec, b.mVec);
            __vector int32_t t1 = vec_sel(mVec, t2, mask.mMask);
            return SIMDVec_i(t1);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_i max(int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_max(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<4> const & mask, int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_max(mVec, t0);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = vec_max(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_max(mVec, b.mVec);
            mVec = vec_sel(mVec, t2, mask.mMask);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            mVec = vec_max(mVec, t0);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<4> const & mask, int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_max(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_min(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_min(mVec, b.mVec);
            __vector int32_t t1 = vec_sel(mVec, t2, mask.mMask);
            return SIMDVec_i(t1);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_i min(int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_min(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<4> const & mask, int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_min(mVec, t0);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = vec_min(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_min(mVec, b.mVec);
            mVec = vec_sel(mVec, t2, mask.mMask);
            return *this;
        }
        // MINSA
        UME_FORCE_INLINE SIMDVec_i & mina(int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            mVec = vec_min(mVec, t0);
            return *this;
        }
        // MMINSA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<4> const & mask, int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_min(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
//         // HMAX
//         UME_FORCE_INLINE int32_t hmax () const {
//             int32_t t0 = mVec[0] > mVec[1] ? mVec[0] : mVec[1];
//             int32_t t1 = mVec[2] > mVec[3] ? mVec[2] : mVec[3];
//             return t0 > t1 ? t0 : t1;
//         }
//         // MHMAX
//         UME_FORCE_INLINE int32_t hmax(SIMDVecMask<4> const & mask) const {
//             int32_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<int32_t>::min();
//             int32_t t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
//             int32_t t2 = (mask.mMask[2] && mVec[2] > t1) ? mVec[2] : t1;
//             int32_t t3 = (mask.mMask[3] && mVec[3] > t2) ? mVec[3] : t2;
//             return t3;
//         }
//         // IMAX
//         UME_FORCE_INLINE uint32_t imax() const {
//             int32_t t0 = mVec[0] > mVec[1] ? 0 : 1;
//             int32_t t1 = mVec[2] > mVec[3] ? 2 : 3;
//             return mVec[t0] > mVec[t1] ? t0 : t1;
//         }
//         // MIMAX
//         UME_FORCE_INLINE uint32_t imax(SIMDVecMask<4> const & mask) const {
//             uint32_t i0 = 0xFFFFFFFF;
//             int32_t t0 = std::numeric_limits<int32_t>::min();
//             if(mask.mMask[0] == true) {
//                 i0 = 0;
//                 t0 = mVec[0];
//             }
//             if((mask.mMask[1] == true) && (mVec[1] > t0)) {
//                 i0 = 1;
//                 t0 = mVec[1];
//             }
//             if ((mask.mMask[2] == true) && (mVec[2] > t0)) {
//                 i0 = 2;
//                 t0 = mVec[2];
//             }
//             if ((mask.mMask[3] == true) && (mVec[3] > t0)) {
//                 i0 = 3;
//             }
//             return i0;
//         }
//         // HMIN
//         UME_FORCE_INLINE int32_t hmin() const {
//             int32_t t0 = mVec[0] < mVec[1] ? mVec[0] : mVec[1];
//             int32_t t1 = mVec[2] < mVec[3] ? mVec[2] : mVec[3];
//             return t0 < t1 ? t0 : t1;
//         }
//         // MHMIN
//         UME_FORCE_INLINE int32_t hmin(SIMDVecMask<4> const & mask) const {
//             int32_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<int32_t>::max();
//             int32_t t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
//             int32_t t2 = (mask.mMask[2] && mVec[2] < t1) ? mVec[2] : t1;
//             int32_t t3 = (mask.mMask[3] && mVec[3] < t2) ? mVec[3] : t2;
//             return t3;
//         }
//         // IMIN
//         UME_FORCE_INLINE uint32_t imin() const {
//             int32_t t0 = mVec[0] < mVec[1] ? 0 : 1;
//             int32_t t1 = mVec[2] < mVec[3] ? 2 : 3;
//             return mVec[t0] < mVec[t1] ? t0 : t1;
//         }
//         // MIMIN
//         UME_FORCE_INLINE uint32_t imin(SIMDVecMask<4> const & mask) const {
//             uint32_t i0 = 0xFFFFFFFF;
//             int32_t t0 = std::numeric_limits<int32_t>::max();
//             if (mask.mMask[0] == true) {
//                 i0 = 0;
//                 t0 = mVec[0];
//             }
//             if ((mask.mMask[1] == true) && mVec[1] < t0) {
//                 i0 = 1;
//                 t0 = mVec[1];
//             }
//             if ((mask.mMask[2] == true) && mVec[2] < t0) {
//                 i0 = 2;
//                 t0 = mVec[2];
//             }
//             if ((mask.mMask[3] == true) && mVec[3] < t0) {
//                 i0 = 3;
//             }
//             return i0;
//         }

        // BANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_and(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_and(mVec, b.mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_i band(int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_and(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (int32_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<4> const & mask, int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_and(mVec, t0);
            __vector int32_t t2 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t2);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec = vec_and(mVec, b.mVec)
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_and(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            mVec = vec_and(mVec, t0);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<4> const & mask, int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_and(mVec, t0);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_or(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_or(mVec, b.mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_i bor(int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_or(mVec, t0.mVec);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (int32_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<4> const & mask, int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_or(mVec, t0.mVec);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec = vec_or(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_or(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_i & bora(int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            mVec = vec_or(mVec, t0.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (int32_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<4> const & mask, int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_or(mVec, t0.mVec);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_xor(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_xor(mVec, b.mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_i bxor(int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_xor(mVec, t0.mVec);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (int32_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<4> const & mask, int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_xor(mVec, t0.mVec);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec = vec_xor(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_xor(mVec, b.mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            mVec = vec_xor(mVec, t0.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (int32_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<4> const & mask, int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_xor(mVec, t0.mVec);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_i bnot() const {
            __vector int32_t t0 = vec_nand(mVec, mVec);
	    return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_i bnot(SIMDVecMask<4> const & mask) const {
            __vector int32_t t0 = vec_nand(mVec, mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota() {
            mVec = vec_nand(mVec, mVec);
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota(SIMDVecMask<4> const & mask) {
            __vector int32_t t0 = vec_nand(mVec, mVec);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
//         // HBAND
//         UME_FORCE_INLINE int32_t hband() const {
//             return mVec[0] & mVec[1] & mVec[2] & mVec[3];
//         }
//         // MHBAND
//         UME_FORCE_INLINE int32_t hband(SIMDVecMask<4> const & mask) const {
//             int32_t t0 = mask.mMask[0] ? mVec[0] : 0xFFFFFFFF;
//             int32_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
//             int32_t t2 = mask.mMask[2] ? mVec[2] & t1 : t1;
//             int32_t t3 = mask.mMask[3] ? mVec[3] & t2 : t2;
//             return t3;
//         }
//         // HBANDS
//         UME_FORCE_INLINE int32_t hband(int32_t b) const {
//             return mVec[0] & mVec[1] & mVec[2] & mVec[3] & b;
//         }
//         // MHBANDS
//         UME_FORCE_INLINE int32_t hband(SIMDVecMask<4> const & mask, int32_t b) const {
//             int32_t t0 = mask.mMask[0] ? mVec[0] & b: b;
//             int32_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
//             int32_t t2 = mask.mMask[2] ? mVec[2] & t1 : t1;
//             int32_t t3 = mask.mMask[3] ? mVec[3] & t2 : t2;
//             return t3;
//         }
//         // HBOR
//         UME_FORCE_INLINE int32_t hbor() const {
//             return mVec[0] | mVec[1] | mVec[2] | mVec[3];
//         }
//         // MHBOR
//         UME_FORCE_INLINE int32_t hbor(SIMDVecMask<4> const & mask) const {
//             int32_t t0 = mask.mMask[0] ? mVec[0] : 0;
//             int32_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
//             int32_t t2 = mask.mMask[2] ? mVec[2] | t1 : t1;
//             int32_t t3 = mask.mMask[3] ? mVec[3] | t2 : t2;
//             return t3;
//         }
//         // HBORS
//         UME_FORCE_INLINE int32_t hbor(int32_t b) const {
//             return mVec[0] | mVec[1] | mVec[2] | mVec[3] | b;
//         }
//         // MHBORS
//         UME_FORCE_INLINE int32_t hbor(SIMDVecMask<4> const & mask, int32_t b) const {
//             int32_t t0 = mask.mMask[0] ? mVec[0] | b : b;
//             int32_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
//             int32_t t2 = mask.mMask[2] ? mVec[2] | t1 : t1;
//             int32_t t3 = mask.mMask[3] ? mVec[3] | t2 : t2;
//             return t3;
//         }
//         // HBXOR
//         UME_FORCE_INLINE int32_t hbxor() const {
//             return mVec[0] ^ mVec[1] ^ mVec[2] ^ mVec[3];
//         }
//         // MHBXOR
//         UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<4> const & mask) const {
//             int32_t t0 = mask.mMask[0] ? mVec[0] : 0;
//             int32_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
//             int32_t t2 = mask.mMask[2] ? mVec[2] ^ t1 : t1;
//             int32_t t3 = mask.mMask[3] ? mVec[3] ^ t2 : t2;
//             return t3;
//         }
//         // HBXORS
//         UME_FORCE_INLINE int32_t hbxor(int32_t b) const {
//             return mVec[0] ^ mVec[1] ^ mVec[2] ^ mVec[3] ^ b;
//         }
//         // MHBXORS
//         UME_FORCE_INLINE int32_t hbxor(SIMDVecMask<4> const & mask, int32_t b) const {
//             int32_t t0 = mask.mMask[0] ? mVec[0] ^ b : b;
//             int32_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
//             int32_t t2 = mask.mMask[2] ? mVec[2] ^ t1 : t1;
//             int32_t t3 = mask.mMask[3] ? mVec[3] ^ t2 : t2;
//             return t3;
//         }

//         // GATHERS
//         UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, uint32_t const * indices) {
//             alignas(16) int32_t raw[4] = {
//                 baseAddr[indices[0]],
//                 baseAddr[indices[1]],
//                 baseAddr[indices[2]],
//                 baseAddr[indices[3]]};
//             mVec = vec_ld(0, raw);
//             return *this;
//         }
//         // MGATHERS
//         UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<4> const & mask, int32_t const * baseAddr, uint32_t const * indices) {
//             alignas(16) int32_t raw[4];
//             alignas(16) uint32_t raw_mask[4];
//             vec_st(mask.mMask, 0, raw_mask);
//             vec_st(mVec, 0, raw);
//             if (raw_mask[0] != 0) raw[0] = baseAddr[indices[0]];
//             if (raw_mask[1] != 0) raw[1] = baseAddr[indices[1]];
//             if (raw_mask[2] != 0) raw[2] = baseAddr[indices[2]];
//             if (raw_mask[3] != 0) raw[3] = baseAddr[indices[3]];
//             mVec = vec_ld(0, raw);
//             return *this;
//         }
//         // GATHERV
//         UME_FORCE_INLINE SIMDVec_i & gather(int32_t const * baseAddr, SIMDVec_i const & indices) {
//             alignas(16) int32_t raw_indices[4];
//             vec_st(indices.mVec, 0, raw_indices);
//             alignas(16) int32_t raw[4] = {
//                 baseAddr[raw_indices[0]],
//                 baseAddr[raw_indices[1]],
//                 baseAddr[raw_indices[2]],
//                 baseAddr[raw_indices[3]]};
//             mVec = vec_ld(0, raw);
//             return *this;
//         }
//         // MGATHERV
//         UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<4> const & mask, int32_t const * baseAddr, SIMDVec_i const & indices) {
//             alignas(16) int32_t raw[4];
//             alignas(16) uint32_t raw_mask[4];
//             alignas(16) int32_t raw_indices[4];
//             vec_st(mask.mMask, 0, raw_mask);
//             vec_st(mVec, 0, raw);
//             vec_st(indices.mVec, 0, raw_indices);
//             if (raw_mask[0] != 0) raw[0] = baseAddr[raw_indices[0]];
//             if (raw_mask[1] != 0) raw[1] = baseAddr[raw_indices[1]];
//             if (raw_mask[2] != 0) raw[2] = baseAddr[raw_indices[2]];
//             if (raw_mask[3] != 0) raw[3] = baseAddr[raw_indices[3]];
//             mVec = vec_ld(0, raw);
//             return *this;
//         }
        
//         // SCATTERS
//         UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, uint32_t* indices) const {
//             baseAddr[indices[0]] = mVec[0];
//             baseAddr[indices[1]] = mVec[1];
//             baseAddr[indices[2]] = mVec[2];
//             baseAddr[indices[3]] = mVec[3];
//             return baseAddr;
//         }
//         // MSCATTERS
//         UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<4> const & mask, int32_t* baseAddr, uint32_t* indices) const {
//             if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
//             if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
//             if (mask.mMask[2] == true) baseAddr[indices[2]] = mVec[2];
//             if (mask.mMask[3] == true) baseAddr[indices[3]] = mVec[3];
//             return baseAddr;
//         }
//         // SCATTERV
//         UME_FORCE_INLINE int32_t* scatter(int32_t* baseAddr, SIMDVec_i const & indices) const {
//             baseAddr[indices.mVec[0]] = mVec[0];
//             baseAddr[indices.mVec[1]] = mVec[1];
//             baseAddr[indices.mVec[2]] = mVec[2];
//             baseAddr[indices.mVec[3]] = mVec[3];
//             return baseAddr;
//         }
//         // MSCATTERV
//         UME_FORCE_INLINE int32_t* scatter(SIMDVecMask<4> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) const {
//             if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
//             if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
//             if (mask.mMask[2] == true) baseAddr[indices.mVec[2]] = mVec[2];
//             if (mask.mMask[3] == true) baseAddr[indices.mVec[3]] = mVec[3];
//             return baseAddr;
//         }

        // LSHV
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_sl(mVec, b);
            return SIMDVec_i(t0);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_sl(mVec, b);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_i lsh(int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_sl(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<4> const & mask, int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_sl(mVec, t0);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // LSHVA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVec_i const & b) {
            mVec = = vec_sl(mVec, b)
            return *this;
        }
        // MLSHVA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_sl(mVec, b);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // LSHSA
        UME_FORCE_INLINE SIMDVec_i & lsha(int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            mVec = vec_sl(mVec, t0);
            return *this;
        }
        // MLSHSA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<4> const & mask, int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_sl(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // RSHV
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_sr(mVec, b);
            return SIMDVec_i(t0);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<4> const & mask, SIMDVec_i const & b) const {
            __vector int32_t t0 = vec_sr(mVec, b);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_i rsh(int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_sr(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<4> const & mask, int32_t b) const {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_sr(mVec, t0);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // RSHVA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVec_i const & b) {
            mVec = = vec_sr(mVec, b)
            return *this;
        }
        // MRSHVA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<4> const & mask, SIMDVec_i const & b) {
            __vector int32_t t0 = vec_sr(mVec, b);
            mVec = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }
        // RSHSA
        UME_FORCE_INLINE SIMDVec_i & rsha(int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            mVec = vec_sr(mVec, t0);
            return *this;
        }
        // MRSHSA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<4> const & mask, int32_t b) {
            SIMDVec_i t0(b, b, b, b);
            __vector int32_t t1 = vec_sr(mVec, t0);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // ROLV
        // MROLV
        // ROLS
        // MROLS
        // ROLVA
        // MROLVA
        // ROLSA
        // MROLSA
        // RORV
        // MRORV
        // RORS
        // MRORS
        // RORVA
        // MRORVA
        // RORSA
        // MRORSA

        // NEG
        UME_FORCE_INLINE SIMDVec_i neg() const {
            __vector int32_t t1 = vec_neg(mVec);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_i neg(SIMDVecMask<4> const & mask) const {
            __vector int32_t t1 = vec_neg(mVec);
            __vector int32_t t2 = vec_sel(mVec, t1, mask.mMask);
            return SIMDVec_i(t2);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_i & nega() {
            mVec = vec_neg(mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_i & nega(SIMDVecMask<4> const & mask) {
            __vector int32_t t1 = vec_neg(mVec);
            mVec = vec_sel(mVec, t1, mask.mMask);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_i abs() const {
            __vector int32_t t0 = vec_abs(mVec);
            return SIMDVec_i(t0);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_i abs(SIMDVecMask<4> const & mask) const {
            __vector int32_t t0 = vec_abs(mVec);
            __vector int32_t t1 = vec_sel(mVec, t0, mask.mMask);
            return SIMDVec_i(t1);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_i & absa() {
            mVec = vec_abs(mVec);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_i & absa(SIMDVecMask<4> const & mask) {
            __vector int32_t t0 = vec_abs(mVec);
            mVec t1 = vec_sel(mVec, t0, mask.mMask);
            return *this;
        }

//         // PACK
//         UME_FORCE_INLINE SIMDVec_i & pack(SIMDVec_i<int32_t, 2> const & a, SIMDVec_i<int32_t, 2> const & b) {
//             mVec[0] = a[0];
//             mVec[1] = a[1];
//             mVec[2] = b[0];
//             mVec[3] = b[1];
//             return *this;
//         }
//         // PACKLO
//         UME_FORCE_INLINE SIMDVec_i & packlo(SIMDVec_i<int32_t, 2> const & a) {
//             mVec[0] = a[0];
//             mVec[1] = a[1];
//             return *this;
//         }
//         // PACKHI
//         UME_FORCE_INLINE SIMDVec_i packhi(SIMDVec_i<int32_t, 2> const & b) {
//             mVec[2] = b[0];
//             mVec[3] = b[1];
//             return *this;
//         }
//         // UNPACK
//         UME_FORCE_INLINE void unpack(SIMDVec_i<int32_t, 2> & a, SIMDVec_i<int32_t, 2> & b) const {
//             a.insert(0, mVec[0]);
//             a.insert(1, mVec[1]);
//             b.insert(0, mVec[2]);
//             b.insert(1, mVec[3]);
//         }
//         // UNPACKLO
//         UME_FORCE_INLINE SIMDVec_i<int32_t, 2> unpacklo() const {
//             return SIMDVec_i<int32_t, 2> (mVec[0], mVec[1]);
//         }
//         // UNPACKHI
//         UME_FORCE_INLINE SIMDVec_i<int32_t, 2> unpackhi() const {
//             return SIMDVec_i<int32_t, 2> (mVec[2], mVec[3]);
//         }

        // PROMOTE
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 4>() const;
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int16_t, 4>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 4>() const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<float, 4>() const;
    };

}
}

#undef SET_I32
#undef SET_UI32

#endif
