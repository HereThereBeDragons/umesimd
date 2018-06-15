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

#ifndef UME_SIMD_VEC_UINT64_2_H_
#define UME_SIMD_VEC_UINT64_2_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_u<uint64_t, 2> :
        public SIMDVecUnsignedInterface<
            SIMDVec_u<uint64_t, 2>,
            uint64_t,
            2,
            SIMDVecMask<2>,
            SIMDSwizzle<2>> ,
        public SIMDVecPackableInterface<
            SIMDVec_u<uint64_t, 2>,
            SIMDVec_u<uint64_t, 1>>
    {
    private:
        uint64x2_t mVec;

        friend class SIMDVec_i<int64_t, 2>;
        friend class SIMDVec_f<double, 2>;

        friend class SIMDVec_u<uint64_t, 4>;

        UME_FORCE_INLINE explicit SIMDVec_u(uint64x2_t const & x) {
            this->mVec = x;
        }
    public:
        constexpr static uint32_t length() { return 2; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_u() {}
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint64_t i) {
            mVec = vdupq_n_u64(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_u(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value && 
                                    !std::is_same<T, uint64_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_u(static_cast<uint64_t>(i)) {}
        // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_u(uint64_t const *p) {
            mVec = vld1q_u64(p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_u(uint64_t i0, uint64_t i1) {
            alignas(16) uint64_t tmp[2] = {i0, i1};

            mVec = vld1q_u64(tmp);
        }

        // EXTRACT
        UME_FORCE_INLINE uint64_t extract(uint32_t index) const {
            if ((index & 1) == 0) {
                return vgetq_lane_u64(mVec, 0);
            }
            return vgetq_lane_u64(mVec, 1);
        }
        UME_FORCE_INLINE uint64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_u & insert(uint32_t index, uint64_t value) {
            if ((index & 1) == 0) {
                mVec = vsetq_lane_u64(value, mVec, 0);
                return *this;
            }
            mVec = vsetq_lane_u64(value, mVec, 1);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_u, uint64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_u, uint64_t>(index, static_cast<SIMDVec_u &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<2>> operator() (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<2>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_u, uint64_t, SIMDVecMask<2>>(mask, static_cast<SIMDVec_u &>(*this));
        }
#endif

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVec_u const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator= (SIMDVec_u const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            mVec = vbslq_u64(mask.mMask, b.mVec, mVec);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(uint64_t b) {
            mVec = vdupq_n_u64(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator=(uint64_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_u & assign(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp =  vdupq_n_u64(b);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_u & load(uint64_t const *p) {
            mVec = vld1q_u64(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_u & load(SIMDVecMask<2> const & mask, uint64_t const *p) {
            uint64x2_t tmp = vld1q_u64(p);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_u & loada(uint64_t const *p) {
            mVec = vld1q_u64(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_u & loada(SIMDVecMask<2> const & mask, uint64_t const *p) {
            uint64x2_t tmp = vld1q_u64(p);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE uint64_t* store(uint64_t* p) const {
            vst1q_u64(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE uint64_t* store(SIMDVecMask<2> const & mask, uint64_t* p) const {
            uint64x2_t tmp = vld1q_u64(p);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, mVec, tmp);
            vst1q_u64(p, tmp2);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE uint64_t* storea(uint64_t* p) const {
            vst1q_u64(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE uint64_t* storea(SIMDVecMask<2> const & mask, uint64_t* p) const {
            uint64x2_t tmp = vld1q_u64(p);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, mVec, tmp);
            vst1q_u64(p, tmp2);
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp = vbslq_u64(mask.mMask, b.mVec, mVec);
            return SIMDVec_u(tmp);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_u blend(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp2);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVec_u const & b) const {
            uint64x2_t tmp = vaddq_u64(mVec, b.mVec);
            return SIMDVec_u(tmp);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (SIMDVec_u const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp = vaddq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp2);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_u add(uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vaddq_u64(mVec, tmp);
            return SIMDVec_u(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_u operator+ (uint64_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_u add(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vaddq_u64(mVec, tmp);
            uint64x2_t tmp3 = vbslq_u64(mask.mMask, tmp2, mVec);
            return SIMDVec_u(tmp3);;
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVec_u const & b) {
            mVec = vaddq_u64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (SIMDVec_u const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = vaddq_u64(mVec, b.mVec);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            mVec = vaddq_u64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator+= (uint64_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_u & adda(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vaddq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // SADDV
        // MSADDV
        // SADDS
        // MSADDS
        // SADDVA
        // MSADDVA
        // SADDSA
        // MSADDSA
        
        // POSTINC
        UME_FORCE_INLINE SIMDVec_u postinc() {
            uint64x2_t tmp = vdupq_n_u64(1);
            uint64x2_t tmp2 = mVec;
            mVec = vaddq_u64(mVec, tmp);
            return SIMDVec_u(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_u operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_u postinc(SIMDVecMask<2> const & mask) {
            uint64x2_t tmp = vdupq_n_u64(1);
            uint64x2_t tmp2 = mVec;
            uint64x2_t tmp3 = vaddq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp3, mVec);
            return SIMDVec_u(tmp2);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc() {
            uint64x2_t tmp = vdupq_n_u64(1);
            mVec = vaddq_u64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_u & prefinc(SIMDVecMask<2> const & mask) {
            uint64x2_t tmp = vdupq_n_u64(1);
            uint64x2_t tmp2 = vaddq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVec_u const & b) const {
            uint64x2_t tmp = vsubq_u64(mVec, b.mVec);
            return SIMDVec_u(tmp);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (SIMDVec_u const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp = vsubq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp2);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_u sub(uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vsubq_u64(mVec, tmp);
            return SIMDVec_u(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_u operator- (uint64_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_u sub(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vsubq_u64(mVec, tmp);
            uint64x2_t tmp3 = vbslq_u64(mask.mMask, tmp2, mVec);
            return SIMDVec_u(tmp3);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVec_u const & b) {
            mVec = vsubq_u64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (SIMDVec_u const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = vsubq_u64(mVec, b.mVec);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            mVec = vsubq_u64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-= (uint64_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_u & suba(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vsubq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // SSUBV
        // MSSUBV
        // SSUBS
        // MSSUBS
        // SSUBVA
        // MSSUBVA
        // SSUBSA
        // MSSUBSA
        
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVec_u const & b) const {
            uint64x2_t tmp = vsubq_u64(b.mVec, mVec);
            return SIMDVec_u(tmp);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp = vsubq_u64(b.mVec, mVec);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp, b.mVec);
            return SIMDVec_u(tmp2);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(int64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vsubq_u64(tmp, mVec);
            return SIMDVec_u(tmp2);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_u subfrom(SIMDVecMask<2> const & mask, int64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vsubq_u64(tmp, mVec);
            uint64x2_t tmp3 = vbslq_u64(mask.mMask, tmp2, tmp);
            return SIMDVec_u(tmp3);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVec_u const & b) {
            mVec = vsubq_u64(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = vsubq_u64(b.mVec, mVec);
            mVec = vbslq_u64(mask.mMask, tmp, b.mVec);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(int64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            mVec = vsubq_u64(tmp, mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_u & subfroma(SIMDVecMask<2> const & mask, int64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vsubq_u64(tmp, mVec);
            mVec = vbslq_u64(mask.mMask, tmp2, tmp);
            return *this;
        }
        
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec() {
            uint64x2_t tmp = vdupq_n_u64(1);
            uint64x2_t tmp2 = mVec;
            mVec = vsubq_u64(mVec, tmp);
            return SIMDVec_u(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_u operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_u postdec(SIMDVecMask<2> const & mask) {
            uint64x2_t tmp = vdupq_n_u64(1);
            uint64x2_t tmp2 = mVec;
            uint64x2_t tmp3 = vsubq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp3, mVec);
            return SIMDVec_u(tmp2);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec() {
            uint64x2_t tmp = vdupq_n_u64(1);
            mVec = vsubq_u64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_u & prefdec(SIMDVecMask<2> const & mask) {
            uint64x2_t tmp = vdupq_n_u64(1);
            uint64x2_t tmp2 = vsubq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp2, mVec);
            return *this;
        }
       
/* NO 64bit integer mul, only 64bit float mul 
        // MULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVec_u const & b) const {
            uint64x2_t t0 = vmulq_u64(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (SIMDVec_u const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t t0 = vmulq_u64(mVec, b.mVec);
            uint64x2_t t1 = vbslq_u64(mask.mMask, t0, mVec);
            return SIMDVec_u(t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_u mul(uint64_t b) const {
            uint64x2_t t0 = vdupq_n_u64(b);
            uint64x2_t t1 = vmulq_u64(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator* (uint64_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_u mul(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t t0 = vdupq_n_u64(b);
            uint64x2_t t1 = vmulq_u64(mVec, t0);
            uint64x2_t t2 = vbslq_u64(mask.mMask, t1, mVec);
            return SIMDVec_u(t2);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVec_u const & b) {
            mVec = vmulq_u64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (SIMDVec_u const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = vmulq_u64(mVec, b.mVec);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_u & mula(uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            mVec = vmulq_u64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator*= (uint64_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_u & mula(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2= vmulq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp2, mVec);
            return *this;
        }
NO 64bit integer mul, only 64bit float        
*/

/* NO integer div in neon
 * see https://community.arm.com/tools/f/discussions/930/division-with-neon
 * vcvt.f32.u32  q0, q0
 * vrecpe.f32        q0, q0
 * vmul.f32    q0, q0, q1   @ q1 = 65536
 * vcvt.u32.f32  q0, q0
        // DIVV
        UME_FORCE_INLINE SIMDVec_u div(SIMDVec_u const & b) const {
            uint64x2_t t0 = vdivq_u64(mVec, b.mVec);
            return SIMDVec_u(t0);
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (SIMDVec_u const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t t0 = vdivq_u64(mVec, b.mVec);
            uint64x2_t t1 = vbslq_u64(mask.mMask, t0, mVec);
            return SIMDVec_u(t1);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_u div(uint64_t b) const {
            uint64x2_t t0 = vdupq_n_u64(b);
            uint64x2_t t1 = vdivq_u64(mVec, t0);
            return SIMDVec_u(t1);
        }
        UME_FORCE_INLINE SIMDVec_u operator/ (uint64_t b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_u div(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t t0 = vdupq_n_u64(b);
            uint64x2_t t1 = vdivq_u64(mVec, t0);
            uint64x2_t t2 = vbslq_u64(mask.mMask, t1, mVec);
            return SIMDVec_u(t2);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVec_u const & b) {
            mVec = vdivq_u64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator/= (SIMDVec_u const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = vdivq_u64(mVec, b.mVec);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_u & diva(uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            mVec = vdivq_u64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator/= (uint64_t b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_u & diva(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2= vdivq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp2, mVec);
            return *this;
        }
        
        // RCP
        UME_FORCE_INLINE SIMDVec_u rcp() const {
            uint64x2_t tmp = vrecpeq_i64(mVec);
            return SIMDVec_u(tmp);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_u rcp(SIMDVecMask<2> const & mask) const {
            uint64x2_t tmp = vrecpeq_u64(mVec);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp2);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_u rcp(uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2= vdivq_u64(tmp, mVec);
            return SIMDVec_u(tmp2);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_u rcp(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2= vdivq_u64(tmp, mVec);
            uint64x2_t tmp3 = vbslq_u64(mask.mMask, tmp2, mVec);
            return SIMDVec_u(tmp3);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_u & rcpa() {
            mVec = vrecpeq_u64(mVec);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_u & rcpa(SIMDVecMask<2> const & mask) {
            uint64x2_t tmp = vrecpeq_u64(mVec);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_u & rcpa(uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            mVec = vdivq_u64(tmp, mVec);
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_u & rcpa(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2= vdivq_u64(tmp, mVec);
            mVec = vbslq_u64(mask.mMask, tmp2, mVec);
            return *this;
        }
* NO integer div in neon
* see https://community.arm.com/tools/f/discussions/930/division-with-neon
*/
        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq (SIMDVec_u const & b) const {
            uint64x2_t tmp = vceqq_u64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (SIMDVec_u const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq (uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vceqq_u64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (uint64_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpne (SIMDVec_u const & b) const {
            uint64x2_t tmp = vceqq_u64(mVec, b.mVec);
            uint64x2_t tmp2 =  vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(tmp)));
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (SIMDVec_u const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<2> cmpne (uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vceqq_u64(mVec, tmp);
            uint64x2_t tmp3 = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(tmp2)));
            return SIMDVecMask<2>(tmp3);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (uint64_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt (SIMDVec_u const & b) const {
            uint64x2_t tmp =vcgtq_u64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (SIMDVec_u const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt (uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vcgtq_u64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (uint64_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<2> cmplt (SIMDVec_u const & b) const {
            uint64x2_t tmp =vcltq_u64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (SIMDVec_u const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<2> cmplt (uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 =vcltq_u64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (uint64_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpge (SIMDVec_u const & b) const {
            uint64x2_t tmp =vcgeq_u64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (SIMDVec_u const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<2> cmpge (uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 =vcgeq_u64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (uint64_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<2> cmple (SIMDVec_u const & b) const {
            uint64x2_t tmp =vcleq_u64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (SIMDVec_u const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<2> cmple (uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 =vcleq_u64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (uint64_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe (SIMDVec_u const & b) const {
            uint64x2_t tmp = vceqq_u64(mVec, b.mVec);
            uint32_t tmp2 = vminvq_u32(vreinterpretq_u32_u64(tmp));
            return tmp2 != 0;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vceqq_u64(mVec, tmp);
            uint32_t tmp3 = vminvq_u32(vreinterpretq_u32_u64(tmp2));
            return tmp3 != 0;
        }
        
        // UNIQUE
//        // HADD
        UME_FORCE_INLINE uint64_t hadd() const {
            return vaddvq_u64(mVec);    
        }
        // MHADD
        UME_FORCE_INLINE uint64_t hadd(SIMDVecMask<2> const & mask) const {
            uint64x2_t tmp0 = vdupq_n_u64(0);
            uint64x2_t tmp = vbslq_u64(mask.mMask, mVec, tmp0);
            return vaddvq_u64(tmp);
        }       
        // HADDS
         UME_FORCE_INLINE uint64_t hadd(uint64_t b) const {
            return vaddvq_u64(mVec) + b;
         }
        // MHADDS
        UME_FORCE_INLINE uint64_t hadd(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp0 = vdupq_n_u64(0);
            uint64x2_t tmp = vbslq_u64(mask.mMask, mVec, tmp0);
            return vaddvq_u64(tmp) + b;
         }
        // HMUL
        // MHMUL
        // HMULS
        // MHMULS
/* no direct functions 
        // FMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64x2_t tmp = vfmaq_u64(c.mVec, mVec, b.mVec);
            return SIMDVec_u(tmp);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_u fmuladd(SIMDVecMask<2> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64x2_t tmp = vfmaq_u64(c.mVec, mVec, b.mVec);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp2);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64x2_t tmp = vmulq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vsubq_u64(tmp, c.mVec);
            return SIMDVec_u(tmp2);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_u fmulsub(SIMDVecMask<2> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64x2_t tmp = vmulq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vsubq_u64(tmp, c.mVec);
            uint64x2_t tmp3 = vbslq_u64(mask.mMask, tmp2, mVec);
            return SIMDVec_u(tmp3);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64x2_t tmp = vaddq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vmulq_u64(tmp, c.mVec);
            return SIMDVec_u(tmp2);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_u faddmul(SIMDVecMask<2> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64x2_t tmp = vaddq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vmulq_u64(tmp, c.mVec);
            uint64x2_t tmp3 = vbslq_u64(mask.mMask, tmp2, mVec);
            return SIMDVec_u(tmp3);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64x2_t tmp = vsubq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vmulq_u64(tmp, c.mVec);
            return SIMDVec_u(tmp2);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_u fsubmul(SIMDVecMask<2> const & mask, SIMDVec_u const & b, SIMDVec_u const & c) const {
            uint64x2_t tmp = vsubq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vmulq_u64(tmp, c.mVec);
            uint64x2_t tmp3 = vbslq_u64(mask.mMask, tmp2, mVec);
            return SIMDVec_u(tmp3);
        }
  no direct functions
*/

/* no min/max 64bit int, only 64bit float
        // MAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVec_u const & b) const {
            uint64x2_t tmp = vmaxq_u64(mVec, b.mVec);
            return SIMDVec_u(tmp);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp = vmaxq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp2);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_u max(uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vmaxq_u64(mVec, tmp);
            return SIMDVec_u(tmp2);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_u max(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vmaxq_u64(mVec, tmp);
            uint64x2_t tmp3 = vbslq_u64(mask.mMask, tmp2, mVec);
            return SIMDVec_u(tmp3);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVec_u const & b) {
            mVec = vmaxq_u64(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = vmaxq_u64(mVec, b.mVec);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            mVec = vmaxq_u64(mVec, tmp);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_u & maxa(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vmaxq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVec_u const & b) const {
            uint64x2_t tmp = vminq_u64(mVec, b.mVec);
            return SIMDVec_u(tmp);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp = vminq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp2);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_u min(uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vminq_u64(mVec, tmp);
            return SIMDVec_u(tmp2);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_u min(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp2 = vminq_u64(mVec, tmp);
            uint64x2_t tmp3 = vbslq_u64(mask.mMask, tmp2, mVec);
            return SIMDVec_u(tmp3);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVec_u const & b) {
            mVec = vminq_u64(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_u & mina(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = vminq_u64(mVec, b.mVec);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
 no min/max for 64bit int, only 64bit float       
*/
        // MINSA
        // MMINSA
        // HMAX
        // MHMAX
        // IMAX
        // MIMAX
        // HMIN
        // MHMIN
        // IMIN
        // MIMIN

        // BANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVec_u const & b) const {
            uint64x2_t tmp = vandq_u64(mVec, b.mVec);
            return SIMDVec_u(tmp);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (SIMDVec_u const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp = vandq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp2);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_u band(uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp1 = vandq_u64(mVec, tmp);
            return SIMDVec_u(tmp1);
        }
        UME_FORCE_INLINE SIMDVec_u operator& (uint64_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_u band(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp1 = vandq_u64(mVec, tmp);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp1, mVec);
            return SIMDVec_u(tmp2);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVec_u const & b) {
            mVec = vandq_u64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (SIMDVec_u const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = vandq_u64(mVec, b.mVec);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            mVec = vandq_u64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_u & banda(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp1 = vandq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp1, mVec);
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVec_u const & b) const {
            uint64x2_t tmp1 = vorrq_u64(mVec, b.mVec);
            return SIMDVec_u(tmp1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (SIMDVec_u const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp1 = vorrq_u64(mVec, b.mVec);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp1, mVec);
            return SIMDVec_u(tmp2);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_u bor(uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp1 = vorrq_u64(mVec, tmp);
            return SIMDVec_u(tmp1);
        }
        UME_FORCE_INLINE SIMDVec_u operator| (uint64_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_u bor(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp1 = vorrq_u64(mVec, tmp);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp1, mVec);
            return SIMDVec_u(tmp2);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVec_u const & b) {
            mVec = vorrq_u64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (SIMDVec_u const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp1 = vorrq_u64(mVec, b.mVec);
            mVec = vbslq_u64(mask.mMask, tmp1, mVec);
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_u & bora(uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            mVec = vorrq_u64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator|= (uint64_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_u & bora(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp1 = vorrq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp1, mVec);
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVec_u const & b) const {
            uint64x2_t tmp = veorq_u64(mVec, b.mVec);
            return SIMDVec_u(tmp);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (SIMDVec_u const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp = veorq_u64(mVec, b.mVec);
            uint64x2_t tmp1 = vbslq_u64(mask.mMask, tmp, mVec);  
            return SIMDVec_u(tmp1);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_u bxor(uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp1 = veorq_u64(mVec, tmp);
            return SIMDVec_u(tmp1);
        }
        UME_FORCE_INLINE SIMDVec_u operator^ (uint64_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_u bxor(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp1 = veorq_u64(mVec, tmp);
            uint64x2_t tmp2 = vbslq_u64(mask.mMask, tmp1, mVec);
            return SIMDVec_u(tmp2);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVec_u const & b) {
            mVec = veorq_u64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (SIMDVec_u const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = veorq_u64(mVec, b.mVec);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            mVec = veorq_u64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_u & operator^= (uint64_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_u & bxora(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vdupq_n_u64(b);
            uint64x2_t tmp1 = veorq_u64(mVec, tmp);
            mVec = vbslq_u64(mask.mMask, tmp1, mVec);
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_u bnot() const {
            uint64x2_t tmp = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(mVec)));
            return SIMDVec_u(tmp);
        }
        UME_FORCE_INLINE SIMDVec_u operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_u bnot(SIMDVecMask<2> const & mask) const {
            uint64x2_t tmp = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(mVec)));
            uint64x2_t tmp1 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota() {
            mVec = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(mVec)));
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_u & bnota(SIMDVecMask<2> const & mask) {
            uint64x2_t tmp = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(mVec)));
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        
        // HBAND
        // MHBAND
        // HBANDS
        // MHBANDS
        // HBOR
        // MHBOR
        // HBORS
        // MHBORS
        // HBXOR
        // MHBXOR
        // HBXORS
        // MHBXORS

        // GATHERS
        // MGATHERS
        // GATHERV
        // MGATHERV
        // SCATTERS
        // MSCATTERS
        // SCATTERV
        // MSCATTERV
/*
 * /opt/gcc-7.2.0/lib/gcc/aarch64-unknown-linux-gnu/7.2.0/include/arm_neon.h:26222:50: internal compiler error: Segmentation fault
   return (uint64x2_t) __builtin_aarch64_lshrv2di ((int64x2_t) __a, __b);
       
        // LSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVec_u const & b) const {
            uint64x2_t tmp = vshlq_u64(mVec, vreinterpretq_s64_u64(b.mVec));
            return SIMDVec_u(tmp);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp = vshlq_u64(mVec, vreinterpretq_s64_u64(b.mVec));
	    uint64x2_t tmp1 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp1);
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_u lsh(uint64_t b) const {
            uint64x2_t tmp = vshlq_n_u64(mVec, b);
            return SIMDVec_u(tmp);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_u lsh(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vshlq_n_u64(mVec, b);
	    uint64x2_t tmp1 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp1);
        }
        // LSHVA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVec_u const & b) {
            mVec = vshlq_u64(mVec, vreinterpretq_s64_u64(b.mVec));
            return *this;
        }
        // MLSHVA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = vshlq_u64(mVec, vreinterpretq_s64_u64(b.mVec));
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // LSHSA
        UME_FORCE_INLINE SIMDVec_u & lsha(uint64_t b) {
            mVec = vshlq_n_u64(mVec, b);
            return *this;
        }
        // MLSHSA
        UME_FORCE_INLINE SIMDVec_u & lsha(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vshlq_n_u64(mVec, b);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // RSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVec_u const & b) const {
            uint64x2_t tmp = vshlq_u64(mVec, -(vreinterpretq_s64_u64(b.mVec)));
            return SIMDVec_u(tmp);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<2> const & mask, SIMDVec_u const & b) const {
            uint64x2_t tmp = vshlq_u64(mVec, -(vreinterpretq_s64_u64(b.mVec)));
            uint64x2_t tmp1 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp1);
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_u rsh(uint64_t b) const {
            uint64x2_t tmp = vshrq_n_u64(mVec, b);
            return SIMDVec_u(tmp);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_u rsh(SIMDVecMask<2> const & mask, uint64_t b) const {
            uint64x2_t tmp = vshrq_n_u64(mVec, b);
	    uint64x2_t tmp1 = vbslq_u64(mask.mMask, tmp, mVec);
            return SIMDVec_u(tmp1);
        }
        // RSHVA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVec_u const & b) {
            mVec = vshlq_u64(mVec, -(vreinterpretq_s64_u64(b.mVec)));
            return *this;
        }
        // MRSHVA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<2> const & mask, SIMDVec_u const & b) {
            uint64x2_t tmp = vshlq_u64(mVec, -(vreinterpretq_s64_u64(b.mVec)));
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
        // RSHSA
        UME_FORCE_INLINE SIMDVec_u & rsha(uint64_t b) {
            mVec = vshrq_n_u64(mVec, b);
            return *this;
        }
        // MRSHSA
        UME_FORCE_INLINE SIMDVec_u & rsha(SIMDVecMask<2> const & mask, uint64_t b) {
            uint64x2_t tmp = vshrq_n_u64(mVec, b);
            mVec = vbslq_u64(mask.mMask, tmp, mVec);
            return *this;
        }
   */     
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

        // PACK
        // PACKLO
        // PACKHI
        // UNPACK
        // UNPACKLO
        // UNPACKHI

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_u<uint32_t, 2>() const;

        // UTOI
        UME_FORCE_INLINE operator SIMDVec_i<int64_t, 2>() const;
        // UTOF
        UME_FORCE_INLINE operator SIMDVec_f<double, 2>() const;
    };

}
}

#endif
