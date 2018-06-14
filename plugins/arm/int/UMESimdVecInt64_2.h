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

#ifndef UME_SIMD_VEC_INT64_2_H_
#define UME_SIMD_VEC_INT64_2_H_

#include <type_traits>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int64_t, 2> :
        public SIMDVecSignedInterface<
            SIMDVec_i<int64_t, 2>,
            SIMDVec_u<uint64_t, 2>,
            int64_t,
            2,
            uint64_t,
            SIMDVecMask<2>,
            SIMDSwizzle<2>> ,
        public SIMDVecPackableInterface<
            SIMDVec_i<int64_t, 2>,
            SIMDVec_i<int64_t, 1 >>
    {
        friend class SIMDVec_u<uint64_t, 2>;
        friend class SIMDVec_f<double, 2>;

        friend class SIMDVec_i<int64_t, 4>;
    private:
        int64x2_t mVec;

    public:
        constexpr static uint32_t length() { return 2; }
        constexpr static uint32_t alignment() { return 16; }

        // ZERO-CONSTR
        UME_FORCE_INLINE SIMDVec_i() {};
        // SET-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i) {
            mVec = vdupq_n_s64(i);
        }
        // This constructor is used to force types other than SCALAR_TYPES
        // to be promoted to SCALAR_TYPE instead of SCALAR_TYPE*. This prevents
        // ambiguity between SET-CONSTR and LOAD-CONSTR.
        template<typename T>
        UME_FORCE_INLINE SIMDVec_i(
            T i, 
            typename std::enable_if< std::is_fundamental<T>::value &&
                                    !std::is_same<T, int64_t>::value,
                                    void*>::type = nullptr)
        : SIMDVec_i(static_cast<int64_t>(i)) {}
       // LOAD-CONSTR
        UME_FORCE_INLINE explicit SIMDVec_i(int64_t const *p) {
            mVec = vld1q_s64(p);
        }
        // FULL-CONSTR
        UME_FORCE_INLINE SIMDVec_i(int64_t i0, int64_t i1) {
            alignas(16) int64_t tmp[2] = {i0, i1};

            mVec = vld1q_s64(tmp);
        }

        // EXTRACT
        UME_FORCE_INLINE int64_t extract(uint32_t index) const {
            if ((index & 1) == 0) {
                return vgetq_lane_s64(mVec, 0);
            }
            return vgetq_lane_s64(mVec, 1);
        }
        UME_FORCE_INLINE int64_t operator[] (uint32_t index) const {
            return extract(index);
        }

        // INSERT
        UME_FORCE_INLINE SIMDVec_i & insert(uint32_t index, int64_t value) {
            if ((index & 1) == 0) {
                mVec = vsetq_lane_s64(value, mVec, 0);
                return *this;
            }
            mVec = vsetq_lane_s64(value, mVec, 1);
            return *this;
        }
        UME_FORCE_INLINE IntermediateIndex<SIMDVec_i, int64_t> operator[] (uint32_t index) {
            return IntermediateIndex<SIMDVec_i, int64_t>(index, static_cast<SIMDVec_i &>(*this));
        }

        // Override Mask Access operators
#if defined(USE_PARENTHESES_IN_MASK_ASSIGNMENT)
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>> operator() (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#else
        UME_FORCE_INLINE IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>> operator[] (SIMDVecMask<2> const & mask) {
            return IntermediateMask<SIMDVec_i, int64_t, SIMDVecMask<2>>(mask, static_cast<SIMDVec_i &>(*this));
        }
#endif

        // ****************************************************************************************
        // Overloading Interface functions starts here!
        // ****************************************************************************************

        // ASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator= (SIMDVec_i const & b) {
            return assign(b);
        }
        // MASSIGNV
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            mVec = vbslq_s64(mask.mMask, b.mVec, mVec);
            return *this;
        }
        // ASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(int64_t b) {
            mVec = vdupq_n_s64(b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator=(int64_t b) {
            return assign(b);
        }
        // MASSIGNS
        UME_FORCE_INLINE SIMDVec_i & assign(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp =  vdupq_n_s64(b);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }

        // PREFETCH0
        // PREFETCH1
        // PREFETCH2

        // LOAD
        UME_FORCE_INLINE SIMDVec_i & load(int64_t const *p) {
            mVec = vld1q_s64(p);
            return *this;
        }
        // MLOAD
        UME_FORCE_INLINE SIMDVec_i & load(SIMDVecMask<2> const & mask, int64_t const *p) {
            int64x2_t tmp = vld1q_s64(p);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // LOADA
        UME_FORCE_INLINE SIMDVec_i & loada(int64_t const *p) {
            mVec = vld1q_s64(p);
            return *this;
        }
        // MLOADA
        UME_FORCE_INLINE SIMDVec_i & loada(SIMDVecMask<2> const & mask, int64_t const *p) {
            int64x2_t tmp = vld1q_s64(p);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // STORE
        UME_FORCE_INLINE int64_t* store(int64_t* p) const {
            vst1q_s64(p, mVec);
            return p;
        }
        // MSTORE
        UME_FORCE_INLINE int64_t* store(SIMDVecMask<2> const & mask, int64_t* p) const {
            int64x2_t tmp = vld1q_s64(p);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, mVec, tmp);
            vst1q_s64(p, tmp2);
            return p;
        }
        // STOREA
        UME_FORCE_INLINE int64_t* storea(int64_t* p) const {
            vst1q_s64(p, mVec);
            return p;
        }
        // MSTOREA
        UME_FORCE_INLINE int64_t* storea(SIMDVecMask<2> const & mask, int64_t* p) const {
            int64x2_t tmp = vld1q_s64(p);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, mVec, tmp);
            vst1q_s64(p, tmp2);
            return p;
        }

        // BLENDV
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp = vbslq_s64(mask.mMask, b.mVec, mVec);
            return SIMDVec_i(tmp);
        }
        // BLENDS
        UME_FORCE_INLINE SIMDVec_i blend(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // SWIZZLE
        // SWIZZLEA

        // ADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVec_i const & b) const {
            int64x2_t tmp = vaddq_s64(mVec, b.mVec);
            return SIMDVec_i(tmp);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (SIMDVec_i const & b) const {
            return add(b);
        }
        // MADDV
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp = vaddq_s64(mVec, b.mVec);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // ADDS
        UME_FORCE_INLINE SIMDVec_i add(int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vaddq_s64(mVec, tmp);
            return SIMDVec_i(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_i operator+ (int64_t b) const {
            return add(b);
        }
        // MADDS
        UME_FORCE_INLINE SIMDVec_i add(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vaddq_s64(mVec, tmp);
            int64x2_t tmp3 = vbslq_s64(mask.mMask, tmp2, mVec);
            return SIMDVec_i(tmp3);;
        }
        // ADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = vaddq_s64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (SIMDVec_i const & b) {
            return adda(b);
        }
        // MADDVA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = vaddq_s64(mVec, b.mVec);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // ADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            mVec = vaddq_s64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator+= (int64_t b) {
            return adda(b);
        }
        // MADDSA
        UME_FORCE_INLINE SIMDVec_i & adda(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vaddq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp2, mVec);
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
        UME_FORCE_INLINE SIMDVec_i postinc() {
            int64x2_t tmp = vdupq_n_s64(1);
            int64x2_t tmp2 = mVec;
            mVec = vaddq_s64(mVec, tmp);
            return SIMDVec_i(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_i operator++ (int) {
            return postinc();
        }
        // MPOSTINC
        UME_FORCE_INLINE SIMDVec_i postinc(SIMDVecMask<2> const & mask) {
            int64x2_t tmp = vdupq_n_s64(1);
            int64x2_t tmp2 = mVec;
            int64x2_t tmp3 = vaddq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp3, mVec);
            return SIMDVec_i(tmp2);
        }
        // PREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc() {
            int64x2_t tmp = vdupq_n_s64(1);
            mVec = vaddq_s64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator++ () {
            return prefinc();
        }
        // MPREFINC
        UME_FORCE_INLINE SIMDVec_i & prefinc(SIMDVecMask<2> const & mask) {
            int64x2_t tmp = vdupq_n_s64(1);
            int64x2_t tmp2 = vaddq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // SUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVec_i const & b) const {
            int64x2_t tmp = vsubq_s64(mVec, b.mVec);
            return SIMDVec_i(tmp);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (SIMDVec_i const & b) const {
            return sub(b);
        }
        // MSUBV
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp = vsubq_s64(mVec, b.mVec);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // SUBS
        UME_FORCE_INLINE SIMDVec_i sub(int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vsubq_s64(mVec, tmp);
            return SIMDVec_i(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_i operator- (int64_t b) const {
            return sub(b);
        }
        // MSUBS
        UME_FORCE_INLINE SIMDVec_i sub(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vsubq_s64(mVec, tmp);
            int64x2_t tmp3 = vbslq_s64(mask.mMask, tmp2, mVec);
            return SIMDVec_i(tmp3);
        }
        // SUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = vsubq_s64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (SIMDVec_i const & b) {
            return suba(b);
        }
        // MSUBVA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = vsubq_s64(mVec, b.mVec);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // SUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            mVec = vsubq_s64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-= (int64_t b) {
            return suba(b);
        }
        // MSUBSA
        UME_FORCE_INLINE SIMDVec_i & suba(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vsubq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp2, mVec);
            return *this;
        }
//         // SSUBV
//         UME_FORCE_INLINE SIMDVec_i ssub(SIMDVec_i const & b) const {
//             int64_t t0 = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
//             int64_t t1 = (mVec[1] < b.mVec[1]) ? 0 : mVec[1] - b.mVec[1];
//             return SIMDVec_i(t0, t1);
//         }
//         // MSSUBV
//         UME_FORCE_INLINE SIMDVec_i ssub(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
//             int64_t t0 = mVec[0], t1 = mVec[1];
//             if (mask.mMask[0] == true) {
//                 t0 = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
//             }
//             if (mask.mMask[1] == true) {
//                 t1 = (mVec[1] < b.mVec[1]) ? 0 : mVec[1] - b.mVec[1];
//             }
//             return SIMDVec_i(t0, t1);
//         }
//         // SSUBS
//         UME_FORCE_INLINE SIMDVec_i ssub(int64_t b) const {
//             int64_t t0 = (mVec[0] < b) ? 0 : mVec[0] - b;
//             int64_t t1 = (mVec[1] < b) ? 0 : mVec[1] - b;
//             return SIMDVec_i(t0, t1);
//         }
//         // MSSUBS
//         UME_FORCE_INLINE SIMDVec_i ssub(SIMDVecMask<2> const & mask, int64_t b) const {
//             int64_t t0 = mVec[0], t1 = mVec[1];
//             if (mask.mMask[0] == true) {
//                 t0 = (mVec[0] < b) ? 0 : mVec[0] - b;
//             }
//             if (mask.mMask[1] == true) {
//                 t1 = (mVec[1] < b) ? 0 : mVec[1] - b;
//             }
//             return SIMDVec_i(t0, t1);
//         }
//         // SSUBVA
//         UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVec_i const & b) {
//             mVec[0] =  (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
//             mVec[1] =  (mVec[1] < b.mVec[1]) ? 0 : mVec[1] - b.mVec[1];
//             return *this;
//         }
//         // MSSUBVA
//         UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
//             if (mask.mMask[0] == true) {
//                 mVec[0] = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
//             }
//             if (mask.mMask[1] == true) {
//                 mVec[0] = (mVec[0] < b.mVec[0]) ? 0 : mVec[0] - b.mVec[0];
//             }
//             return *this;
//         }
//         // SSUBSA
//         UME_FORCE_INLINE SIMDVec_i & ssuba(int64_t b) {
//             mVec[0] = (mVec[0] < b) ? 0 : mVec[0] - b;
//             mVec[1] = (mVec[1] < b) ? 0 : mVec[1] - b;
//             return *this;
//         }
//         // MSSUBSA
//         UME_FORCE_INLINE SIMDVec_i & ssuba(SIMDVecMask<2> const & mask, int64_t b)  {
//             if (mask.mMask[0] == true) {
//                 mVec[0] = (mVec[0] < b) ? 0 : mVec[0] - b;
//             }
//             if (mask.mMask[1] == true) {
//                 mVec[1] = (mVec[1] < b) ? 0 : mVec[1] - b;
//             }
//             return *this;
//         }
        // SUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVec_i const & b) const {
            int64x2_t tmp = vsubq_s64(b.mVec, mVec);
            return SIMDVec_i(tmp);
        }
        // MSUBFROMV
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp = vsubq_s64(b.mVec, mVec);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, b.mVec);
            return SIMDVec_i(tmp2);
        }
        // SUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vsubq_s64(tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // MSUBFROMS
        UME_FORCE_INLINE SIMDVec_i subfrom(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vsubq_s64(tmp, mVec);
            int64x2_t tmp3 = vbslq_s64(mask.mMask, tmp2, tmp);
            return SIMDVec_i(tmp3);
        }
        // SUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = vsubq_s64(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = vsubq_s64(b.mVec, mVec);
            mVec = vbslq_s64(mask.mMask, tmp, b.mVec);
            return *this;
        }
        // SUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            mVec = vsubq_s64(tmp, mVec);
            return *this;
        }
        // MSUBFROMSA
        UME_FORCE_INLINE SIMDVec_i & subfroma(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vsubq_s64(tmp, mVec);
            mVec = vbslq_s64(mask.mMask, tmp2, tmp);
            return *this;
        }
        // POSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec() {
            int64x2_t tmp = vdupq_n_s64(1);
            int64x2_t tmp2 = mVec;
            mVec = vsubq_s64(mVec, tmp);
            return SIMDVec_i(tmp2);
        }
        UME_FORCE_INLINE SIMDVec_i operator-- (int) {
            return postdec();
        }
        // MPOSTDEC
        UME_FORCE_INLINE SIMDVec_i postdec(SIMDVecMask<2> const & mask) {
            int64x2_t tmp = vdupq_n_s64(1);
            int64x2_t tmp2 = mVec;
            int64x2_t tmp3 = vsubq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp3, mVec);
            return SIMDVec_i(tmp2);
        }
        // PREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec() {
            int64x2_t tmp = vdupq_n_s64(1);
            mVec = vsubq_s64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator-- () {
            return prefdec();
        }
        // MPREFDEC
        UME_FORCE_INLINE SIMDVec_i & prefdec(SIMDVecMask<2> const & mask) {
            int64x2_t tmp = vdupq_n_s64(1);
            int64x2_t tmp2 = vsubq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // MULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVec_i const & b) const {
            int64x2_t t0 = vmulq_s64(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (SIMDVec_i const & b) const {
            return mul(b);
        }
        // MMULV
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t t0 = vmulq_s64(mVec, b.mVec);
            int64x2_t t1 = vbslq_s64(mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // MULS
        UME_FORCE_INLINE SIMDVec_i mul(int64_t b) const {
            int64x2_t t0 = vdupq_n_s64(b);
            int64x2_t t1 = vmulq_s64(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator* (int64_t b) const {
            return mul(b);
        }
        // MMULS
        UME_FORCE_INLINE SIMDVec_i mul(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t t0 = vdupq_n_s64(b);
            int64x2_t t1 = vmulq_s64(mVec, t0);
            int64x2_t t2 = vbslq_s64(mask.mMask, t1, mVec);
            return SIMDVec_i(t2);
        }
        // MULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec = vmulq_s64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (SIMDVec_i const & b) {
            return mula(b);
        }
        // MMULVA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = vmulq_s64(mVec, b.mVec);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // MULSA
        UME_FORCE_INLINE SIMDVec_i & mula(int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            mVec = vmulq_s64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator*= (int64_t b) {
            return mula(b);
        }
        // MMULSA
        UME_FORCE_INLINE SIMDVec_i & mula(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2= vmulq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // DIVV
        UME_FORCE_INLINE SIMDVec_i div(SIMDVec_i const & b) const {
            int64x2_t t0 = vdivq_s64(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (SIMDVec_i const & b) const {
            return div(b);
        }
        // MDIVV
        UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t t0 = vdivq_s64(mVec, b.mVec);
            int64x2_t t1 = vbslq_s64(mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // DIVS
        UME_FORCE_INLINE SIMDVec_i div(int64_t b) const {
            int64x2_t t0 = vdupq_n_s64(b);
            int64x2_t t1 = vdivq_s64(mVec, t0);
            return SIMDVec_i(t1);
        }
        UME_FORCE_INLINE SIMDVec_i operator/ (int64_t b) const {
            return div(b);
        }
        // MDIVS
        UME_FORCE_INLINE SIMDVec_i div(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t t0 = vdupq_n_s64(b);
            int64x2_t t1 = vdivq_s64(mVec, t0);
            int64x2_t t2 = vbslq_s64(mask.mMask, t1, mVec);
            return SIMDVec_i(t2);
        }
        // DIVVA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVec_i const & b) {
            mVec = vdivq_s64(mVec, b.mVec);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (SIMDVec_i const & b) {
            return diva(b);
        }
        // MDIVVA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = vdivq_s64(mVec, b.mVec);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // DIVSA
        UME_FORCE_INLINE SIMDVec_i & diva(int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            mVec = vdivq_s64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator/= (int64_t b) {
            return diva(b);
        }
        // MDIVSA
        UME_FORCE_INLINE SIMDVec_i & diva(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2= vdivq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // RCP
        UME_FORCE_INLINE SIMDVec_i rcp() const {
            int64x2_t tmp = vrecpeq_s64(mVec);
            return SIMDVec_i(tmp);
        }
        // MRCP
        UME_FORCE_INLINE SIMDVec_i rcp(SIMDVecMask<2> const & mask) const {
            int64x2_t tmp = vrecpeq_s64(mVec);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // RCPS
        UME_FORCE_INLINE SIMDVec_i rcp(int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2= vdivq_s64(tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // MRCPS
        UME_FORCE_INLINE SIMDVec_i rcp(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2= vdivq_s64(tmp, mVec);
            int64x2_t tmp3 = vbslq_s64(mask.mMask, tmp2, mVec);
            return SIMDVec_i(tmp3);
        }
        // RCPA
        UME_FORCE_INLINE SIMDVec_i & rcpa() {
            mVec = vrecpeq_s64(mVec);
            return *this;
        }
        // MRCPA
        UME_FORCE_INLINE SIMDVec_i & rcpa(SIMDVecMask<2> const & mask) {
            int64x2_t tmp = vrecpeq_s64(mVec);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // RCPSA
        UME_FORCE_INLINE SIMDVec_i & rcpa(int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            mVec = vdivq_s64(tmp, mVec);
            return *this;
        }
        // MRCPSA
        UME_FORCE_INLINE SIMDVec_i & rcpa(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2= vdivq_s64(tmp, mVec);
            mVec = vbslq_s64(mask.mMask, tmp2, mVec);
            return *this;
        }

        // CMPEQV
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq (SIMDVec_i const & b) const {
            uint64x2_t tmp = vceqq_s64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (SIMDVec_i const & b) const {
            return cmpeq(b);
        }
        // CMPEQS
        UME_FORCE_INLINE SIMDVecMask<2> cmpeq (int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            uint64x2_t tmp2 = vceqq_s64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator== (int64_t b) const {
            return cmpeq(b);
        }
        // CMPNEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpne (SIMDVec_i const & b) const {
            uint64x2_t tmp = vceqq_s64(mVec, b.mVec);
            uint64x2_t tmp2 =  vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(tmp)));
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (SIMDVec_i const & b) const {
            return cmpne(b);
        }
        // CMPNES
        UME_FORCE_INLINE SIMDVecMask<2> cmpne (int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            uint64x2_t tmp2 = vceqq_s64(mVec, tmp);
            uint64x2_t tmp3 = vreinterpretq_u64_u32(vmvnq_u32(vreinterpretq_u32_u64(tmp2)));
            return SIMDVecMask<2>(tmp3);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator!= (int64_t b) const {
            return cmpne(b);
        }
        // CMPGTV
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt (SIMDVec_i const & b) const {
            uint64x2_t tmp =vcgtq_s64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (SIMDVec_i const & b) const {
            return cmpgt(b);
        }
        // CMPGTS
        UME_FORCE_INLINE SIMDVecMask<2> cmpgt (int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            uint64x2_t tmp2 = vcgtq_s64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator> (int64_t b) const {
            return cmpgt(b);
        }
        // CMPLTV
        UME_FORCE_INLINE SIMDVecMask<2> cmplt (SIMDVec_i const & b) const {
            uint64x2_t tmp =vcltq_s64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (SIMDVec_i const & b) const {
            return cmplt(b);
        }
        // CMPLTS
        UME_FORCE_INLINE SIMDVecMask<2> cmplt (int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            uint64x2_t tmp2 =vcltq_s64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator< (int64_t b) const {
            return cmplt(b);
        }
        // CMPGEV
        UME_FORCE_INLINE SIMDVecMask<2> cmpge (SIMDVec_i const & b) const {
            uint64x2_t tmp =vcgeq_s64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (SIMDVec_i const & b) const {
            return cmpge(b);
        }
        // CMPGES
        UME_FORCE_INLINE SIMDVecMask<2> cmpge (int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            uint64x2_t tmp2 =vcgeq_s64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator>= (int64_t b) const {
            return cmpge(b);
        }
        // CMPLEV
        UME_FORCE_INLINE SIMDVecMask<2> cmple (SIMDVec_i const & b) const {
            uint64x2_t tmp =vcleq_s64(mVec, b.mVec);
            return SIMDVecMask<2>(tmp);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (SIMDVec_i const & b) const {
            return cmple(b);
        }
        // CMPLES
        UME_FORCE_INLINE SIMDVecMask<2> cmple (int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            uint64x2_t tmp2 =vcleq_s64(mVec, tmp);
            return SIMDVecMask<2>(tmp2);
        }
        UME_FORCE_INLINE SIMDVecMask<2> operator<= (int64_t b) const {
            return cmple(b);
        }
        // CMPEV
        UME_FORCE_INLINE bool cmpe (SIMDVec_i const & b) const {
            uint64x2_t tmp = vceqq_s64(mVec, b.mVec);
            uint32_t tmp2 = vminvq_u32(vreinterpretq_u32_u64(tmp));
            return tmp2 != 0;
        }
        // CMPES
        UME_FORCE_INLINE bool cmpe(int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            uint64x2_t tmp2 = vceqq_s64(mVec, tmp);
            uint32_t tmp3 = vminvq_u32(vreinterpretq_u32_u64(tmp2));
            return tmp3 != 0;
        }
//         // UNIQUE
//         UME_FORCE_INLINE bool unique() const {
//             return mVec[0] != mVec[1];
//         }
        // HADD
        UME_FORCE_INLINE int64_t hadd() const {
            return vaddvq_s64(mVec);    
        }
        // MHADD
        UME_FORCE_INLINE int64_t hadd(SIMDVecMask<2> const & mask) const {
            int64x2_t tmp0 = vdupq_n_s64(0);
            int64x2_t tmp = vbslq_s64(mask.mMask, tmp0, mVec);
            return vaddvq_s64(tmp);
        }
//         // HADDS
//         UME_FORCE_INLINE int64_t hadd(int64_t b) const {
//             return mVec[0] + mVec[1] + b;
//         }
//         // MHADDS
//         UME_FORCE_INLINE int64_t hadd(SIMDVecMask<2> const & mask, int64_t b) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] + b : b;
//             int64_t t1 = mask.mMask[1] ? mVec[1] + t0 : t0;
//             return t1;
//         }
//         // HMUL
//         UME_FORCE_INLINE int64_t hmul() const {
//             return mVec[0] * mVec[1];
//         }
//         // MHMUL
//         UME_FORCE_INLINE int64_t hmul(SIMDVecMask<2> const & mask) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] : 1;
//             int64_t t1 = mask.mMask[1] ? mVec[1]*t0 : t0;
//             return t1;
//         }
//         // HMULS
//         UME_FORCE_INLINE int64_t hmul(int64_t b) const {
//             return mVec[0] * mVec[1] * b;
//         }
//         // MHMULS
//         UME_FORCE_INLINE int64_t hmul(SIMDVecMask<2> const & mask, int64_t b) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] * b : b;
//             int64_t t1 = mask.mMask[1] ? mVec[1] * t0 : t0;
//             return t1;
//         }

        // FMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64x2_t tmp = vfmaq_s64(c.mVec, mVec, b.mVec);
            return SIMDVec_i(tmp);
        }
        // MFMULADDV
        UME_FORCE_INLINE SIMDVec_i fmuladd(SIMDVecMask<2> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64x2_t tmp = vfmaq_s64(c.mVec, mVec, b.mVec);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // FMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64x2_t tmp = vmulq_s64(mVec, b.mVec);
            int64x2_t tmp2 = vsubq_s64(tmp, c.mVec);
            return SIMDVec_i(tmp2);
        }
        // MFMULSUBV
        UME_FORCE_INLINE SIMDVec_i fmulsub(SIMDVecMask<2> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64x2_t tmp = vmulq_s64(mVec, b.mVec);
            int64x2_t tmp2 = vsubq_s64(tmp, c.mVec);
            int64x2_t tmp3 = vbslq_s64(mask.mMask, tmp2, mVec);
            return SIMDVec_i(tmp3);
        }
        // FADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64x2_t tmp = vaddq_s64(mVec, b.mVec);
            int64x2_t tmp2 = vmulq_s64(tmp, c.mVec);
            return SIMDVec_i(tmp2);
        }
        // MFADDMULV
        UME_FORCE_INLINE SIMDVec_i faddmul(SIMDVecMask<2> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64x2_t tmp = vaddq_s64(mVec, b.mVec);
            int64x2_t tmp2 = vmulq_s64(tmp, c.mVec);
            int64x2_t tmp3 = vbslq_s64(mask.mMask, tmp2, mVec);
            return SIMDVec_i(tmp3);
        }
        // FSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64x2_t tmp = vsubq_s64(mVec, b.mVec);
            int64x2_t tmp2 = vmulq_s64(tmp, c.mVec);
            return SIMDVec_i(tmp2);
        }
        // MFSUBMULV
        UME_FORCE_INLINE SIMDVec_i fsubmul(SIMDVecMask<2> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            int64x2_t tmp = vsubq_s64(mVec, b.mVec);
            int64x2_t tmp2 = vmulq_s64(tmp, c.mVec);
            int64x2_t tmp3 = vbslq_s64(mask.mMask, tmp2, mVec);
            return SIMDVec_i(tmp3);
        }

        // MAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVec_i const & b) const {
            int64x2_t tmp = vmaxq_s64(mVec, b.mVec);
            return SIMDVec_i(tmp);
        }
        // MMAXV
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp = vmaxq_s64(mVec, b.mVec);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // MAXS
        UME_FORCE_INLINE SIMDVec_i max(int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vmaxq_s64(mVec, tmp);
            return SIMDVec_i(tmp2);
        }
        // MMAXS
        UME_FORCE_INLINE SIMDVec_i max(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vmaxq_s64(mVec, tmp);
            int64x2_t tmp3 = vbslq_s64(mask.mMask, tmp2, mVec);
            return SIMDVec_i(tmp3);
        }
        // MAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = vmaxq_s64(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = vmaxq_s64(mVec, b.mVec);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // MAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            mVec = vmaxq_s64(mVec, tmp);
            return *this;
        }
        // MMAXSA
        UME_FORCE_INLINE SIMDVec_i & maxa(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vmaxq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp2, mVec);
            return *this;
        }
        // MINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVec_i const & b) const {
            int64x2_t tmp = vminq_s64(mVec, b.mVec);
            return SIMDVec_i(tmp);
        }
        // MMINV
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp = vminq_s64(mVec, b.mVec);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // MINS
        UME_FORCE_INLINE SIMDVec_i min(int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vminq_s64(mVec, tmp);
            return SIMDVec_i(tmp2);
        }
        // MMINS
        UME_FORCE_INLINE SIMDVec_i min(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp2 = vminq_s64(mVec, tmp);
            int64x2_t tmp3 = vbslq_s64(mask.mMask, tmp2, mVec);
            return SIMDVec_i(tmp3);
        }
        // MINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = vminq_s64(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = vminq_s64(mVec, b.mVec);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
//         // MINSA
//         UME_FORCE_INLINE SIMDVec_i & mina(int64_t b) {
//             if(mVec[0] > b) mVec[0] = b;
//             if(mVec[1] > b) mVec[1] = b;
//             return *this;
//         }
//         // MMINSA
//         UME_FORCE_INLINE SIMDVec_i & mina(SIMDVecMask<2> const & mask, int64_t b) {
//             if (mask.mMask[0] == true && mVec[0] > b) {
//                 mVec[0] = b;
//             }
//             if (mask.mMask[1] == true && mVec[1] > b) {
//                 mVec[1] = b;
//             }
//             return *this;
//         }
//         // HMAX
//         UME_FORCE_INLINE int64_t hmax () const {
//             return mVec[0] > mVec[1] ? mVec[0] : mVec[1];
//         }
//         // MHMAX
//         UME_FORCE_INLINE int64_t hmax(SIMDVecMask<2> const & mask) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<int64_t>::min();
//             int64_t t1 = (mask.mMask[1] && mVec[1] > t0) ? mVec[1] : t0;
//             return t1;
//         }
//         // IMAX
//         UME_FORCE_INLINE uint32_t imax() const {
//             return mVec[0] > mVec[1] ? 0 : 1;
//         }
//         // MIMAX
//         UME_FORCE_INLINE uint32_t imax(SIMDVecMask<2> const & mask) const {
//             uint32_t i0 = 0xFFFFFFFF;
//             int64_t t0 = std::numeric_limits<int64_t>::min();
//             if(mask.mMask[0] == true) {
//                 i0 = 0;
//                 t0 = mVec[0];
//             }
//             if(mask.mMask[1] == true && mVec[1] > t0) {
//                 i0 = 1;
//             }
//             return i0;
//         }
//         // HMIN
//         UME_FORCE_INLINE int64_t hmin() const {
//             return mVec[0] < mVec[1] ? mVec[0] : mVec[1];
//         }
//         // MHMIN
//         UME_FORCE_INLINE int64_t hmin(SIMDVecMask<2> const & mask) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] : std::numeric_limits<int64_t>::max();
//             int64_t t1 = (mask.mMask[1] && mVec[1] < t0) ? mVec[1] : t0;
//             return t1;
//         }
//         // IMIN
//         UME_FORCE_INLINE uint32_t imin() const {
//             return mVec[0] < mVec[1] ? 0 : 1;
//         }
//         // MIMIN
//         UME_FORCE_INLINE uint32_t imin(SIMDVecMask<2> const & mask) const {
//             uint32_t i0 = 0xFFFFFFFF;
//             int64_t t0 = std::numeric_limits<int64_t>::max();
//             if(mask.mMask[0] == true) {
//                 i0 = 0;
//                 t0 = mVec[0];
//             }
//             if(mask.mMask[1] == true && mVec[1] < t0) {
//                 i0 = 1;
//             }
//             return i0;
//         }

        // BANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVec_i const & b) const {
            int64x2_t tmp = vandq_s64(mVec, b);
            return SIMDVec_i(tmp);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (SIMDVec_i const & b) const {
            return band(b);
        }
        // MBANDV
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp = vandq_s64(mVec, b);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // BANDS
        UME_FORCE_INLINE SIMDVec_i band(int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp1 = vandq_s64(mVec, tmp1);
            return SIMDVec_i(tmp1);
        }
        UME_FORCE_INLINE SIMDVec_i operator& (int64_t b) const {
            return band(b);
        }
        // MBANDS
        UME_FORCE_INLINE SIMDVec_i band(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp1 = vandq_s64(mVec, tmp1);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp1, mVec);
            return SIMDVec_i(tmp2);
        }
        // BANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec = vandq_s64(mVec, b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (SIMDVec_i const & b) {
            return banda(b);
        }
        // MBANDVA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = vandq_s64(mVec, b);
            mVec = vbslq_s64(mask.mMask, b, mVec);
            return *this;
        }
        // BANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            mVec = vandq_s64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator&= (bool b) {
            return banda(b);
        }
        // MBANDSA
        UME_FORCE_INLINE SIMDVec_i & banda(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp1 = vandq_s64(mVec, tmp1);
            mVec = vbslq_s64(mask.mMask, tmp1, mVec);
            return *this;
        }
        // BORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVec_i const & b) const {
            int64x2_t tmp1 = vorrq_s64(mVec, b);
            return SIMDVec_i(tmp1);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (SIMDVec_i const & b) const {
            return bor(b);
        }
        // MBORV
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp1 = vorrq_s64(mVec, b);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp1, mVec);
            return SIMDVec_i(tmp2);
        }
        // BORS
        UME_FORCE_INLINE SIMDVec_i bor(int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp1 = vorrq_s64(mVec, tmp);
            return SIMDVec_i(tmp1);
        }
        UME_FORCE_INLINE SIMDVec_i operator| (int64_t b) const {
            return bor(b);
        }
        // MBORS
        UME_FORCE_INLINE SIMDVec_i bor(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp1 = vorrq_s64(mVec, tmp);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp1, mVec);
            return SIMDVec_i(tmp2);
        }
        // BORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec = vorrq_s64(mVec, b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (SIMDVec_i const & b) {
            return bora(b);
        }
        // MBORVA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp1 = vorrq_s64(mVec, b);
            mVec = vbslq_s64(mask.mMask, tmp1, mVec);
            return *this;
        }
        // BORSA
        UME_FORCE_INLINE SIMDVec_i & bora(int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            mVec = vorrq_s64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator|= (int64_t b) {
            return bora(b);
        }
        // MBORSA
        UME_FORCE_INLINE SIMDVec_i & bora(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp1 = vorrq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp1, mVec);
            return *this;
        }
        // BXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVec_i const & b) const {
            int64x2_t tmp = veorq_s64(mVec, b);
            int64x2_t tmp1 = vbslq_s64(mask.mMask, tmp, mVec);  
            return SIMDVec_i(tmp1);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (SIMDVec_i const & b) const {
            return bxor(b);
        }
        // MBXORV
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp = veorq_s64(mVec, b);
            return SIMDVec_i(tmp);
        }
        // BXORS
        UME_FORCE_INLINE SIMDVec_i bxor(int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp1 = veorq_s64(mVec, tmp);
            return SIMDVec_i(tmp1);
        }
        UME_FORCE_INLINE SIMDVec_i operator^ (int64_t b) const {
            return bxor(b);
        }
        // MBXORS
        UME_FORCE_INLINE SIMDVec_i bxor(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp1 = veorq_s64(mVec, tmp);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp1, mVec);
            return SIMDVec_i(tmp2);
        }
        // BXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec = veorq_s64(mVec, b);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (SIMDVec_i const & b) {
            return bxora(b);
        }
        // MBXORVA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = veorq_s64(mVec, b);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // BXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            mVec = veorq_s64(mVec, tmp);
            return *this;
        }
        UME_FORCE_INLINE SIMDVec_i & operator^= (int64_t b) {
            return bxora(b);
        }
        // MBXORSA
        UME_FORCE_INLINE SIMDVec_i & bxora(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vdupq_n_s64(b);
            int64x2_t tmp1 = veorq_s64(mVec, tmp);
            mVec = vbslq_s64(mask.mMask, tmp1, mVec);
            return *this;
        }
        // BNOT
        UME_FORCE_INLINE SIMDVec_i bnot() const {
            int64x2_t tmp = vmvnq_s64(mVec);
            return SIMDVec_i(tmp);
        }
        UME_FORCE_INLINE SIMDVec_i operator~ () const {
            return bnot();
        }
        // MBNOT
        UME_FORCE_INLINE SIMDVec_i bnot(SIMDVecMask<2> const & mask) const {
            int64x2_t tmp = vmvnq_s64(mVec);
            int64x2_t tmp1 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp1);
        }
        // BNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota() {
            mVec = vmvnq_s64(mVec);
            return *this;
        }
        // MBNOTA
        UME_FORCE_INLINE SIMDVec_i & bnota(SIMDVecMask<2> const & mask) {
            int64x2_t tmp = vmvnq_s64(mVec);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
//         // HBAND
//         UME_FORCE_INLINE int64_t hband() const {
//             return mVec[0] & mVec[1];
//         }
//         // MHBAND
//         UME_FORCE_INLINE int64_t hband(SIMDVecMask<2> const & mask) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] : 0xFFFFFFFFFFFFFFFF;
//             int64_t t1 = mask.mMask[1] ? mVec[1] & t0 : t0;
//             return t1;
//         }
//         // HBANDS
//         UME_FORCE_INLINE int64_t hband(int64_t b) const {
//             return mVec[0] & mVec[1] & b;
//         }
//         // MHBANDS
//         UME_FORCE_INLINE int64_t hband(SIMDVecMask<2> const & mask, int64_t b) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] & b: b;
//             int64_t t1 = mask.mMask[1] ? mVec[1] & t0: t0;
//             return t1;
//         }
//         // HBOR
//         UME_FORCE_INLINE int64_t hbor() const {
//             return mVec[0] | mVec[1];
//         }
//         // MHBOR
//         UME_FORCE_INLINE int64_t hbor(SIMDVecMask<2> const & mask) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] : 0;
//             int64_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
//             return t1;
//         }
//         // HBORS
//         UME_FORCE_INLINE int64_t hbor(int64_t b) const {
//             return mVec[0] | mVec[1] | b;
//         }
//         // MHBORS
//         UME_FORCE_INLINE int64_t hbor(SIMDVecMask<2> const & mask, int64_t b) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] | b : b;
//             int64_t t1 = mask.mMask[1] ? mVec[1] | t0 : t0;
//             return t1;
//         }
//         // HBXOR
//         UME_FORCE_INLINE int64_t hbxor() const {
//             return mVec[0] ^ mVec[1];
//         }
//         // MHBXOR
//         UME_FORCE_INLINE int64_t hbxor(SIMDVecMask<2> const & mask) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] : 0;
//             int64_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
//             return t1;
//         }
//         // HBXORS
//         UME_FORCE_INLINE int64_t hbxor(int64_t b) const {
//             return mVec[0] ^ mVec[1] ^ b;
//         }
//         // MHBXORS
//         UME_FORCE_INLINE int64_t hbxor(SIMDVecMask<2> const & mask, int64_t b) const {
//             int64_t t0 = mask.mMask[0] ? mVec[0] ^ b : b;
//             int64_t t1 = mask.mMask[1] ? mVec[1] ^ t0 : t0;
//             return t1;
//         }
// 
//         // GATHERS
//         UME_FORCE_INLINE SIMDVec_i & gather(int64_t const * baseAddr, uint64_t const * indices) {
//             mVec[0] = baseAddr[indices[0]];
//             mVec[1] = baseAddr[indices[1]];
//             return *this;
//         }
//         // MGATHERS
//         UME_FORCE_INLINE SIMDVec_i & gather(SIMDVecMask<2> const & mask, int64_t const * baseAddr, uint64_t const * indices) {
//             if (mask.mMask[0] == true) mVec[0] = baseAddr[indices[0]];
//             if (mask.mMask[1] == true) mVec[1] = baseAddr[indices[1]];
//             return *this;
//         }
//         // GATHERV
//         UME_FORCE_INLINE SIMDVec_i gather(int64_t const * baseAddr, SIMDVec_u<uint64_t, 2> const & indices) {
//             mVec[0] = baseAddr[indices.mVec[0]];
//             mVec[1] = baseAddr[indices.mVec[1]];
//             return *this;
//         }
//         // MGATHERV
//         UME_FORCE_INLINE SIMDVec_i gather(SIMDVecMask<2> const & mask, int64_t const * baseAddr, SIMDVec_u<uint64_t, 2> const & indices) {
//             if (mask.mMask[0] == true) mVec[0] = baseAddr[indices.mVec[0]];
//             if (mask.mMask[1] == true) mVec[1] = baseAddr[indices.mVec[1]];
//             return *this;
//         }
//         // SCATTERS
//         UME_FORCE_INLINE int64_t* scatter(int64_t* baseAddr, uint64_t* indices) const {
//             baseAddr[indices[0]] = mVec[0];
//             baseAddr[indices[1]] = mVec[1];
//             return baseAddr;
//         }
//         // MSCATTERS
//         UME_FORCE_INLINE int64_t*  scatter(SIMDVecMask<2> const & mask, int64_t* baseAddr, uint64_t* indices) const {
//             if (mask.mMask[0] == true) baseAddr[indices[0]] = mVec[0];
//             if (mask.mMask[1] == true) baseAddr[indices[1]] = mVec[1];
//             return baseAddr;
//         }
//         // SCATTERV
//         UME_FORCE_INLINE int64_t*  scatter(int64_t* baseAddr, SIMDVec_u<uint64_t, 2> const & indices) const {
//             baseAddr[indices.mVec[0]] = mVec[0];
//             baseAddr[indices.mVec[1]] = mVec[1];
//             return baseAddr;
//         }
//         // MSCATTERV
//         UME_FORCE_INLINE int64_t*  scatter(SIMDVecMask<2> const & mask, int64_t* baseAddr, SIMDVec_u<uint64_t, 2> const & indices) const {
//             if (mask.mMask[0] == true) baseAddr[indices.mVec[0]] = mVec[0];
//             if (mask.mMask[1] == true) baseAddr[indices.mVec[1]] = mVec[1];
//             return baseAddr;
//         }

        // LSHV
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVec_i const & b) const {
            int64x2_t tmp = vshlq_s64(mVec, b);
            return SIMDVec_i(tmp);
        }
        // MLSHV
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp = vshlq_s64(mVec, b);
	    int64x2_t tmp1 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp1);
        }
        // LSHS
        UME_FORCE_INLINE SIMDVec_i lsh(int64_t b) const {
            int64x2_t tmp = vshlq_n_s64(mVec, b);
            return SIMDVec_i(tmp);
        }
        // MLSHS
        UME_FORCE_INLINE SIMDVec_i lsh(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vshlq_n_s64(mVec, b);
	    int64x2_t tmp1 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp1);
        }
        // LSHVA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVec_i const & b) {
            mVec = vshlq_s64(mVec, b);
            return *this;
        }
        // MLSHVA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = vshlq_s64(mVec, b);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // LSHSA
        UME_FORCE_INLINE SIMDVec_i & lsha(int64_t b) {
            mVec = vshlq_n_s64(mVec, b);
            return *this;
        }
        // MLSHSA
        UME_FORCE_INLINE SIMDVec_i & lsha(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vshlq_n_s64(mVec, b);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // RSHV
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVec_i const & b) const {
            int64x2_t tmp = vshrq_s64(mVec, b);
            return SIMDVec_i(tmp);
        }
        // MRSHV
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<2> const & mask, SIMDVec_i const & b) const {
            int64x2_t tmp = vshrq_s64(mVec, b);
            int64x2_t tmp1 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp1);
        }
        // RSHS
        UME_FORCE_INLINE SIMDVec_i rsh(int64_t b) const {
            int64x2_t tmp = vshrq_n_s64(mVec, b);
            return SIMDVec_i(tmp);
        }
        // MRSHS
        UME_FORCE_INLINE SIMDVec_i rsh(SIMDVecMask<2> const & mask, int64_t b) const {
            int64x2_t tmp = vshrq_n_s64(mVec, b);
	    int64x2_t tmp1 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp1);
        }
        // RSHVA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVec_i const & b) {
            mVec = vshrq_s64(mVec, b);
            return *this;
        }
        // MRSHVA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<2> const & mask, SIMDVec_i const & b) {
            int64x2_t tmp = vshrq_s64(mVec, b);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // RSHSA
        UME_FORCE_INLINE SIMDVec_i & rsha(int64_t b) {
            mVec = vshrq_n_s64(mVec, b);
            return *this;
        }
        // MRSHSA
        UME_FORCE_INLINE SIMDVec_i & rsha(SIMDVecMask<2> const & mask, int64_t b) {
            int64x2_t tmp = vshrq_n_s64(mVec, b);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
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
            int64x2_t tmp = vnegq_s64(mVec);
            return SIMDVec_i(tmp);
        }
        UME_FORCE_INLINE SIMDVec_i operator- () const {
            return neg();
        }
        // MNEG
        UME_FORCE_INLINE SIMDVec_i neg(SIMDVecMask<2> const & mask) const {
            int64x2_t tmp = vnegq_s64(mVec);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // NEGA
        UME_FORCE_INLINE SIMDVec_i & nega() {
            mVec = vnegq_s64(mVec);
            return *this;
        }
        // MNEGA
        UME_FORCE_INLINE SIMDVec_i & nega(SIMDVecMask<2> const & mask) {
            int64x2_t tmp = vnegq_s64(mVec);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }
        // ABS
        UME_FORCE_INLINE SIMDVec_i abs() const {
            int64x2_t tmp = vabsq_s64(mVec);
            return SIMDVec_i(tmp);
        }
        // MABS
        UME_FORCE_INLINE SIMDVec_i abs(SIMDVecMask<2> const & mask) const {
            int64x2_t tmp = vabsq_s64(mVec);
            int64x2_t tmp2 = vbslq_s64(mask.mMask, tmp, mVec);
            return SIMDVec_i(tmp2);
        }
        // ABSA
        UME_FORCE_INLINE SIMDVec_i & absa() {
            mVec = vabsq_s64(mVec);
            return *this;
        }
        // MABSA
        UME_FORCE_INLINE SIMDVec_i & absa(SIMDVecMask<2> const & mask) {
            int64x2_t tmp = vabsq_s64(mVec);
            mVec = vbslq_s64(mask.mMask, tmp, mVec);
            return *this;
        }

//         // PACK
//         UME_FORCE_INLINE SIMDVec_i & pack(SIMDVec_i<int64_t, 1> const & a, SIMDVec_i<int64_t, 1> const & b) {
//             mVec[0] = a[0];
//             mVec[1] = b[0];
//             return *this;
//         }
//         // PACKLO
//         UME_FORCE_INLINE SIMDVec_i & packlo(SIMDVec_i<int64_t, 1> const & a) {
//             mVec[0] = a[0];
//             return *this;
//         }
//         // PACKHI
//         UME_FORCE_INLINE SIMDVec_i packhi(SIMDVec_i<int64_t, 1> const & b) {
//             mVec[1] = b[0];
//             return *this;
//         }
//         // UNPACK
//         void unpack(SIMDVec_i<int64_t, 1> & a, SIMDVec_i<int64_t, 1> & b) const {
//             a.insert(0, mVec[0]);
//             b.insert(0, mVec[1]);
//         }
//         // UNPACKLO
//         SIMDVec_i<int64_t, 1> unpacklo() const {
//             return SIMDVec_i<int64_t, 1> (mVec[0]);
//         }
//         // UNPACKHI
//         SIMDVec_i<int64_t, 1> unpackhi() const {
//             return SIMDVec_i<int64_t, 1> (mVec[1]);
//         }

        // PROMOTE
        // -
        // DEGRADE
        UME_FORCE_INLINE operator SIMDVec_i<int32_t, 2>() const;

        // ITOU
        UME_FORCE_INLINE operator SIMDVec_u<uint64_t, 2>() const;
        // ITOF
        UME_FORCE_INLINE operator SIMDVec_f<double, 2>() const;
    };

}
}

#endif
