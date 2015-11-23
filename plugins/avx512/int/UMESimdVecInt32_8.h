// The MIT License (MIT)
//
// Copyright (c) 2015 CERN
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

#ifndef UME_SIMD_VEC_INT32_8_H_
#define UME_SIMD_VEC_INT32_8_H_

#include <type_traits>
#include <immintrin.h>

#include "../../../UMESimdInterface.h"

namespace UME {
namespace SIMD {

    template<>
    class SIMDVec_i<int32_t, 8> final :
        public SIMDVecSignedInterface<
            SIMDVec_i<int32_t, 8>,
            SIMDVec_u<uint32_t, 8>,
            int32_t,
            8,
            uint32_t,
            SIMDVecMask<8>,
            SIMDVecSwizzle<8 >> ,
        public SIMDVecPackableInterface<
           SIMDVec_i<int32_t, 8>,
           SIMDVec_i<int32_t, 4 >>
    {
        friend class SIMDVec_u<uint32_t, 8>;
        friend class SIMDVec_f<float, 8>;
        friend class SIMDVec_f<double, 8>;
    private:
        __m256i mVec;

        inline explicit SIMDVec_i(__m256i & x) { mVec = x; }
        inline explicit SIMDVec_i(const __m256i & x) { mVec = x; }
    public:

        constexpr static uint32_t length() { return 8; }
        constexpr static uint32_t alignment() { return 32; }

        // ZERO-CONSTR
        inline SIMDVec_i() {};

        // SET-CONSTR
        inline explicit SIMDVec_i(int32_t i) {
            mVec = _mm256_set1_epi32(i);
        }
        // LOAD-CONSTR
        inline explicit SIMDVec_i(int32_t const * p) {
            mVec = _mm256_mask_load_epi32(mVec, 0xFF, (void *)p);
        }
        // FULL-CONSTR
        inline SIMDVec_i(int32_t i0, int32_t i1, int32_t i2, int32_t i3,
            int32_t i4, int32_t i5, int32_t i6, int32_t i7)
        {
            mVec = _mm256_setr_epi32(i0, i1, i2, i3, i4, i5, i6, i7);
        }

        // EXTRACT
        inline int32_t extract(uint32_t index) const {
            //return _mm256_extract_epi32(mVec, index); // TODO: this can be implemented in ICC
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            return raw[index];
        }
        inline int32_t operator[] (uint32_t index) const {
            return extract(index);
        }
        // Override Mask Access operators
        inline IntermediateMask<SIMDVec_i, SIMDVecMask<8>> operator[] (SIMDVecMask<8> const & mask) {
            return IntermediateMask<SIMDVec_i, SIMDVecMask<8>>(mask, static_cast<SIMDVec_i &>(*this));
        }
        // INSERT
        inline SIMDVec_i & insert(uint32_t index, int32_t value) {
            //UME_PERFORMANCE_UNOPTIMAL_WARNING()
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            raw[index] = value;
            mVec = _mm256_load_si256((__m256i *)raw);
            return *this;
        }
        // ASSIGNV
        inline SIMDVec_i & assign(SIMDVec_i const & b) {
            mVec = b.mVec;
            return *this;
        }
        // MASSIGNV
        inline SIMDVec_i & assign(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return *this;
        }
        // ASSIGNS
        inline SIMDVec_i & assign(int32_t b) {
            mVec = _mm256_set1_epi32(b);
            return *this;
        }
        // MASSIGNS
        inline SIMDVec_i & assign(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_mov_epi32(mVec, mask.mMask, t0);
            return *this;
        }
        // PREFETCH0
        // PREFETCH1
        // PREFETCH2
        // LOAD
        inline SIMDVec_i & load(int32_t const * p) {
            mVec = _mm256_mask_loadu_epi32(mVec, 0xFF, p);
            return *this;
        }
        // MLOAD
        inline SIMDVec_i & load(SIMDVecMask<8> const & mask, int32_t const * p) {
            mVec = _mm256_mask_loadu_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // LOADA
        inline SIMDVec_i & loada(int32_t const * p) {
            mVec = _mm256_load_si256((__m256i*)p);
            return *this;
        }
        // MLOADA
        inline SIMDVec_i & loada(SIMDVecMask<8> const & mask, int32_t const * p) {
            mVec = _mm256_mask_load_epi32(mVec, mask.mMask, p);
            return *this;
        }
        // STORE
        inline int32_t * store(int32_t * p) const {
            _mm256_mask_storeu_epi32(p, 0xFF, mVec);
            return p;
        }
        // MSTORE
        inline int32_t * store(SIMDVecMask<8> const & mask, int32_t * p) const {
            _mm256_mask_storeu_epi32(p, mask.mMask, mVec);
            return p;
        }
        // STOREA
        inline int32_t * storea(int32_t * addrAligned) {
            _mm256_store_si256((__m256i*)addrAligned, mVec);
            return addrAligned;
        }
        // MSTOREA
        inline int32_t * storea(SIMDVecMask<8> const & mask, int32_t * p) const {
            _mm256_mask_store_epi32(p, mask.mMask, mVec);
            return p;
        }
        // BLENDV
        inline SIMDVec_i blend(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_mov_epi32(mVec, mask.mMask, b.mVec);
            return SIMDVec_i(t0);
        }
        // BLENDS
        inline SIMDVec_i blend(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_mov_epi32(mVec, mask.mMask, t0);
            return SIMDVec_i(t1);
        }
        // SWIZZLE
        // SWIZZLEA
        // ADDV
        inline SIMDVec_i add(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        inline SIMDVec_i operator+ (SIMDVec_i const & b) const {
            __m256i t0 = _mm256_add_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MADDV
        inline SIMDVec_i add(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // ADDS
        inline SIMDVec_i add(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_add_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MADDS
        inline SIMDVec_i add(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // ADDVA
        inline SIMDVec_i & adda(SIMDVec_i const & b) {
            mVec = _mm256_add_epi32(mVec, b.mVec);
            return *this;
        }
        // MADDVA
        inline SIMDVec_i & adda(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ADDSA 
        inline SIMDVec_i & adda(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_add_epi32(mVec, t0);
            return *this;
        }
        // MADDSA
        inline SIMDVec_i & adda(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t0);
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
        inline SIMDVec_i postinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_add_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MPOSTINC
        inline SIMDVec_i postinc(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // PREFINC
        inline SIMDVec_i & prefinc() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_add_epi32(mVec, t0);
            return *this;
        }
        // MPREFINC
        inline SIMDVec_i & prefinc(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_mask_add_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // SUBV
        inline SIMDVec_i sub(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_sub_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MSUBV
        inline SIMDVec_i sub(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // SUBS
        inline SIMDVec_i sub(int32_t b) const {
            __m256i t0 = _mm256_sub_epi32(mVec, _mm256_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        // MSUBS
        inline SIMDVec_i sub(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // SUBVA
        inline SIMDVec_i & suba(SIMDVec_i const & b) {
            mVec = _mm256_sub_epi32(mVec, b.mVec);
            return *this;
        }
        // MSUBVA
        inline SIMDVec_i & suba(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // SUBSA
        inline SIMDVec_i & suba(int32_t b) {
            mVec = _mm256_sub_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        // MSUBSA
        inline SIMDVec_i & suba(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
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
        inline SIMDVec_i subfrom(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_sub_epi32(b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMV
        inline SIMDVec_i subfrom(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return SIMDVec_i(t0);
        }
        // SUBFROMS
        inline SIMDVec_i subfrom(int32_t b) const {
            __m256i t0 = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec);
            return SIMDVec_i(t0);
        }
        // MSUBFROMS
        inline SIMDVec_i subfrom(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // SUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVec_i const & b) {
            mVec = _mm256_sub_epi32(b.mVec, mVec);
            return *this;
        }
        // MSUBFROMVA
        inline SIMDVec_i & subfroma(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_sub_epi32(b.mVec, mask.mMask, b.mVec, mVec);
            return *this;
        }
        // SUBFROMSA
        inline SIMDVec_i & subfroma(int32_t b) {
            mVec = _mm256_sub_epi32(_mm256_set1_epi32(b), mVec);
            return *this;
        }
        // MSUBFROMSA
        inline SIMDVec_i subfroma(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_sub_epi32(t0, mask.mMask, t0, mVec);
            return *this;
        }

        // POSTDEC
        inline SIMDVec_i postdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_sub_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MPOSTDEC
        inline SIMDVec_i postdec(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            __m256i t1 = mVec;
            mVec = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // PREFDEC
        inline SIMDVec_i & prefdec() {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_sub_epi32(mVec, t0);
            return *this;
        }
        // MPREFDEC
        inline SIMDVec_i & prefdec(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi32(1);
            mVec = _mm256_mask_sub_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MULV
        inline SIMDVec_i mul(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMULV
        inline SIMDVec_i mul(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MULS
        inline SIMDVec_i mul(int32_t b) const {
            __m256i t0 = _mm256_mullo_epi32(mVec, _mm256_set1_epi32(b));
            return SIMDVec_i(t0);
        }
        // MMULS
        inline SIMDVec_i mul(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MULVA
        inline SIMDVec_i & mula(SIMDVec_i const & b) {
            mVec = _mm256_mullo_epi32(mVec, b.mVec);
            return *this;
        }
        // MMULVA
        inline SIMDVec_i & mula(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MULSA
        inline SIMDVec_i & mula(int32_t b) {
            mVec = _mm256_mullo_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        // MMULSA
        inline SIMDVec_i & mula(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_mullo_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // DIVV
        // MDIVV
        // DIVS
        // MDIVS
        // DIVVA
        // MDIVVA
        // DIVSA
        // MDIVSA
        // RCP
        // MRCP
        // RCPS
        // MRCPS
        // RCPA
        // MRCPA
        // RCPSA
        // MRCPSA
        // CMPEQV
        inline SIMDVecMask<8> cmpeq(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm256_cmpeq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<8>(t0);
        }
        // CMPEQS
        inline SIMDVecMask<8> cmpeq(int32_t b) const {
            __mmask8 t0 = _mm256_cmpeq_epi32_mask(mVec, _mm256_set1_epi32(b)) & 0xFF;
            return SIMDVecMask<8>(t0);
        }
        // CMPNEV
        inline SIMDVecMask<8> cmpne(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm256_cmpneq_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<8>(t0);
        }
        // CMPNES
        inline SIMDVecMask<8> cmpne(int32_t b) const {
            __mmask8 t0 = _mm256_cmpneq_epi32_mask(mVec, _mm256_set1_epi32(b));
            return SIMDVecMask<8>(t0);
        }
        // CMPGTV
        inline SIMDVecMask<8> cmpgt(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm256_cmpgt_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<8>(t0);
        }
        // CMPGTS
        inline SIMDVecMask<8> cmpgt(int32_t b) const {
            __mmask8 t0 = _mm256_cmpgt_epi32_mask(mVec, _mm256_set1_epi32(b));
            return SIMDVecMask<8>(t0);
        }
        // CMPLTV
        inline SIMDVecMask<8> cmplt(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm256_cmplt_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<8>(t0);
        }
        // CMPLTS
        inline SIMDVecMask<8> cmplt(int32_t b) const {
            __mmask8 t0 = _mm256_cmplt_epi32_mask(mVec, _mm256_set1_epi32(b));
            return SIMDVecMask<8>(t0);
        }
        // CMPGEV
        inline SIMDVecMask<8> cmpge(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm256_cmpge_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<8>(t0);
        }
        // CMPGES
        inline SIMDVecMask<8> cmpge(int32_t b) const {
            __mmask8 t0 = _mm256_cmpge_epi32_mask(mVec, _mm256_set1_epi32(b));
            return SIMDVecMask<8>(t0);
        }
        // CMPLEV
        inline SIMDVecMask<8> cmple(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm256_cmple_epi32_mask(mVec, b.mVec);
            return SIMDVecMask<8>(t0);
        }
        // CMPLES
        inline SIMDVecMask<8> cmple(int32_t b) const {
            __mmask8 t0 = _mm256_cmple_epi32_mask(mVec, _mm256_set1_epi32(b));
            return SIMDVecMask<8>(t0);
        }
        // CMPEV
        inline bool cmpe(SIMDVec_i const & b) const {
            __mmask8 t0 = _mm256_cmple_epi32_mask(mVec, b.mVec);
            return (t0 == 0x0F);
        }
        // CMPES
        inline bool cmpe(int32_t b) const {
            __mmask8 t0 = _mm256_cmpeq_epi32_mask(mVec, _mm256_set1_epi32(b));
            return (t0 == 0x0F);
        }
        // UNIQUE
        inline bool unique() const {
            __m256i t0 = _mm256_conflict_epi32(mVec);
            __mmask8 t1 = _mm256_cmpeq_epi32_mask(t0, _mm256_set1_epi32(1));
            return (t1 == 0x00);
        }
        // HADD
        inline int32_t hadd() const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] + raw[1] + raw[2] + raw[3] +
                   raw[4] + raw[5] + raw[6] + raw[7];
        }
        // MHADD
        inline int32_t hadd(SIMDVecMask<8> const mask) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = 0;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            if (mask.mMask & 0x10) t0 += raw[4];
            if (mask.mMask & 0x20) t0 += raw[5];
            if (mask.mMask & 0x40) t0 += raw[6];
            if (mask.mMask & 0x80) t0 += raw[7];
            return t0;
        }
        // HADDS
        inline int32_t hadd(int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b + raw[0] + raw[1] + raw[2] + raw[3] + 
                       raw[4] + raw[5] + raw[6] + raw[7];
        }
        // MHADDS
        inline int32_t hadd(SIMDVecMask<8> const mask, int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x01) t0 += raw[0];
            if (mask.mMask & 0x02) t0 += raw[1];
            if (mask.mMask & 0x04) t0 += raw[2];
            if (mask.mMask & 0x08) t0 += raw[3];
            if (mask.mMask & 0x10) t0 += raw[4];
            if (mask.mMask & 0x20) t0 += raw[5];
            if (mask.mMask & 0x40) t0 += raw[6];
            if (mask.mMask & 0x80) t0 += raw[7];
            return t0;
        }
        // HMUL
        inline int32_t hmul() const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] * raw[1] * raw[2] * raw[3] *
                   raw[4] * raw[5] * raw[6] * raw[7];
        }
        // MHMUL
        inline int32_t hmul(SIMDVecMask<8> const mask) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = 1;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            if (mask.mMask & 0x10) t0 *= raw[4];
            if (mask.mMask & 0x20) t0 *= raw[5];
            if (mask.mMask & 0x40) t0 *= raw[6];
            if (mask.mMask & 0x80) t0 *= raw[7];
            return t0;
        }
        // HMULS
        inline int32_t hmul(int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b * raw[0] * raw[1] * raw[2] * raw[3] +
                       raw[4] + raw[5] + raw[6] + raw[7];
        }
        // MHMULS
        inline int32_t hmul(SIMDVecMask<8> const mask, int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = 1;
            if (mask.mMask & 0x01) t0 *= raw[0];
            if (mask.mMask & 0x02) t0 *= raw[1];
            if (mask.mMask & 0x04) t0 *= raw[2];
            if (mask.mMask & 0x08) t0 *= raw[3];
            if (mask.mMask & 0x10) t0 *= raw[4];
            if (mask.mMask & 0x20) t0 *= raw[5];
            if (mask.mMask & 0x40) t0 *= raw[6];
            if (mask.mMask & 0x80) t0 *= raw[7];
            return b * t0;
        }
        // FMULADDV
        /*inline SIMDVec_i fmuladd(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm_mul_epi32(mVec, b.mVec);
            __m256i t1 = _mm_add_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }*/
        // MFMULADDV
        /*inline SIMDVec_i fmuladd(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm_mask_mul_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m256i t1 = _mm_mask_add_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }*/
        // FMULSUBV
        /*inline SIMDVec_i fmulsub(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm_mul_epi32(mVec, b.mVec);
            __m256i t1 = _mm_sub_epi32(t0, c.mVec);
            return SIMDVec_i(t1);
        }*/
        // MFMULSUBV
        /*inline SIMDVec_i fmulsub(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm_mask_mul_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m256i t1 = _mm_mask_sub_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }*/
        // FADDMULV
        /*inline SIMDVec_i faddmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm_add_epi32(t0, b.mVec);
            __m256i t1 = _mm_mul_epi32(mVec, c.mVec);
            return SIMDVec_i(t1);
        }*/
        // MFADDMULV
        /*inline SIMDVec_i faddmul(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm_mask_add_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m256i t1 = _mm_mask_mul_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }*/
        // FSUBMULV
        /*inline SIMDVec_i fsubmul(SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm_sub_epi32(t0, b.mVec);
            __m256i t1 = _mm_mul_epi32(mVec, c.mVec);
            return SIMDVec_i(t1);
        }*/
        // MFSUBMULV
        /*inline SIMDVec_i fsubmul(SIMDVecMask<8> const & mask, SIMDVec_i const & b, SIMDVec_i const & c) const {
            __m256i t0 = _mm_mask_sub_epi32(mVec, mask.mMask, mVec, b.mVec);
            __m256i t1 = _mm_mask_mul_epi32(t0, mask.mMask, t0, c.mVec);
            return SIMDVec_i(t1);
        }*/
        // MAXV
        inline SIMDVec_i max(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_max_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMAXV
        inline SIMDVec_i max(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MAXS
        inline SIMDVec_i max(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_max_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMAXS
        inline SIMDVec_i max(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_max_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MAXVA
        inline SIMDVec_i & maxa(SIMDVec_i const & b) {
            mVec = _mm256_max_epi32(mVec, b.mVec);
            return *this;
        }
        // MMAXVA
        inline SIMDVec_i & maxa(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_max_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MAXSA
        inline SIMDVec_i & maxa(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_max_epi32(mVec, t0);
            return *this;
        }
        // MMAXSA
        inline SIMDVec_i & maxa(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_max_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // MINV
        inline SIMDVec_i min(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_min_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MMINV
        inline SIMDVec_i min(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MINS
        inline SIMDVec_i min(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_min_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MMINS
        inline SIMDVec_i min(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_min_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MINVA
        inline SIMDVec_i & mina(SIMDVec_i const & b) {
            mVec = _mm256_min_epi32(mVec, b.mVec);
            return *this;
        }
        // MMINVA
        inline SIMDVec_i & mina(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_min_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // MINSA
        inline SIMDVec_i & mina(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_min_epi32(mVec, t0);
            return *this;
        }
        // MMINSA
        inline SIMDVec_i & mina(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_min_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HMAX
        // MHMAX
        // IMAX
        // MIMAX
        // HMIN
        // MHMIN
        // IMIN
        // MIMIN

        // BANDV
        inline SIMDVec_i band(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_and_si256(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MBANDV
        inline SIMDVec_i band(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BANDS
        inline SIMDVec_i band(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_and_si256(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBANDS
        inline SIMDVec_i band(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BANDVA
        inline SIMDVec_i & banda(SIMDVec_i const & b) {
            mVec = _mm256_and_si256(mVec, b.mVec);
            return *this;
        }
        // MBANDVA
        inline SIMDVec_i & banda(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_and_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BANDSA
        inline SIMDVec_i & banda(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_and_si256(mVec, t0);
            return *this;
        }
        // MBANDSA
        inline SIMDVec_i & banda(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_and_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BORV
        inline SIMDVec_i bor(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_or_si256(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MBORV
        inline SIMDVec_i bor(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BORS
        inline SIMDVec_i bor(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_or_si256(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBORS
        inline SIMDVec_i bor(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BORVA
        inline SIMDVec_i & bora(SIMDVec_i const & b) {
            mVec = _mm256_or_si256(mVec, b.mVec);
            return *this;
        }
        // MBORVA
        inline SIMDVec_i & bora(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_or_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BORSA
        inline SIMDVec_i & bora(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_or_si256(mVec, t0);
            return *this;
        }
        // MBORSA
        inline SIMDVec_i & bora(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_or_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BXORV
        inline SIMDVec_i bxor(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_xor_si256(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MBXORV
        inline SIMDVec_i bxor(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // BXORS
        inline SIMDVec_i bxor(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_xor_si256(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBXORS
        inline SIMDVec_i bxor(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BXORVA
        inline SIMDVec_i & bxora(SIMDVec_i const & b) {
            mVec = _mm256_xor_si256(mVec, b.mVec);
            return *this;
        }
        // MBXORVA
        inline SIMDVec_i & bxora(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // BXORSA
        inline SIMDVec_i & bxora(int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_xor_si256(mVec, t0);
            return *this;
        }
        // MBXORSA
        inline SIMDVec_i & bxora(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_xor_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // BNOT
        inline SIMDVec_i bnot() const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_mask_andnot_epi32(mVec, 0xFF, mVec, t0);
            return SIMDVec_i(t1);
        }
        // MBNOT
        inline SIMDVec_i bnot(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            __m256i t1 = _mm256_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // BNOTA
        inline SIMDVec_i & bnota() {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            mVec = _mm256_mask_andnot_epi32(mVec, 0xFF, mVec, t0);
            return *this;
        }
        // MBNOTA
        inline SIMDVec_i bnota(SIMDVecMask<8> const & mask) {
            __m256i t0 = _mm256_set1_epi32(0xFFFFFFFF);
            mVec = _mm256_mask_andnot_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // HBAND
        inline int32_t hband() const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] & raw[1] & raw[2] & raw[3] &
                   raw[4] & raw[5] & raw[6] & raw[7];
        }
        // MHBAND
        inline int32_t hband(SIMDVecMask<8> const mask) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = 0xFFFFFFFF;
            if (mask.mMask & 0x01) t0 &= raw[0];
            if (mask.mMask & 0x02) t0 &= raw[1];
            if (mask.mMask & 0x04) t0 &= raw[2];
            if (mask.mMask & 0x08) t0 &= raw[3];
            if (mask.mMask & 0x10) t0 &= raw[4];
            if (mask.mMask & 0x20) t0 &= raw[5];
            if (mask.mMask & 0x40) t0 &= raw[6];
            if (mask.mMask & 0x80) t0 &= raw[7];
            return t0;
        }
        // HBANDS
        inline int32_t hband(int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b & raw[0] & raw[1] & raw[2] & raw[3] &
                       raw[4] & raw[5] & raw[6] & raw[7];
        }
        // MHBANDS
        inline int32_t hband(SIMDVecMask<8> const mask, int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x01) t0 &= raw[0];
            if (mask.mMask & 0x02) t0 &= raw[1];
            if (mask.mMask & 0x04) t0 &= raw[2];
            if (mask.mMask & 0x08) t0 &= raw[3];
            if (mask.mMask & 0x10) t0 &= raw[4];
            if (mask.mMask & 0x20) t0 &= raw[5];
            if (mask.mMask & 0x40) t0 &= raw[6];
            if (mask.mMask & 0x80) t0 &= raw[7];
            return t0;
        }
        // HBOR
        inline int32_t hbor() const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] | raw[1] | raw[2] | raw[3] |
                   raw[4] | raw[5] | raw[6] | raw[7];
        }
        // MHBOR
        inline int32_t hbor(SIMDVecMask<8> const mask) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = 0;
            if (mask.mMask & 0x01) t0 |= raw[0];
            if (mask.mMask & 0x02) t0 |= raw[1];
            if (mask.mMask & 0x04) t0 |= raw[2];
            if (mask.mMask & 0x08) t0 |= raw[3];
            if (mask.mMask & 0x10) t0 |= raw[4];
            if (mask.mMask & 0x20) t0 |= raw[5];
            if (mask.mMask & 0x40) t0 |= raw[6];
            if (mask.mMask & 0x80) t0 |= raw[7];
            return t0;
        }
        // HBORS
        inline int32_t hbor(int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b | raw[0] | raw[1] | raw[2] | raw[3] |
                       raw[4] | raw[5] | raw[6] | raw[7];
        }
        // MHBORS
        inline int32_t hbor(SIMDVecMask<8> const mask, int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x01) t0 |= raw[0];
            if (mask.mMask & 0x02) t0 |= raw[1];
            if (mask.mMask & 0x04) t0 |= raw[2];
            if (mask.mMask & 0x08) t0 |= raw[3];
            if (mask.mMask & 0x10) t0 |= raw[4];
            if (mask.mMask & 0x20) t0 |= raw[5];
            if (mask.mMask & 0x40) t0 |= raw[6];
            if (mask.mMask & 0x80) t0 |= raw[7];
            return t0;
        }
        // HBXOR
        inline int32_t hbxor() const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                   raw[4] ^ raw[5] ^ raw[6] ^ raw[7];
        }
        // MHBXOR
        inline int32_t hbxor(SIMDVecMask<8> const mask) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = 0;
            if (mask.mMask & 0x01) t0 ^= raw[0];
            if (mask.mMask & 0x02) t0 ^= raw[1];
            if (mask.mMask & 0x04) t0 ^= raw[2];
            if (mask.mMask & 0x08) t0 ^= raw[3];
            if (mask.mMask & 0x10) t0 ^= raw[4];
            if (mask.mMask & 0x20) t0 ^= raw[5];
            if (mask.mMask & 0x40) t0 ^= raw[6];
            if (mask.mMask & 0x80) t0 ^= raw[7];
            return t0;
        }
        // HBXORS
        inline int32_t hbxor(int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            return b ^ raw[0] ^ raw[1] ^ raw[2] ^ raw[3] ^
                       raw[4] ^ raw[5] ^ raw[6] ^ raw[7];
        }
        // MHBXORS
        inline int32_t hbxor(SIMDVecMask<8> const mask, int32_t b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i*)raw, mVec);
            int32_t t0 = b;
            if (mask.mMask & 0x01) t0 ^= raw[0];
            if (mask.mMask & 0x02) t0 ^= raw[1];
            if (mask.mMask & 0x04) t0 ^= raw[2];
            if (mask.mMask & 0x08) t0 ^= raw[3];
            if (mask.mMask & 0x10) t0 ^= raw[4];
            if (mask.mMask & 0x20) t0 ^= raw[5];
            if (mask.mMask & 0x40) t0 ^= raw[6];
            if (mask.mMask & 0x80) t0 ^= raw[7];
            return t0;
        }
        // GATHERS
        inline SIMDVec_i & gather(int32_t* baseAddr, uint64_t* indices) {
            alignas(32) int32_t raw[8] = { 
                baseAddr[indices[0]], baseAddr[indices[1]], 
                baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]],
                baseAddr[indices[6]], baseAddr[indices[7]], };
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // MGATHERS
        inline SIMDVec_i & gather(SIMDVecMask<8> const & mask, int32_t* baseAddr, uint64_t* indices) {
            alignas(32) int32_t raw[8] = { 
                baseAddr[indices[0]], baseAddr[indices[1]], 
                baseAddr[indices[2]], baseAddr[indices[3]],
                baseAddr[indices[4]], baseAddr[indices[5]],
                baseAddr[indices[6]], baseAddr[indices[7]] };
            mVec = _mm256_mask_load_epi32(mVec, mask.mMask, raw);
            return *this;
        }
        // GATHERV
        inline SIMDVec_i & gather(int32_t* baseAddr, SIMDVec_i const & indices) {
            alignas(32) int32_t rawIndices[4];
            alignas(32) int32_t rawData[4];
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            rawData[0] = baseAddr[rawIndices[0]];
            rawData[1] = baseAddr[rawIndices[1]];
            rawData[2] = baseAddr[rawIndices[2]];
            rawData[3] = baseAddr[rawIndices[3]];
            rawData[4] = baseAddr[rawIndices[4]];
            rawData[5] = baseAddr[rawIndices[5]];
            rawData[6] = baseAddr[rawIndices[6]];
            rawData[7] = baseAddr[rawIndices[7]];
            mVec = _mm256_load_si256((__m256i*)rawData);
            return *this;
        }
        // MGATHERV
        inline SIMDVec_i & gather(SIMDVecMask<8> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) {
            alignas(32) int32_t rawIndices[8];
            alignas(32) int32_t rawData[8];
            _mm256_store_si256((__m256i*) rawIndices, indices.mVec);
            rawData[0] = baseAddr[rawIndices[0]];
            rawData[1] = baseAddr[rawIndices[1]];
            rawData[2] = baseAddr[rawIndices[2]];
            rawData[3] = baseAddr[rawIndices[3]];
            rawData[4] = baseAddr[rawIndices[4]];
            rawData[5] = baseAddr[rawIndices[5]];
            rawData[6] = baseAddr[rawIndices[6]];
            rawData[7] = baseAddr[rawIndices[7]];
            mVec = _mm256_mask_load_epi32(mVec, mask.mMask, rawData);
            return *this;
        }
        // SCATTERS
        inline int32_t* scatter(int32_t* baseAddr, uint64_t* indices) {
            alignas(32) int32_t rawIndices[8] = { 
                indices[0], indices[1], indices[2], indices[3],
                indices[4], indices[5], indices[6], indices[7] };
            __m256i t0 = _mm256_load_si256((__m256i *) rawIndices);
            _mm256_i32scatter_epi32(baseAddr, t0, mVec, 1);
            return baseAddr;
        }
        // MSCATTERS
        inline int32_t* scatter(SIMDVecMask<8> const & mask, int32_t* baseAddr, uint64_t* indices) {
            alignas(32) int32_t rawIndices[8] = { 
                indices[0], indices[1], indices[2], indices[3],
                indices[4], indices[5], indices[6], indices[7] };
            __m256i t0 = _mm256_mask_load_epi32(_mm256_set1_epi32(0), mask.mMask, (__m256i *) rawIndices);
            _mm256_mask_i32scatter_epi32(baseAddr, mask.mMask, t0, mVec, 1);
            return baseAddr;
        }
        // SCATTERV
        inline int32_t* scatter(int32_t* baseAddr, SIMDVec_i const & indices) {
            _mm256_i32scatter_epi32(baseAddr, indices.mVec, mVec, 1);
            return baseAddr;
        }
        // MSCATTERV
        inline int32_t* scatter(SIMDVecMask<8> const & mask, int32_t* baseAddr, SIMDVec_i const & indices) {
            _mm256_mask_i32scatter_epi32(baseAddr, mask.mMask, indices.mVec, mVec, 1);
            return baseAddr;
        }
        // LSHV
        // MLSHV
        // LSHS
        // MLSHS
        // LSHVA
        // MLSHVA
        // LSHSA
        // MLSHSA
        // RSHV
        // MRSHV
        // RSHS
        // MRSHS
        // RSHVA
        // MRSHVA
        // RSHSA
        // MRSHSA
        // ROLV
        inline SIMDVec_i rol(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_rolv_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MROLV
        inline SIMDVec_i rol(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // ROLS
        inline SIMDVec_i rol(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_rolv_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MROLS
        inline SIMDVec_i rol(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // ROLVA
        inline SIMDVec_i & rola(SIMDVec_i const & b) {
            mVec = _mm256_rolv_epi32(mVec, b.mVec);
            return *this;
        }
        // MROLVA
        inline SIMDVec_i & rola(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_rolv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // ROLSA
        inline SIMDVec_i & rola(int32_t b) {
            mVec = _mm256_rolv_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        // MROLSA
        inline SIMDVec_i & rola(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_rolv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // RORV
        inline SIMDVec_i ror(SIMDVec_i const & b) const {
            __m256i t0 = _mm256_rorv_epi32(mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // MRORV
        inline SIMDVec_i ror(SIMDVecMask<8> const & mask, SIMDVec_i const & b) const {
            __m256i t0 = _mm256_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return SIMDVec_i(t0);
        }
        // RORS
        inline SIMDVec_i ror(int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_rorv_epi32(mVec, t0);
            return SIMDVec_i(t1);
        }
        // MRORS
        inline SIMDVec_i ror(SIMDVecMask<8> const & mask, int32_t b) const {
            __m256i t0 = _mm256_set1_epi32(b);
            __m256i t1 = _mm256_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return SIMDVec_i(t1);
        }
        // RORVA
        inline SIMDVec_i & rora(SIMDVec_i const & b) {
            mVec = _mm256_rorv_epi32(mVec, b.mVec);
            return *this;
        }
        // MRORVA
        inline SIMDVec_i & rora(SIMDVecMask<8> const & mask, SIMDVec_i const & b) {
            mVec = _mm256_mask_rorv_epi32(mVec, mask.mMask, mVec, b.mVec);
            return *this;
        }
        // RORSA
        inline SIMDVec_i & rora(int32_t b) {
            mVec = _mm256_rorv_epi32(mVec, _mm256_set1_epi32(b));
            return *this;
        }
        // MRORSA
        inline SIMDVec_i & rora(SIMDVecMask<8> const & mask, int32_t b) {
            __m256i t0 = _mm256_set1_epi32(b);
            mVec = _mm256_mask_rorv_epi32(mVec, mask.mMask, mVec, t0);
            return *this;
        }
        // NEG
        inline SIMDVec_i neg() const {
            __m256i t0 = _mm256_sub_epi32(_mm256_setzero_si256(), mVec);
            return SIMDVec_i(t0);
        }
        // MNEG
        inline SIMDVec_i neg(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_setzero_si256();
            __m256i t1 = _mm256_mask_sub_epi32(mVec, mask.mMask, t0, mVec);
            return SIMDVec_i(t1);
        }
        // NEGA
        inline SIMDVec_i & nega() {
            mVec = _mm256_sub_epi32(_mm256_setzero_si256(), mVec);
            return *this;
        }
        // MNEGA
        inline SIMDVec_i & nega(SIMDVecMask<8> const & mask) {
            mVec = _mm256_mask_sub_epi32(mVec, mask.mMask, _mm256_setzero_si256(), mVec);
            return *this;
        }
        // ABS
        inline SIMDVec_i abs() const {
            __m256i t0 = _mm256_abs_epi32(mVec);
            return SIMDVec_i(t0);
        }
        // MABS
        inline SIMDVec_i abs(SIMDVecMask<8> const & mask) const {
            __m256i t0 = _mm256_mask_abs_epi32(mVec, mask.mMask, mVec);
            return SIMDVec_i(t0);
        }
        // ABSA
        inline SIMDVec_i & absa() {
            mVec = _mm256_abs_epi32(mVec);
            return *this;
        }
        // MABSA
        inline SIMDVec_i & absa(SIMDVecMask<8> const & mask) {
            mVec = _mm256_mask_abs_epi32(mVec, mask.mMask, mVec);
            return *this;
        }
        // PACK
        inline SIMDVec_i & pack(SIMDVec_i<int32_t, 4> const & a, SIMDVec_i<int32_t, 4> const & b) {
            alignas(32) int32_t raw[8];
            _mm_store_si128((__m128i*)&raw[0], a.mVec);
            _mm_store_si128((__m128i*)&raw[4], b.mVec);
            mVec = _mm256_load_si256((__m256i*)raw);
            return *this;
        }
        // PACKLO
        inline SIMDVec_i & packlo(SIMDVec_i<int32_t, 4> const & a) {
            alignas(32) int32_t raw[8];
            _mm_store_si128((__m128i*)&raw[0], a.mVec);
            mVec = _mm256_mask_load_epi32(mVec, 0xF, (__m256i*)raw);
            return *this;
        }
        // PACKHI
        inline SIMDVec_i & packhi(SIMDVec_i<int32_t, 4> const & b) {
            alignas(32) int32_t raw[8];
            _mm_store_si128((__m128i*)&raw[4], b.mVec);
            mVec = _mm256_mask_load_epi32(mVec, 0xF0, (__m256i*)raw);
            return *this;
        }
        // UNPACK
        inline void unpack(SIMDVec_i<int32_t, 4> & a, SIMDVec_i<int32_t, 4> & b) const {
            alignas(32) int32_t raw[8];
            _mm256_store_si256((__m256i *)raw, mVec);
            a.mVec = _mm_load_si128((__m128i*)&raw[0]);
            b.mVec = _mm_load_si128((__m128i*)&raw[4]);
        }
        // UNPACKLO
        inline SIMDVec_i<int32_t, 4> unpacklo() const {
            alignas(32) int32_t raw[8];
            _mm256_mask_store_epi32((__m256i*)raw, 0xF, mVec);
            __m128i t0 = _mm_load_si128((__m128i*)&raw[0]);
            return SIMDVec_i<int32_t, 4>(t0);
        }
        // UNPACKHI
        inline SIMDVec_i<int32_t, 4> unpackhi() const {
            alignas(32) int32_t raw[8];
            _mm256_mask_store_epi32(raw, 0xF0, mVec);
            __m128i t0 = _mm_load_si128((__m128i*)&raw[4]);
            return SIMDVec_i<int32_t, 4>(t0);
        }

        // ITOU
        inline operator SIMDVec_u<uint32_t, 8> () const;
        // ITOF
        inline operator SIMDVec_f<float, 8>() const;

    };

}
}

#endif
