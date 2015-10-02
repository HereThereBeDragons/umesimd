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
// This piece of code was developed as part of ICE-DIP project at CERN.
//  "ICE-DIP is a European Industrial Doctorate project funded by the European Community's 
//  7th Framework programme Marie Curie Actions under grant PITN-GA-2012-316596".
//
#ifndef UME_UNIT_TEST_DATA_SETS_H_
#define UME_UNIT_TEST_DATA_SETS_H_

#include "../UMEBasicTypes.h"

struct DataSet_1_32u {
    struct inputs {
        static const uint32_t inputA[32];
        static const uint32_t inputB[32];
        static const uint32_t inputC[32];
        static const uint32_t scalarA;
        static const bool  maskA[32];
    };

    struct outputs {
        static const uint32_t ADDV[32];
        static const uint32_t MADDV[32];
        static const uint32_t ADDS[32];
        static const uint32_t MADDS[32];
        static const uint32_t POSTPREFINC[32];
        static const uint32_t MPOSTPREFINC[32];
        static const uint32_t SUBV[32];
        static const uint32_t MSUBV[32];
        static const uint32_t SUBS[32];
        static const uint32_t MSUBS[32];
        static const uint32_t SUBFROMV[32];
        static const uint32_t MSUBFROMV[32];
        static const uint32_t SUBFROMS[32];
        static const uint32_t MSUBFROMS[32];
        static const uint32_t POSTPREFDEC[32];
        static const uint32_t MPOSTPREFDEC[32];
        static const uint32_t MULV[32];
        static const uint32_t MMUL[32];
        static const uint32_t MULS[32];
        static const uint32_t MMULS[32];
        static const uint32_t DIVV[32];
        static const uint32_t MDIVV[32];
        static const uint32_t DIVS[32];
        static const uint32_t MDIVS[32];
        static const uint32_t RCP[32];
        static const uint32_t MRCP[32];
        static const uint32_t RCPS[32];
        static const uint32_t MRCPS[32];
        static const uint32_t FMULADD[32];
        static const uint32_t MFMULADD[32];
        static const uint32_t FMULSUB[32];
        static const uint32_t MFMULSUBV[32];
        static const uint32_t FADDMULV[32];
        static const uint32_t MFADDMULV[32];
        static const uint32_t FSUBMULV[32];
        static const uint32_t MFSUBMULV[32];
        static const uint32_t MAXV[32];
        static const uint32_t MMAXV[32];
        static const uint32_t MAXS[32];
        static const uint32_t MMAXS[32];
        static const uint32_t MINV[32];
        static const uint32_t MMINV[32];
        static const uint32_t MINS[32];
        static const uint32_t MMINS[32];
        static const uint32_t SQR[32];
        static const uint32_t MSQR[32];
        static const uint32_t SQRT[32];
        static const uint32_t MSQRT[32];
    };
};

struct DataSet_1_32f {
    struct inputs {
        static const float inputA[32];
        static const float inputB[32];
        static const float inputC[32];
        static const float scalarA;
        static const bool  maskA[32];
    };

    struct outputs {
        static const float ADDV[32];
        static const float MADDV[32];
        static const float ADDS[32];
        static const float MADDS[32];
        static const float POSTPREFINC[32];
        static const float MPOSTPREFINC[32];
        static const float SUBV[32];
        static const float MSUBV[32];
        static const float SUBS[32];
        static const float MSUBS[32];
        static const float SUBFROMV[32];
        static const float MSUBFROMV[32];
        static const float SUBFROMS[32];
        static const float MSUBFROMS[32];
        static const float POSTPREFDEC[32];
        static const float MPOSTPREFDEC[32];
        static const float MULV[32];
        static const float MMUL[32];
        static const float MULS[32];
        static const float MMULS[32];
        static const float DIVV[32];
        static const float MDIVV[32];
        static const float DIVS[32];
        static const float MDIVS[32];
        static const float RCP[32];
        static const float MRCP[32];
        static const float RCPS[32];
        static const float MRCPS[32];
        static const float FMULADD[32];
        static const float MFMULADD[32];
        static const float FMULSUB[32];
        static const float MFMULSUBV[32];
        static const float FADDMULV[32];
        static const float MFADDMULV[32];
        static const float FSUBMULV[32];
        static const float MFSUBMULV[32];
        static const float MAXV[32];
        static const float MMAXV[32];
        static const float MAXS[32];
        static const float MMAXS[32];
        static const float MINV[32];
        static const float MMINV[32];
        static const float MINS[32];
        static const float MMINS[32];
        static const float NEG[32];
        static const float MNEG[32];
        static const float ABS[32];
        static const float MABS[32];
        static const float SQR[32];
        static const float MSQR[32];
        static const float SQRT[32];
        static const float MSQRT[32];
        static const float ROUND[32];
        static const int32_t TRUNC[32];
        static const int32_t MTRUNC[32];
        static const float FLOOR[32];
        static const float MFLOOR[32];
        static const float CEIL[32];
        static const float MCEIL[32];
        static const float SIN[32];
        static const float MSIN[32];
        static const float COS[32];
        static const float MCOS[32];
        static const float TAN[32];
        static const float MTAN[32];
        static const float CTAN[32];
        static const float MCTAN[32];
    };
};

#endif
