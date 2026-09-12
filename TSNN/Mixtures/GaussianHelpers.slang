

// SPDX-License-Identifier: BSD-2-Clause
//
// Slang/SPIR-V translation of Norbert Juffa's CUDA erff approximation.
// The translation changes target-language intrinsics but retains the
// original approximation and coefficients.
//
// Copyright (c) 2016-2021, Norbert Juffa
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
//    notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimer in the
//    documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
// Based on Norbert Juffa's BSD-2-Clause erff approximation.
// Maximum error reported for the original CUDA implementation: 1.29532 ULP.
[ForceInline]
[BackwardDerivative(erf_bwd)]
float erf(float x)
{
    float ax = abs(x);
    float x2 = x * x;
    float r;

    if (ax <= 1.0f)
    {
        r = -5.58853149e-4f;
        r = fma(r, x2,  4.90735564e-3f);
        r = fma(r, x2, -2.67030653e-2f);
        r = fma(r, x2,  1.12799220e-1f);
        r = fma(r, x2, -3.76123011e-1f);
        r = fma(r, x2,  1.28379121e-1f);

        return fma(r, x, x);
    }

    r = 2.58907676e-5f;
    r = fma(r, ax, -5.64874616e-4f);

    float u = 5.66795561e-3f;
    u = fma(u, ax, -3.51759829e-2f);

    r = fma(r, x2, u);
    r = fma(r, ax, 1.54329389e-1f);
    r = fma(r, ax, 9.15674746e-1f);
    r = fma(r, ax, 6.28459513e-1f);
    r = fma(r, ax, ax);

    r = 1.0f - exp2(-r);

    // Apply x's sign without a floating-point comparison or divergent branch.
    return asfloat(asuint(r) | (asuint(x) & 0x80000000u));
}

void erf_bwd(inout DifferentialPair<float> dpx, float dOut)
{
    float x = dpx.p;
    float dx = dOut * (2.0f / sqrt(float.getPi())) * exp(-x * x);
    dpx = diffPair(x, dx);
}

[ForceInline]
float normalCDF(float x, float mean, float logSigma)
{
    float k = exp(-logSigma) / sqrt(2.0f);
    return 0.5f * (1.0f + erf((x - mean) * k));
}

// Fast inverse error function approximation
// Source: "A handy approximation for the error function and its inverse" by Sergei Winitzki.
// Code: https://stackoverflow.com/questions/27229371/inverse-error-function-in-c
// From: https://gpuopen.com/learn/sampling-normal-gaussian-distribution-gpus/
[ForceInline]
float erfinv_fast(float x)
{
    float tt1, tt2, lnx, sgn;
    sgn = (x < 0.0f) ? -1.0f : 1.0f;
 
    x = (1.0f - x) * (1.0f + x);
    lnx = log(x);
 
    tt1 = 2.0f / (float.getPi() * 0.147f) + 0.5f * lnx;
    tt2 = 1.0f / (0.147f) * lnx;
 
    return (sgn * sqrt(-tt1 + sqrt(tt1 * tt1 - tt2)));
}

// Precise inverse error function approximation using polynomials
// Source: "Approximating the erfinv function"
// https://people.maths.ox.ac.uk/gilesm/files/gems_erfinv.pdf
[ForceInline]
float erfinv(float x)
{
    float w, p;
    w = -log((1.0f - x) * (1.0f + x));
    if (w < 5.000000f)
    {
        w = w - 2.500000f;
        p = 2.81022636e-08f;
        p = 3.43273939e-07f + p * w;
        p = -3.5233877e-06f + p * w;
        p = -4.39150654e-06f + p * w;
        p = 0.00021858087f + p * w;
        p = -0.00125372503f + p * w;
        p = -0.00417768164f + p * w;
        p = 0.246640727f + p * w;
        p = 1.50140941f + p * w;
    }
    else
    {
        w = sqrt(w) - 3.000000f;
        p = -0.000200214257f;
        p = 0.000100950558f + p * w;
        p = 0.00134934322f + p * w;
        p = -0.00367342844f + p * w;
        p = 0.00573950773f + p * w;
        p = -0.0076224613f + p * w;
        p = 0.00943887047f + p * w;
        p = 1.00167406f + p * w;
        p = 2.83297682f + p * w;
    }
    return p * x;
}

[ForceInline]
float normalCDFInverse(float p, float mean, float logSigma)
{
    float k = exp(logSigma) * sqrt(2.0f);
    return mean + k * erfinv(2.0f * p - 1.0f);
}