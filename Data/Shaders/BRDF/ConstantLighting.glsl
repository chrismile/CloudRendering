/**
 * MIT License
 *
 * Copyright (c) 2024, Christoph Neuhauser
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// ---------- BRDF Helper Functions ----------

// 1. Helper functions for sampling


// 2. Helper functions for evaluation


// ---------- BRDF Interface ----------

// Compute importance-sampled light vector
vec3 sampleBrdf(mat3 frame, flags hitFlags) {
    hitFlags.specularHit = false;
    hitFlags.clearcoatHit = false;
    
    // Sampling PDF: 1/(2pi)
    return frame * sampleHemisphere(vec2(random(), random()));
}

// Evaluate BRDF with compensation of importance sampling and a fixed sampling PDF
vec3 evaluateBrdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColor) {
    float theta = dot(lightVector, normalVector);

    // BRDF: R/(2pi * theta) = isosurfaceColor / (2pi * theta).
    // Sampling PDF: 1/(2pi)
    // sum BRDF * theta / PDF = sum R
    return isoSurfaceColor;
}

vec3 evaluateBrdfNee(vec3 viewVector, vec3 dirOut, vec3 dirNee, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, vec3 isoSurfaceColor, bool useMIS, float samplingPDF, flags hitFlags, out float pdfSamplingOut, out float pdfSamplingNee) {
    if(useMIS) {
        pdfSamplingNee = 1.0 / (2.0 * M_PI);
    }
    pdfSamplingOut = 1.0 / (2.0 * M_PI);
    return isoSurfaceColor / (2.0 * M_PI);
}

// Combined call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {
    lightVector = sampleBrdf(frame, hitFlags);
    return evaluateBrdf(viewVector, lightVector, normalVector, isoSurfaceColor);
}
