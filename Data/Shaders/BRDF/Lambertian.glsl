/**
 * MIT License
 *
 * Copyright (c) 2024, Christoph Neuhauser, Jonas Itt
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
    
#ifdef UNIFORM_SAMPLING
    
    // Sampling PDF: 1/(2pi)
    return frame * sampleHemisphere(vec2(random(), random()));
    
#else
    
    // Sampling PDF: cos(theta) / pi
    return frame * sampleHemisphereCosineWeighted(vec2(random(), random()));
    
#endif
}

// Evaluate BRDF with compensation of importance sampling and a fixed sampling PDF
vec3 evaluateBrdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColor) {
    float theta = dot(lightVector, normalVector);

    // BRDF: R/pi = isosurfaceColor / pi, i.e., sampling PDF is multiplied with pi.
#ifdef UNIFORM_SAMPLING
    
    // Sampling PDF: 1/(2pi)
    float pdfFactor = 0.5;
    
#else
    
    // Sampling PDF: cos(theta) / pi
    float pdfFactor = theta;

#endif

    return (isoSurfaceColor * theta) / pdfFactor;
}

vec3 evaluateBrdfNee(vec3 viewVector, vec3 dirOut, vec3 dirNee, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, vec3 isoSurfaceColor, bool useMIS, float samplingPDF, flags hitFlags, out float pdfSamplingOut, out float pdfSamplingNee) {
    float thetaNee = dot(normalVector, dirNee);
#ifdef UNIFORM_SAMPLING
    if(useMIS) {
        pdfSamplingNee = 1.0 / (2.0 * M_PI);
    }
    pdfSamplingOut = 1.0 / (2.0 * M_PI);
#else
    if(useMIS) {
        pdfSamplingNee = thetaNee / M_PI;
    }
    pdfSamplingOut = dot(dirOut, normalVector) / M_PI;
#endif
    
    return isoSurfaceColor / M_PI * thetaNee;
}

// Combined call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {
    lightVector = sampleBrdf(frame, hitFlags);
    return evaluateBrdf(viewVector, lightVector, normalVector, isoSurfaceColor);
}
