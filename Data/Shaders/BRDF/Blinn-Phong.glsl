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

// Compute Importance Sampled light vector
vec3 sampleBrdf(mat3 frame, out flags hitFlags) {
    hitFlags.specularHit = false;
    hitFlags.clearcoatHit = false;
	return frame * sampleHemisphere(vec2(random(), random()));
}

// Evaluate BRDF with compensation of Importance Sampling and a fixed sampling PDF


// Evaluate BRDF and prepare for Importance Sampling compensation
vec3 evaluateBrdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColor) {
    // http://www.thetenthplanet.de/archives/255
    vec3 halfwayVector = normalize(viewVector + lightVector);
    const float n = 10.0;
    float norm = clamp(
            (n + 2.0) / (4.0 * M_PI * (exp2(-0.5 * n))),
            (n + 2.0) / (8.0 * M_PI), (n + 4.0) / (8.0 * M_PI));

    return 2.0 * M_PI * isoSurfaceColor * (pow(max(dot(normalVector, halfwayVector), 0.0), n) / norm);
}

vec3 evaluateBrdfNee(vec3 viewVector, vec3 dirOut, vec3 dirNee, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, vec3 isoSurfaceColor, bool useMIS, float samplingPDF, flags hitFlags, out float pdfSamplingOut, out float pdfSamplingNee) {
    pdfSamplingNee = 1.0 / (2.0 * M_PI);
    pdfSamplingOut = 1.0 / (2.0 * M_PI);
        
    // http://www.thetenthplanet.de/archives/255
    const float n = 10.0;
    float norm = clamp(
        (n + 2.0) / (4.0 * M_PI * (exp2(-0.5 * n))),
        (n + 2.0) / (8.0 * M_PI), (n + 4.0) / (8.0 * M_PI));
    vec3 halfwayVectorNee = normalize(viewVector + dirNee);
    return isoSurfaceColor * (pow(max(dot(normalVector, halfwayVectorNee), 0.0), n) / norm);
}

// Combined Call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {
    
    lightVector = sampleBrdf(frame, hitFlags);
    return evaluateBrdf(viewVector, lightVector, normalVector, isoSurfaceColor);
}