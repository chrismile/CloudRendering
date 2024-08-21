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

#define BRDF_SUPPORTS_SPECULAR

// ---------- BRDF Helper Functions ----------

// 1. Helper functions for sampling

// Return the value further away from 0
float avoidZero(float x, float y){
    if ((abs(x) > abs(y))) {
        return abs(x);
    } else {
        return abs(y);
    }
}

vec3 sample_GGX(vec3 viewVector, float roughness, mat3 frameMatrix) {
    // Source (mathematical derivation): https://www.youtube.com/watch?v=MkFS6lw6aEs
    // Generate random u and v between 0.0 and 1.0
    float u = random();
    float v = random();

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    // Compute spherical angles
    float theta = acos(sqrt((1-u)/(u*(alpha2 - 1.0)+1.0)));
    float phi = 2 * M_PI * v;
    
    //pdf_ggx = cos(theta);
    // Compute halfway vector h
    vec3 halfwayVector = frameMatrix*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

    // Compute light Vector l
    vec3 lightVector = 2*avoidZero(dot(viewVector,halfwayVector),0.001)*halfwayVector - viewVector;
    return lightVector;
}

vec3 sample_Lambertian(vec3 viewVector, mat3 frameMatrix) {
    // Source (mathematical derivation): https://www.youtube.com/watch?v=xFsJMUS94Fs
    // Generate random u and v between 0.0 and 1.0
    float u = random();
    float v = random();

    // Compute spherical angles
    float theta = asin(sqrt(u));
    float phi = 2 * M_PI * v;

    // pdf
    //pdf_diffuse = (sin(theta)*cos(theta)/M_PI);
    vec3 lightVector = frameMatrix*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

    return lightVector;
}

// 2. Helper functions for evaluation
// Source: https://www.youtube.com/watch?v=gya7x9H3mV0
// Paper: https://onlinelibrary.wiley.com/doi/abs/10.1111/1467-8659.1330233
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1-F0) * pow((1.0 - cosTheta), 5.0);
}

// Source: https://www.youtube.com/watch?v=gya7x9H3mV0
// Paper: https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
float D_GGX(float NdotH, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;
    float b = (NdotH2 * (alpha2 - 1.0) + 1.0);
    return (1 / M_PI) * alpha2 / (b * b);
}

// Paper: https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
float D_Beckmann(float NdotH, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float first = 1.0 / (M_PI*alpha2*pow(NdotH,4.0));
    float ex = exp((NdotH*NdotH -1)/(alpha2*NdotH*NdotH));
    return first*ex;
}

// Source: https://www.youtube.com/watch?v=gya7x9H3mV0
// Paper: https://www.graphics.cornell.edu/~bjw/microfacetbsdf.pdf
float G1_GGX_Schlick(float NdotV, float roughness) {
    float alpha = roughness * roughness;
    float k = alpha / 2.0;
    return max(NdotV, 0.001) / (max(NdotV, 0.001) * (1.0 - k) + k);
}

// Source: https://www.youtube.com/watch?v=gya7x9H3mV0
// Paper: https://ieeexplore.ieee.org/abstract/document/1138991
float G_Smith(float NdotV, float NdotL, float roughness) {
    return G1_GGX_Schlick(NdotL, roughness) * G1_GGX_Schlick(NdotV, roughness);
}


// ---------- BRDF Interface ----------

// Compute Importance Sampled light vector
vec3 sampleBrdf(float metallic, float specular, float roughness, vec3 viewVector, mat3 frameMatrix, out flags hitFlags) {
    // Idea adapted from https://schuttejoe.github.io/post/disneybsdf/
    // Calculate probabilies for sampling the lobes
    float metal = metallic;
    float spec = (1.0 - roughness) * (1.0 + specular);
    float dielectric = (1.0 - metallic);

    float specularW = metal + spec;
    float diffuseW = dielectric;

    float norm = 1.0/(specularW + diffuseW);

    float specularP = specularW * norm;
    float diffuseP = diffuseW * norm;

    float u = random();

    if(u < specularP) {
        hitFlags.specularHit = true;
        return sample_GGX(viewVector, roughness, frameMatrix);
    } else {
        hitFlags.specularHit = false;
        return sample_Lambertian(viewVector, frameMatrix);
    }
}

// Evaluate BRDF with compensation of Importance Sampling and a fixed sampling PDF
vec3 evaluateBrdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColorDef, flags hitFlags, out float samplingPDF) {
    vec3 halfwayVector = normalize(lightVector + viewVector);

    float NdotH = dot(halfwayVector, normalVector);
    float LdotH = dot(lightVector, halfwayVector);
    float VdotN = dot(viewVector, normalVector);
    float LdotN = dot(lightVector, normalVector);
    float LdotV = dot(lightVector, normalVector);
    float VdotH = dot(viewVector, halfwayVector);

    vec3 baseColor = isoSurfaceColorDef;
    // Specular F: Schlick Fresnel Approximation
    vec3 f0 = vec3(0.16 * (sqr(parameters.specular)));
    f0 = mix(f0, baseColor, parameters.metallic);

    vec3 F = fresnelSchlick(VdotH, f0);
    // Specular D: GGX Distribuition
    
    float D = D_GGX(NdotH, clamp(parameters.roughness,0.05, 1.0));
    // Speuclar G: G_Smith with G_1Smith-GGX
    float G = G_Smith(VdotN, LdotN, clamp(parameters.roughness,0.05, 1.0));
    // Result
    vec3 spec = (F * G * VdotH)/(NdotH*VdotN);
    
    // Diffuse Part
    vec3 rhoD = baseColor;
    
    rhoD *= vec3(1.0) - F;
    rhoD *= (1.0 - parameters.metallic);
    vec3 diff = rhoD;
    // Whitout importance sampling: rhoD *= (1.0 / M_PI)
    float sinThetaH = sqrt(1-(min(NdotH*NdotH,0.95)));
    float cosThetaH = NdotH;

    if (hitFlags.specularHit) {
        samplingPDF = D;
    } else {
        samplingPDF = (1.0/M_PI);
    }

    vec3 colorOut = diff + spec;
    return colorOut;
}

// Evaluate BRDF and prepare for Importance Sampling compensation
vec3 evaluateBrdfPdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColorDef) {
    vec3 halfwayVector = normalize(lightVector + viewVector);

    float NdotH = dot(halfwayVector, normalVector);
    float LdotH = dot(lightVector, halfwayVector);
    float VdotN = dot(viewVector, normalVector);
    float LdotN = dot(lightVector, normalVector);
    float LdotV = dot(lightVector, normalVector);
    float VdotH = dot(viewVector, halfwayVector);

    vec3 baseColor = isoSurfaceColorDef;
    
    // Specular Part
    // Specular F: Schlick Fresnel Approximation
    vec3 f0 = vec3(0.16 * (sqr(parameters.specular)));
    f0 = mix(f0, baseColor, parameters.metallic);

    vec3 F = fresnelSchlick(VdotH, f0);
    // Specular D: GGX Distribuition
    
    float D = D_GGX(NdotH, clamp(parameters.roughness,0.05, 1.0));
    // Speuclar G: G_Smith with G_1Smith-GGX
    float G = G_Smith(VdotN, LdotN, clamp(parameters.roughness,0.05, 1.0));

    float sinThetaH = sqrt(1-(min(NdotH*NdotH,0.95)));
    vec3 spec = (F * G * D * VdotH * sinThetaH)/(VdotN);
    
    // Diffuse Part
    // Importance Sampling pdf: 1/PI sin(theta) cos(theta)
    vec3 rhoD = baseColor;
    
    rhoD *= vec3(1.0) - F;
    rhoD *= (1.0 - parameters.metallic);
    rhoD *= (1.0 / M_PI);

    vec3 diff = rhoD;

    vec3 colorOut = diff + spec;
    return colorOut;
}

vec3 evaluateBrdfNee(vec3 viewVector, vec3 dirOut, vec3 dirNee, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, vec3 isoSurfaceColor, bool useMIS, float samplingPDF, flags hitFlags, out float pdfSamplingOut, out float pdfSamplingNee) {
        vec3 halfwayVector = normalize(dirOut + viewVector);
        float cosThetaH = dot(halfwayVector, normalVector);
        float sinThetaH = sqrt(1.0/(cosThetaH*cosThetaH));

        vec3 halfwayVectorNee  = normalize(dirNee + viewVector);
        float cosThetaHNee = dot(halfwayVectorNee, normalVector);
        float sinThetaHNee = sqrt(1.0 - min(cosThetaHNee*cosThetaHNee,0.95));

        pdfSamplingOut = samplingPDF * cosThetaH * sinThetaH;
        pdfSamplingNee = samplingPDF * cosThetaHNee * sinThetaHNee;

        return evaluateBrdfPdf(viewVector, dirNee, normalVector, isoSurfaceColor) * dot(dirNee, normalVector);
}

// Combined Call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {
    // Source (Explanation): https://www.youtube.com/watch?v=gya7x9H3mV0
    // Paper: https://dl.acm.org/doi/pdf/10.1145/357290.357293
    float roughness = clamp(parameters.roughness,0.05, 1.0);

    // Sampling and evaluating
    lightVector = sampleBrdf(parameters.metallic, parameters.specular, roughness, viewVector, frame, hitFlags);
    return evaluateBrdf(viewVector, lightVector, normalVector, isoSurfaceColor, hitFlags, samplingPDF);
}
