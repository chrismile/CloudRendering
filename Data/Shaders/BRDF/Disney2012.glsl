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

/**
* Copyright Disney Enterprises, Inc.  All rights reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License
* and the following modification to it: Section 6 Trademarks.
* deleted and replaced with:
*
* 6. Trademarks. This License does not grant permission to use the
* trade names, trademarks, service marks, or product names of the
* Licensor and its affiliates, except as required for reproducing
* the content of the NOTICE file.
*
* You may obtain a copy of the License at
* http://www.apache.org/licenses/LICENSE-2.0
*/

#define BRDF_SUPPORTS_SPECULAR

// ---------- BRDF Helper Functions ----------

// 1. Helper functions for sampling

vec3 sample_clearcoat_disney(vec3 viewVector, mat3 frameMatrix, float alpha, out float theta_h) {
    // Adapated from derviations here: https://www.youtube.com/watch?v=xFsJMUS94Fs
    // Generate random u and v between 0.0 and 1.0
    float u = random();
    float v = random();
    float alpha2 = alpha * alpha;

    // Compute spherical angles
    float theta = acos(sqrt((1 - pow(alpha2, (1- u)))/(1-alpha2)));
    float phi = 2 * M_PI * v;
    theta_h = theta;
    vec3 halfwayVector = frameMatrix*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

    // Compute light Vector l
    vec3 lightVector = 2*dot(viewVector,halfwayVector)*halfwayVector - viewVector;
    return lightVector;
}

vec3 sample_diffuse_disney(vec3 viewVector, mat3 frameMatrix, out float theta_h) {
    // Source: https://www.youtube.com/watch?v=xFsJMUS94Fs
    // Generate random u and v between 0.0 and 1.0
    float u = random();
    float v = random();

    // Compute spherical angles
    float theta = asin(sqrt(u));
    float phi = 2 * M_PI * v;
    theta_h = theta;

    vec3 halfwayVector = frameMatrix*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

    // Compute light Vector l
    vec3 lightVector = 2*dot(viewVector,halfwayVector)*halfwayVector - viewVector;
    return lightVector;
}

vec3 sample_specular_disney(vec3 viewVector, mat3 frameMatrix, float ax, float ay, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, out float theta_h) {
    // Adapated from derviations here: https://www.youtube.com/watch?v=xFsJMUS94Fs
    float u = clamp(random(),0.05, 0.95);
    float v = clamp(random(),0.05, 0.95);
    float phi = atan((ay/ax) * tan(2*M_PI*u));
    float theta = acos(sqrt((1-v)/(1+((cos(phi)*cos(phi)/(ax*ax))+(sin(phi)*sin(phi)/(ay*ay)))*v)));
    theta_h = theta;

    // Pronblem:
    // sqrt macht Problem für negativ
    // NMormalize macht problem für nahe 0
    vec3 halfwayVector = normalize(sqrt((v)/(1-v))*(ax*cos(2*M_PI*u)*tangentVector + ay*sin(2*M_PI*u)*bitangentVector) + normalVector);
    vec3 lightVector = 2*dot(viewVector,halfwayVector)*halfwayVector - viewVector;
    return lightVector;
}

// 2. Helper functions for evaluation

float GTR1(float NdotH, float a) {
    // Source: https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdfs/disney.brdf#L49C1-L55C2

    if (a >= 1) return 1 / PI;
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

// Source: https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdfs/disney.brdf#L69C1-L74C2
float smithG_GGX(float NdotV, float alphaG) {
    // Source: https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdfs/disney.brdf#L69C1-L74C2

    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

// ---------- BRDF Interface ----------

// Compute Importance Sampled light vector
vec3 sampleBrdf(float metallic, float specular, float clearcoat, float clearcoatGloss, float roughness, float subsurface, vec3 viewVector, mat3 frameMatrix, float ax, float ay, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, out float theta_h, out flags hitFlags) {
    // Idea adapted from https://schuttejoe.github.io/post/disneybsdf/
    
    float metal = metallic;
    float spec = (1.0 + specular) * (1.0 - roughness);
    float dielectric = (1.0 - metallic) * (1.0 + roughness);

    float specularW = metal + spec;
    float diffuseW = dielectric + subsurface;
    float clearcoatW = clamp(clearcoat, 0.0, 1.0);

    float norm = 1.0/(specularW + diffuseW + clearcoatW);

    float specularP = specularW * norm;
    float diffuseP = diffuseW * norm;
    float clearcoatP = clearcoatW * norm;

    float u = random();
    if(u < specularP) {
        hitFlags.specularHit = true;
        hitFlags.clearcoatHit = false;
        return sample_specular_disney(viewVector, frameMatrix, ax, ay, normalVector, tangentVector, bitangentVector, theta_h);
    } else if (u < specularP + diffuseP) {
        hitFlags.specularHit = false;
        hitFlags.clearcoatHit = false;
        return sample_diffuse_disney(viewVector, frameMatrix, theta_h);
    } else {
        hitFlags.specularHit = true;
        hitFlags.clearcoatHit = true;

        float alpha = mix(.1, .001, clearcoatGloss);
        return sample_clearcoat_disney(viewVector, frameMatrix, alpha, theta_h);
    }
}

// Evaluate BRDF with compensation of Importance Sampling and a fixed sampling PDF
vec3 evaluateBrdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColor, float th, float ax, float ay, vec3 x, vec3 y, flags hitFlags, out float samplingPDF) {
    // Paper: https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
    // Alternative Example Implementation (without Importance Sampling): https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdf/BRDFBase.cpp#L409
    vec3 halfwayVector = normalize(lightVector + viewVector);


    // Base Angles
    float theta_h = dot(halfwayVector, normalVector);
    float theta_d = dot(lightVector, halfwayVector);
    float theta_v = dot(viewVector, normalVector);
    float theta_l = dot(lightVector, normalVector);

    float NdotL = dot(lightVector, normalVector);
    float VdotH = dot(viewVector, halfwayVector);
    float NdotH = dot(halfwayVector, normalVector);

    // Base colors and values
    vec3 baseColor = isoSurfaceColor;
    // vec3(pow(baseColor[0], 2.2), pow(baseColor[1], 2.2), pow(baseColor[2], 2.2));
    vec3 col = baseColor;
    float lum = 0.3 * col[0] + 0.6 * col[1] + 0.1 * col[2];

    vec3 col_tint = lum > 0 ? col/lum: vec3(1.0);
    vec3 col_spec0 = mix(parameters.specular*0.08*mix(vec3(1.0),col_tint,parameters.specularTint), col, parameters.metallic);
    vec3 col_sheen = mix(vec3(1.0),col_tint,parameters.sheenTint);
    // Diffuse
    
    // Base Diffuse
    float f_d90 = 0.5 + 2 * parameters.roughness * pow(cos(theta_d), 2);
    float f_d = (1 + (f_d90 - 1) * (pow(1 - cos(theta_l), 5.0))) * (1 + (f_d90 - 1) * (pow(1 - cos(theta_v), 5.0)));
    
    // Subsurface Approximation: Inspired by Hanrahan-Krueger subsurface BRDF
    float f_d_subsurface_90 = parameters.roughness * pow(cos(theta_d), 2);
    float f_subsurface = (1.0 + (f_d_subsurface_90 - 1) * (pow(1 - cos(theta_l), 5.0))) * (1.0 + (f_d_subsurface_90 - 1.0) * (pow(1.0 - cos(theta_v), 5.0)));
    float f_d_subsurface = 1.25 * (f_subsurface * (1/(theta_l + theta_v) - 0.5) + 0.5);

    // Sheen
    float f_h = pow((1 - theta_d), 5.0);
    vec3 sheen = f_h * parameters.sheen * col_sheen;

    vec3 diffuse = (col * mix(f_d, f_d_subsurface, parameters.subsurface) + sheen) * (1.0 - parameters.metallic);
    
    // Specular F (Schlick Fresnel Approximation)
    vec3 f_specular = mix(col_spec0, vec3(1.0), f_h);
    
    // GTR2
    float d_specular = 1 / (M_PI * ax * ay * sqr(sqr(dot(x, halfwayVector) / ax) + sqr(dot(y, halfwayVector) / ay)) + pow(theta_h, 2.0));

    // Specular G Smith G GGX
    float g_specular = 1 / (theta_v + sqrt(sqr(dot(x,lightVector)*ax) + sqr(dot(y,lightVector)*ay) + sqr(theta_v)));
    g_specular *= (1 / (theta_v + sqrt(sqr(dot(x, viewVector) * ax) + sqr(dot(y, viewVector) * ay) + sqr(theta_v))));

    float sinThetaH = sin(th);
    float cosThetaH = NdotH;

    vec3 specular = f_specular * g_specular * 4 * NdotL * VdotH * sin(th)/ NdotH;

    // Clearcoat
    float f_clearcoat = mix(0.04,1.0,f_h);
    float d_clearcoat = GTR1(theta_h, mix(.1, .001, parameters.clearcoatGloss));
    float g_clearcoat = smithG_GGX(theta_l, 0.25) * smithG_GGX(theta_v, 0.25);
        
    float clear = parameters.clearcoat*f_clearcoat*g_clearcoat*NdotL*VdotH * sin(th)/NdotH;
    // Result

    if (hitFlags.specularHit) {
        if(hitFlags.clearcoatHit) {
            samplingPDF = d_clearcoat;
        } else {
            samplingPDF = d_specular;
        }
    } else {
        samplingPDF = (1.0/M_PI);
    }

    return diffuse + specular + clear;
}

// Evaluate BRDF and prepare for Importance Sampling compensation
vec3 evaluateBrdfPdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColor, float ax, float ay, vec3 x, vec3 y) {
    // Paper: https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
    // Alternative Example Implementation (without Importance Sampling): https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdf/BRDFBase.cpp#L409
    vec3 halfwayVector = normalize(lightVector + viewVector);

    // Base Angles
    float theta_h = dot(halfwayVector, normalVector);
    float theta_d = dot(lightVector, halfwayVector);
    float theta_v = dot(viewVector, normalVector);
    float theta_l = dot(lightVector, normalVector);

    float NdotL = dot(lightVector, normalVector);
    float VdotH = dot(viewVector, halfwayVector);
    float NdotH = dot(halfwayVector, normalVector);

    // Base colors and values
    vec3 baseColor = isoSurfaceColor;

    vec3 col = baseColor;
    float lum = 0.3 * col[0] + 0.6 * col[1] + 0.1 * col[2];

    vec3 col_tint = lum > 0 ? col/lum: vec3(1.0);
    vec3 col_spec0 = mix(parameters.specular*0.08*mix(vec3(1.0),col_tint,parameters.specularTint), col, parameters.metallic);
    vec3 col_sheen = mix(vec3(1.0),col_tint,parameters.sheenTint);
    // Diffuse
    
    // Base Diffuse
    float f_d90 = 0.5 + 2 * parameters.roughness * pow(cos(theta_d), 2);
    float f_d = (1 + (f_d90 - 1) * (pow(1 - cos(theta_l), 5.0))) * (1 + (f_d90 - 1) * (pow(1 - cos(theta_v), 5.0)));
    
    // Subsurface Approximation: Inspired by Hanrahan-Krueger subsurface BRDF
    float f_d_subsurface_90 = parameters.roughness * pow(cos(theta_d), 2);
    float f_subsurface = (1.0 + (f_d_subsurface_90 - 1) * (pow(1 - cos(theta_l), 5.0))) * (1.0 + (f_d_subsurface_90 - 1.0) * (pow(1.0 - cos(theta_v), 5.0)));
    float f_d_subsurface = 1.25 * (f_subsurface * (1/(theta_l + theta_v) - 0.5) + 0.5);

    // Sheen
    float f_h = pow((1 - theta_d), 5.0);
    vec3 sheen = f_h * parameters.sheen * col_sheen;

    vec3 diffuse = (col * mix(f_d, f_d_subsurface, parameters.subsurface) + sheen) * (1.0 - parameters.metallic);
    diffuse *= (1.0/M_PI);
    
    // Specular F (Schlick Fresnel Approximation)
    vec3 f_specular = mix(col_spec0, vec3(1.0), f_h);
    
    // GTR2
    float d_specular = 1 / (M_PI * ax * ay * sqr(sqr(dot(x, halfwayVector) / ax) + sqr(dot(y, halfwayVector) / ay)) + pow(theta_h, 2.0));

    // Specular G Smith G GGX
    float g_specular = 1 / (theta_v + sqrt(sqr(dot(x,lightVector)*ax) + sqr(dot(y,lightVector)*ay) + sqr(theta_v)));
    g_specular *= (1 / (theta_v + sqrt(sqr(dot(x, viewVector) * ax) + sqr(dot(y, viewVector) * ay) + sqr(theta_v))));

    float sinThetaH = sqrt(1-(NdotH*NdotH));

    vec3 specular = f_specular * d_specular * g_specular * 4 * NdotL * VdotH * sinThetaH;

    // Clearcoat
    float f_clearcoat = mix(0.04,1.0,f_h);
    float d_clearcoat = GTR1(theta_h, mix(.1, .001, parameters.clearcoatGloss));
    float g_clearcoat = smithG_GGX(theta_l, 0.25) * smithG_GGX(theta_v, 0.25);
        
    float clear = parameters.clearcoat*f_clearcoat*g_clearcoat * NdotL * VdotH * sinThetaH;
    
    // Result        
    return diffuse + specular + clear;
}

vec3 evaluateBrdfNee(vec3 viewVector, vec3 dirOut, vec3 dirNee, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, vec3 isoSurfaceColor, bool useMIS, float samplingPDF, flags hitFlags, out float pdfSamplingOut, out float pdfSamplingNee) {
    float aspect = sqrt(1 - parameters.anisotropic * 0.9);
    float ax = max(0.001, sqr(parameters.roughness) / aspect);
    float ay = max(0.001, sqr(parameters.roughness) * aspect);

    vec3 halfwayVector = normalize(dirOut + viewVector);
    float cosThetaH = dot(halfwayVector, normalVector);
    float sinThetaH = sqrt(1.0/(cosThetaH*cosThetaH));

    vec3 halfwayVectorNee  = normalize(dirNee + viewVector);
    float cosThetaHNee = dot(halfwayVectorNee, normalVector);
    float sinThetaHNee = sqrt(1.0 - (cosThetaHNee*cosThetaHNee));

    if (hitFlags.specularHit) {
            pdfSamplingOut = samplingPDF * cosThetaH;
            pdfSamplingNee = samplingPDF * cosThetaHNee;
    } else {
        pdfSamplingOut = samplingPDF * cosThetaH * sinThetaH;
        pdfSamplingNee = samplingPDF * cosThetaHNee * sinThetaHNee;
    }       
    
    return evaluateBrdfPdf(viewVector, dirNee, normalVector, isoSurfaceColor, ax, ay, tangentVector, bitangentVector) * dot(dirNee, normalVector);

}

// Combined Call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {
    // 1. Paper: https://media.disneyanimation.com/uploads/production/publication_asset/48/asset/s2012_pbs_disney_brdf_notes_v3.pdf
    // 2. BRDF Example Implementation (without Importance Sampling): https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
    float aspect = sqrt(1 - parameters.anisotropic * 0.9);
    float ax = max(0.001, sqr(parameters.roughness) / aspect);
    float ay = max(0.001, sqr(parameters.roughness) * aspect);

    float th;
    lightVector = sampleBrdf(parameters.metallic, parameters.specular, parameters.clearcoat, parameters.clearcoatGloss, parameters.roughness, parameters.subsurface, viewVector, frame, ax, ay, normalVector, tangentVector, bitangentVector, th, hitFlags);
    return evaluateBrdf(viewVector, lightVector, normalVector, isoSurfaceColor, th, ax, ay, tangentVector, bitangentVector, hitFlags, samplingPDF);
}