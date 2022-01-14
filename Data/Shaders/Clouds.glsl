/**
 * MIT License
 *
 * Copyright (c) 2021, Christoph Neuhauser, Ludwig Leonard
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

-- Compute

#version 450

layout (local_size_x = 32, local_size_y = 32, local_size_z = 1) in;

layout (binding = 0, rgba32f) uniform image2D resultImage;

layout (binding = 1) uniform sampler3D gridImage;

layout (binding = 2) uniform Parameters {
    // Transform from normalized device coordinates to world space.
    mat4 inverseViewProjMatrix;

    // Cloud properties.
    vec3 boxMin;
    vec3 boxMax;

    vec3 extinction;
    vec3 scatteringAlbedo;
    float phaseG;

    // Sky properties.
    vec3 sunDirection;
    vec3 sunIntensity;

    // For residual ratio tracking and decomposition tracking.
    ivec3 superVoxelSize;
    ivec3 superVoxelGridSize;
} parameters;

layout (binding = 3) uniform FrameInfo
{
    uint frameCount;
    uvec3 other;
} frameInfo;

layout (binding = 4, rgba32f) uniform image2D accImage;

layout (binding = 5, rgba32f) uniform image2D firstX;

layout (binding = 6, rgba32f) uniform image2D firstW;

#ifdef COMPUTE_PRIMARY_RAY_ABSORPTION_MOMENTS
layout (binding = 7, r32f) uniform image2DArray primaryRayAbsorptionMomentsImage;
#endif

#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
layout (binding = 8, r32f) uniform image2DArray scatterRayAbsorptionMomentsImage;
#endif

#if defined(USE_RESIDUAL_RATIO_TRACKING) || defined(USE_DECOMPOSITION_TRACKING)
layout (binding = 9) uniform sampler3D superVoxelGridImage;
layout (binding = 10) uniform usampler3D superVoxelGridOccupancyImage;
#endif

/**
 * This code is part of an GLSL port of the HLSL code accompanying the paper "Moment-Based Order-Independent
 * Transparency" by Münstermann, Krumpen, Klein, and Peters (http://momentsingraphics.de/?page_id=210).
 * The original code was released in accordance to CC0 (https://creativecommons.org/publicdomain/zero/1.0/).
 *
 * This port is released under the terms of the MIT License.
 */
/*! This function implements complex multiplication.*/
layout(std140, binding = 11) uniform MomentUniformData {
    vec4 wrapping_zone_parameters;
    //float overestimation;
    //float moment_bias;
};
const float ABSORBANCE_MAX_VALUE = 10.0;

vec2 Multiply(vec2 LHS, vec2 RHS){
    return vec2(LHS.x*RHS.x-LHS.y*RHS.y,LHS.x*RHS.y+LHS.y*RHS.x);
}


// --- Constants
const float PI = 3.14159265359;
const float TWO_PI = 2*3.14159265359;

//--- Random Number Generator (Hybrid Taus)

uvec4 rngState = uvec4(0);
uint tausStep(uint z, int S1, int S2, int S3, uint M) { uint b = (((z << S1) ^ z) >> S2); return ((z & M) << S3) ^ b; }
uint lcgStep(uint z, uint A, uint C) { return A * z + C; }

float random() {
    rngState.x = tausStep(rngState.x, 13, 19, 12, 4294967294);
    rngState.y = tausStep(rngState.y, 2, 25, 4, 4294967288);
    rngState.z = tausStep(rngState.z, 3, 11, 17, 4294967280);
    rngState.w = lcgStep(rngState.w, 1664525, 1013904223);
    return 2.3283064365387e-10 * (rngState.x ^ rngState.y ^ rngState.z ^ rngState.w);
}

void initializeRandom(uint seed) {
    rngState = uvec4(seed);
    for (int i = 0; i < seed % 7 + 2; i++) {
        random();
    }
}

void createOrthonormalBasis(vec3 D, out vec3 B, out vec3 T) {
    vec3 other = abs(D.z) >= 0.9999 ? vec3(1, 0, 0) : vec3(0, 0, 1);
    B = normalize(cross(other, D));
    T = normalize(cross(D, B));
}

vec3 randomDirection(vec3 D) {
    float r1 = random();
    float r2 = random() * 2 - 1;
    float sqrR2 = r2 * r2;
    float two_pi_by_r1 = TWO_PI * r1;
    float sqrt_of_one_minus_sqrR2 = sqrt(1.0 - sqrR2);
    float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
    float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
    float z = r2;

    vec3 t0, t1;
    createOrthonormalBasis(D, t0, t1);

    return t0 * x + t1 * y + D * z;
}

//--- Scattering functions

#define oneMinusG2 (1.0 - (GFactor) * (GFactor))
#define onePlusG2 (1.0 + (GFactor) * (GFactor))
#define oneOver2G (0.5 / (GFactor))

float invertcdf(float GFactor, float xi) {
    float t = (oneMinusG2) / (1.0f - GFactor + 2.0f * GFactor * xi);
    return oneOver2G * (onePlusG2 - t * t);
}

vec3 importanceSamplePhase(float GFactor, vec3 D, out float pdf) {
    if (abs(GFactor) < 0.001) {
        pdf = 1.0 / (4 * PI);
        return randomDirection(-D);
    }

    float phi = random() * 2 * PI;
    float cosTheta = invertcdf(GFactor, random());
    float sinTheta = sqrt(max(0, 1.0f - cosTheta * cosTheta));

    vec3 t0, t1;
    createOrthonormalBasis(D, t0, t1);

    pdf = 0.25 / PI * (oneMinusG2) / pow(onePlusG2 - 2 * GFactor * cosTheta, 1.5);

    return sinTheta * sin(phi) * t0 + sinTheta * cos(phi) * t1 + cosTheta * D;
}



//--- Tools

vec3 sampleSkybox(in vec3 dir) {
    vec3 L = dir;

    vec3 BG_COLORS[5] = {
        vec3(0.1, 0.05, 0.01), // GROUND DARKER BLUE
        vec3(0.01, 0.05, 0.2), // HORIZON GROUND DARK BLUE
        vec3(0.8, 0.9, 1.0), // HORIZON SKY WHITE
        vec3(0.1, 0.3, 1.0),  // SKY LIGHT BLUE
        vec3(0.01, 0.1, 0.7)  // SKY BLUE
    };

    float BG_DISTS[5] = {
        -1.0,
        -0.1,
        0.0,
        0.4,
        1.0
    };

    vec3 col = BG_COLORS[0];
    col = mix(col, BG_COLORS[1], vec3(smoothstep(BG_DISTS[0], BG_DISTS[1], L.y)));
    col = mix(col, BG_COLORS[2], vec3(smoothstep(BG_DISTS[1], BG_DISTS[2], L.y)));
    col = mix(col, BG_COLORS[3], vec3(smoothstep(BG_DISTS[2], BG_DISTS[3], L.y)));
    col = mix(col, BG_COLORS[4], vec3(smoothstep(BG_DISTS[3], BG_DISTS[4], L.y)));

    return col;
}

vec3 sampleLight(in vec3 dir) {
    int N = 10;
    float phongNorm = (N + 2) / (2 * 3.14159);
    return parameters.sunIntensity * pow(max(0, dot(dir, parameters.sunDirection)), N) * phongNorm;
}

float sampleCloud(in vec3 pos) {
    ivec3 dim = textureSize(gridImage, 0);
    vec3 coord = (pos - parameters.boxMin)/(parameters.boxMax - parameters.boxMin);
    coord += vec3(random() - 0.5, random() - 0.5, random() - 0.5)/ dim;
    return texture(gridImage, coord).x;
}

void createCameraRay(in vec2 coord, out vec3 x, out vec3 w) {
    vec4 ndcP = vec4(coord, 0, 1);
    vec4 ndcT = ndcP + vec4(0, 0, 1, 0);

    vec4 viewP = parameters.inverseViewProjMatrix * ndcP;
    viewP.xyz /= viewP.w;
    vec4 viewT = parameters.inverseViewProjMatrix * ndcT;
    viewT.xyz /= viewT.w;

    x = viewP.xyz;
    w = normalize(viewT.xyz - viewP.xyz);
}

bool rayBoxIntersect(vec3 bMin, vec3 bMax, vec3 P, vec3 D, out float tMin, out float tMax) {
    // Un-parallelize D.
    D.x = abs(D).x <= 0.000001 ? 0.000001 : D.x;
    D.y = abs(D).y <= 0.000001 ? 0.000001 : D.y;
    D.z = abs(D).z <= 0.000001 ? 0.000001 : D.z;
    vec3 C_Min = (bMin - P)/D;
    vec3 C_Max = (bMax - P)/D;
    tMin = max(max(min(C_Min[0], C_Max[0]), min(C_Min[1], C_Max[1])), min(C_Min[2], C_Max[2]));
    tMin = max(0.0, tMin);
    tMax = min(min(max(C_Min[0], C_Max[0]), max(C_Min[1], C_Max[1])), max(C_Min[2], C_Max[2]));
    if (tMax <= tMin || tMax <= 0) {
        return false;
    }
    return true;
}

//--- Volume Pathtracer

float maxComponent(vec3 v) {
    return max(v.x, max(v.y, v.z));
}

// Pathtracing with Delta tracking and Spectral tracking.
vec3 pathtraceSpectral(vec3 x, vec3 w) {
    float majorant = maxComponent(parameters.extinction);

    vec3 weights = vec3(1,1,1);

    vec3 absorptionAlbedo = vec3(1,1,1) - parameters.scatteringAlbedo;
    vec3 scatteringAlbedo = parameters.scatteringAlbedo;
    float PA = maxComponent (absorptionAlbedo * parameters.extinction);
    float PS = maxComponent (scatteringAlbedo * parameters.extinction);

    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax))
    {
        x += w * tMin;
        float d = tMax - tMin;
        while (true) {
            float t = -log(max(0.0000000001, 1 - random()))/majorant;

            if (t > d) {
                break;
            }

            x += w * t;

            float density = sampleCloud(x);

            vec3 sigma_a = absorptionAlbedo * parameters.extinction * density;
            vec3 sigma_s = scatteringAlbedo * parameters.extinction * density;
            vec3 sigma_n = vec3(majorant) - parameters.extinction * density;

            float Pa = maxComponent(sigma_a);
            float Ps = maxComponent(sigma_s);
            float Pn = maxComponent(sigma_n);
            float C = Pa + Ps + Pn;
            Pa /= C;
            Ps /= C;
            Pn /= C;

            float xi = random();

            if (xi < Pa)
            return vec3(0); // weights * sigma_a / (majorant * Pa) * L_e; // 0 - No emission

            if (xi < 1 - Pn) // scattering event
            {
                float pdf_w;
                w = importanceSamplePhase(parameters.phaseG, w, pdf_w);
                if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
                    x += w*tMin;
                    d = tMax - tMin;
                }
                weights *= sigma_s / (majorant * Ps);
            } else {
                d -= t;
                weights *= sigma_n / (majorant * Pn);
            }
        }
    }

    return min(weights, vec3(100000, 100000, 100000)) * (sampleSkybox(w) + sampleLight(w));
}

struct ScatterEvent
{
    bool hasValue;
    vec3 x; float pdf_x;
    vec3 w; float pdf_w;
};

vec3 pathtrace(
        vec3 x, vec3 w, out ScatterEvent firstEvent
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
        , out float scatterRayAbsorptionMoments[NUM_SCATTER_RAY_ABSORPTION_MOMENTS + 1]
#endif
) {
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
    for (int i = 0; i <= NUM_SCATTER_RAY_ABSORPTION_MOMENTS; i++) {
        scatterRayAbsorptionMoments[i] = 0.0;
    }
    float depth = 0.0;
#endif

    firstEvent = ScatterEvent(false, x, 0.0, w, 0.0);

    float majorant = parameters.extinction.x;
    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;
    float PA = absorptionAlbedo * parameters.extinction.x;
    float PS = scatteringAlbedo * parameters.extinction.x;

    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
        //depth += tMin;
#endif
        float d = tMax - tMin;

        float pdf_x = 1;
        float transmittance = 1.0;

        while (true) {
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
            float absorbance = -log(transmittance);
            if (absorbance > ABSORBANCE_MAX_VALUE) {
                absorbance = ABSORBANCE_MAX_VALUE;
            }
#ifdef USE_POWER_MOMENTS_SCATTER_RAY
            for (int i = 0; i <= NUM_SCATTER_RAY_ABSORPTION_MOMENTS; i++) {
                scatterRayAbsorptionMoments[i] += absorbance * pow(depth, i);
            }
#else
            float phase = fma(depth, wrapping_zone_parameters.y, wrapping_zone_parameters.y);
            vec2 circlePoint = vec2(cos(phase), sin(phase));
            scatterRayAbsorptionMoments[0] = absorbance;
            scatterRayAbsorptionMoments[1] = absorbance * circlePoint.x;
            scatterRayAbsorptionMoments[2] = absorbance * circlePoint.y;
            vec2 circlePointNext = circlePoint;
            for (int i = 2; i <= NUM_SCATTER_RAY_ABSORPTION_MOMENTS / 2; i++) {
                circlePointNext = Multiply(circlePointNext, circlePoint);
                scatterRayAbsorptionMoments[i * 2] = absorbance * circlePointNext.x;
                scatterRayAbsorptionMoments[i * 2 + 1] = absorbance * circlePointNext.y;
            }
#endif
            transmittance = 1.0;
#endif
            float t = -log(max(0.0000000001, 1 - random()))/majorant;

            if (t > d)
            break;

            x += w * t;
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
            depth += t;
#endif

            float density = sampleCloud(x);
            transmittance *= 1.0 - density;

            float sigma_a = PA * density;
            float sigma_s = PS * density;
            float sigma_n = majorant - parameters.extinction.x * density;

            float Pa = sigma_a / majorant;
            float Ps = sigma_s / majorant;
            float Pn = sigma_n / majorant;

            float xi = random();

            if (xi < Pa) {
                return vec3(0); // weights * sigma_a / (majorant * Pa) * L_e; // 0 - No emission
            }

            if (xi < 1 - Pn) // scattering event
            {
                float pdf_w;
                w = importanceSamplePhase(parameters.phaseG, w, pdf_w);

                pdf_x *= exp(-majorant * t) * majorant * density;

                if (!firstEvent.hasValue) {
                    firstEvent.x = x;
                    firstEvent.pdf_x = sigma_s * pdf_x;
                    firstEvent.w = w;
                    firstEvent.pdf_w = pdf_w;
                    firstEvent.hasValue = true;
                }

                if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
                    x += w*tMin;
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
                    depth += tMin;
#endif
                    d = tMax - tMin;
                }
            } else {
                pdf_x *= exp(-majorant * t) * majorant * (1 - density);
                d -= t;
            }
        }
    }

    return sampleSkybox(w) + sampleLight(w);
}


//---------------------------------------------------------
// Ratio tracking
//---------------------------------------------------------

#ifdef USE_RATIO_TRACKING
vec3 ratioTracking(vec3 x, vec3 w, out ScatterEvent firstEvent) {
    firstEvent = ScatterEvent(false, x, 0.0, w, 0.0);

    float majorant = parameters.extinction.x;
    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;
    float PA = absorptionAlbedo * parameters.extinction.x;
    float PS = scatteringAlbedo * parameters.extinction.x;

    float transmittance = 1.0;

    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
        float d = tMax - tMin;
        float pdf_x = 1;

        while (true) {
            float t = -log(max(0.0000000001, 1 - random()))/majorant;

            if (t > d)
                break;

            x += w * t;

            float density = sampleCloud(x);

            float sigma_a = PA * density;
            float sigma_s = PS * density;
            float sigma_n = majorant - parameters.extinction.x * density;

            float Pa = sigma_a / majorant;
            float Ps = sigma_s / majorant;
            float Pn = sigma_n / majorant;

            float xi = random();

            transmittance *= 1.0 - Pa;
            //if (xi < Pa) {
            //    return vec3(0); // weights * sigma_a / (majorant * Pa) * L_e; // 0 - No emission
            //}

            if (xi < 1 - Pn) // scattering event
            {
                float pdf_w;
                w = importanceSamplePhase(parameters.phaseG, w, pdf_w);

                pdf_x *= exp(-majorant * t) * majorant * density;

                if (!firstEvent.hasValue) {
                    firstEvent.x = x;
                    firstEvent.pdf_x = sigma_s * pdf_x;
                    firstEvent.w = w;
                    firstEvent.pdf_w = pdf_w;
                    firstEvent.hasValue = true;
                }

                if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
                    x += w*tMin;
                    d = tMax - tMin;
                }
            } else {
                pdf_x *= exp(-majorant * t) * majorant * (1 - density);
                d -= t;
            }
        }
    }

    return transmittance * (sampleSkybox(w) + sampleLight(w));
}
#endif


//---------------------------------------------------------
// Residual ratio tracking
//---------------------------------------------------------

/**
 * For more details on residual ratio tracking, please refer to:
 * J. Novák, A. Selle, and W. Jarosz. Residual ratio tracking for estimating attenuation in participating media.
 * ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia) , 33(6), Nov. 2014.
 */

#ifdef USE_RESIDUAL_RATIO_TRACKING
float residualRatioTrackingEstimator(
        vec3 x, vec3 w, float dStart, float dEnd, float T,
        inout float reservoirWeightSum, out float reservoirT, out float reservoirDist,
        float absorptionAlbedo, float mu_c, float mu_r_bar) {
    float T_c = exp(-mu_c * (dEnd - dStart));
    float T_r = 1.0;
    float dTravelled = dStart;

    //if (mu_r_bar < 1e-5) {
    //    return T_c;
    //}

    do {
        float t = -log(max(0.0000000001, 1 - random())) / mu_r_bar;
        x += w * t;

        dTravelled += t;
        if (dTravelled >= dEnd) {
            break;
        }

        float density = sampleCloud(x);
        float mu = parameters.extinction.x * density;
        //T_r *= (1.0 - absorptionAlbedo * (mu - mu_c) / mu_r_bar);
        //T_r *= (1.0 - density) * (mu - mu_c) / mu_r_bar;
        T_r *= 1.0 - (mu - mu_c) / mu_r_bar;

        float Ps = parameters.scatteringAlbedo.x * density;
        //float T_c_local = exp(-absorptionAlbedo * parameters.scatteringAlbedo.x * mu_c * (dTravelled - dStart));
        float T_c_local = exp(-mu_c * (dTravelled - dStart));
        float T_local = T * T_r * T_c_local;
        // https://developer.download.nvidia.com//ray-tracing-gems/rtg2-chapter22-preprint.pdf
        float reservoirWeight = T_local * Ps;
        reservoirWeightSum += reservoirWeight;
        float xi = random();
        if (xi < reservoirWeight / reservoirWeightSum) {
            reservoirT = T_local;
            reservoirDist = dTravelled;
        }
    } while(true);

    return T_c * T_r;
}

vec3 residualRatioTracking(vec3 x, vec3 w, out ScatterEvent firstEvent) {
    firstEvent = ScatterEvent(false, x, 0.0, w, 0.0);

    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;

    // Functional representation of the transmittance using reservoir sampling.
    // cf.: https://developer.nvidia.com/blog/learn-more-about-reservoir-sampling-in-free-ray-tracing-gems-ii-chapter/
    float reservoirWeightSum = 0.0;
    float reservoirT = 0.0;
    float reservoirDist = 0.0;

    float T = 1.0;

    vec3 accumulatedColor = vec3(0.0, 0.0, 0.0);

    const vec3 EPSILON_VEC = vec3(1e-6);
    float tMinVal, tMaxVal;
    vec3 oldX;

    ivec3 voxelGridSize = textureSize(gridImage, 0);
    vec3 boxDelta = parameters.boxMax - parameters.boxMin;

    float tMaxX, tMaxY, tMaxZ, tDeltaX, tDeltaY, tDeltaZ;
    ivec3 superVoxelIndex;

    // Loop over all in-scattering rays.
    int iteration = 0;
    while (true) {
        /// Does in-scattering ray intersect the box?
        if (rayBoxIntersect(parameters.boxMin + EPSILON_VEC, parameters.boxMax - EPSILON_VEC, x, w, tMinVal, tMaxVal)) {
            x += w * tMinVal;
            oldX = x;
            float dTotal = tMaxVal - tMinVal;

            vec3 startPoint = (x - parameters.boxMin) / boxDelta * voxelGridSize / parameters.superVoxelSize;
            vec3 endPoint = (x + w * dTotal - parameters.boxMin) / boxDelta * voxelGridSize / parameters.superVoxelSize;

            int stepX = int(sign(endPoint.x - startPoint.x));
            if (stepX != 0)
            tDeltaX = min(stepX / (endPoint.x - startPoint.x), 1e7);
            else
            tDeltaX = 1e7; // inf
            if (stepX > 0)
            tMaxX = tDeltaX * (1.0 - fract(startPoint.x));
            else
            tMaxX = tDeltaX * fract(startPoint.x);
            superVoxelIndex.x = int(floor(startPoint.x));

            int stepY = int(sign(endPoint.y - startPoint.y));
            if (stepY != 0)
            tDeltaY = min(stepY / (endPoint.y - startPoint.y), 1e7);
            else
            tDeltaY = 1e7; // inf
            if (stepY > 0)
            tMaxY = tDeltaY * (1.0 - fract(startPoint.y));
            else
            tMaxY = tDeltaY * fract(startPoint.y);
            superVoxelIndex.y = int(floor(startPoint.y));

            int stepZ = int(sign(endPoint.z - startPoint.z));
            if (stepZ != 0)
            tDeltaZ = min(stepZ / (endPoint.z - startPoint.z), 1e7);
            else
            tDeltaZ = 1e7; // inf
            if (stepZ > 0)
            tMaxZ = tDeltaZ * (1.0 - fract(startPoint.z));
            else
            tMaxZ = tDeltaZ * fract(startPoint.z);
            superVoxelIndex.z = int(floor(startPoint.z));

            if (stepX == 0 && stepY == 0 && stepZ == 0) {
                break;
            }
            ivec3 step = ivec3(stepX, stepY, stepZ);
            vec3 tMax = vec3(tMaxX, tMaxY, tMaxZ);
            vec3 tDelta = vec3(tDeltaX, tDeltaY, tDeltaZ);

            ivec3 startVoxelInt = clamp(ivec3(floor(startPoint)), ivec3(0), parameters.superVoxelGridSize - ivec3(1));
            ivec3 endVoxelInt = clamp(ivec3(ceil(endPoint)), ivec3(0), parameters.superVoxelGridSize - ivec3(1));

            // Loop over all super voxels along the ray.
            while (all(greaterThanEqual(superVoxelIndex, ivec3(0))) && all(lessThan(superVoxelIndex, parameters.superVoxelGridSize))) {
                vec2 superVoxel = texelFetch(superVoxelGridImage, superVoxelIndex, 0).rg;
                float mu_c = superVoxel.x;
                float mu_r_bar = superVoxel.y;
                //bool superVoxelEmpty = texelFetch(superVoxelGridOccupancyImage, ivec3(0), 0).r != 0;

                vec3 minVoxelPos = superVoxelIndex * parameters.superVoxelSize;
                vec3 maxVoxelPos = minVoxelPos + parameters.superVoxelSize;
                minVoxelPos = minVoxelPos / voxelGridSize * boxDelta + parameters.boxMin;
                maxVoxelPos = maxVoxelPos / voxelGridSize * boxDelta + parameters.boxMin;
                float tMinVoxel = 0.0, tMaxVoxel = 0.0;
                rayBoxIntersect(minVoxelPos, maxVoxelPos, oldX, w, tMinVoxel, tMaxVoxel);

                x = oldX + w * tMinVoxel;
                T *= residualRatioTrackingEstimator(
                        x, w, tMinVoxel, tMaxVoxel, T,
                        reservoirWeightSum, reservoirT, reservoirDist,
                        absorptionAlbedo, mu_c, mu_r_bar);

                if (tMaxX < tMaxY) {
                    if (tMaxX < tMaxZ) {
                        superVoxelIndex.x += stepX;
                        tMaxX += tDeltaX;
                    } else {
                        superVoxelIndex.z += stepZ;
                        tMaxZ += tDeltaZ;
                    }
                } else {
                    if (tMaxY < tMaxZ) {
                        superVoxelIndex.y += stepY;
                        tMaxY += tDeltaY;
                    } else {
                        superVoxelIndex.z += stepZ;
                        tMaxZ += tDeltaZ;
                    }
                }
            }
        } else {
            break;
        }

        float xi = random();
        if (xi > reservoirWeightSum || iteration >= 10) {
            break;
        }
        accumulatedColor += T * (sampleSkybox(w) + sampleLight(w));
        iteration++;

        //float pdf_w;
        //x = x + w * (tMinVal + xi * (tMaxVal - tMinVal)); // Uniform sampling.
        //w = importanceSamplePhase(parameters.phaseG, w, pdf_w);

        // https://developer.download.nvidia.com//ray-tracing-gems/rtg2-chapter22-preprint.pdf
        T = reservoirT;
        float pdf_w;
        x = oldX + w * reservoirDist;
        w = importanceSamplePhase(parameters.phaseG, w, pdf_w);

        reservoirWeightSum = 0.0;
        reservoirT = 0.0;
        reservoirDist = 0.0;
    }

    accumulatedColor += T * (sampleSkybox(w) + sampleLight(w));
    return accumulatedColor;
}
#endif


//---------------------------------------------------------
// Decomposition tracking
//---------------------------------------------------------

/**
 * For more details on decomposition tracking, please refer to:
 * P. Kutz, R. Habel, Y. K. Li, and J. Novák. Spectral and decomposition tracking for rendering heterogeneous volumes.
 * ACM Trans. Graph., 36(4), Jul. 2017.
 */

#ifdef USE_DECOMPOSITION_TRACKING
vec3 analogDecompositionTracking(vec3 x, vec3 w, out ScatterEvent firstEvent) {
    firstEvent = ScatterEvent(false, x, 0.0, w, 0.0);

    int it = 0;
    const vec3 EPSILON_VEC = vec3(1e-6);
    float tMinVal, tMaxVal;
    if (rayBoxIntersect(parameters.boxMin + EPSILON_VEC, parameters.boxMax - EPSILON_VEC, x, w, tMinVal, tMaxVal)) {
        float majorant = parameters.extinction.x;
        float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;

        ivec3 voxelGridSize = textureSize(gridImage, 0);
        vec3 boxDelta = parameters.boxMax - parameters.boxMin;
        vec3 superVoxelSize = parameters.superVoxelSize * boxDelta / voxelGridSize;

        x += w * tMinVal;
        vec3 startPoint = (x - parameters.boxMin) / boxDelta * voxelGridSize / parameters.superVoxelSize;
        ivec3 superVoxelIndex = ivec3(floor(startPoint));

        ivec3 cachedSuperVoxelIndex = ivec3(-1, -1, -1);
        vec2 superVoxelMinMaxDensity = vec2(0.0, 0.0);

        // Loop over all super voxels along the ray.
        while (all(greaterThanEqual(superVoxelIndex, ivec3(0))) && all(lessThan(superVoxelIndex, parameters.superVoxelGridSize))) {
            vec3 minSuperVoxelPos = parameters.boxMin + superVoxelIndex * superVoxelSize;
            vec3 maxSuperVoxelPos = minSuperVoxelPos + superVoxelSize;

            float tMinSuperVoxel = 0.0, tMaxSuperVoxel = 0.0;
            rayBoxIntersect(minSuperVoxelPos, maxSuperVoxelPos, x, w, tMinSuperVoxel, tMaxSuperVoxel);
            float d_max = tMaxSuperVoxel - tMinSuperVoxel; // + 1e-7
            x += w * tMinSuperVoxel;

            if (cachedSuperVoxelIndex != superVoxelIndex) {
                superVoxelMinMaxDensity = texelFetch(superVoxelGridImage, superVoxelIndex, 0).xy;
                cachedSuperVoxelIndex = superVoxelIndex;
            }
            if (superVoxelMinMaxDensity.y < 1e-5) {
                x += w * d_max;
            } else {
                float mu_c_t = max(0.0000000001, majorant * superVoxelMinMaxDensity.x);
                float majorant_r_local = max(0.0000000001, majorant * superVoxelMinMaxDensity.y - mu_c_t);
                //float mu_c_t = 0.0000000001;
                //float majorant_r_local = max(0.0000000001, majorant * 1 - mu_c_t);

                bool isNullCollision;
                float t_c = -log(max(0.0000000001, 1 - random())) / mu_c_t;
                float t_r = 0.0;
                isNullCollision = false;
                while (true) {
                    t_r -= log(max(0.0000000001, 1 - random())) / majorant_r_local;

                    if (t_c >= d_max && t_r >= d_max) {
                        x = x + d_max * w;
                        break; // null collision, proceed to next super voxel
                    }

                    vec3 xs = x + w * min(t_c, t_r);
                    bool isCollision = false;
                    if (t_c <= t_r) {
                        isCollision = true;
                    } else {
                        float density = sampleCloud(xs);
                        isCollision = random() * majorant_r_local < parameters.extinction.x * density - mu_c_t;
                    }

                    if (isCollision) {
                        x = xs;

                        if (random() < absorptionAlbedo) {
                            return vec3(0.0); // absorption event/emission
                        }

                        float pdf_w;
                        w = importanceSamplePhase(parameters.phaseG, w, pdf_w);
                        t_r = 0.0;
                        t_c = -log(max(0.0000000001, 1 - random())) / mu_c_t;
                        rayBoxIntersect(minSuperVoxelPos, maxSuperVoxelPos, x, w, tMinSuperVoxel, tMaxSuperVoxel);
                        d_max = tMaxSuperVoxel - tMinSuperVoxel; // + 1e-7
                    }
                }
            }

            vec3 cellCenter = (minSuperVoxelPos + maxSuperVoxelPos) * 0.5;
            vec3 mov = x + w * 0.00001 - cellCenter;
            vec3 smov = sign(mov);
            mov *= smov;

            ivec3 dims = ivec3(mov.x >= mov.y && mov.x >= mov.z, mov.y >= mov.x && mov.y >= mov.z, mov.z >= mov.x && mov.z >= mov.y);
            superVoxelIndex += dims * ivec3(smov);
        }
    }

    return sampleSkybox(w) + sampleLight(w);
}
#endif


//---------------------------------------------------------
// Absorption moment computation
//---------------------------------------------------------

#ifdef COMPUTE_PRIMARY_RAY_ABSORPTION_MOMENTS
void computePrimaryRayAbsorptionMoments(
        vec3 x, vec3 w, out float primaryRayAbsorptionMoments[NUM_PRIMARY_RAY_ABSORPTION_MOMENTS + 1]) {
    for (int i = 0; i <= NUM_PRIMARY_RAY_ABSORPTION_MOMENTS; i++) {
        primaryRayAbsorptionMoments[i] = 0.0;
    }
    float depth = 0.0;

    float majorant = parameters.extinction.x;
    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;
    float PA = absorptionAlbedo * parameters.extinction.x;
    float PS = scatteringAlbedo * parameters.extinction.x;

    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax))
    {
        x += w * tMin;
        //depth += tMin;
        float d = tMax - tMin;

        float pdf_x = 1.0;
        float transmittance = 1.0;

        while (true) {
            float absorbance = -log(transmittance);
            if (absorbance > ABSORBANCE_MAX_VALUE) {
                absorbance = ABSORBANCE_MAX_VALUE;
            }
#ifdef USE_POWER_MOMENTS_PRIMARY_RAY
            for (int i = 0; i <= NUM_PRIMARY_RAY_ABSORPTION_MOMENTS; i++) {
                primaryRayAbsorptionMoments[i] += absorbance * pow(depth, i);
            }
#else
            float phase = fma(depth, wrapping_zone_parameters.y, wrapping_zone_parameters.y);
            vec2 circlePoint = vec2(cos(phase), sin(phase));
            primaryRayAbsorptionMoments[0] = absorbance;
            primaryRayAbsorptionMoments[1] = absorbance * circlePoint.x;
            primaryRayAbsorptionMoments[2] = absorbance * circlePoint.y;
            vec2 circlePointNext = circlePoint;
            for (int i = 2; i <= NUM_PRIMARY_RAY_ABSORPTION_MOMENTS / 2; i++) {
                circlePointNext = Multiply(circlePointNext, circlePoint);
                primaryRayAbsorptionMoments[i * 2] = absorbance * circlePointNext.x;
                primaryRayAbsorptionMoments[i * 2 + 1] = absorbance * circlePointNext.y;
            }
#endif
            transmittance = 1.0;

            float t = -log(max(0.0000000001, 1 - random())) / majorant;

            if (t > d)
            break;

            x += w * t;
            depth += t;

            float density = sampleCloud(x);
            transmittance *= 1.0 - density;

            float sigma_a = PA * density;
            float sigma_s = PS * density;
            float sigma_n = majorant - parameters.extinction.x * density;

            float Pa = sigma_a / majorant;
            float Ps = sigma_s / majorant;
            float Pn = sigma_n / majorant;

            float xi = random();

            if (xi < Pa)
            return; // weights * sigma_a / (majorant * Pa) * L_e; // 0 - No emission

            if (xi < 1 - Pn) // scattering event
            {
                if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax))
                {
                    x += w*tMin;
                    depth += tMin;
                    d = tMax - tMin;
                }
            } else {
                pdf_x *= exp(-parameters.extinction.x * density);
                d -= t;
            }
        }
    }
}
#endif


//---------------------------------------------------------
// Main
//---------------------------------------------------------
void main()
{
    uint frame = frameInfo.frameCount;

    ivec2 dim = imageSize(resultImage);
    ivec2 imageCoord = ivec2(gl_GlobalInvocationID.xy);

    initializeRandom(frame * dim.x * dim.y + gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * dim.x);

    vec2 screenCoord = 2.0 * (gl_GlobalInvocationID.xy + vec2(random(), random())) / dim - 1;

    // Get ray direction and volume entry point
    vec3 x, w;
    createCameraRay(screenCoord, x, w);

    // Perform a single path and get radiance
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
    float scatterRayAbsorptionMoments[NUM_SCATTER_RAY_ABSORPTION_MOMENTS + 1];
#endif

#if defined(USE_DELTA_TRACKING)
    ScatterEvent firstEvent;
    vec3 result = pathtrace(
            x, w, firstEvent
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
            , scatterRayAbsorptionMoments
#endif
    );
#elif defined(USE_SPECTRAL_DELTA_TRACKING)
    ScatterEvent firstEvent = ScatterEvent(false, x, 0.0, w, 0.0);
    vec3 result = pathtraceSpectral(x, w);
#elif defined(USE_RATIO_TRACKING)
    ScatterEvent firstEvent;
    vec3 result = ratioTracking(x, w, firstEvent);
#elif defined(USE_RESIDUAL_RATIO_TRACKING)
    ScatterEvent firstEvent;
    vec3 result = residualRatioTracking(x, w, firstEvent);
#elif defined(USE_DECOMPOSITION_TRACKING)
    ScatterEvent firstEvent;
    vec3 result = analogDecompositionTracking(x, w, firstEvent);
#endif

#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
    for (int i = 0; i <= NUM_SCATTER_RAY_ABSORPTION_MOMENTS; i++) {
        float moment = scatterRayAbsorptionMoments[i];
        float momentOld = frame == 0 ? 0.0 : imageLoad(scatterRayAbsorptionMomentsImage, ivec3(imageCoord, i)).x;
        moment = mix(momentOld, moment, 1.0 / float(frame + 1));
        imageStore(scatterRayAbsorptionMomentsImage, ivec3(imageCoord, i), vec4(moment));
    }
#endif

    // Accumulate result
    vec3 resultOld = frame == 0 ? vec3(0) : imageLoad(accImage, imageCoord).xyz;
    result = mix(resultOld, result, 1.0 / float(frame + 1));
    imageStore(accImage, imageCoord, vec4(result, 1));
    imageStore(resultImage, imageCoord, vec4(result,1));

    //vec3 resultOld = frame == 0 ? vec3(0) : imageLoad(accImage, imageCoord).xyz;
    //result += resultOld;
    //imageStore(accImage, imageCoord, vec4(result, 1));
    //imageStore(resultImage, imageCoord, vec4(result/(frame + 1),1));

    // return; Uncomment this if want to execute faster a while(true) loop PT

    // Saving the first scatter position and direction
    if (firstEvent.hasValue) {
        imageStore(firstX, imageCoord, vec4(firstEvent.x, firstEvent.pdf_x));
        imageStore(firstW, imageCoord, vec4(firstEvent.w, firstEvent.pdf_w));
    } else {
        imageStore(firstX, imageCoord, vec4(0));
        imageStore(firstW, imageCoord, vec4(0));
    }

#ifdef COMPUTE_PRIMARY_RAY_ABSORPTION_MOMENTS
    float primaryRayAbsorptionMoments[NUM_PRIMARY_RAY_ABSORPTION_MOMENTS + 1];
    computePrimaryRayAbsorptionMoments(x, w, primaryRayAbsorptionMoments);
    for (int i = 0; i <= NUM_PRIMARY_RAY_ABSORPTION_MOMENTS; i++) {
        float moment = primaryRayAbsorptionMoments[i];
        float momentOld = frame == 0 ? 0.0 : imageLoad(primaryRayAbsorptionMomentsImage, ivec3(imageCoord, i)).x;
        moment = mix(momentOld, moment, 1.0 / float(frame + 1));
        imageStore(primaryRayAbsorptionMomentsImage, ivec3(imageCoord, i), vec4(moment));
    }
#endif
}
