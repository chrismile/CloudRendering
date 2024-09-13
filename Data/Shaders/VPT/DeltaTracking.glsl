/**
 * MIT License
 *
 * Copyright (c) 2021-2022, Christoph Neuhauser, Ludwig Leonard
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

// Pathtracing with Delta tracking and Spectral tracking.

#ifdef USE_SPECTRAL_DELTA_TRACKING
/**
 * For more details on spectral delta tracking, please refer to:
 * P. Kutz, R. Habel, Y. K. Li, and J. Novák. Spectral and decomposition tracking for rendering heterogeneous volumes.
 * ACM Trans. Graph., 36(4), Jul. 2017.
 */
vec3 deltaTrackingSpectral(vec3 x, vec3 w, inout ScatterEvent firstEvent) {
    float majorant = maxComponent(parameters.extinction);

    vec3 weights = vec3(1, 1, 1);
#ifdef USE_ISOSURFACES
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;
#endif
#ifdef CLOSE_ISOSURFACES
    bool isFirstPointFromOutside = true;
#endif

    vec3 absorptionAlbedo = vec3(1, 1, 1) - parameters.scatteringAlbedo;
    vec3 scatteringAlbedo = parameters.scatteringAlbedo;
    float PA = maxComponent(absorptionAlbedo * parameters.extinction);
    float PS = maxComponent(scatteringAlbedo * parameters.extinction);

    int i = 0;
    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
        float d = tMax - tMin;
        while (true) {
#ifdef USE_ISOSURFACES
            i++;
            if (i == 1000) {
                return vec3(0.0, 0.0, 0.0);
            }
#endif

            float t = -log(max(0.0000000001, 1 - random()))/majorant;

            if (t > d) {
                break;
            }

            vec3 xNew = x + w * t;

#ifdef USE_TRANSFER_FUNCTION
            vec4 densityEmission = sampleCloudDensityEmission(xNew);
            float density = densityEmission.a;
#else
            float density = sampleCloud(xNew);
#endif

#include "CheckIsosurfaceHit.glsl"

            x = xNew;

            vec3 sigma_a = absorptionAlbedo * parameters.extinction * density;
            vec3 sigma_s = scatteringAlbedo * parameters.extinction * density;
            vec3 sigma_n = vec3(majorant) - parameters.extinction * density;

#if defined(MAX_BASED_PROBABILITY)
            float Pa = maxComponent(sigma_a);
            float Ps = maxComponent(sigma_s);
            float Pn = maxComponent(sigma_n);
#elif defined(AVG_BASED_PROBABILITY)
            float Pa = avgComponent(sigma_a);
            float Ps = avgComponent(sigma_s);
            float Pn = avgComponent(sigma_n);
#else // Path history average-based probability
            float Pa = avgComponent(sigma_a * weights);
            float Ps = avgComponent(sigma_s * weights);
            float Pn = avgComponent(sigma_n * weights);
#endif
            float C = Pa + Ps + Pn;
            Pa /= C;
            Ps /= C;
            Pn /= C;

            float xi = random();

            if (xi < Pa) {
                if (!firstEvent.hasValue) {
                    firstEvent.x = x;
                    firstEvent.pdf_x = 0; // TODO
                    firstEvent.w = vec3(0.);
                    firstEvent.pdf_w = 0;
                    firstEvent.hasValue = true;
                    firstEvent.density = density * maxComponent(parameters.extinction);
                    firstEvent.depth = tMax - d + t;
                }

#ifdef USE_TRANSFER_FUNCTION
                vec3 L_e = parameters.emissionStrength * densityEmission.rgb;
                return weights * sigma_a / (majorant * Pa) * L_e;
#else
                return vec3(0); // weights * sigma_a / (majorant * Pa) * L_e; // 0 - No emission
#endif
            }

            if (xi < Pa + Ps) { // scattering event
                float pdf_w;
                w = importanceSamplePhase(parameters.phaseG, w, pdf_w);

                if (!firstEvent.hasValue) {
                    firstEvent.x = x;
                    firstEvent.pdf_x = 0; // TODO
                    firstEvent.w = w;
                    firstEvent.pdf_w = pdf_w;
                    firstEvent.hasValue = true;
                    firstEvent.density = density * maxComponent(parameters.extinction);
                    firstEvent.depth = tMax - d + t;
                }

                if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
                    x += w*tMin;
                    d = tMax - tMin;
                }
                weights *= sigma_s / (majorant * Ps);
            } else {
                d -= t;
                weights *= sigma_n / (majorant * Pn);
            }
#if !defined(MAX_BASED_PROBABILITY) && !defined(AVG_BASED_PROBABILITY)
            weights = min(weights, vec3(100.0, 100.0, 100.0));
#endif
        }
    }

    return min(weights, vec3(100000, 100000, 100000)) * (sampleSkybox(w) + sampleLight(w));
}
#endif


#ifdef USE_DELTA_TRACKING
vec3 deltaTracking(
        vec3 x, vec3 w, inout ScatterEvent firstEvent
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

    float majorant = parameters.extinction.x;
    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;
    float PA = absorptionAlbedo * parameters.extinction.x;
    float PS = scatteringAlbedo * parameters.extinction.x;

#ifdef USE_ISOSURFACES
    vec3 weights = vec3(1, 1, 1);
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;
#endif
#ifdef CLOSE_ISOSURFACES
    bool isFirstPointFromOutside = true;
#endif

    int i = 0;
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
#ifdef USE_ISOSURFACES
            i++;
            if (i == 1000) {
                return vec3(0.0, 0.0, 0.0);
            }
#endif
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
            float t = -log(max(0.0000000001, 1 - random())) / majorant;

            if (t > d) {
                break;
            }

            vec3 xNew = x + w * t;
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
            depth += t;
#endif

#ifdef USE_TRANSFER_FUNCTION
            vec4 densityEmission = sampleCloudDensityEmission(xNew);
            float density = densityEmission.a;
#else
            float density = sampleCloud(xNew);
#endif

#include "CheckIsosurfaceHit.glsl"

            x = xNew;
            transmittance *= 1.0 - density;

            float sigma_a = PA * density;
            float sigma_s = PS * density;
            float sigma_n = majorant - parameters.extinction.x * density;

            float Pa = sigma_a / majorant;
            float Ps = sigma_s / majorant;
            float Pn = sigma_n / majorant;

            float xi = random();

            if (xi < Pa) {
                if (!firstEvent.hasValue) {
                    firstEvent.x = x;
                    firstEvent.pdf_x = 0;
                    firstEvent.w = vec3(0.);
                    firstEvent.pdf_w = 0;
                    firstEvent.hasValue = true;
                    firstEvent.density = density * parameters.extinction.x;
                    firstEvent.depth = tMax - d + t;
                }
#if defined(USE_EMISSION) || defined(USE_TRANSFER_FUNCTION)
                vec3 L_e = vec3(0.0);
#endif
#ifdef USE_EMISSION
                L_e += sampleEmission(x);
#ifdef USE_ISOSURFACES
                return weights * emission;
#else
                return emission;
#endif
#endif
#ifdef USE_TRANSFER_FUNCTION
                L_e += parameters.emissionStrength * densityEmission.rgb;
                //return weights * sigma_a / (majorant * Pa) * L_e;
#endif
#if defined(USE_EMISSION) || defined(USE_TRANSFER_FUNCTION)
#ifdef USE_ISOSURFACES
                return weights * L_e;
#else
                return L_e;
#endif
#else
                return vec3(0); // weights * sigma_a / (majorant * Pa) * L_e; // 0 - No emission
#endif
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
                    firstEvent.density = density * parameters.extinction.x;
                    firstEvent.depth = tMax - d + t;
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

#ifdef USE_ISOSURFACES
    return weights * (sampleSkybox(w) + sampleLight(w));
#else
    return sampleSkybox(w) + sampleLight(w);
#endif
}
#endif
