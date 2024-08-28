/**
 * MIT License
 *
 * Copyright (c) 2021-2022, Christoph Neuhauser, Timm Knörle, Ludwig Leonard
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
 * For more details on spectral delta tracking, please refer to:
 * P. Kutz, R. Habel, Y. K. Li, and J. Novák. Spectral and decomposition tracking for rendering heterogeneous volumes.
 * ACM Trans. Graph., 36(4), Jul. 2017.
 */
#ifdef USE_NEXT_EVENT_TRACKING_SPECTRAL
vec3 nextEventTrackingSpectral(vec3 x, vec3 w, inout ScatterEvent firstEvent, bool onlyFirstEvent) {
    float majorant = maxComponent(parameters.extinction);

    vec3 weights = vec3(1, 1, 1);
#ifdef USE_ISOSURFACES
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;
#endif

    vec3 absorptionAlbedo = vec3(1, 1, 1) - parameters.scatteringAlbedo;
    vec3 scatteringAlbedo = parameters.scatteringAlbedo;
    float PA = maxComponent(absorptionAlbedo * parameters.extinction);
    float PS = maxComponent(scatteringAlbedo * parameters.extinction);

    vec3 color = vec3(0);
    float bw_phase = 1.;

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
            vec4 densityEmission = sampleCloudDensityEmission(x);
            float density = densityEmission.a;
            //scatteringAlbedo = densityEmission.rgb;
            //absorptionAlbedo = vec3(1) - scatteringAlbedo;
            //float PA = maxComponent(absorptionAlbedo * parameters.extinction);
            //float PS = maxComponent(scatteringAlbedo * parameters.extinction);
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
                //return color;
#ifdef USE_EMISSION
                vec3 emission = sampleEmission(x);
#ifdef USE_ISOSURFACES
                return color + weights * emission;
#else
                return color + emission;
#endif
#else
#ifdef USE_TRANSFER_FUNCTION
#ifdef USE_ISOSURFACES
                return weights * densityEmission.rgb * parameters.emissionStrength;
#else
                return densityEmission.rgb * parameters.emissionStrength;
#endif
#endif
                return color; // weights * sigma_a / (majorant * Pa) * L_e; // 0 - No emission
#endif
            }

            if (xi < Pa + Ps) { // scattering event
                float pdf_w;
                vec3 next_w = importanceSamplePhase(parameters.phaseG, w, pdf_w);

                if (!firstEvent.hasValue) {
                    firstEvent.x = x;
                    firstEvent.pdf_x = 0; // TODO
                    firstEvent.w = next_w;
                    firstEvent.pdf_w = pdf_w;
                    firstEvent.hasValue = true;
                    firstEvent.density = density * maxComponent(parameters.extinction);
                    firstEvent.depth = tMax - d + t;
                }
                if (onlyFirstEvent){
                    return vec3(0);
                }

                float pdfLightNee; // only used for skybox.
                vec3 dirLightNee;
#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
                // We are sampling the environment map or headlight with 50/50 chance.
                bool isSamplingHeadlight = (parameters.isEnvMapBlack != 0u) ? true : (random() > 0.5);
                float lightProbabilityFactor = parameters.isEnvMapBlack != 0u ? 1.0 : 2.0;
                float lightDistance = 0.0;
                uint lightIdx = 0;
                if (isSamplingHeadlight) {
                    dirLightNee = getHeadlightDirection(x, lightIdx, lightProbabilityFactor, lightDistance);
                } else {
#endif
                    dirLightNee = importanceSampleSkybox(pdfLightNee);
#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
                }
#endif

                float pdf_nee_phase = evaluatePhase(parameters.phaseG, w, dirLightNee);
                w = next_w;

                weights *= sigma_s / (majorant * Ps);

#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
                vec3 commonFactor = (lightProbabilityFactor * pdf_nee_phase) * min(weights, vec3(100000, 100000, 100000));
                if (isSamplingHeadlight) {
                    color +=
                            commonFactor * calculateTransmittanceDistance(x, dirLightNee, lightDistance)
                            * sampleHeadlight(x, lightIdx);
                } else {
                    color +=
                            commonFactor / pdfLightNee * calculateTransmittance(x, dirLightNee)
                            * (sampleSkybox(dirLightNee) + sampleLight(dirLightNee));
                }
#else
                // Normal NEE.
                color +=
                        (pdf_nee_phase / pdfLightNee * calculateTransmittance(x, dirLightNee))
                        * min(weights, vec3(100000, 100000, 100000)) * (sampleSkybox(dirLightNee) + sampleLight(dirLightNee));
#endif

                if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
                    x += w*tMin;
                    d = tMax - tMin;
                }
            } else {
                d -= t;
                weights *= sigma_n / (majorant * Pn);
            }
#if !defined(MAX_BASED_PROBABILITY) && !defined(AVG_BASED_PROBABILITY)
            weights = min(weights, vec3(100.0, 100.0, 100.0));
#endif
        }
    }

    if (!firstEvent.hasValue){
        color += bw_phase * min(weights, vec3(100000, 100000, 100000)) * (sampleSkybox(w) + sampleLight(w));
    }
    return color;
}
#endif

#ifdef USE_NEXT_EVENT_TRACKING
vec3 nextEventTracking(vec3 x, vec3 w, inout ScatterEvent firstEvent, bool onlyFirstEvent) {
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

    vec3 color = vec3(0.);

    int i = 0;
    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
        float d = tMax - tMin;
        float pdf_x = 1;

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
                    firstEvent.pdf_x = sigma_s * pdf_x;
                    firstEvent.w = w;
                    firstEvent.pdf_w = 0;
                    firstEvent.hasValue = true;
                    firstEvent.density = density * maxComponent(parameters.extinction);
                    firstEvent.depth = tMax - d + t;
                }
#ifdef USE_EMISSION
                vec3 emission = sampleEmission(x);
#ifdef USE_ISOSURFACES
                return color + weights * emission;
#else
                return color + emission;
#endif
#else
#ifdef USE_TRANSFER_FUNCTION
#ifdef USE_ISOSURFACES
                return color + weights * parameters.emissionStrength * densityEmission.rgb;
#else
                return color + parameters.emissionStrength * densityEmission.rgb;
#endif
#endif
                return color; // weights * sigma_a / (majorant * Pa) * L_e; // 0 - No emission
#endif
            }

            if (xi < 1 - Pn)// scattering event
            {
                float pdf_w;
                vec3 next_w = importanceSamplePhase(parameters.phaseG, w, pdf_w);

                if (!firstEvent.hasValue) {
                    firstEvent.x = x;
                    firstEvent.pdf_x = sigma_s * pdf_x;
                    firstEvent.w = next_w;
                    firstEvent.pdf_w = pdf_w;
                    firstEvent.hasValue = true;
                    firstEvent.density = density * maxComponent(parameters.extinction);
                    firstEvent.depth = tMax - d + t;
                }
                if (onlyFirstEvent){
                    return vec3(0);
                }

                float pdfLightNee; // only used for skybox.
                vec3 dirLightNee;
#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
                // We are sampling the environment map or headlight with 50/50 chance.
                bool isSamplingHeadlight = (parameters.isEnvMapBlack != 0u) ? true : (random() > 0.5);
                float lightProbabilityFactor = parameters.isEnvMapBlack != 0u ? 1.0 : 2.0;
                float lightDistance = 0.0;
                uint lightIdx = 0;
                if (isSamplingHeadlight) {
                    dirLightNee = getHeadlightDirection(x, lightIdx, lightProbabilityFactor, lightDistance);
                } else {
#endif
                    dirLightNee = importanceSampleSkybox(pdfLightNee);
#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
                }
#endif

                float pdf_nee_phase = evaluatePhase(parameters.phaseG, w, dirLightNee);
                w = next_w;

#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
                float commonFactor = lightProbabilityFactor * pdf_nee_phase;
                vec3 colorNew;
                if (isSamplingHeadlight) {
                    colorNew =
                            (commonFactor * calculateTransmittanceDistance(x, dirLightNee, lightDistance))
                            * sampleHeadlight(x, lightIdx);
                } else {
                    colorNew =
                            (commonFactor / pdfLightNee * calculateTransmittance(x, dirLightNee))
                            * (sampleSkybox(dirLightNee) + sampleLight(dirLightNee));
                }
#else
                // Normal NEE.
                vec3 colorNew =
                        (pdf_nee_phase / pdfLightNee * calculateTransmittance(x, dirLightNee))
                        * (sampleSkybox(dirLightNee) + sampleLight(dirLightNee));
#endif

#ifdef USE_ISOSURFACES
                colorNew *= weights;
#endif
                color += colorNew;
                pdf_x *= exp(-majorant * t) * majorant * density;

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

    if (!firstEvent.hasValue){
        color += sampleSkybox(w) + sampleLight(w);
    }
    return color;
}
#endif
