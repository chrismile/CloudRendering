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

vec4 rayMarchingEmissionAbsorption(vec3 x, vec3 w, inout ScatterEvent firstEvent) {
    const float majorant = parameters.extinction.x;
    const ivec3 gridImgSize = textureSize(gridImage, 0);
    const float stepSize = 0.25 / max(gridImgSize.x, max(gridImgSize.y, gridImgSize.z));
    const float attenuationCoefficient = majorant * stepSize;
    vec3 absorptionAlbedo = vec3(1, 1, 1) - parameters.scatteringAlbedo;
    vec3 scatteringAlbedo = parameters.scatteringAlbedo;

#ifdef USE_ISOSURFACES
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;
#endif
#ifdef CLOSE_ISOSURFACES
    bool isFirstPointFromOutside = true;
#endif

    float alphaAccum = 0.0;
    vec3 colorAccum = vec3(0.0);
    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        float t = tMin;
        while (true) {
            bool hasHit = false;
            vec3 xOld = x + w * t;
            t += stepSize;
#if defined(COMPOSITION_MODEL_ALPHA_BLENDING)
            if (t > tMax || alphaAccum > 0.999) {
                break;
            }
#else
            if (t > tMax) {
                break;
            }
#endif
            vec3 xNew = x + w * t;

#ifdef USE_TRANSFER_FUNCTION
            vec4 densityEmission = sampleCloudDensityEmission(xNew);
            float density = densityEmission.a;
            vec3 color = densityEmission.rgb;
#else
            float density = sampleCloud(xNew);
            vec3 color = scatteringAlbedo;
#endif

            float alpha = 0.0;

#ifdef USE_ISOSURFACES
            const int isoSubdivs = 2;
            bool foundHit = false;
            for (int subdiv = 0; subdiv < isoSubdivs; subdiv++) {
                vec3 x0 = mix(xOld, xNew, float(subdiv) / float(isoSubdivs));
                vec3 x1 = mix(xOld, xNew, float(subdiv + 1) / float(isoSubdivs));
                float scalarValue = sampleCloudDirect(x1);

                currentScalarSign = sign(scalarValue - parameters.isoValue);
                if (isFirstPoint) {
                    isFirstPoint = false;
                    lastScalarSign = currentScalarSign;
                }

                if (lastScalarSign != currentScalarSign) {
                    refineIsoSurfaceHit(x1, x0, currentScalarSign);
                    xNew = x1;
                    vec3 surfaceNormal;
                    color = getIsoSurfaceHitDirect(xNew, w, surfaceNormal);
                    //alpha = color.a; // TODO
                    alpha = 1.0;
                    density = 1e9;
                    hasHit = true;
                    break;
                }
            }
#endif

            if (!hasHit) {
                if (density > 1e-4) {
                    hasHit = true;
                }
                alpha = 1.0 - exp(-density * attenuationCoefficient);
            }

            if (hasHit && !firstEvent.hasValue) {
                firstEvent.x = x;
                firstEvent.pdf_x = 0;
                firstEvent.w = vec3(0.0);
                firstEvent.pdf_w = 0;
                firstEvent.hasValue = true;
                firstEvent.density = density * parameters.extinction.x;
                firstEvent.depth = t;
            }

#if defined(COMPOSITION_MODEL_ALPHA_BLENDING)
            colorAccum = colorAccum + (1.0 - alphaAccum) * alpha * color;
            alphaAccum = alphaAccum + (1.0 - alphaAccum) * alpha;
#elif defined(COMPOSITION_MODEL_AVERAGE)
            colorAccum += color;
            alphaAccum += 1.0;
#elif defined(COMPOSITION_MODEL_MAXIMUM_INTENSITY_PROJECTION)
            alpha = clamp(density, 0.0, 1.0);
            colorAccum = max(colorAccum, color);
            alphaAccum = max(alphaAccum, alpha);
#endif
        }
    }

#if defined(COMPOSITION_MODEL_ALPHA_BLENDING)
    return vec4(colorAccum, alphaAccum);
#elif defined(COMPOSITION_MODEL_AVERAGE)
    return vec4(colorAccum / (alphaAccum + 1e-5), 1.0);
#elif defined(COMPOSITION_MODEL_MAXIMUM_INTENSITY_PROJECTION)
    return vec4(colorAccum, 1.0);
#endif
}
