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

-- Compute

#version 450

#include "VptHeader.glsl"

#ifdef USE_NANOVDB
#define PNANOVDB_GLSL
#include "PNanoVDB.glsl"
#endif

#include "VptUtils.glsl"
#include "VptMomentUtils.glsl"
#include "DeltaTracking.glsl"
#include "RatioTracking.glsl"
#include "ResidualRatioTracking.glsl"
#include "DecompositionTracking.glsl"
#include "NextEventTracking.glsl"

void pathTraceSample(int i, bool onlyFirstEvent, out ScatterEvent firstEvent){
    uint frame = frameInfo.frameCount + i;

    ivec2 dim = imageSize(resultImage);
    ivec2 imageCoord = ivec2(gl_GlobalInvocationID.xy);

    uint seed = frame * dim.x * dim.y + gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * dim.x;
#ifdef CUSTOM_SEED_OFFSET
    seed += CUSTOM_SEED_OFFSET;
#endif
    initializeRandom(seed);

    vec2 screenCoord = 2.0 * (gl_GlobalInvocationID.xy + vec2(random(), random())) / dim - 1;

    // Get ray direction and volume entry point
    vec3 x, w;
    createCameraRay(screenCoord, x, w);

    // Perform a single path and get radiance
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
    float scatterRayAbsorptionMoments[NUM_SCATTER_RAY_ABSORPTION_MOMENTS + 1];
#endif

#if defined(USE_DELTA_TRACKING)
    vec3 result = deltaTracking(
            x, w, firstEvent
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
            , scatterRayAbsorptionMoments
#endif
    );
#elif defined(USE_SPECTRAL_DELTA_TRACKING)
    vec3 result = deltaTrackingSpectral(x, w, firstEvent);
#elif defined(USE_RATIO_TRACKING)
    vec3 result = ratioTracking(x, w, firstEvent);
#elif defined(USE_RESIDUAL_RATIO_TRACKING)
    vec3 result = residualRatioTracking(x, w, firstEvent);
#elif defined(USE_DECOMPOSITION_TRACKING)
    vec3 result = analogDecompositionTracking(x, w, firstEvent);
#elif defined(USE_NEXT_EVENT_TRACKING)
    vec3 result = nextEventTracking(x, w, firstEvent, onlyFirstEvent);
#elif defined(USE_NEXT_EVENT_TRACKING_SPECTRAL)
    vec3 result = nextEventTrackingSpectral(x, w, firstEvent, onlyFirstEvent);
#endif

    if (!onlyFirstEvent) {
        // Accumulate cloudOnly
        vec4 cloudOnlyOld = frame == 0 ? vec4(0) : imageLoad(cloudOnlyImage, imageCoord);
        vec4 cloudOnly = firstEvent.hasValue ? vec4(result, 1) : vec4(0);
        cloudOnly = mix(cloudOnlyOld, cloudOnly, 1.0 / float(frame + 1));
        imageStore(cloudOnlyImage, imageCoord, cloudOnly);

        // Accumulate background
        vec4 backgroundOld = frame == 0 ? vec4(0) : imageLoad(backgroundImage, imageCoord);
        vec4 background = firstEvent.hasValue ? vec4(sampleSkybox(w), 1) : vec4(result, 1);
        background = mix(backgroundOld, background, 1.0 / float(frame + 1));
        imageStore(backgroundImage, imageCoord, background);


        // Accumulate result
        vec3 resultOld = frame == 0 ? vec3(0) : imageLoad(accImage, imageCoord).xyz;
        result = mix(resultOld, result, 1.0 / float(frame + 1));
        imageStore(accImage, imageCoord, vec4(result, 1));
        imageStore(resultImage, imageCoord, vec4(result, 1));
    }

    vec4 positionOld = frame == 0 ? vec4(0) : imageLoad(firstX, imageCoord);
    vec4 position = firstEvent.hasValue ? vec4(firstEvent.x, 1) : vec4(0);
    position = mix(positionOld, position, 1.0 / float(frame + 1));
    imageStore(firstX, imageCoord, position);

    vec2 depthOld = frame == 0 ? vec2(0) : imageLoad(depthImage, imageCoord).xy;
    depthOld.y = depthOld.y * depthOld.y + depthOld.x * depthOld.x;
    vec2 depth = firstEvent.hasValue ? vec2(firstEvent.depth, firstEvent.depth * firstEvent.depth) : vec2(0);
    depth = mix(depthOld, depth, 1.0 / float(frame + 1));
    imageStore(depthImage, imageCoord, vec4(depth.x, sqrt(max(0.,depth.y - depth.x * depth.x)),0,0));

    vec2 densityOld = frame == 0 ? vec2(0) : imageLoad(densityImage, imageCoord).xy;
    densityOld.y = densityOld.y * densityOld.y + densityOld.x * densityOld.x;
    vec2 density = firstEvent.hasValue ? vec2(firstEvent.density * .001, firstEvent.density * firstEvent.density * .001 * .001) : vec2(0);
    density = mix(densityOld, density, 1.0 / float(frame + 1));
    imageStore(densityImage, imageCoord, vec4(density.x, sqrt(max(0.,density.y - density.x * density.x)),0,0));

    vec2 oldReprojUV = frame == 0 ? vec2(-1,-1) : imageLoad(reprojUVImage, imageCoord).xy;
    vec4 prevClip = (parameters.previousViewProjMatrix * vec4(firstEvent.x, 1));
    vec2 reprojUV = prevClip.xy / prevClip.w;
    reprojUV = reprojUV * .5 + .5;
    reprojUV = firstEvent.hasValue? reprojUV : oldReprojUV;
    imageStore(reprojUVImage, imageCoord, vec4(reprojUV, 0, 0));

    // Saving the first scatter position and direction
    if (firstEvent.hasValue) {
        vec3 diff = getCloudFiniteDifference(firstEvent.x);

        vec3 diffOld = frame == 0 ? vec3(0) : imageLoad(normalImage, imageCoord).xyz;
        diff = mix(diffOld, diff, 1.0 / float(frame + 1));
        imageStore(normalImage, imageCoord, vec4(diff,1));

        //imageStore(firstX, imageCoord, vec4(firstEvent.x, firstEvent.pdf_x));
        imageStore(firstW, imageCoord, vec4(firstEvent.w, firstEvent.pdf_w));
    } else {
        //imageStore(firstX, imageCoord, vec4(0));
        imageStore(normalImage, imageCoord, vec4(0));
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

void main() {
    for (int i = 0; i < parameters.numFeatureMapSamplesPerFrame; i++){
        ScatterEvent firstEvent;
        pathTraceSample(i, i > 0, firstEvent);
    }
}