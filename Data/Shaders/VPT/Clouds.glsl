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

//#extension GL_EXT_debug_printf : enable
//#extension GL_EXT_shader_explicit_arithmetic_types_int64 : enable

#include "VptHeader.glsl"

#ifdef USE_NANOVDB
#define PNANOVDB_GLSL
//#define PNANOVDB_ADDRESS_64
#include "PNanoVDB.glsl"
#endif

#ifdef USE_NANOVDB
pnanovdb_readaccessor_t accessor;
#endif

#if defined(USE_ISOSURFACES) || defined(USE_HEADLIGHT)
vec3 cameraPosition;
#endif

#include "VptUtils.glsl"
#include "VptMomentUtils.glsl"
#include "DeltaTracking.glsl"
#include "RatioTracking.glsl"
#include "ResidualRatioTracking.glsl"
#include "DecompositionTracking.glsl"
#include "NextEventTracking.glsl"
#include "IsosurfaceRendering.glsl"

#ifdef USE_RAY_MARCHING_EMISSION_ABSORPTION
#include "RayMarchingEmissionAbsorption.glsl"
#endif

#ifdef WRITE_DEPTH_BLENDED_MAP
#include "DepthBlended.glsl"
#endif

#ifdef WRITE_DEPTH_NEAREST_OPAQUE_MAP
#include "DepthNearestOpaque.glsl"
#endif

#ifdef WRITE_TRANSMITTANCE_VOLUME
#include "TransmittanceVolume.glsl"
#endif

void pathTraceSample(int i, bool onlyFirstEvent, out ScatterEvent firstEvent){
    uint frame = frameInfo.frameCount + i;
    uint frameGlobal = frameInfo.globalFrameNumber + i;

    ivec2 dim = imageSize(resultImage);
    ivec2 imageCoord = ivec2(gl_GlobalInvocationID.xy);

    uint seed = frameGlobal * dim.x * dim.y + gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * dim.x;
#ifdef CUSTOM_SEED_OFFSET
    seed += CUSTOM_SEED_OFFSET;
#endif
    initializeRandom(seed);

    vec2 screenCoord = 2.0 * (gl_GlobalInvocationID.xy + vec2(random(), random())) / dim - 1;

    // Get ray direction and volume entry point
    vec3 x, w;
    createCameraRay(screenCoord, x, w);
    firstEvent = ScatterEvent(
            false, x, 0.0, w, 0.0, 0.0, 0.0
#ifdef CLOSE_ISOSURFACES
            , vec3(0.0), false
#endif
    );

#if defined(USE_ISOSURFACES) || defined(USE_HEADLIGHT)
    cameraPosition = x;
#endif

#ifdef USE_NANOVDB
    accessor = createAccessor();
#endif

    // Perform a single path and get radiance
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
    float scatterRayAbsorptionMoments[NUM_SCATTER_RAY_ABSORPTION_MOMENTS + 1];
#endif

#if defined(WRITE_TRANSMITTANCE_VOLUME)
    vec3 result = vec3(0.0);
#elif defined(USE_DELTA_TRACKING)
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
#elif defined(USE_ISOSURFACE_RENDERING)
    vec3 result = isosurfaceRendering(x, w, firstEvent);
#elif defined(USE_RAY_MARCHING_EMISSION_ABSORPTION)
    vec4 colorRayOut = rayMarchingEmissionAbsorption(x, w, firstEvent);
    vec3 bgColor = sampleSkybox(w);
    vec3 result = bgColor.rgb * (1.0 - colorRayOut.a) + colorRayOut.rgb;
#endif

#ifdef WRITE_TRANSMITTANCE_VOLUME
    computeTransmittanceVolume(x, w);
#endif

    if (!onlyFirstEvent) {
#ifdef WRITE_CLOUDONLY_MAP
        // Accumulate cloudOnly
#if defined(USE_RAY_MARCHING_EMISSION_ABSORPTION) && !defined(WRITE_TRANSMITTANCE_VOLUME)
        vec4 cloudOnly = firstEvent.hasValue ? colorRayOut : vec4(0);
#else
        vec4 cloudOnly = firstEvent.hasValue ? vec4(result, 1) : vec4(0);
#endif
#ifndef DISABLE_ACCUMULATION
        vec4 cloudOnlyOld = frame == 0 ? vec4(0) : imageLoad(cloudOnlyImage, imageCoord);
        cloudOnly = mix(cloudOnlyOld, cloudOnly, 1.0 / float(frame + 1));
#endif
        imageStore(cloudOnlyImage, imageCoord, cloudOnly);
#endif // WRITE_CLOUDONLY_MAP

#ifdef WRITE_BACKGROUND_MAP
        // Accumulate background
        vec4 background = firstEvent.hasValue ? vec4(sampleSkybox(w), 1) : vec4(result, 1);
#ifndef DISABLE_ACCUMULATION
        vec4 backgroundOld = frame == 0 ? vec4(0) : imageLoad(backgroundImage, imageCoord);
        background = mix(backgroundOld, background, 1.0 / float(frame + 1));
#endif
        imageStore(backgroundImage, imageCoord, background);
#endif // WRITE_BACKGROUND_MAP

        // Accumulate result
#ifndef OUTPUT_FOREGROUND_MAP
#ifndef DISABLE_ACCUMULATION
        vec3 resultOld = frame == 0 ? vec3(0) : imageLoad(accImage, imageCoord).xyz;
        result = mix(resultOld, result, 1.0 / float(frame + 1));
#endif
        imageStore(accImage, imageCoord, vec4(result, 1));
        imageStore(resultImage, imageCoord, vec4(result, 1));
#else
#if defined(USE_RAY_MARCHING_EMISSION_ABSORPTION) && !defined(WRITE_TRANSMITTANCE_VOLUME)
        vec4 resultRgba = firstEvent.hasValue ? colorRayOut : vec4(0);
#else
        vec4 resultRgba = firstEvent.hasValue ? vec4(result, 1) : vec4(0);
#endif
#ifndef DISABLE_ACCUMULATION
        vec4 resultOld = frame == 0 ? vec4(0) : imageLoad(accImage, imageCoord);
        resultRgba = mix(resultOld, resultRgba, 1.0 / float(frame + 1));
#endif
        imageStore(accImage, imageCoord, resultRgba);
        imageStore(resultImage, imageCoord, resultRgba);
#endif
    }

#ifdef WRITE_POSITION_MAP
    vec4 position = firstEvent.hasValue ? vec4(firstEvent.x, 1) : vec4(0);
#ifndef DISABLE_ACCUMULATION
    vec4 positionOld = frame == 0 ? vec4(0) : imageLoad(firstX, imageCoord);
    position = mix(positionOld, position, 1.0 / float(frame + 1));
#endif
    imageStore(firstX, imageCoord, position);
#endif

#ifdef WRITE_DEPTH_MAP
#ifndef DISABLE_ACCUMULATION
    vec2 depth = firstEvent.hasValue ? vec2(firstEvent.depth, firstEvent.depth * firstEvent.depth) : vec2(0);
    vec2 depthOld = frame == 0 ? vec2(0) : imageLoad(depthImage, imageCoord).xy;
    depthOld.y = depthOld.y * depthOld.y + depthOld.x * depthOld.x;
    depth = mix(depthOld, depth, 1.0 / float(frame + 1));
#else
    vec2 depth = firstEvent.hasValue ? vec2(firstEvent.depth, firstEvent.depth * firstEvent.depth) : vec2(parameters.farDistance,0);
#endif
    imageStore(depthImage, imageCoord, vec4(depth.x, sqrt(max(0.,depth.y - depth.x * depth.x)),0,0));
#endif

#ifdef WRITE_DENSITY_MAP
    vec2 density = firstEvent.hasValue ? vec2(firstEvent.density * 0.001, firstEvent.density * firstEvent.density * 0.001 * 0.001) : vec2(0.0);
#ifndef DISABLE_ACCUMULATION
    vec2 densityOld = frame == 0 ? vec2(0.0) : imageLoad(densityImage, imageCoord).xy;
    densityOld.y = densityOld.y * densityOld.y + densityOld.x * densityOld.x;
    density = mix(densityOld, density, 1.0 / float(frame + 1));
#endif
    imageStore(densityImage, imageCoord, vec4(density.x, sqrt(max(0.0, density.y - density.x * density.x)), 0.0, 0.0));
#endif

#ifdef WRITE_REPROJ_UV_MAP
    vec2 oldReprojUV = frame == 0 ? vec2(-1.0, -1.0) : imageLoad(reprojUVImage, imageCoord).xy;
    vec4 prevClip = (parameters.previousViewProjMatrix * vec4(firstEvent.x, 1));
    vec2 reprojUV = prevClip.xy / prevClip.w;
    reprojUV = reprojUV * 0.5 + 0.5;
    reprojUV = firstEvent.hasValue ? reprojUV : oldReprojUV;
    imageStore(reprojUVImage, imageCoord, vec4(reprojUV, 0, 0));
#endif

#ifdef WRITE_ALBEDO_MAP
    vec4 albedo = firstEvent.hasValue ? vec4(parameters.scatteringAlbedo, 1.0) : vec4(0.0);
#ifndef DISABLE_ACCUMULATION
    vec4 albedoOld = frame == 0.0 ? vec4(0) : imageLoad(albedoImage, imageCoord);
    albedo = mix(albedoOld, albedo, 1.0 / float(frame + 1));
#endif
    imageStore(albedoImage, imageCoord, albedo);
#endif

#if defined(WRITE_DEPTH_NABLA_MAP) || defined(WRITE_DEPTH_FWIDTH_MAP)
    vec2 nabla = vec2(0.0, 0.0);
#endif

    // Saving the first scatter position and direction
    if (firstEvent.hasValue) {
        //imageStore(firstX, imageCoord, vec4(firstEvent.x, firstEvent.pdf_x));

#ifdef WRITE_NORMAL_MAP
#if defined(CLOSE_ISOSURFACES)
        vec3 diff;
        if (firstEvent.isIsosurface) {
            diff = firstEvent.normal;
        } else {
            diff = -computeGradient((firstEvent.x - parameters.boxMin) / (parameters.boxMax -parameters.boxMin));
        }
#elif defined(USE_ISOSURFACES)
        vec3 diff = -computeGradient((firstEvent.x - parameters.boxMin) / (parameters.boxMax -parameters.boxMin));
#else // !defined(USE_ISOSURFACES)
        vec3 diff = getCloudFiniteDifference(firstEvent.x);
#endif // USE_ISOSURFACES
#ifndef DISABLE_ACCUMULATION
        vec3 diffOld = frame == 0 ? vec3(0.0) : imageLoad(normalImage, imageCoord).xyz;
        diff = mix(diffOld, diff, 1.0 / float(frame + 1.0));
#endif
        imageStore(normalImage, imageCoord, vec4(diff, 1.0));
#endif

#ifdef WRITE_FIRST_W_MAP
        imageStore(firstW, imageCoord, vec4(firstEvent.w, firstEvent.pdf_w));
#endif

#if defined(WRITE_DEPTH_NABLA_MAP) || defined(WRITE_DEPTH_FWIDTH_MAP)
        vec3 camNormalFlat = (parameters.inverseTransposedViewMatrix * vec4(diff, 0.0)).xyz;
        // A = cos(diff, camX)
        // cot(acos(A)) = cos(acos(A)) / sin(acos(A)) = A / sin(acos(A)) = A / sqrt(1 - A^2)
        const float A = camNormalFlat.x; // dot(camNormalFlat, vec3(1.0, 0.0, 0.0))
        const float B = camNormalFlat.y; // dot(camNormalFlat, vec3(0.0, 1.0, 0.0))
        nabla = vec2(A / sqrt(1.0 - A * A), B / sqrt(1.0 - B * B));
#endif
    } else {
        //imageStore(firstX, imageCoord, vec4(0));

//#ifdef WRITE_NORMAL_MAP
//        imageStore(normalImage, imageCoord, vec4(0));
//#endif

#ifdef WRITE_NORMAL_MAP
#ifdef DISABLE_ACCUMULATION
        imageStore(normalImage, imageCoord, vec4(0.0));
#else
        vec3 diff = frame == 0 ? vec3(0.0) : imageLoad(normalImage, imageCoord).xyz * (float(frame) / float(frame + 1.0));
        imageStore(normalImage, imageCoord, vec4(diff, 1.0));
#endif
#endif

#ifdef WRITE_FIRST_W_MAP
        imageStore(firstW, imageCoord, vec4(0.0));
#endif
    }

#ifdef WRITE_FLOW_MAP
    vec2 flowVector = vec2(0.0);
    if (firstEvent.hasValue) {
        vec4 lastFramePositionNdc = parameters.previousViewProjMatrix * vec4(firstEvent.x, 1.0);
        lastFramePositionNdc.xyz /= lastFramePositionNdc.w;
        vec2 pixelPositionLastFrame = (0.5 * lastFramePositionNdc.xy + vec2(0.5)) * vec2(dim) - vec2(0.5);
        flowVector = vec2(imageCoord) - pixelPositionLastFrame;
    }
    imageStore(flowImage, imageCoord, vec4(flowVector, 0.0, 0.0));
#endif

#ifdef WRITE_FLOW_REVERSE_MAP
    // Motion vectors as expected by NVIDIA libraries.
    //vec2 screenCoord = 2.0 * (gl_GlobalInvocationID.xy + vec2(random(), random())) / dim - 1;
    vec2 flowReverseVector = vec2(0.0);
    if (firstEvent.hasValue) {
        vec4 lastFramePositionNdc = parameters.previousViewProjMatrix * vec4(firstEvent.x, 1.0);
        lastFramePositionNdc.xyz /= lastFramePositionNdc.w;
        vec2 pixelPositionLastFrame = (0.5 * lastFramePositionNdc.xy + vec2(0.5)) * vec2(dim) - vec2(0.5);
        flowReverseVector = pixelPositionLastFrame - vec2(imageCoord);
    }
    imageStore(flowReverseImage, imageCoord, vec4(flowReverseVector, 0.0, 0.0));
#endif

#ifdef WRITE_DEPTH_NABLA_MAP
#ifndef DISABLE_ACCUMULATION
    if (frame != 0) {
        vec2 nablaOld = imageLoad(depthNablaImage, imageCoord).xy;
        nabla = mix(nablaOld, nabla, 1.0 / float(frame + 1));
    }
#endif
    imageStore(depthNablaImage, imageCoord, vec4(nabla, 0.0, 0.0));
#endif

#ifdef WRITE_DEPTH_FWIDTH_MAP
    float fwidthValue = abs(nabla.x) + abs(nabla.y);
#ifndef DISABLE_ACCUMULATION
    if (frame != 0) {
        float fwidthValueOld = imageLoad(depthFwidthImage, imageCoord).x;
        fwidthValue = mix(fwidthValueOld, fwidthValue, 1.0 / float(frame + 1));
    }
#endif
    imageStore(depthFwidthImage, imageCoord, vec4(fwidthValue));
#endif

#ifdef WRITE_DEPTH_BLENDED_MAP
    vec2 depthBlended = computeDepthBlended(x, w);
#ifndef DISABLE_ACCUMULATION
    vec2 depthBlendedOld = frame == 0 ? vec2(0) : imageLoad(depthBlendedImage, imageCoord).xy;
    depthBlended = mix(depthBlendedOld, depthBlended, 1.0 / float(frame + 1));
#endif
    imageStore(depthBlendedImage, imageCoord, vec4(depthBlended, 0.0, 1.0));
#endif

#ifdef WRITE_DEPTH_NEAREST_OPAQUE_MAP
    vec2 depthNO = computeDepthNearestOpaque(x, w);
#ifndef DISABLE_ACCUMULATION
    vec2 depthNOOld = frame == 0 ? vec2(0) : imageLoad(depthNearestOpaqueImage, imageCoord).xy;
    depthNO = mix(depthNOOld, depthNO, 1.0 / float(frame + 1));
#endif
    imageStore(depthNearestOpaqueImage, imageCoord, vec4(depthNO, 0.0, 1.0));
#endif

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
