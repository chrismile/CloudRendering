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

void main() {
    uint frame = frameInfo.frameCount;

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
    ScatterEvent firstEvent;
    vec3 result = deltaTracking(
            x, w, firstEvent
#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
            , scatterRayAbsorptionMoments
#endif
    );
#elif defined(USE_SPECTRAL_DELTA_TRACKING)
    ScatterEvent firstEvent = ScatterEvent(false, x, 0.0, w, 0.0, 0.0, 0.0);
    vec3 result = deltaTrackingSpectral(x, w, firstEvent);
#elif defined(USE_RATIO_TRACKING)
    ScatterEvent firstEvent;
    vec3 result = ratioTracking(x, w, firstEvent);
#elif defined(USE_RESIDUAL_RATIO_TRACKING)
    ScatterEvent firstEvent;
    vec3 result = residualRatioTracking(x, w, firstEvent);
#elif defined(USE_DECOMPOSITION_TRACKING)
    ScatterEvent firstEvent;
    vec3 result = analogDecompositionTracking(x, w, firstEvent);
#elif defined(USE_NEXT_EVENT_TRACKING)
    ScatterEvent firstEvent = ScatterEvent(false, x, 0.0, w, 0.0, 0.0, 0.0);
    vec3 result = nextEventTracking(x, w, firstEvent);
#elif defined(USE_NEXT_EVENT_TRACKING_SPECTRAL)
    ScatterEvent firstEvent = ScatterEvent(false, x, 0.0, w, 0.0, 0.0, 0.0);
    vec3 result = nextEventTrackingSpectral(x, w, firstEvent);
#endif

#ifdef COMPUTE_SCATTER_RAY_ABSORPTION_MOMENTS
    for (int i = 0; i <= NUM_SCATTER_RAY_ABSORPTION_MOMENTS; i++) {
        float moment = scatterRayAbsorptionMoments[i];
        float momentOld = frame == 0 ? 0.0 : imageLoad(scatterRayAbsorptionMomentsImage, ivec3(imageCoord, i)).x;
        moment = mix(momentOld, moment, 1.0 / float(frame + 1));
        imageStore(scatterRayAbsorptionMomentsImage, ivec3(imageCoord, i), vec4(moment));
    }
#endif

    vec4 prevClip = (parameters.previousViewProjMatrix * vec4(firstEvent.x, 1));
    vec2 lastUV = prevClip.xy / prevClip.w;
    lastUV = lastUV * .5 + .5;
    lastUV = firstEvent.hasValue? lastUV : vec2(-1,-1);
    imageStore(reprojUVImage, imageCoord, vec4(lastUV, 0, 0));

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

    /*for (int i = 0; i < 2; i++) {
        float pdf_skybox;

        vec3 dir = importanceSampleSkybox(10, 0, pdf_skybox);
        if (length(dir- w) < .01) {
            result = vec3(100000,0,100000);
        }
    }*/

    /*if (firstEvent.hasValue){
        float pdf_skybox;
        w = importanceSampleSkybox(10, 0, pdf_skybox);
        result = sampleSkybox(w);
    }*/

    //result.x = evaluateSkyboxPDF(10,0,w);
    //result.yz = vec2(0);

    /*float l = length(imageCoord - ivec2(70,70));
    if (l >= 20 && l < 40){
        vec3 expected = vec3(0.);
        for (int i = 0; i < 1000; i++){
            float pdf_w;
            vec3 sim_w = w;
            for (int j = 0; j < 3; j++){
                vec3 phase_sample = importanceSamplePhase(parameters.phaseG, sim_w, pdf_w);
                sim_w = phase_sample;
            }
            expected += sampleSkybox(sim_w);
        }
        result = expected / 1000;
    }
    if (l < 20){
        vec3 expected = vec3(0.);
        for (int i = 0; i < 1000; i++){
            float pdf_w;
            vec3 sim_w = w;
            float weight = 1.;
            for (int j = 0; j < 3; j++){
                vec3 uni_sample = importanceSamplePhase(0, sim_w, pdf_w);
                float pdf_eval = evaluatePhase(parameters.phaseG, sim_w, uni_sample);
                //pdf_w = evaluateSkyboxPDF(10, 0, sky_sample);
                sim_w = uni_sample;
                weight *= pdf_eval / pdf_w;
            }
            expected += sampleSkybox(sim_w) * weight;
        }
        result = expected / 1000;
    }

    if (l >= 40 && l < 60){
        vec3 expected = vec3(0.);
        for (int i = 0; i < 1000; i++){
            float pdf_w;
            vec3 sim_w = w;
            float weight = 1.;
            for (int j = 0; j < 3; j++){
                vec3 sky_sample = importanceSampleSkybox(10, 0, pdf_w);
                float pdf_eval = evaluatePhase(parameters.phaseG, sim_w, sky_sample);
                //pdf_w = evaluateSkyboxPDF(10, 0, sky_sample);
                sim_w = sky_sample;
                weight *= pdf_eval / pdf_w;
            }
            expected += sampleSkybox(sim_w) * weight;
        }
        result = expected / 1000;
    }*/
    //result -= parameters.sunIntensity * 1000;

    //if (isinf(result.x)) {
    //    result = vec3(result.x, 0, result.x);
    //}


    // Accumulate result
    vec3 resultOld = frame == 0 ? vec3(0) : imageLoad(accImage, imageCoord).xyz;
    result = mix(resultOld, result, 1.0 / float(frame + 1));
    imageStore(accImage, imageCoord, vec4(result, 1));
    imageStore(resultImage, imageCoord, vec4(result,1));

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



    //vec2 octaUV = worldToOctahedralUV(w);
    //vec3 octaCol = textureLod(environmentMapOctahedralTexture, octaUV, parameters.phaseG * 8.).rrr;
    //octaCol = octaCol * (1.-cloudOnly.a) + cloudOnly.rgb;
    //octaCol = octahedralUVToWorld(octaUV);
    //octaCol -= parameters.sunIntensity * 1000;

    //if (isinf(octaCol.r)){
    //    octaCol = vec3(100000,0,100000);
    //}

    //octaCol.r = octaUV.x > .5?1.:0.;
    //octaCol.g = octaUV.y > .5?1.:0.;
    //imageStore(depthDensityImage, imageCoord, vec4(octaCol,1.));

    //vec3 resultOld = frame == 0 ? vec3(0) : imageLoad(accImage, imageCoord).xyz;
    //result += resultOld;
    //imageStore(accImage, imageCoord, vec4(result, 1));
    //imageStore(resultImage, imageCoord, vec4(result/(frame + 1),1));

    // return; Uncomment this if want to execute faster a while(true) loop PT

    // Saving the first scatter position and direction
    if (firstEvent.hasValue) {
        //imageStore(firstX, imageCoord, vec4(firstEvent.x, firstEvent.pdf_x));
        imageStore(firstW, imageCoord, vec4(firstEvent.w, firstEvent.pdf_w));
    } else {
        //imageStore(firstX, imageCoord, vec4(0));
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
