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
#ifdef USE_NEXT_EVENT_TRACKING
float calculateTransmittance(vec3 x, vec3 w) {
    float majorant = parameters.extinction.x;
    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;
    float PA = absorptionAlbedo * parameters.extinction.x;
    float PS = scatteringAlbedo * parameters.extinction.x;

    float transmittance = 1.0;
    float rr_prob = 1.;

    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
        float d = tMax - tMin;
        float pdf_x = 1;

        while (true) {
            float t = -log(max(0.0000000001, 1 - random()))/majorant;

            if (t > d) {
                break;
            }

            if (random() > min(1., transmittance * 3.)) {
                return 0;
            }else {
                rr_prob *= min(1., transmittance * 3.);
            }

            x += w * t;

            #ifdef USE_NANOVDB
            float density = sampleCloud(accessor, x);
            #else
            float density = sampleCloud(x);
            #endif

            float sigma_a = PA * density;
            float sigma_s = PS * density;
            float sigma_n = majorant - parameters.extinction.x * density;

            float Pa = sigma_a / majorant;
            float Ps = sigma_s / majorant;
            float Pn = sigma_n / majorant;

            if (random() > Pn) {
                return 0;
            }
            //transmittance *= 1.0 - Pa - Ps;

            d -= t;
        }
    }
    return transmittance / rr_prob;
}



/**
 * For more details on spectral delta tracking, please refer to:
 * P. Kutz, R. Habel, Y. K. Li, and J. Novák. Spectral and decomposition tracking for rendering heterogeneous volumes.
 * ACM Trans. Graph., 36(4), Jul. 2017.
 */
vec3 nextEventTrackingSpectral(vec3 x, vec3 w, out ScatterEvent firstEvent) {
#ifdef USE_NANOVDB
    pnanovdb_readaccessor_t accessor = createAccessor();
#endif

    firstEvent = ScatterEvent(false, x, 0.0, w, 0.0, 0.0, 0.0);

    float majorant = maxComponent(parameters.extinction);

    vec3 weights = vec3(1, 1, 1);

    vec3 absorptionAlbedo = vec3(1, 1, 1) - parameters.scatteringAlbedo;
    vec3 scatteringAlbedo = parameters.scatteringAlbedo;
    float PA = maxComponent(absorptionAlbedo * parameters.extinction);
    float PS = maxComponent(scatteringAlbedo * parameters.extinction);

    vec3 color = vec3(0,0,0);

    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
        float d = tMax - tMin;
        while (true) {
            float t = -log(max(0.0000000001, 1 - random()))/majorant;

            if (t > d) {
                break;
            }

            x += w * t;

#ifdef USE_NANOVDB
            float density = sampleCloud(accessor, x);
#else
            float density = sampleCloud(x);
#endif

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
                    firstEvent.x =  0 * x;
                    firstEvent.pdf_x = 0; // TODO
                    firstEvent.w = vec3(0.);
                    firstEvent.pdf_w = 0;
                    firstEvent.hasValue = true;
                    firstEvent.density = density * maxComponent(parameters.extinction);
                    firstEvent.depth = tMax - d + t;
                }

                return vec3(0); // weights * sigma_a / (majorant * Pa) * L_e; // 0 - No emission
            }

            if (xi < 1 - Pn) { // scattering event
                float pdf_w;
                float pdf_skybox;

                vec3 old_w= w;

                vec3 sky_sample = importanceSampleSkybox(10, 0, pdf_skybox);
                //sky_sample = importanceSamplePhase(parameters.phaseG, w, pdf_skybox);
                float pdf_eval = evaluatePhase(parameters.phaseG, w, sky_sample);
                color += weights * calculateTransmittance(x,sky_sample) * (sampleSkybox(sky_sample) + sampleLight(sky_sample)) * pdf_eval / pdf_skybox;

                w = importanceSamplePhase(parameters.phaseG, w, pdf_w);

                /*if (random() > 1.1){
                }else{
                    // sample uniform
                    //vec3 sky_sample = importanceSamplePhase(0, w, pdf_w);
                    vec3 sky_sample = importanceSampleSkybox(10, 0, pdf_w);
                    float pdf_eval = evaluatePhase(parameters.phaseG, w, sky_sample);
                    //pdf_w = evaluateSkyboxPDF(10, 0, sky_sample);
                    w = sky_sample;
                    weights *= pdf_eval / pdf_w;
                }*/

                if (!firstEvent.hasValue) {
                    firstEvent.x = calculateTransmittance(x, sky_sample) * sampleSkybox(sky_sample) * pdf_eval / pdf_skybox;
                    //firstEvent.x = calculateTransmittance(x, w) * sampleSkybox(w);
                    firstEvent.pdf_x = 1.; // TODO
                    firstEvent.w = importanceSampleSkybox(10, 0, pdf_w);
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

    if (!firstEvent.hasValue){
        color += sampleSkybox(w) + sampleLight(w);
    }
    //return min(weights, vec3(100000, 100000, 100000)) * (sampleSkybox(w) + sampleLight(w));
    return color * .5 + .5 * min(weights, vec3(100000, 100000, 100000)) * (sampleSkybox(w) + sampleLight(w));
}


vec3 nextEventTracking(vec3 x, vec3 w, out ScatterEvent firstEvent) {
    firstEvent = ScatterEvent(false, x, 0.0, w, 0.0, 0.0, 0.0);

    #ifdef USE_NANOVDB
    pnanovdb_readaccessor_t accessor = createAccessor();
    #endif

    float majorant = parameters.extinction.x;
    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;
    float PA = absorptionAlbedo * parameters.extinction.x;
    float PS = scatteringAlbedo * parameters.extinction.x;

    float transmittance = 1.0;

    float bw_phase = 1.;
    float cum_wp = 1.;
    vec3 color = vec3(0.);

    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
        float d = tMax - tMin;
        float pdf_x = 1;

        while (true) {
            float t = -log(max(0.0000000001, 1 - random()))/majorant;

            if (t > d) {
                break;
            }

            x += w * t;

            #ifdef USE_NANOVDB
            float density = sampleCloud(accessor, x);
            #else
            float density = sampleCloud(x);
            #endif

            float sigma_a = PA * density;
            float sigma_s = PS * density;
            float sigma_n = majorant - parameters.extinction.x * density;

            float Pa = sigma_a / majorant;
            float Ps = sigma_s / majorant;
            float Pn = sigma_n / majorant;

            float xi = random();

            //transmittance *= 1.0 - Pa;
            if (xi < Pa) {
                return color;
                //return vec3(0); // weights * sigma_a / (majorant * Pa) * L_e; // 0 - No emission
            }

            if (xi < 1 - Pn)// scattering event
            {
                float pdf_w, pdf_nee;
                vec3 next_w = importanceSamplePhase(parameters.phaseG, w, pdf_w);
                vec3 nee_w = importanceSampleSkybox(10, 0, pdf_nee);

                //next_w = importanceSamplePhase(0.5, w, pdf_w);
                //next_w = importanceSampleSkybox(10, 0, pdf_w);
                float pdf_nee_phase = evaluatePhase(parameters.phaseG, w, nee_w);
                float pdf_phase_nee = evaluateSkyboxPDF(10, 0, next_w);
                w = next_w;
                //transmittance *= pdf_eval / pdf_w;

                bw_phase = pdf_w * pdf_w / (pdf_w * pdf_w + pdf_phase_nee * pdf_phase_nee);
                float bw_nee = pdf_nee * pdf_nee / (pdf_nee * pdf_nee + pdf_phase_nee * pdf_nee_phase);

                color += bw_nee * transmittance * calculateTransmittance(x,nee_w) * (sampleSkybox(nee_w) + sampleLight(nee_w)) * pdf_nee_phase / pdf_nee;
                //color += bw_phase * transmittance * calculateTransmittance(x,next_w) * (sampleSkybox(next_w) + sampleLight(next_w));
                cum_wp *= bw_phase;
                //return color;
                pdf_x *= exp(-majorant * t) * majorant * density;

                if (!firstEvent.hasValue) {
                    firstEvent.x = x;
                    //firstEvent.x = calculateTransmittance(x, sky_sample) * sampleSkybox(sky_sample) * pdf_eval / pdf_skybox;
                    //firstEvent.x = calculateTransmittance(x, w) * sampleSkybox(w);
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
    if (!firstEvent.hasValue){
        //color += sampleSkybox(w) + sampleLight(w);
    }
    return color + bw_phase * transmittance * (sampleSkybox(w) + sampleLight(w));
    return transmittance * (sampleSkybox(w) + sampleLight(w));
}
#endif