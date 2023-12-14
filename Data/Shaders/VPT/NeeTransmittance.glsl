// Pathtracing with Delta tracking and Spectral tracking.
#if defined(USE_NEXT_EVENT_TRACKING) || defined(USE_NEXT_EVENT_TRACKING_SPECTRAL)
float calculateTransmittance(vec3 x, vec3 w) {
    float majorant = parameters.extinction.x;
    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;
    float PA = absorptionAlbedo * parameters.extinction.x;
    float PS = scatteringAlbedo * parameters.extinction.x;

    float transmittance = 1.0;
    float rr_factor = 1.;

#ifdef USE_ISOSURFACES
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;
#endif

    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
        float d = tMax - tMin;
        float pdf_x = 1;

        float targetThroughput = .1;

        while (true) {
            float t = -log(max(0.0000000001, 1 - random()))/majorant;

            if (t > d) {
                break;
            }

            /*if (transmittance * rr_factor < targetThroughput) {
                float rr_survive_prob = (transmittance * rr_factor) / targetThroughput;
                if (random() > rr_survive_prob) {
                    return 0;
                }else {
                    rr_factor /= rr_survive_prob;
                }
            }*/

            vec3 xNew = x + w * t;

#ifdef USE_TRANSFER_FUNCTION
            vec4 densityEmission = sampleCloudDensityEmission(xNew);
            float density = densityEmission.a;
#else
            float density = sampleCloud(xNew);
#endif

#ifdef USE_ISOSURFACES
            //#if defined(ISOSURFACE_TYPE_DENSITY) && !defined(USE_TRANSFER_FUNCTION)
            //            float scalarValue = density;
            //#else
            //            float scalarValue = sampleCloudDirect(xNew);
            //#endif
            const int isoSubdivs = 2;
            bool foundHit = false;
            for (int subdiv = 0; subdiv < isoSubdivs; subdiv++) {
                vec3 x0 = mix(x, xNew, float(subdiv) / float(isoSubdivs));
                vec3 x1 = mix(x, xNew, float(subdiv + 1) / float(isoSubdivs));
                float scalarValue = sampleCloudDirect(x1);

                currentScalarSign = sign(scalarValue - parameters.isoValue);
                if (isFirstPoint) {
                    isFirstPoint = false;
                    lastScalarSign = currentScalarSign;
                }

                if (lastScalarSign != currentScalarSign) {
                    refineIsoSurfaceHit(x1, x0, currentScalarSign);
                    x = x1;
                    return 0;
                }
            }
#endif

            x = xNew;

            float sigma_a = PA * density;
            float sigma_s = PS * density;
            float sigma_n = majorant - parameters.extinction.x * density;

            float Pa = sigma_a / majorant;
            float Ps = sigma_s / majorant;
            float Pn = sigma_n / majorant;

            if (random() > Pn) {
                // switches between ratio and delta tracking
                return 0;
            }
            //transmittance *= 1.0 - Pa - Ps;

            d -= t;
        }
    }
    return transmittance * rr_factor;
}
#endif
