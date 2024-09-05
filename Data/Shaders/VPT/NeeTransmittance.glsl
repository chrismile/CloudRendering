// Pathtracing with Delta tracking and Spectral tracking.
#if defined(USE_NEXT_EVENT_TRACKING) || defined(USE_NEXT_EVENT_TRACKING_SPECTRAL)

#ifndef USE_OCCUPANCY_GRID
float calculateTransmittance(vec3 x, vec3 w) {
    float majorant = parameters.extinction.x;
    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;
    float PA = absorptionAlbedo * parameters.extinction.x;
    float PS = scatteringAlbedo * parameters.extinction.x;

    float transmittance = 1.0;
    float rr_factor = 1.0;

#ifdef USE_ISOSURFACES
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;
#endif

    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
        float d = tMax - tMin;

        float targetThroughput = 0.1;

        while (true) {
            float t = -log(max(0.0000000001, 1 - random())) / majorant;

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
            //            float scalarValue = sampleCloudIso(xNew);
            //#endif
            const int isoSubdivs = 2;
            bool foundHit = false;
            for (int subdiv = 0; subdiv < isoSubdivs; subdiv++) {
                vec3 x0 = mix(x, xNew, float(subdiv) / float(isoSubdivs));
                vec3 x1 = mix(x, xNew, float(subdiv + 1) / float(isoSubdivs));
                float scalarValue = sampleCloudIso(x1);

                currentScalarSign = sign(scalarValue - parameters.isoValue);
                if (isFirstPoint) {
                    isFirstPoint = false;
                    lastScalarSign = currentScalarSign;
                }

                if (lastScalarSign != currentScalarSign) {
                    refineIsoSurfaceHit(x1, x0, currentScalarSign);
                    x = x1;
#define USE_TRANSMITTANCE_DELTA_TRACKING
#if defined(ISOSURFACE_USE_TF) && defined(USE_TRANSFER_FUNCTION)
                    float opacity = 1.0 - sampleIsoOpacityTF(x);
                    if (opacity < 1e-4) {
                        return 0;
                    }
#ifdef USE_TRANSMITTANCE_DELTA_TRACKING
                    if (random() > opacity) {
                        return 0;
                    }
#else
                    transmittance *= opacity;
#endif
#endif
                    isFirstPoint = true;
                    break;
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

#ifdef USE_TRANSMITTANCE_DELTA_TRACKING
            if (random() > Pn) {
                return 0;
            }
#else
            transmittance *= Pn;
#endif

            d -= t;
        }
    }
    return transmittance * rr_factor;
}
#else // defined(USE_OCCUPANCY_GRID) - use empty space skipping
float calculateTransmittance(vec3 x, vec3 w) {
    float majorant = parameters.extinction.x;
    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;
    float PA = absorptionAlbedo * parameters.extinction.x;
    float PS = scatteringAlbedo * parameters.extinction.x;

    float transmittance = 1.0;
    float rr_factor = 1.0;

#ifdef USE_ISOSURFACES
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;
#endif

    const vec3 EPSILON_VEC = vec3(1e-6);
    float tMinVal, tMaxVal;
    if (rayBoxIntersect(parameters.boxMin + EPSILON_VEC, parameters.boxMax - EPSILON_VEC, x, w, tMinVal, tMaxVal)) {
        vec3 boxDelta = parameters.boxMax - parameters.boxMin;
        vec3 superVoxelSize = parameters.superVoxelSize * boxDelta / parameters.gridResolution;

        x += w * tMinVal;
        vec3 startPoint = (x - parameters.boxMin) / boxDelta * parameters.gridResolution / parameters.superVoxelSize;
        ivec3 superVoxelIndex = ivec3(floor(startPoint));

        // We know that the first super voxel cannot be empty, as we start on a volume or surface event.
        ivec3 cachedSuperVoxelIndex = superVoxelIndex;
        bool isSuperVoxelEmpty = false;

        float targetThroughput = 0.1;

        // Loop over all super voxels along the ray.
        while (all(greaterThanEqual(superVoxelIndex, ivec3(0))) && all(lessThan(superVoxelIndex, parameters.superVoxelGridSize))) {
            vec3 minSuperVoxelPos = parameters.boxMin + superVoxelIndex * superVoxelSize;
            vec3 maxSuperVoxelPos = minSuperVoxelPos + superVoxelSize;

            float tMinSuperVoxel = 0.0, tMaxSuperVoxel = 0.0;
            rayBoxIntersect(minSuperVoxelPos, maxSuperVoxelPos, x, w, tMinSuperVoxel, tMaxSuperVoxel);
            float d_max = tMaxSuperVoxel - tMinSuperVoxel; // + 1e-7
            x += w * tMinSuperVoxel;

            if (cachedSuperVoxelIndex != superVoxelIndex) {
                isSuperVoxelEmpty = imageLoad(occupancyGridImage, superVoxelIndex).x == 0u;
                cachedSuperVoxelIndex = superVoxelIndex;
            }
            //isSuperVoxelEmpty = true;
            if (isSuperVoxelEmpty) {
                x += w * d_max;
            } else {
                float t = 0.0;
                while (true) {
                    t -= log(max(0.0000000001, 1 - random())) / majorant;

                    if (t >= d_max) {
                        x = x + d_max * w;
                        break; // null collision, proceed to next super voxel
                    }

                    vec3 xNew = x + w * t;

#ifdef USE_TRANSFER_FUNCTION
                    vec4 densityEmission = sampleCloudDensityEmission(xNew);
                    float density = densityEmission.a;
#else
                    float density = sampleCloud(xNew);
#endif

#ifdef USE_ISOSURFACES
                    const int isoSubdivs = 2;
                    bool foundHit = false;
                    for (int subdiv = 0; subdiv < isoSubdivs; subdiv++) {
                        vec3 x0 = mix(x, xNew, float(subdiv) / float(isoSubdivs));
                        vec3 x1 = mix(x, xNew, float(subdiv + 1) / float(isoSubdivs));
                        float scalarValue = sampleCloudIso(x1);

                        currentScalarSign = sign(scalarValue - parameters.isoValue);
                        if (isFirstPoint) {
                            isFirstPoint = false;
                            lastScalarSign = currentScalarSign;
                        }

                        if (lastScalarSign != currentScalarSign) {
                            refineIsoSurfaceHit(x1, x0, currentScalarSign);
                            x = x1;
#define USE_TRANSMITTANCE_DELTA_TRACKING
#if defined(ISOSURFACE_USE_TF) && defined(USE_TRANSFER_FUNCTION)
                            float opacity = 1.0 - sampleIsoOpacityTF(x);
                            if (opacity < 1e-4) {
                                return 0;
                            }
#ifdef USE_TRANSMITTANCE_DELTA_TRACKING
                            if (random() > opacity) {
                                return 0;
                            }
#else
                            transmittance *= opacity;
#endif
#endif
                            isFirstPoint = true;
                            break;
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

#ifdef USE_TRANSMITTANCE_DELTA_TRACKING
                    if (random() > Pn) {
                        return 0;
                    }
#else
                    transmittance *= Pn;
#endif
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

    return transmittance * rr_factor;
}
#endif


#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
float calculateTransmittanceDistance(vec3 x, vec3 w, float maxDist) {
    float majorant = parameters.extinction.x;
    float absorptionAlbedo = 1.0 - parameters.scatteringAlbedo.x;
    float scatteringAlbedo = parameters.scatteringAlbedo.x;
    float PA = absorptionAlbedo * parameters.extinction.x;
    float PS = scatteringAlbedo * parameters.extinction.x;

    float transmittance = 1.0;
    float rr_factor = 1.0;

#ifdef USE_ISOSURFACES
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;
#endif

    float tMin, tMax;
    float currDist = 0.0;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
        currDist += tMin;
        float d = tMax - tMin;

        float targetThroughput = 0.1;

        while (true) {
            float t = -log(max(0.0000000001, 1 - random())) / majorant;
            currDist += t;

            if (t > d || currDist > maxDist) {
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
            //            float scalarValue = sampleCloudIso(xNew);
            //#endif
            const int isoSubdivs = 2;
            bool foundHit = false;
            for (int subdiv = 0; subdiv < isoSubdivs; subdiv++) {
                vec3 x0 = mix(x, xNew, float(subdiv) / float(isoSubdivs));
                vec3 x1 = mix(x, xNew, float(subdiv + 1) / float(isoSubdivs));
                float scalarValue = sampleCloudIso(x1);

                currentScalarSign = sign(scalarValue - parameters.isoValue);
                if (isFirstPoint) {
                    isFirstPoint = false;
                    lastScalarSign = currentScalarSign;
                }

                if (lastScalarSign != currentScalarSign) {
                    refineIsoSurfaceHit(x1, x0, currentScalarSign);
                    x = x1;
#define USE_TRANSMITTANCE_DELTA_TRACKING
#if defined(ISOSURFACE_USE_TF) && defined(USE_TRANSFER_FUNCTION)
                    float opacity = 1.0 - sampleIsoOpacityTF(x);
                    if (opacity < 1e-4) {
                        return 0;
                    }
#ifdef USE_TRANSMITTANCE_DELTA_TRACKING
                    if (random() > opacity) {
                        return 0;
                    }
#else
                    transmittance *= opacity;
#endif
#endif
                    isFirstPoint = true;
                    break;
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

#ifdef USE_TRANSMITTANCE_DELTA_TRACKING
            if (random() > Pn) {
                return 0;
            }
#else
            transmittance *= Pn;
#endif

            d -= t;
        }
    }
    return transmittance * rr_factor;
}
#endif

#endif
