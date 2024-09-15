#ifdef USE_ISOSURFACES
        bool foundHit = false;
        for (int subdiv = 0; subdiv < NUM_ISOSURFACE_SUBDIVISIONS; subdiv++) {
            vec3 x0 = mix(x, xNew, float(subdiv) / float(NUM_ISOSURFACE_SUBDIVISIONS));
            vec3 x1 = mix(x, xNew, float(subdiv + 1) / float(NUM_ISOSURFACE_SUBDIVISIONS));
            float scalarValue = sampleCloudIso(x1);

            currentScalarSign = sign(scalarValue - parameters.isoValue);

#ifdef CLOSE_ISOSURFACES
            if (isFirstPointFromOutside) {
                isFirstPoint = false;
                lastScalarSign = sign(-parameters.isoValue);
            } else
#endif
            if (isFirstPoint) {
                isFirstPoint = false;
                lastScalarSign = currentScalarSign;
            }

            if (lastScalarSign != currentScalarSign) {
                refineIsoSurfaceHit(x1, x0, currentScalarSign);
                x = x1;
                foundHit = getIsoSurfaceHit(
                        x, w, weights
#if defined(USE_NEXT_EVENT_TRACKING_SPECTRAL) || defined(USE_NEXT_EVENT_TRACKING)
                        , color
#endif
#ifdef CLOSE_ISOSURFACES
                        , isFirstPointFromOutside
#endif
                );
                x += w * 1e-4;
                isFirstPoint = true;
                if (foundHit) {
                    if (!firstEvent.hasValue) {
                        firstEvent.x = x;
                        firstEvent.pdf_x = 0;
                        firstEvent.w = w;
                        firstEvent.pdf_w = 0;
                        firstEvent.hasValue = true;
                        firstEvent.density = parameters.extinction.x;
                        firstEvent.depth = tMax - d + distance(x1, x);
#ifdef CLOSE_ISOSURFACES
                        firstEvent.normal = surfaceNormalGlobal;
                        firstEvent.isIsosurface = true;
#endif
                    }
                    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
                        x += w * tMin;
                        d = tMax - tMin;
                    }
                }
                //foundHit = true;
#ifdef CLOSE_ISOSURFACES
                isFirstPointFromOutside = false;
#endif
                break;
            }

#ifdef CLOSE_ISOSURFACES
            isFirstPointFromOutside = false;
#endif
        }
        if (foundHit) {
            if (weights.r < 1e-3 && weights.g < 1e-3 && weights.b < 1e-3) {
                return vec3(0.0, 0.0, 0.0);
            }
            // TODO: Russian roulette (https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting)
            //float maxWeight = max(weights.r, max(weights.g, weights.b));
            //if (maxWeight < 1e-3) {
            //float q = 0.5;
            //if (random() <= q) return vec3(0.0, 0.0, 0.0);
            //else weights = weights / (1.0 - q);
            //}
            continue;
        }
#endif
