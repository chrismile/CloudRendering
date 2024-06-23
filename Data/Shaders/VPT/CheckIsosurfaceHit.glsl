#ifdef USE_ISOSURFACES
        bool foundHit = false;
        for (int subdiv = 0; subdiv < NUM_ISOSURFACE_SUBDIVISIONS; subdiv++) {
            vec3 x0 = mix(x, xNew, float(subdiv) / float(NUM_ISOSURFACE_SUBDIVISIONS));
            vec3 x1 = mix(x, xNew, float(subdiv + 1) / float(NUM_ISOSURFACE_SUBDIVISIONS));
            float scalarValue = sampleCloudIso(x1);

            currentScalarSign = sign(scalarValue - parameters.isoValue);
            if (isFirstPoint) {
                isFirstPoint = false;
                lastScalarSign = currentScalarSign;
            }

            if (lastScalarSign != currentScalarSign) {
                refineIsoSurfaceHit(x1, x0, currentScalarSign);
                x = x1;
#if defined(USE_NEXT_EVENT_TRACKING_SPECTRAL) || defined(USE_NEXT_EVENT_TRACKING)
                foundHit = getIsoSurfaceHit(x, w, weights, color);
#else
                foundHit = getIsoSurfaceHit(x, w, weights);
#endif
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
                    }
                    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
                        x += w * tMin;
                        d = tMax - tMin;
                    }
                }
                //foundHit = true;
                break;
            }
        }
        if (foundHit) {
            if (weights.r < 1e-3 && weights.g < 1e-3 && weights.b < 1e-3) {
                return vec3(0.0, 0.0, 0.0);
            }
            // TODO: Russian roulette (https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Russian_Roulette_and_Splitting)
            //float q = max(weights.r, max(weights.g, weights.b));
            //if (random() <= q) return vec3(0.0, 0.0, 0.0);
            //else weights = (weights - q) / (1.0 - q);
            continue;
        }
#endif
