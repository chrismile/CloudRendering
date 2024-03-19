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
                        firstEvent.w = vec3(0.);
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
            continue;
        }
#endif
