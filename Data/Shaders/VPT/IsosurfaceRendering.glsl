#ifdef USE_ISOSURFACE_RENDERING
vec3 isosurfaceRendering(vec3 x, vec3 w, out ScatterEvent firstEvent) {
    firstEvent = ScatterEvent(false, x, 0.0, w, 0.0, 0.0, 0.0);

    vec3 weights = vec3(1, 1, 1);
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;

    ivec3 voxelGridSize = textureSize(gridImage, 0);
    vec3 boxDelta = parameters.boxMax - parameters.boxMin;
    vec3 voxelSize3d = boxDelta / voxelGridSize;
    float t = max(voxelSize3d.x, max(voxelSize3d.y, voxelSize3d.z)) * parameters.isoStepWidth;

    bool foundHit = false;
    int i = 0;
    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        x += w * tMin;
        float d = tMax - tMin;
        while (t <= d) {
            vec3 xNew = x + w * t;
            float scalarValue = sampleCloudDirect(xNew);

            currentScalarSign = sign(scalarValue - parameters.isoValue);
            if (isFirstPoint) {
                isFirstPoint = false;
                lastScalarSign = currentScalarSign;
            }

            if (lastScalarSign != currentScalarSign) {
                if (!firstEvent.hasValue) {
                    firstEvent.x = x;
                    firstEvent.pdf_x = 0;
                    firstEvent.w = vec3(0.);
                    firstEvent.pdf_w = 0;
                    firstEvent.hasValue = true;
                    firstEvent.density = parameters.extinction.x;
                    firstEvent.depth = tMax - d + t;
                }
                refineIsoSurfaceHit(xNew, x, currentScalarSign);
                x = xNew;
                foundHit = true;
                break;
            }

            x = xNew;
            d -= t;
        }
    }

    if (foundHit) {
        vec3 surfaceNormal;
        vec3 color = getIsoSurfaceHitDirect(x, w, surfaceNormal);
        weights *= color;
        x += surfaceNormal * 1e-4;

        vec3 surfaceTangent;
        vec3 surfaceBitangent;
        ComputeDefaultBasis(surfaceNormal, surfaceTangent, surfaceBitangent);
        mat3 frame = mat3(surfaceTangent, surfaceBitangent, surfaceNormal);

        const int numAoSamples = 4;
        const float MAX_DIST = 0.05;
        float weight = 1.0;

        for (int i = 0; i < numAoSamples; i++) {
            w = frame * sampleHemisphere(vec2(random(), random()));

            if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
                x += w * tMin;
                float d = tMax - tMin;
                d = max(d, MAX_DIST);
                while (t <= d) {
                    vec3 xNew = x + w * t;
                    float scalarValue = sampleCloudDirect(xNew);

                    currentScalarSign = sign(scalarValue - parameters.isoValue);
                    if (isFirstPoint) {
                        isFirstPoint = false;
                        lastScalarSign = currentScalarSign;
                    }

                    if (lastScalarSign != currentScalarSign) {
                        weight -= 1.0 / float(numAoSamples);
                        break;
                    }

                    x = xNew;
                    d -= t;
                }
            }
        }

        return weights * weight;
        //return surfaceNormal;
    }

    return sampleSkybox(w) + sampleLight(w);
}
#endif
