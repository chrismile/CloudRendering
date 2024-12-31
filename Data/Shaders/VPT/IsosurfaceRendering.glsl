#ifdef USE_ISOSURFACE_RENDERING
vec3 isosurfaceRendering(vec3 x, vec3 w, inout ScatterEvent firstEvent) {
    vec3 weights = vec3(1, 1, 1);
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;
#ifdef CLOSE_ISOSURFACES
    bool isFirstPointFromOutside = true;
#endif

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
            float scalarValue = sampleCloudIso(xNew);

            currentScalarSign = sign(scalarValue - parameters.isoValue);
#ifdef CLOSE_ISOSURFACES
            if (isFirstPoint) {
                isFirstPoint = false;
                lastScalarSign = sign(-parameters.isoValue);
            }
#else
            if (isFirstPoint) {
                isFirstPoint = false;
                lastScalarSign = currentScalarSign;
            }
#endif

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

#ifdef CLOSE_ISOSURFACES
            isFirstPointFromOutside = false;
#endif
            x = xNew;
            d -= t;
        }
    }

    if (foundHit) {
        vec3 surfaceNormal;
        vec3 color = getIsoSurfaceHitDirect(
                x, w, surfaceNormal
#ifdef CLOSE_ISOSURFACES
                , isFirstPointFromOutside
#endif
        );
        weights *= color;
        x += surfaceNormal * 1e-3;

        vec3 surfaceTangent;
        vec3 surfaceBitangent;
        ComputeDefaultBasis(surfaceNormal, surfaceTangent, surfaceBitangent);
        mat3 frame = mat3(surfaceTangent, surfaceBitangent, surfaceNormal);

        float weight = 1.0;

        for (int i = 0; i < parameters.numAoSamples; i++) {
            w = frame * sampleHemisphere(vec2(random(), random()));

            if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
                x += w * tMin;
                float d = tMax - tMin;
                d = min(d, parameters.maxAoDist);
#ifdef USE_AO_DIST
                vec3 xStart = x;
#endif
                while (t <= d) {
                    vec3 xNew = x + w * t;
                    float scalarValue = sampleCloudIso(xNew);

                    currentScalarSign = sign(scalarValue - parameters.isoValue);
                    if (isFirstPoint) {
                        isFirstPoint = false;
                        lastScalarSign = currentScalarSign;
                    }

                    if (lastScalarSign != currentScalarSign) {
#ifdef USE_AO_DIST
                        refineIsoSurfaceHit(xNew, x, currentScalarSign);
                        weight -= (1.0 - length(xNew - xStart) / d) / float(parameters.numAoSamples);
#else
                        weight -= 1.0 / float(parameters.numAoSamples);
#endif
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
