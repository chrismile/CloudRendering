vec2 computeDepthBlended(vec3 x, vec3 w) {
    const float majorant = parameters.extinction.x;
    const float stepSize = 0.001;
    const float attenuationCoefficient = majorant * stepSize;
    // stepSize * attenuationCoefficient

#ifdef USE_ISOSURFACES
    float lastScalarSign, currentScalarSign;
    bool isFirstPoint = true;
#endif

    float depthAccum = 0.0, alphaAccum = 0.0;
    float tMin, tMax;
    if (rayBoxIntersect(parameters.boxMin, parameters.boxMax, x, w, tMin, tMax)) {
        float t = tMin;
        while (true) {
            //t += -log(max(0.0000000001, 1 - random())) / majorant;
            t += stepSize;
            if (t > tMax || alphaAccum > 0.999) {
                break;
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
                float scalarValue = sampleCloudDirect(x1);

                currentScalarSign = sign(scalarValue - parameters.isoValue);
                if (isFirstPoint) {
                    isFirstPoint = false;
                    lastScalarSign = currentScalarSign;
                }

                if (lastScalarSign != currentScalarSign) {
                    refineIsoSurfaceHit(x1, x0, currentScalarSign);
                    x = x1;
                    density = 1e6;
                }
            }
#endif

            float alpha = 1.0 - exp(-density * stepSize * attenuationCoefficient);
            depthAccum = depthAccum + (1.0 - alphaAccum) * alpha * t;
            alphaAccum = alphaAccum + (1.0 - alphaAccum) * alpha;
        }
    }

    return vec2(depthAccum / max(alphaAccum, 1e-5), alphaAccum);
    //return vec2(depthAccum, alphaAccum);
}
