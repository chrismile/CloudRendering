// ---------- BRDF Helper Functions ----------

// 1. Helper functions for sampling


// 2. Helper functions for evaluation


// ---------- BRDF Interface ----------

// Compute Importance Sampled light vector
vec3 sampleBrdf(mat3 frame, out flags hitFlags) {
    hitFlags.specularHit = false;
    hitFlags.clearcoatHit = false;
	return frame * sampleHemisphere(vec2(random(), random()));
}

// Evaluate BRDF with compensation of Importance Sampling and a fixed sampling PDF


// Evaluate BRDF and prepare for Importance Sampling compensation
vec3 evaluateBrdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColor, float samplingPDF) {
    // http://www.thetenthplanet.de/archives/255
    vec3 halfwayVector = normalize(viewVector + lightVector);
    const float n = 10.0;
    float norm = clamp(
            (n + 2.0) / (4.0 * M_PI * (exp2(-0.5 * n))),
            (n + 2.0) / (8.0 * M_PI), (n + 4.0) / (8.0 * M_PI));

    return 2.0 * M_PI * isoSurfaceColor * (pow(max(dot(normalVector, halfwayVector), 0.0), n) / norm);
}

// Combined Call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {
    
    lightVector = sampleBrdf(frame, hitFlags);
    return evaluateBrdf(viewVector, lightVector, normalVector, isoSurfaceColor, samplingPDF);
}