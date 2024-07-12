// ---------- BRDF Helper Functions ----------

// 1. Helper functions for sampling


// 2. Helper functions for evaluation


// ---------- BRDF Interface ----------

// Compute Importance Sampled light vector
vec3 sampleBrdf(mat3 frame, flags hitFlags) {
    hitFlags.specularHit = false;
    hitFlags.clearcoatHit = false;
    
    #ifdef UNIFORM_SAMPLING
    
    // Sampling PDF: 1/(2pi)
    return frame * sampleHemisphere(vec2(random(), random()));
    
    #else
    
    // Sampling PDF: cos(theta) / pi
    return frame * sampleHemisphereCosineWeighted(vec2(random(), random()));
    
    #endif
}

// Evaluate BRDF with compensation of Importance Sampling and a fixed sampling PDF
vec3 evaluateBrdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColor) {
    float theta = dot(lightVector, normalVector);
    #ifdef UNIFORM_SAMPLING
    
    // Sampling PDF: 1/(2pi)
    float pdf = 0.5;
    
    #else
    
    // Sampling PDF: cos(theta) / pi
    float pdf = theta;

    #endif

    return (isoSurfaceColor * theta) / pdf;
}

// Evaluate BRDF and prepare for Importance Sampling compensation


// Combined Call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {
    lightVector = sampleBrdf(frame, hitFlags);
    return evaluateBrdf(viewVector, lightVector, normalVector, isoSurfaceColor);
}