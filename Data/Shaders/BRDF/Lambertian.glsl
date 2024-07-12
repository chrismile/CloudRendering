// ---------- BRDF Helper Functions ----------

// 1. Helper functions for sampling


// 2. Helper functions for evaluation


// ---------- BRDF Interface ----------

// Compute Importance Sampled light vector


// Evaluate BRDF with compensation of Importance Sampling and a fixed sampling PDF


// Evaluate BRDF and prepare for Importance Sampling compensation


// Combined Call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {

}