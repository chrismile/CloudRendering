// ---------- BRDF Helper Functions ----------

// 1. Helper functions for sampling
vec3 sample_GGX(vec3 viewVector, float roughness, mat3 frameMatrix) {
    // https://www.youtube.com/watch?v=MkFS6lw6aEs
    // Generate random u and v between 0.0 and 1.0
    float u = random();
    float v = random();

    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;

    // Compute spherical angles
    float theta = acos(sqrt((1-u)/(u*(alpha2 - 1.0)+1.0)));
    float phi = 2 * M_PI * v;
    
    //pdf_ggx = cos(theta);
    // Compute halfway vector h
    vec3 halfwayVector = frameMatrix*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

    // Compute light Vector l
    vec3 lightVector = 2*dot(viewVector,halfwayVector)*halfwayVector - viewVector;
    return lightVector;
}

vec3 sample_Lambertian(vec3 viewVector, mat3 frameMatrix) {
    // https://www.youtube.com/watch?v=xFsJMUS94Fs
    // Generate random u and v between 0.0 and 1.0
    float u = random();
    float v = random();

    // Compute spherical angles
    float theta = asin(sqrt(u));
    float phi = 2 * M_PI * v;

    // pdf
    //pdf_diffuse = (sin(theta)*cos(theta)/M_PI);
    vec3 halfwayVector = frameMatrix*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

    // Compute light Vector l
    vec3 lightVector = 2*dot(viewVector,halfwayVector)*halfwayVector - viewVector;
    return lightVector;
}

// 2. Helper functions for evaluation
// Source: https://www.youtube.com/watch?v=gya7x9H3mV0
vec3 fresnelSchlick(float cosTheta, vec3 F0) {
    return F0 + (1-F0) * pow((1.0 - cosTheta), 5.0);
}

// Source: https://www.youtube.com/watch?v=gya7x9H3mV0
float D_GGX(float NdotH, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float NdotH2 = NdotH * NdotH;
    float b = (NdotH2 * (alpha2 - 1.0) + 1.0);
    return (1 / M_PI) * alpha2 / (b * b);
}

float D_Beckmann(float NdotH, float roughness) {
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;
    float first = 1.0 / (M_PI*alpha2*pow(NdotH,4.0));
    float ex = exp((NdotH*NdotH -1)/(alpha2*NdotH*NdotH));
    return first*ex;
}

// Source: https://www.youtube.com/watch?v=gya7x9H3mV0
float G1_GGX_Schlick(float NdotV, float roughness) {
    float alpha = roughness * roughness;
    float k = alpha / 2.0;
    return max(NdotV, 0.001) / (NdotV * (1.0 - k) + k);
}

// Source: https://www.youtube.com/watch?v=gya7x9H3mV0
float G_Smith(float NdotV, float NdotL, float roughness) {
    return G1_GGX_Schlick(NdotL, roughness) * G1_GGX_Schlick(NdotV, roughness);
}


// ---------- BRDF Interface ----------

// Compute Importance Sampled light vector
vec3 sampleBrdf(float metallic, float specular, float roughness, vec3 viewVector, mat3 frameMatrix, out flags hitFlags) {
    // Idea adapted from https://schuttejoe.github.io/post/disneybsdf/
    // Calculate probabilies for sampling the lobes
    float metal = metallic;
    float spec = (1.0 - roughness) * (1.0 + specular);
    float dielectric = (1.0 - metallic);

    float specularW = metal + spec;
    float diffuseW = dielectric;

    float norm = 1.0/(specularW + diffuseW);

    float specularP = specularW * norm;
    float diffuseP = diffuseW * norm;

    float u = random();

    if(u < specularP) {
        hitFlags.specularHit = true;
        return sample_GGX(viewVector, roughness, frameMatrix);
    } else {
        hitFlags.specularHit = false;
        return sample_Lambertian(viewVector, frameMatrix);
    }
}

// Evaluate BRDF with compensation of Importance Sampling and a fixed sampling PDF
vec3 evaluateBrdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColorDef, flags hitFlags, out float samplingPDF) {
    vec3 halfwayVector = normalize(lightVector + viewVector);

    // ----------- Evaluating the BRDF
    // Base Angles
    float NdotH = dot(halfwayVector, normalVector);
    float LdotH = dot(lightVector, halfwayVector);
    float VdotN = dot(viewVector, normalVector);
    float LdotN = dot(lightVector, normalVector);
    float LdotV = dot(lightVector, normalVector);
    float VdotH = dot(viewVector, halfwayVector);

    vec3 baseColor = isoSurfaceColorDef;
    // Diffuse:
    // Specular F: Schlick Fresnel Approximation
    vec3 f0 = vec3(0.16 * (sqr(parameters.specular)));
    f0 = mix(f0, baseColor, parameters.metallic);

    vec3 F = fresnelSchlick(VdotH, f0);
    // Specular D: GGX Distribuition
    
    float D = D_GGX(NdotH, clamp(parameters.roughness,0.05, 1.0));
    // Speuclar G: G_Smith with G_1Smith-GGX
    float G = G_Smith(VdotN, LdotN, clamp(parameters.roughness,0.05, 1.0));
    // Result
    vec3 spec = (F * G * VdotH)/(NdotH*VdotN);
    
    // Diffuse Part
    // Importance Sampling pdf: 1/PI sin(theta) cos(theta)
    vec3 rhoD = baseColor;
    
    // Debug: if (gl_GlobalInvocationID.x == 500 && gl_GlobalInvocationID.y == 500) { debugPrintfEXT("Specular D: %f Specular F: %f Specular G: %f", D, F, G); }
    rhoD *= vec3(1.0) - F;
    rhoD *= (1.0 - parameters.metallic);
    vec3 diff = rhoD;
    // Whitout importance sampling: rhoD *= (1.0 / M_PI)
    //diff /= pdf_diffuse;
    float sinThetaH = sqrt(1-(NdotH*NdotH));
    float cosThetaH = NdotH;

    if (hitFlags.specularHit) {
        samplingPDF = D;
    } else {
        samplingPDF = (1.0/M_PI);
    }

    // ----------- Weighting in the PDF
    vec3 colorOut = diff + spec;
    return colorOut;
}

// Evaluate BRDF and prepare for Importance Sampling compensation
vec3 evaluateBrdfPdf(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColorDef) {
    vec3 halfwayVector = normalize(lightVector + viewVector);

    // ----------- Evaluating the BRDF
    // Base Angles
    float NdotH = dot(halfwayVector, normalVector);
    float LdotH = dot(lightVector, halfwayVector);
    float VdotN = dot(viewVector, normalVector);
    float LdotN = dot(lightVector, normalVector);
    float LdotV = dot(lightVector, normalVector);
    float VdotH = dot(viewVector, halfwayVector);

    vec3 baseColor = isoSurfaceColorDef;
    // Diffuse:
    // Specular F: Schlick Fresnel Approximation
    vec3 f0 = vec3(0.16 * (sqr(parameters.specular)));
    f0 = mix(f0, baseColor, parameters.metallic);

    vec3 F = fresnelSchlick(VdotH, f0);
    // Specular D: GGX Distribuition
    
    float D = D_GGX(NdotH, clamp(parameters.roughness,0.05, 1.0));
    // Speuclar G: G_Smith with G_1Smith-GGX
    float G = G_Smith(VdotN, LdotN, clamp(parameters.roughness,0.05, 1.0));
    // Result
    float sinThetaH = sqrt(1-(NdotH*NdotH));
    vec3 spec = (F * G * D * VdotH * sinThetaH)/(VdotN);
    
    // Diffuse Part
    // Importance Sampling pdf: 1/PI sin(theta) cos(theta)
    vec3 rhoD = baseColor;
    rhoD *= sinThetaH * NdotH;
    
    // Debug: if (gl_GlobalInvocationID.x == 500 && gl_GlobalInvocationID.y == 500) { debugPrintfEXT("Specular D: %f Specular F: %f Specular G: %f", D, F, G); }
    rhoD *= vec3(1.0) - F;
    rhoD *= (1.0 - parameters.metallic);
    rhoD *= (1.0 / M_PI);

    vec3 diff = rhoD;
    //diff /= pdf_diffuse;

    // ----------- Weighting in the PDF
    vec3 colorOut = diff + spec;
    return colorOut;
}

vec3 evaluateBrdfNee(vec3 viewVector, vec3 dirOut, vec3 dirNee, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, vec3 isoSurfaceColor, bool useMIS, float samplingPDF, flags hitFlags, out float pdfSamplingOut, out float pdfSamplingNee) {
        vec3 halfwayVector = normalize(dirOut + viewVector);
        float cosThetaH = dot(halfwayVector, normalVector);
        float sinThetaH = sqrt(1.0/(cosThetaH*cosThetaH));

        vec3 halfwayVectorNee  = normalize(dirNee + viewVector);
        float cosThetaHNee = dot(halfwayVectorNee, normalVector);
        float sinThetaHNee = sqrt(1.0 - (cosThetaHNee*cosThetaHNee));

        pdfSamplingOut = samplingPDF * cosThetaH * sinThetaH;
        pdfSamplingNee = samplingPDF * cosThetaHNee * sinThetaHNee;

        return evaluateBrdfPdf(viewVector, dirNee, normalVector, isoSurfaceColor) * dot(dirNee, normalVector);
}

// Combined Call to importance sample and evaluate BRDF

vec3 computeBrdf(vec3 viewVector, out vec3 lightVector, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, mat3 frame, vec3 isoSurfaceColor, out flags hitFlags, out float samplingPDF) {
    // Source: https://www.youtube.com/watch?v=gya7x9H3mV0
    float roughness = clamp(parameters.roughness,0.05, 1.0);

    // ----------- Importance Sampling
    // Importance Sampling for Diffuse: PDF 1/PI * cos(theta) * sin(theta)

    // Sampling and evaluating
    lightVector = sampleBrdf(parameters.metallic, parameters.specular, roughness, viewVector, frame, hitFlags);
    return evaluateBrdf(viewVector, lightVector, normalVector, isoSurfaceColor, hitFlags, samplingPDF);
}