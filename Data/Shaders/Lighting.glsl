/**
 * Simplified Blinn-Phong shading assuming the ambient and diffuse color are equal and the specular color is white.
 * Additionaly, Fresnel shading is used to enhance the outlines.
 * Assumes the following global variables are given: cameraPosition.
 * The camera position is assumed to be the source of a point light.
*/
vec3 blinnPhongShadingSurface(
        in vec3 baseColor, in vec3 fragmentPositionWorld, in vec3 fragmentNormal) {
    // Blinn-Phong Shading
    const vec3 lightColor = vec3(1.0);
    const vec3 ambientColor = baseColor;
    const vec3 diffuseColor = ambientColor;
    vec3 phongColor = vec3(0.0);

#ifdef SURFACE_BRDF_AMBIENT
    const float kA = 1.0;
    const float kD = 0.0;
    const float kS = 0.0;
#elif defined(SURFACE_BRDF_LAMBERTIAN)
    const float kA = 0.4;
    const float kD = 0.6;
    const float kS = 0.0;
#else
    const float kA = 0.4;
    const float kD = 0.6;
    const float kS = 0.2;
#endif
    const float s = 30;
    const vec3 Ia = kA * ambientColor;

    const vec3 n = normalize(fragmentNormal);
    const vec3 v = normalize(cameraPosition - fragmentPositionWorld);
    const vec3 l = v;//normalize(lightDirection);
    const vec3 h = normalize(v + l);

    vec3 Id = kD * clamp(abs(dot(n, l)), 0.0, 1.0) * diffuseColor;
    vec3 Is = kS * pow(clamp(abs(dot(n, h)), 0.0, 1.0), s) * lightColor;

    phongColor = Ia + Id + Is;

    return phongColor;
}
