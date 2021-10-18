-- RayGen

#version 460
#extension GL_EXT_ray_tracing : require

layout (binding = 0) uniform CameraSettings {
    mat4 inverseViewMatrix;
    mat4 inverseProjectionMatrix;
} camera;

layout(binding = 1, rgba8) uniform image2D outputImage;

layout(binding = 2) uniform accelerationStructureEXT topLevelAS;

layout(location = 0) rayPayloadEXT bool hasHit;

void main() {
    vec2 fragNdc = 2.0 * ((vec2(gl_LaunchIDEXT.xy) + vec2(0.5)) / vec2(gl_LaunchSizeEXT.xy)) - 1.0;

    vec3 rayOrigin = (camera.inverseViewMatrix * vec4(0.0, 0.0, 0.0, 1.0)).xyz;
    vec3 rayTarget = (camera.inverseProjectionMatrix * vec4(fragNdc.xy, 1.0, 1.0)).xyz;
    vec3 rayDirection = (camera.inverseViewMatrix * vec4(normalize(rayTarget.xyz), 0.0)).xyz;

    // Alternative:
    //float scale = tan(camera.fov * 0.5);
    //vec2 rayDirCameraSpace = vec2(fragNdc.x * camera.aspectRatio * scale, fragNdc.y * scale);
    //vec3 rayDirection = normalize((camera.inverseViewMatrix * vec4(rayDirCameraSpace, -1.0, 0.0)).xyz);

    float tMin = 0.0001f;
    float tMax = 1000.0f;
    hasHit = true;
    traceRayEXT(
            topLevelAS, gl_RayFlagsTerminateOnFirstHitEXT | gl_RayFlagsOpaqueEXT | gl_RayFlagsSkipClosestHitShaderEXT,
            0xFF, 0, 0, 0, rayOrigin, tMin, rayDirection, tMax, 0);

    vec4 color;
    if (hasHit) {
        color = vec4(1.0, 0.0, 0.0, 1.0);
    } else {
        color = vec4(0.0, 1.0, 0.0, 1.0);
    }

    //ivec2 outpos = ivec2(gl_LaunchIDEXT.xy);
    //outpos.y = imageSize(outputImage).y - outpos.y - 1;
    //imageStore(outputImage, outpos, color);
    imageStore(outputImage, ivec2(gl_LaunchIDEXT.xy), color);
}

-- Miss

#version 460
#extension GL_EXT_ray_tracing : require

layout(location = 0) rayPayloadInEXT bool hasHit;

void main() {
    hasHit = false;
}
