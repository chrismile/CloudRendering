/**
 * MIT License
 *
 * Copyright (c) 2021-2022, Christoph Neuhauser, Timm Knörle, Ludwig Leonard
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

//--- Constants

const float PI = 3.14159265359;
const float TWO_PI = 2*3.14159265359;


//--- Structs

struct ScatterEvent {
    bool hasValue;
    vec3 x; float pdf_x;
    vec3 w; float pdf_w;
    float depth;
    float density;
#ifdef CLOSE_ISOSURFACES
    vec3 normal;
    bool isIsosurface;
#endif
};


//--- Random Number Generator (Hybrid Taus)

uvec4 rngState = uvec4(0);
uint tausStep(uint z, int S1, int S2, int S3, uint M) { uint b = (((z << S1) ^ z) >> S2); return ((z & M) << S3) ^ b; }
uint lcgStep(uint z, uint A, uint C) { return A * z + C; }

float random() {
    rngState.x = tausStep(rngState.x, 13, 19, 12, 4294967294);
    rngState.y = tausStep(rngState.y, 2, 25, 4, 4294967288);
    rngState.z = tausStep(rngState.z, 3, 11, 17, 4294967280);
    rngState.w = lcgStep(rngState.w, 1664525, 1013904223);
    return 2.3283064365387e-10 * (rngState.x ^ rngState.y ^ rngState.z ^ rngState.w);
}

void initializeRandom(uint seed) {
    rngState = uvec4(seed);
    for (int i = 0; i < seed % 7 + 2; i++) {
        random();
    }
}

void createOrthonormalBasis(vec3 D, out vec3 B, out vec3 T) {
    vec3 other = abs(D.z) >= 0.9999 ? vec3(1, 0, 0) : vec3(0, 0, 1);
    B = normalize(cross(other, D));
    T = normalize(cross(D, B));
}

vec3 randomDirection(vec3 D) {
    float r1 = random();
    float r2 = random() * 2 - 1;
    float sqrR2 = r2 * r2;
    float two_pi_by_r1 = TWO_PI * r1;
    float sqrt_of_one_minus_sqrR2 = sqrt(1.0 - sqrR2);
    float x = cos(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
    float y = sin(two_pi_by_r1) * sqrt_of_one_minus_sqrR2;
    float z = r2;

    vec3 t0, t1;
    createOrthonormalBasis(D, t0, t1);

    return t0 * x + t1 * y + D * z;
}


//--- Scattering functions

#define oneMinusG2 (1.0 - (GFactor) * (GFactor))
#define onePlusG2 (1.0 + (GFactor) * (GFactor))
#define oneOver2G (0.5 / (GFactor))

float invertcdf(float GFactor, float xi) {
    float t = (oneMinusG2) / (1.0f - GFactor + 2.0f * GFactor * xi);
    return oneOver2G * (onePlusG2 - t * t);
}

vec3 importanceSamplePhase(float GFactor, vec3 D, out float pdf) {
    if (abs(GFactor) < 0.001) {
        pdf = 1.0 / (4 * PI);
        return randomDirection(-D);
    }

    float phi = random() * 2 * PI;
    float cosTheta = invertcdf(GFactor, random());
    float sinTheta = sqrt(max(0, 1.0f - cosTheta * cosTheta));

    vec3 t0, t1;
    createOrthonormalBasis(D, t0, t1);

    pdf = 0.25 / PI * (oneMinusG2) / pow(onePlusG2 - 2 * GFactor * cosTheta, 1.5);

    return sinTheta * sin(phi) * t0 + sinTheta * cos(phi) * t1 + cosTheta * D;
}

float evaluatePhase(float GFactor, vec3 org_dir, vec3 scatter_dir) {
    float cosTheta = dot(normalize(org_dir), normalize(scatter_dir));
    float pdf = 0.25 / PI * (oneMinusG2) / pow(onePlusG2 - 2 * GFactor * cosTheta, 1.5);
    return pdf;
}

#ifdef USE_ENVIRONMENT_MAP_IMAGE

vec3 octahedralUVToWorld(vec2 uv) {
    uv = uv * 2.0 - 1.0;
    vec2 sgn = sign(uv);
    uv = abs(uv);

    float r = 1.0 - abs(1.0 - uv.x - uv.y);
    float phi = 0.25 * PI * ((uv.y - uv.x) / r + 1.0);
    if (r == 0.0){
        phi = 0.0;
    }

    const float r2 = r * r;
    float x = sgn.x * cos(phi) * r * sqrt(2.0 - r2);
    float y = sgn.y * sin(phi) * r * sqrt(2.0 - r2);
    float z = sign(1.0 - uv.x - uv.y) * (1.0 - r2);
#ifdef USE_ENV_MAP_ROTATION
    return parameters.invEnvMapDirRot * vec3(x, y, z);
#else
    return vec3(x, y, z);
#endif
}

vec2 worldToOctahedralUV(vec3 dir) {
    dir = normalize(dir);
#ifdef USE_ENV_MAP_ROTATION
    dir = parameters.envMapDirRot * dir;
#endif
    vec3 sgn = sign(dir);
    dir = abs(dir);

    float phi = atan(dir.y, dir.x);
    float r = sqrt(1.0 - dir.z);

    float v = r * 2.0 / PI * phi;
    float u = r - v;

    vec2 uv = vec2(u, v);
    if (sgn.z < 0.0){
        uv = vec2(1.0 - v, 1.0 - u);
    }
    uv *= sgn.xy;// + 1.0 - abs(sgn.xy);

    return uv * 0.5 + 0.5;
}

vec3 importanceSampleSkybox(out float pdf) {
    vec2 rnd = vec2(random(), random());
    ivec2 pos = ivec2(0,0);

    int maxMip = textureQueryLevels(environmentMapOctahedralTexture);
    int minMip = 0;

    pdf = 0.25 / PI;

    for (int mip = maxMip - 1; mip >= minMip; mip--) {
        pos *= 2;
        pdf *= 4;
        float ul = texelFetch(environmentMapOctahedralTexture, pos + ivec2(0,0), mip).r + .001;
        float ur = texelFetch(environmentMapOctahedralTexture, pos + ivec2(1,0), mip).r + .001;
        float dl = texelFetch(environmentMapOctahedralTexture, pos + ivec2(0,1), mip).r + .001;
        float dr = texelFetch(environmentMapOctahedralTexture, pos + ivec2(1,1), mip).r + .001;

        float l = ul + dl;
        float r = ur + dr;
        float total = l + r;

        float hSplit = l / total;
        float vSplit;
        if (rnd.x < hSplit) {
            rnd.x = rnd.x / hSplit;
            vSplit = ul / l;
            pdf *= hSplit;
        } else{
            pos.x += 1;
            rnd.x = (rnd.x - hSplit) / (1. - hSplit);
            vSplit = ur / r;
            pdf *= (1. - hSplit);
        }

        if (rnd.y < vSplit) {
            rnd.y = rnd.y / vSplit;
            pdf *= vSplit;
        } else {
            pos.y += 1;
            rnd.y = (rnd.y - vSplit) / (1-vSplit);
            pdf *= (1. - vSplit);
        }

    }
    vec2 uv = (vec2(pos) + rnd) / vec2(1 << (maxMip - minMip));
    //uv = vec2(random(), random());
    //pdf = .25 / PI;
    return octahedralUVToWorld(uv);
}

float evaluateSkyboxPDF(vec3 sampledDir) {
    vec2 uv = worldToOctahedralUV(sampledDir);
    ivec2 pos = ivec2(0,0);

    float pdf = 0.25 / PI;

    int maxMip = textureQueryLevels(environmentMapOctahedralTexture);
    int minMip = 0;

    for (int mip = maxMip - 1; mip >= minMip; mip--) {
        pos *= 2;
        pdf *= 4;
        float ul = texelFetch(environmentMapOctahedralTexture, pos + ivec2(0,0), mip).r + .001;
        float ur = texelFetch(environmentMapOctahedralTexture, pos + ivec2(1,0), mip).r + .001;
        float dl = texelFetch(environmentMapOctahedralTexture, pos + ivec2(0,1), mip).r + .001;
        float dr = texelFetch(environmentMapOctahedralTexture, pos + ivec2(1,1), mip).r + .001;

        float l = ul + dl;
        float r = ur + dr;
        float total = l + r;

        float hSplit = l / total;
        float vSplit;
        if (uv.x < .5) {
            vSplit = ul / l;
            pdf *= hSplit;
            uv.x *= 2;
        } else{
            pos.x += 1;
            vSplit = ur / r;
            pdf *= (1. - hSplit);
            uv.x = uv.x * 2. - 1.;
        }

        if (uv.y < .5) {
            pdf *= vSplit;
            uv.y = uv.y * 2.;
        } else {
            pos.y += 1;
            pdf *= (1. - vSplit);
            uv.y = uv.y * 2. - 1.;
        }

    }
    return pdf;
}

#else

vec3 importanceSampleSkybox(out float pdf) {
    pdf = 1.0 / (4.0 * PI);
    return randomDirection(vec3(1.0, 0.0, 0.0));
}

float evaluateSkyboxPDF(vec3 sampledDir) {
    return 1.0 / (4.0 * PI);
}

#endif

//--- Tools

/**
 * Converts linear RGB to sRGB.
 * For more details see: https://en.wikipedia.org/wiki/SRGB
 */
vec3 linearRGBTosRGB(in vec3 sRGB) {
    return mix(1.055 * pow(sRGB, vec3(1.0 / 2.4)) - 0.055, sRGB * 12.92, lessThanEqual(sRGB, vec3(0.0031308)));
}

/**
 * Converts sRGB to linear RGB.
 * For more details see: https://en.wikipedia.org/wiki/SRGB
 */
vec3 sRGBToLinearRGB(in vec3 linearRGB) {
    return mix(pow((linearRGB + 0.055) / 1.055, vec3(2.4)), linearRGB / 12.92, lessThanEqual(linearRGB, vec3(0.04045)));
}


#ifdef USE_ENVIRONMENT_MAP_IMAGE

vec3 sampleSkybox(in vec3 dir) {
#ifdef USE_ENV_MAP_ROTATION
    dir = parameters.envMapDirRot * dir;
#endif

    // Sample from equirectangular projection.
    vec2 texcoord = vec2(atan(dir.z, dir.x) / TWO_PI + 0.5, -asin(dir.y) / PI + 0.5);
    vec3 textureColor = texture(environmentMapTexture, texcoord).rgb;
    
    // Make sure there is no 'inf' value in the skybox.
    textureColor = min(textureColor , vec3(100000, 100000, 100000));

#ifdef ENV_MAP_IMAGE_USES_LINEAR_RGB
#ifdef USE_LINEAR_RGB
    return parameters.environmentMapIntensityFactor * textureColor;
#else
    return parameters.environmentMapIntensityFactor * linearRGBTosRGB(textureColor);
#endif
#else // !ENV_MAP_IMAGE_USES_LINEAR_RGB
#ifdef USE_LINEAR_RGB
    return parameters.environmentMapIntensityFactor * sRGBToLinearRGB(textureColor);
#else
    return parameters.environmentMapIntensityFactor * textureColor;
#endif
#endif
}
vec3 sampleLight(in vec3 dir) {
    return vec3(0.0);
}

#else // !defined(USE_ENVIRONMENT_MAP_IMAGE)

#ifdef USE_ENVIRONMENT_MAP_DEFAULT
vec3 sampleSkybox(in vec3 dir) {
    vec3 L = dir;

    vec3 BG_COLORS[5] = {
            vec3(0.1, 0.05, 0.01), // GROUND DARKER BLUE
            vec3(0.01, 0.05, 0.2), // HORIZON GROUND DARK BLUE
            vec3(0.8, 0.9, 1.0), // HORIZON SKY WHITE
            vec3(0.1, 0.3, 1.0),  // SKY LIGHT BLUE
            vec3(0.01, 0.1, 0.7)  // SKY BLUE
    };

    float BG_DISTS[5] = {
            -1.0,
            -0.1,
            0.0,
            0.4,
            1.0
    };

    vec3 col = BG_COLORS[0];
    col = mix(col, BG_COLORS[1], vec3(smoothstep(BG_DISTS[0], BG_DISTS[1], L.y)));
    col = mix(col, BG_COLORS[2], vec3(smoothstep(BG_DISTS[1], BG_DISTS[2], L.y)));
    col = mix(col, BG_COLORS[3], vec3(smoothstep(BG_DISTS[2], BG_DISTS[3], L.y)));
    col = mix(col, BG_COLORS[4], vec3(smoothstep(BG_DISTS[3], BG_DISTS[4], L.y)));

#ifdef USE_LINEAR_RGB
    return sRGBToLinearRGB(col);
#else
    return col;
#endif
}
/*
 * See, e.g.: https://www.cs.princeton.edu/courses/archive/fall16/cos526/papers/importance.pdf
 */
vec3 sampleLight(in vec3 dir) {
    int N = 10;
    float phongNorm = (N + 1) / (2 * 3.14159);
    return parameters.sunIntensity * pow(max(0, dot(dir, parameters.sunDirection)), N) * phongNorm;
}
#elif defined(USE_ENVIRONMENT_MAP_BLACK)
vec3 sampleSkybox(in vec3 dir) {
    return vec3(0.0);
}
vec3 sampleLight(in vec3 dir) {
    return vec3(0.0);
}
#else // defined(USE_ENVIRONMENT_MAP_SINGLE_COLOR)
vec3 sampleSkybox(in vec3 dir) {
    return parameters.sunIntensity;
}
vec3 sampleLight(in vec3 dir) {
    return vec3(0.0);
}
#endif

#endif // USE_ENVIRONMENT_MAP_DEFAULT


#if NUM_LIGHTS > 0

vec3 getHeadlightDirection(vec3 pos, inout uint lightIdx, inout float lightProbabilityFactor, inout float lightDistance) {
#if NUM_LIGHTS > 1
    // Sample a random light.
    lightIdx = min(uint(random() * float(NUM_LIGHTS)), NUM_LIGHTS - 1u);
    lightProbabilityFactor *= float(NUM_LIGHTS);
#endif

    Light light = lights[lightIdx];
    vec3 lightPosition = light.position;
    if (light.lightSpace == LIGHT_SPACE_VIEW || light.lightSpace == LIGHT_SPACE_VIEW_ORIENTATION) {
        const float homComp =
                light.lightType == LIGHT_TYPE_DIRECTIONAL || light.lightSpace == LIGHT_SPACE_VIEW_ORIENTATION
                ? 0.0 : 1.0;
        lightPosition = (parameters.inverseViewMatrix * vec4(lightPosition, homComp)).xyz;
    }
    vec3 dir;
    if (light.lightType == LIGHT_TYPE_POINT || light.lightType == LIGHT_TYPE_SPOT) {
        dir = lightPosition - pos;
    } else {
        dir = lightPosition;
    }
    lightDistance = length(dir);
    return dir / (lightDistance + 1e-5);
}

vec3 sampleHeadlight(vec3 pos, uint lightIdx) {
    Light light = lights[lightIdx];
    vec3 lightPosition = light.position;
    if (light.lightSpace == LIGHT_SPACE_VIEW || light.lightSpace == LIGHT_SPACE_VIEW_ORIENTATION) {
        const float homComp =
                light.lightType == LIGHT_TYPE_DIRECTIONAL || light.lightSpace == LIGHT_SPACE_VIEW_ORIENTATION
                ? 0.0 : 1.0;
        lightPosition = (parameters.inverseViewMatrix * vec4(lightPosition, homComp)).xyz;
    }

    float lightFactor = light.intensity;
    if ((light.lightType == LIGHT_TYPE_POINT || light.lightType == LIGHT_TYPE_SPOT) && light.useDistance != 0u) {
        const vec3 diff = lightPosition - pos;
        const float distFactor = max(dot(diff, diff), 1e-3);
        lightFactor /= distFactor;
    }
    if (light.lightType == LIGHT_TYPE_SPOT) {
        const float totalWidth = cos(light.spotTotalWidth);
        const float falloffStart =
                light.spotTotalWidth > light.spotFalloffStart ? cos(light.spotFalloffStart) : totalWidth;
        vec3 p = pos - lightPosition;
        //vec3 z = light.lightSpace == LIGHT_SPACE_VIEW ? parameters.camForward : normalize(-lightPosition);
        vec3 z = normalize(light.spotDirection);
        if (light.lightSpace == LIGHT_SPACE_VIEW || light.lightSpace == LIGHT_SPACE_VIEW_ORIENTATION) {
            z = (parameters.inverseViewMatrix * vec4(z, 0.0)).xyz;
        }
        float cosTheta = dot(z, p) / (length(z) * length(p));
        float coneFactor = smoothstep(totalWidth, falloffStart, cosTheta);
        lightFactor *= coneFactor;
    }

    return lightFactor * light.color;
}

#elif defined(USE_HEADLIGHT)

vec3 getHeadlightDirection(vec3 pos, inout uint lightIdx, inout float lightProbabilityFactor, inout float lightDistance) {
    const vec3 diff = cameraPosition - pos;
    lightDistance = length(diff);
    return diff / (lightDistance + 1e-5);
    //return normalize(cameraPosition - pos);
}

vec3 sampleHeadlight(vec3 pos, inout uint lightIdx) {
#ifdef HEADLIGHT_TYPE_POINT
    #ifdef USE_HEADLIGHT_DISTANCE
        const vec3 diff = cameraPosition - pos;
        const float distFactor = max(dot(diff, diff), 1e-3);
        return (parameters.headlightIntensity / distFactor) * parameters.headlightColor;
    #else
        return parameters.headlightIntensity * parameters.headlightColor;
    #endif
#endif

#ifdef HEADLIGHT_TYPE_SPOT
    const float totalWidth = cos(parameters.headlightSpotTotalWidth);
    const float falloffStart = parameters.headlightSpotTotalWidth > parameters.headlightSpotFalloffStart ? cos(parameters.headlightSpotFalloffStart) : totalWidth;
  
    vec3 z = parameters.camForward;
    vec3 p = pos-cameraPosition;
    float cosTheta = dot(z, p)/(length(z)*length(p));
    float coneFactor = smoothstep(totalWidth, falloffStart, cosTheta);

    #ifdef USE_HEADLIGHT_DISTANCE
        const vec3 diff = cameraPosition - pos;
        const float distFactor = max(dot(diff, diff), 1e-3);
        return coneFactor * (parameters.headlightIntensity/distFactor) * parameters.headlightColor;
    #else
        return coneFactor * (parameters.headlightIntensity) * parameters.headlightColor;
    #endif
#endif
}

#endif // defined(USE_HEADLIGHT) || NUM_LIGHTS > 0

#ifdef USE_NANOVDB
pnanovdb_readaccessor_t createAccessor() {
    pnanovdb_buf_t buf = pnanovdb_buf_t(0);
    pnanovdb_readaccessor_t accessor;
    pnanovdb_grid_handle_t gridHandle;
    gridHandle.address = pnanovdb_address_null();
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, gridHandle));
    pnanovdb_readaccessor_init(accessor, root);
    return accessor;
}
pnanovdb_readaccessor_t createEmissionAccessor() {
    pnanovdb_buf_t buf = pnanovdb_buf_t(0);
    pnanovdb_readaccessor_t accessor;
    pnanovdb_grid_handle_t gridHandle;
    gridHandle.address = pnanovdb_address_null();
    pnanovdb_root_handle_t root = pnanovdb_tree_get_root(buf, pnanovdb_grid_get_tree(buf, gridHandle));
    pnanovdb_readaccessor_init(accessor, root);
    return accessor;
}
#if defined(GRID_INTERPOLATION_NEAREST)
float sampleCloudRaw(in vec3 pos) {
    pnanovdb_buf_t buf = pnanovdb_buf_t(0);
    pnanovdb_grid_handle_t gridHandle = pnanovdb_grid_handle_t(pnanovdb_address_null());
    vec3 posIndex = pnanovdb_grid_world_to_indexf(buf, gridHandle, pos);
    posIndex = floor(posIndex);
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, ivec3(posIndex));
    return pnanovdb_read_float(buf, address);
}
#elif defined(GRID_INTERPOLATION_STOCHASTIC)
float sampleCloudRaw(in vec3 pos) {
    pnanovdb_buf_t buf = pnanovdb_buf_t(0);
    pnanovdb_grid_handle_t gridHandle = pnanovdb_grid_handle_t(pnanovdb_address_null());
    vec3 posIndex = pnanovdb_grid_world_to_indexf(buf, gridHandle, pos);
    posIndex = floor(posIndex + vec3(random() - 0.5, random() - 0.5, random() - 0.5));
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, ivec3(posIndex));
    return pnanovdb_read_float(buf, address);
}
#elif defined(GRID_INTERPOLATION_TRILINEAR)
float sampleCloudRaw(in vec3 pos) {
    pnanovdb_buf_t buf = pnanovdb_buf_t(0);
    pnanovdb_grid_handle_t gridHandle = pnanovdb_grid_handle_t(pnanovdb_address_null());
    vec3 posIndex = pnanovdb_grid_world_to_indexf(buf, gridHandle, pos) - vec3(0.5);
    ivec3 posIndexInt = ivec3(floor(posIndex));
    vec3 posIndexFrac = posIndex - vec3(posIndexInt);

    pnanovdb_address_t address000 = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, posIndexInt + ivec3(0, 0, 0));
    float f000 = pnanovdb_read_float(buf, address000);
    pnanovdb_address_t address100 = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, posIndexInt + ivec3(1, 0, 0));
    float f100 = pnanovdb_read_float(buf, address100);
    float f00 = mix(f000, f100, posIndexFrac.x);

    pnanovdb_address_t address010 = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, posIndexInt + ivec3(0, 1, 0));
    float f010 = pnanovdb_read_float(buf, address010);
    pnanovdb_address_t address110 = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, posIndexInt + ivec3(1, 1, 0));
    float f110 = pnanovdb_read_float(buf, address110);
    float f10 = mix(f010, f110, posIndexFrac.x);

    float f0 = mix(f00, f10, posIndexFrac.y);

    pnanovdb_address_t address001 = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, posIndexInt + ivec3(0, 0, 1));
    float f001 = pnanovdb_read_float(buf, address001);
    pnanovdb_address_t address101 = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, posIndexInt + ivec3(1, 0, 1));
    float f101 = pnanovdb_read_float(buf, address101);
    float f01 = mix(f001, f101, posIndexFrac.x);

    pnanovdb_address_t address011 = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, posIndexInt + ivec3(0, 1, 1));
    float f011 = pnanovdb_read_float(buf, address011);
    pnanovdb_address_t address111 = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, posIndexInt + ivec3(1, 1, 1));
    float f111 = pnanovdb_read_float(buf, address111);
    float f11 = mix(f011, f111, posIndexFrac.x);

    float f1 = mix(f01, f11, posIndexFrac.y);

    return mix(f0, f1, posIndexFrac.z);
}
#endif
#else
#define sampleGridImage(coord) (texture(gridImage, coord).x - parameters.voxelValueMin) / (parameters.voxelValueMax - parameters.voxelValueMin)
#define sampleGridImageOffset(coord, offset) (textureOffset(gridImage, coord, offset).x - parameters.voxelValueMin) / (parameters.voxelValueMax - parameters.voxelValueMin)
float sampleCloudRaw(in vec3 coord) {
#if defined(GRID_INTERPOLATION_STOCHASTIC)
    ivec3 dim = textureSize(gridImage, 0);
    coord += vec3(random() - 0.5, random() - 0.5, random() - 0.5) / dim;
#endif
    return sampleGridImage(coord);
}

#if defined(USE_ISOSURFACE_RENDERING) || defined(USE_ISOSURFACES)
#if defined(ISOSURFACE_TYPE_DENSITY)
#define isoImage gridImage
#elif defined(ISOSURFACE_TYPE_GRADIENT)
#define isoImage gradientImage
#endif
#define sampleIsoImage(coord) (texture(isoImage, coord).x - parameters.voxelValueMin) / (parameters.voxelValueMax - parameters.voxelValueMin)
#define sampleIsoImageOffset(coord, offset) (textureOffset(isoImage, coord, offset).x - parameters.voxelValueMin) / (parameters.voxelValueMax - parameters.voxelValueMin)
#endif // defined(USE_ISOSURFACE_RENDERING) || defined(USE_ISOSURFACES)

#endif

#ifdef USE_EMISSION
float sampleEmissionRaw(in vec3 coord) {
#if defined(GRID_INTERPOLATION_STOCHASTIC)
    ivec3 dim = textureSize(emissionImage, 0);
    coord += vec3(random() - 0.5, random() - 0.5, random() - 0.5) / dim;
#endif
    return texture(emissionImage, coord).x;
}
vec3 sampleEmission(in vec3 pos){
    // transform world pos to density grid pos
    vec3 coord = (pos - parameters.emissionBoxMin) / (parameters.emissionBoxMax - parameters.emissionBoxMin);
#if defined(FLIP_YZ)
    coord = coord.xzy;
#endif
    coord = coord * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;

    float t = sampleEmissionRaw(coord);
    t = clamp(t * parameters.emissionCap,0,1);
    vec3 col = vec3(t*t);
    col.g = col.r * col.r;
    col.b = col.g * col.g;
    return col * parameters.emissionStrength;
}
#endif

#ifdef USE_TRANSFER_FUNCTION
vec4 sampleCloudColorAndDensity(in vec3 pos) {
    // Idea: Returns (color.rgb, density).
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
#if defined(FLIP_YZ)
    coord = coord.xzy;
#endif
    coord = coord * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;
    float densityRaw = sampleCloudRaw(coord);
    //return densityRaw;
    //return texture(transferFunctionTexture, densityRaw);
    return texture(transferFunctionTexture, vec2(densityRaw, 0.0));
}
#endif

#ifdef USE_TRANSFER_FUNCTION
float sampleCloud(in vec3 pos) {
    // Idea: Returns density.
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
#if defined(FLIP_YZ)
    coord = coord.xzy;
#endif
    coord = coord * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;
    float densityRaw = sampleCloudRaw(coord);
    //return texture(transferFunctionTexture, densityRaw).a;
    return texture(transferFunctionTexture, vec2(densityRaw, 0.0)).a;
}
vec4 sampleCloudDensityEmission(in vec3 pos) {
    // Idea: Returns (color.rgb, density).
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
    float densityRaw = sampleCloudRaw(coord);
    //return texture(transferFunctionTexture, densityRaw);
    return texture(transferFunctionTexture, vec2(densityRaw, 0.0));
}
float sampleCloudDirect(in vec3 pos) {
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
#if defined(FLIP_YZ)
    coord = coord.xzy;
#endif
    coord = coord * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;
    float densityRaw = sampleCloudRaw(coord);
    return densityRaw;
}
#else // !defined(USE_TRANSFER_FUNCTION)
float sampleCloud(in vec3 pos) {
    // Transform world position to density grid position.
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
#if defined(FLIP_YZ)
    coord = coord.xzy;
#endif
    coord = coord * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;

    return sampleCloudRaw(coord);// + parameters.extinction.g / parameters.extinction.r * .01;
}
#define sampleCloudDirect sampleCloud
#endif

#if defined(ISOSURFACE_TYPE_GRADIENT) || (defined(ISOSURFACE_TYPE_DENSITY) && defined(ISOSURFACE_USE_TF) && defined(USE_TRANSFER_FUNCTION))
float sampleCloudGradient(in vec3 pos) {
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
#if defined(FLIP_YZ)
    coord = coord.xzy;
#endif
    coord = coord * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;
#if defined(GRID_INTERPOLATION_STOCHASTIC)
    ivec3 dim = textureSize(gridImage, 0);
    coord += vec3(random() - 0.5, random() - 0.5, random() - 0.5) / dim;
#endif
    return texture(gradientImage, coord).x;
}
#endif

#if defined(ISOSURFACE_TYPE_DENSITY)
#define sampleCloudIso sampleCloudDirect
#elif defined(ISOSURFACE_TYPE_GRADIENT)
#define sampleCloudIso sampleCloudGradient
#endif

/**
 * sampleIsoColorTF/sampleIsoOpacityTF use the opposite entry of the
 */
#if defined(ISOSURFACE_USE_TF) && defined(USE_TRANSFER_FUNCTION)
vec4 sampleIsoColorTF(in vec3 pos) {
#if defined(ISOSURFACE_TYPE_DENSITY)
    float densityRaw = (sampleCloudGradient(pos) - parameters.minGradientVal) / (parameters.maxGradientVal - parameters.minGradientVal);
#else // defined(ISOSURFACE_TYPE_GRADIENT)
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
    float densityRaw = sampleCloudRaw(coord);
#endif
    return texture(transferFunctionTexture, vec2(densityRaw, 1.0));
}
float sampleIsoOpacityTF(in vec3 pos) {
    return sampleIsoColorTF(pos).a;
}
#endif

vec3 getCloudFiniteDifference(in vec3 pos) {
#ifdef USE_NANOVDB
    return vec3(0);
#else

    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
    ivec3 dim = textureSize(gridImage, 0);
#if defined(GRID_INTERPOLATION_STOCHASTIC)
    coord += vec3(random() - 0.5, random() - 0.5, random() - 0.5) / dim;
#endif
    float density = texture(gridImage, coord).x;
    vec3 dFdpos = vec3(
        sampleGridImage(coord - vec3(1, 0, 0) / dim) - sampleGridImage(coord + vec3(1, 0, 0) / dim),
        sampleGridImage(coord - vec3(0, 1, 0) / dim) - sampleGridImage(coord + vec3(0, 1, 0) / dim),
        sampleGridImage(coord - vec3(0, 0, 1) / dim) - sampleGridImage(coord + vec3(0, 0, 1) / dim)
    ) / dim * 100;
    const float l = length(dFdpos);
    if (l > 1e-5) {
        dFdpos /= l;
    }
    return dFdpos;
#endif
}

void createCameraRay(in vec2 coord, out vec3 x, out vec3 w) {
    vec4 ndcP = vec4(coord, 0, 1);
    vec4 ndcT = ndcP + vec4(0, 0, 1, 0);

    vec4 viewP = parameters.inverseViewProjMatrix * ndcP;
    viewP.xyz /= viewP.w;
    vec4 viewT = parameters.inverseViewProjMatrix * ndcT;
    viewT.xyz /= viewT.w;

    x = viewP.xyz;
    w = normalize(viewT.xyz - viewP.xyz);
}

bool rayBoxIntersect(vec3 bMin, vec3 bMax, vec3 P, vec3 D, out float tMin, out float tMax) {
    // Un-parallelize D.
    D.x = abs(D).x <= 0.000001 ? 0.000001 : D.x;
    D.y = abs(D).y <= 0.000001 ? 0.000001 : D.y;
    D.z = abs(D).z <= 0.000001 ? 0.000001 : D.z;
    vec3 C_Min = (bMin - P) / D;
    vec3 C_Max = (bMax - P) / D;
    tMin = max(max(min(C_Min.x, C_Max.x), min(C_Min.y, C_Max.y)), min(C_Min.z, C_Max.z));
    tMin = max(0.0, tMin);
    tMax = min(min(max(C_Min.x, C_Max.x), max(C_Min.y, C_Max.y)), max(C_Min.z, C_Max.z));
    if (tMax <= tMin || tMax <= 0) {
        return false;
    }
#ifdef USE_CLIP_PLANE
    // Compute the intersection of the ray with the clip plane.
    float denom = dot(parameters.clipPlaneNormal, D);
    float d = dot(parameters.clipPlaneNormal, P) - parameters.clipPlaneDistance;

    if (abs(denom) < 0.001) {
        // Ray is parallel to the clip plane, return whether we are on the positive side (visible)
        return d > 0.0;
    } else {
        float tClip = (parameters.clipPlaneDistance - dot(parameters.clipPlaneNormal, P)) / denom;
        if (tClip <= 0.0) {
            return d > 0.0;
        }
        if (d > 0.0) {
            tMax = min(tClip, tMax);
        } else {
            tMin = max(tClip, tMin);
        }
    }

    if (tMax <= tMin || tMax <= 0) {
        return false;
    }
#endif
    return true;
}

#ifdef CLOSE_ISOSURFACES
void rayBoxIntersectionNormal(vec3 bMin, vec3 bMax, vec3 P, vec3 D, inout vec3 surfaceNormal) {
    float tMinTest, tMaxTest;
    D.x = abs(D).x <= 1e-6 ? 1e-6 : D.x;
    D.y = abs(D).y <= 1e-6 ? 1e-6 : D.y;
    D.z = abs(D).z <= 1e-6 ? 1e-6 : D.z;
    vec3 C_Min = (bMin - P) / D;
    vec3 C_Max = (bMax - P) / D;
    float minX = min(C_Min.x, C_Max.x);
    float minY = min(C_Min.y, C_Max.y);
    float minZ = min(C_Min.z, C_Max.z);
    float tMin = max(max(minX, minY), minZ);
    tMin = max(0.0, tMin);
    float tMax = min(min(max(C_Min.x, C_Max.x), max(C_Min.y, C_Max.y)), max(C_Min.z, C_Max.z));
    if (tMax <= tMin || tMax <= 0) {
        return;
    }
    vec3 normalBox;
    if (minX > minY && minX > minZ) {
        normalBox = vec3(-sign(D.x), 0.0, 0.0);
    } else if (minY > minZ) {
        normalBox = vec3(0.0, -sign(D.y), 0.0);
    } else {
        normalBox = vec3(0.0, 0.0, -sign(D.z));
    }
#ifdef USE_CLIP_PLANE
    // Compute the intersection of the ray with the clip plane.
    float denom = dot(parameters.clipPlaneNormal, D);
    float d = dot(parameters.clipPlaneNormal, P) - parameters.clipPlaneDistance;

    if (abs(denom) < 1e-3) {
        // Ray is parallel to the clip plane, check whether we are on the visible side.
        if (d > 0.0) {
            surfaceNormal = normalBox;
        }
        return;
    } else {
        float tClip = (parameters.clipPlaneDistance - dot(parameters.clipPlaneNormal, P)) / denom;
        if (tClip <= 0.0) {
            // Clip plane is behind camera position. Check whether we are on the visible side.
            if (d > 0.0) {
                surfaceNormal = normalBox;
            }
            return;
        }
        if (d <= 0.0 && tClip >= tMin) {
            surfaceNormal = -parameters.clipPlaneNormal;
            return;
        }
    }
#endif
    surfaceNormal = normalBox;
}
#endif

float maxComponent(vec3 v) {
    return max(v.x, max(v.y, v.z));
}

float avgComponent(vec3 v) {
    return (v.x + v.y + v.z) / 3.0;
}


#define M_PI 3.14159265358979323846

#ifdef USE_ISOSURFACES
#include "RayTracingUtilities.glsl"

// Simple method for computing surface normals
vec3 getGradient(vec3 texCoords) {
    const float dx = 0.1;
    const float dy = 0.1;
    const float dz = 0.1;
    float gradX =
            (sampleIsoImageOffset(texCoords, ivec3(-1, 0, 0))
            - sampleIsoImageOffset(texCoords, ivec3(1, 0, 0))) * 0.5 / dx;
    float gradY =
            (sampleIsoImageOffset(texCoords, ivec3(0, -1, 0))
            - sampleIsoImageOffset(texCoords, ivec3(0, 1, 0))) * 0.5 / dy;
    float gradZ =
            (sampleIsoImageOffset(texCoords, ivec3(0, 0, -1))
            - sampleIsoImageOffset(texCoords, ivec3(0, 0, 1))) * 0.5 / dz;

    vec3 grad = vec3(gradX, gradY, gradZ);
    float gradLength = length(grad);
    if (gradLength < 1e-4) {
        return vec3(0.0, 0.0, 1.0);
    }
    return grad / gradLength;
}

// Get the multiple of x closest to n, where k*x < n
float getNearestMultiple(float n, float x, bool smaller) {
    float rest = mod(n,x);
    if (smaller) {
        return n - rest;
    } else {
        if (rest == 0.0) {
            return n;
        }
        return x + n - rest;
    }
}

float trilinearInterpolationDensity(vec3 texCoords) {
    float x = texCoords.x;
    float y = texCoords.y;
    float z = texCoords.z;
    
    // Compute texel sizes
    ivec3 sizeTexture = textureSize(isoImage, 0);
    float deltaX = 1.0 / (float(sizeTexture.x) - 1.0);
    float deltaY = 1.0 / (float(sizeTexture.y) - 1.0);
    float deltaZ = 1.0 / (float(sizeTexture.z) - 1.0);

    // Compute the 6 nearest texel values
    float x0 = getNearestMultiple(x, deltaX, true);
    float x1 = getNearestMultiple(x, deltaX, false);
    float y0 = getNearestMultiple(y, deltaY, true);
    float y1 = getNearestMultiple(y, deltaY, false);
    float z0 = getNearestMultiple(z, deltaZ, true);
    float z1 = getNearestMultiple(z, deltaZ, false);
    
    // Get the 8 nearest edges
    vec3 n000 = vec3(x0,y0,z0);
    vec3 n100 = vec3(x1,y0,z0);
    vec3 n110 = vec3(x1,y1,z0);
    vec3 n010 = vec3(x0,y1,z0);
    vec3 n001 = vec3(x0,y0,z1);
    vec3 n101 = vec3(x1,y0,z1);
    vec3 n111 = vec3(x1,y1,z1);
    vec3 n011 = vec3(x0,y1,z1);

    // Compute gradients for these nearest edges
    float d000 = sampleIsoImage(n000);
    float d100 = sampleIsoImage(n100);
    float d110 = sampleIsoImage(n110);
    float d010 = sampleIsoImage(n010);
    float d001 = sampleIsoImage(n001);
    float d101 = sampleIsoImage(n101);
    float d111 = sampleIsoImage(n111);
    float d011 = sampleIsoImage(n011);

    vec3 posIndex = vec3(x*(float(sizeTexture.x)-1.0),y*(float(sizeTexture.y)-1.0),z*(float(sizeTexture.z)-1.0));
    ivec3 posIndexInt = ivec3(floor(posIndex));
    vec3 posIndexFrac = posIndex - vec3(posIndexInt);

    float t00 = mix(d000, d100, posIndexFrac.x);
    float t10 = mix(d010, d110, posIndexFrac.x);
    float t01 = mix(d001, d101, posIndexFrac.x);
    float t11 = mix(d011, d111, posIndexFrac.x);

    float t0 = mix(t00, t10, posIndexFrac.y);
    float t1 = mix(t01, t11, posIndexFrac.y);

    return mix(t0, t1, posIndexFrac.z);
}

#define DIFFERENCES_NEIGHBOR

// Compute surface normals using trilinear interpolaton for 6 points arround the current point
vec3 computeGradient(vec3 texCoords) {
#ifdef DIFFERENCES_NEIGHBOR
    // Idea from: https://math.stackexchange.com/questions/2452416/compute-gradient-of-scalar-field-defined-by-trilinear-interpolation-of-sample-gr
    const float dx = 4.0 * parameters.voxelTexelSize.x;
    const float dy = 4.0 * parameters.voxelTexelSize.y;
    const float dz = 4.0 * parameters.voxelTexelSize.z;

    vec3 gX0 = vec3(-1.0,0.0,0.0)*trilinearInterpolationDensity(texCoords + dx*vec3(-1.0,0.0,0.0));
    vec3 gX1 = vec3(1.0,0.0,0.0)*trilinearInterpolationDensity(texCoords + dx*vec3(1.0,0.0,0.0));
    vec3 gY0 = vec3(0.0,-1.0,0.0)*trilinearInterpolationDensity(texCoords + dy*vec3(0.0,-1.0,0.0));
    vec3 gY1 = vec3(0.0,1.0,0.0)*trilinearInterpolationDensity(texCoords + dy*vec3(0.0,1.0,0.0));
    vec3 gZ0 = vec3(0.0,0.0,-1.0)*trilinearInterpolationDensity(texCoords + dz*vec3(0.0,0.0,-1.0));
    vec3 gZ1 = vec3(0.0,0.0,1.0)*trilinearInterpolationDensity(texCoords + dz*vec3(0.0,0.0,1.0));

    vec3 grad = 0.5*(gX0 + gX1 + gY0 + gY1 + gZ0 + gZ1);
    float gradLength = length(grad);
    if (gradLength < 1e-3) {
        return vec3(0.0, 0.0, 1.0);
    }
    return grad / gradLength;
#else
    const float dx = parameters.voxelTexelSize.x * 1;
    const float dy = parameters.voxelTexelSize.y * 1;
    const float dz = parameters.voxelTexelSize.z * 1;
    float gradX =
            (sampleIsoImage(texCoords - vec3(dx, 0.0, 0.0))
            - sampleIsoImage(texCoords + vec3(dx, 0.0, 0.0))) * 0.5;
    float gradY =
            (sampleIsoImage(texCoords - vec3(0.0, dy, 0.0))
            - sampleIsoImage(texCoords + vec3(0.0, dy, 0.0))) * 0.5;
    float gradZ =
            (sampleIsoImage(texCoords - vec3(0.0, 0.0, dz))
            - sampleIsoImage(texCoords + vec3(0.0, 0.0, dz))) * 0.5;
    vec3 grad = vec3(gradX, gradY, gradZ);
    float gradLength = length(grad);
    if (gradLength < 1e-4) {
        return vec3(0.0, 0.0, 1.0);
    }
    return grad / gradLength;
#endif
}

// Version before commit ec19aebe3fd8df5571cca8ee7449c00d5874e613
vec3 computeGradientLegacy(vec3 texCoords) {
#ifdef DIFFERENCES_NEIGHBOR
    const float dx = 1.0;
    const float dy = 1.0;
    const float dz = 1.0;
    float gradX =
            (sampleIsoImageOffset(texCoords, ivec3(-1, 0, 0))
            - sampleIsoImageOffset(texCoords, ivec3(1, 0, 0))) * 0.5 / dx;
    float gradY =
            (sampleIsoImageOffset(texCoords, ivec3(0, -1, 0))
            - sampleIsoImageOffset(texCoords, ivec3(0, 1, 0))) * 0.5 / dy;
    float gradZ =
            (sampleIsoImageOffset(texCoords, ivec3(0, 0, -1))
            - sampleIsoImageOffset(texCoords, ivec3(0, 0, 1))) * 0.5 / dz;
#else
    const float dx = parameters.voxelTexelSize.x * 1;
    const float dy = parameters.voxelTexelSize.y * 1;
    const float dz = parameters.voxelTexelSize.z * 1;
    float gradX =
            (sampleIsoImage(texCoords - vec3(dx, 0.0, 0.0))
            - sampleIsoImage(texCoords + vec3(dx, 0.0, 0.0))) * 0.5;
    float gradY =
            (sampleIsoImage(texCoords - vec3(0.0, dy, 0.0))
            - sampleIsoImage(texCoords + vec3(0.0, dy, 0.0))) * 0.5;
    float gradZ =
            (sampleIsoImage(texCoords - vec3(0.0, 0.0, dz))
            - sampleIsoImage(texCoords + vec3(0.0, 0.0, dz))) * 0.5;
#endif

    vec3 grad = vec3(gradX, gradY, gradZ);
    float gradLength = length(grad);
    if (gradLength < 1e-3) {
        return vec3(0.0, 0.0, 1.0);
    }
    return grad / gradLength;
}

const int MAX_NUM_REFINEMENT_STEPS = 8;

void refineIsoSurfaceHit(inout vec3 currentPoint, vec3 lastPoint, float stepSign) {
    for (int i = 0; i < MAX_NUM_REFINEMENT_STEPS; i++) {
        vec3 midPoint = (currentPoint + lastPoint) * 0.5;
        //vec3 texCoordsMidPoint = (midPoint - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
        //float scalarValueMidPoint = texture(scalarField, texCoordsMidPoint).r;
        float scalarValueMidPoint = sampleCloudIso(midPoint);
        if ((scalarValueMidPoint - parameters.isoValue) * stepSign >= 0.0) {
            currentPoint = midPoint;
        } else {
            lastPoint = midPoint;
        }
    }
}
#endif

#include "NeeTransmittance.glsl"

#ifdef USE_ISOSURFACES
/**
 * Uniformly samples a direction on the upper hemisphere for the surface normal vector n = (0, 0, 1)^T.
 * For more details see:
 * https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/2D_Sampling_with_Multidimensional_Transformations
 * @param xi Two random numbers uniformly sampled in the range [0, 1).
 */
vec3 sampleHemisphere(vec2 xi) {
    // Uniform Hemisphere PDF: 1 / (2 * pi)
    //float theta = acos(xi.x);
    float phi = 2.0 * M_PI * xi.y;
    float r = sqrt(1.0 - xi.x * xi.x);
    return vec3(cos(phi) * r, sin(phi) * r, xi.x);
}
vec2 concentricSampleDisk(vec2 xi) {
    vec2 xiOffset = 2.0 * xi - vec2(1.0);
    if (xiOffset.x == 0 && xiOffset.y == 0) {
        return vec2(0.0);
    }
    float theta, r;
    if (abs(xiOffset.x) > abs(xiOffset.y)) {
        r = xiOffset.x;
        theta = M_PI / 4.0 * (xiOffset.y / xiOffset.x);
    } else {
        r = xiOffset.y;
        theta = M_PI / 2.0 - M_PI / 4.0 * (xiOffset.x / xiOffset.y);
    }
    return r * vec2(cos(theta), sin(theta));
}
vec3 sampleHemisphereCosineWeighted(vec2 xi) {
    // Cosine Hemisphere PDF: cos(theta) / pi
    vec2 d = concentricSampleDisk(xi);
    float z = sqrt(1.0 - d.x * d.x - d.y * d.y);
    return vec3(d.x, d.y, z);
}

// -------------- Helper Functions and Structs ------------------

struct flags{
    bool specularHit;
    bool clearcoatHit;
};

float sqr(float x) {
    return x * x;
}

// -------------- BRDF Functions ------------------

#ifdef SURFACE_BRDF_DISNEY
#include "Disney2012.glsl"
#endif

#ifdef SURFACE_BRDF_COOK_TORRANCE
#include "CookTorrance.glsl"
#endif

#ifdef SURFACE_BRDF_LAMBERTIAN
#include "Lambertian.glsl"
#endif

#ifdef SURFACE_BRDF_BLINN_PHONG
#include "BlinnPhong.glsl"
#endif

#ifdef SURFACE_BRDF_AMBIENT
#include "ConstantLighting.glsl"
#endif


#if !defined(ISOSURFACE_USE_TF) || !defined(USE_TRANSFER_FUNCTION)
#define isoSurfaceColorDef parameters.isoSurfaceColor
#define DEFINE_ISO_SURFACE_COLOR float isoSurfaceOpacity = 1.0;
#else // defined(ISOSURFACE_TYPE_GRADIENT)
#define DEFINE_ISO_SURFACE_COLOR \
vec4 isoSurfaceColorAll = sampleIsoColorTF(currentPoint); \
vec3 isoSurfaceColorDef = isoSurfaceColorAll.rgb; \
float isoSurfaceOpacity = isoSurfaceColorAll.a;
#endif

#define UNIFORM_SAMPLING
//#define USE_MIS // only for specular BRDF sampling

#ifdef CLOSE_ISOSURFACES
vec3 surfaceNormalGlobal;
#endif

bool getIsoSurfaceHit(
        vec3 currentPoint, inout vec3 w, inout vec3 throughput
#if defined(USE_NEXT_EVENT_TRACKING_SPECTRAL) || defined(USE_NEXT_EVENT_TRACKING)
        , inout vec3 colorNee
#endif
#ifdef CLOSE_ISOSURFACES
        , bool isFirstPointFromOutside
#endif
) {
    // -------------- Abort Conditions ------------------

    DEFINE_ISO_SURFACE_COLOR;
    if (isoSurfaceOpacity < 1e-4) {
        return false;
    }
    if (isoSurfaceOpacity < random()) {
        return false;
    }

    // -------------- Declarations and Preparations ------------------

    flags hitFlags = flags(false, false);
    bool useMIS = false;

    vec3 texCoords = (currentPoint - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
    texCoords = texCoords * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;
#ifdef USE_LEGACY_NORMALS
    vec3 surfaceNormal = computeGradientLegacy(texCoords);
#else
    vec3 surfaceNormal = computeGradient(texCoords);
#endif
#ifdef CLOSE_ISOSURFACES
    if (isFirstPointFromOutside) {
        rayBoxIntersectionNormal(parameters.boxMin, parameters.boxMax, cameraPosition, w, surfaceNormal);
    } else
#endif

    if (dot(w, surfaceNormal) > 0.0) {
        surfaceNormal = -surfaceNormal;
    }

#ifdef CLOSE_ISOSURFACES
    surfaceNormalGlobal = surfaceNormal;
#endif

    vec3 surfaceTangent;
    vec3 surfaceBitangent;
    ComputeDefaultBasis(surfaceNormal, surfaceTangent, surfaceBitangent);
    mat3 frame = mat3(surfaceTangent, surfaceBitangent, surfaceNormal);

    vec3 colorOut;
    vec3 dirOut;

    // -------------- BRDF Sampling and Evaluation ------------------

    float samplingPDF;
    colorOut = computeBrdf(normalize(-w), dirOut, surfaceNormal, surfaceTangent,
                           surfaceBitangent, frame, isoSurfaceColorDef,
                           hitFlags, samplingPDF);

    // -------------- Next Event Tracking (NEE) ------------------

#if defined(USE_ISOSURFACE_NEE) && (defined(USE_NEXT_EVENT_TRACKING_SPECTRAL) || defined(USE_NEXT_EVENT_TRACKING))

#ifdef BRDF_SUPPORTS_SPECULAR

    if (hitFlags.specularHit == true) {
        useMIS = true;
    } else {
        useMIS = false;
    }

#endif

    float pdfLightNee;  // only used for skybox.
    vec3 dirLightNee;

#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
    // We are sampling the environment map or headlight with 50/50 chance.
    bool isSamplingHeadlight = (parameters.isEnvMapBlack != 0u) ? true : (random() > 0.5);
    float lightProbabilityFactor = parameters.isEnvMapBlack != 0u ? 1.0 : 2.0;
    float lightDistance = 0.0;
    uint lightIdx = 0;
    if (isSamplingHeadlight) {
        dirLightNee = getHeadlightDirection(currentPoint, lightIdx, lightProbabilityFactor, lightDistance);
    } else {
#endif
        dirLightNee = importanceSampleSkybox(pdfLightNee);
#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
    }
#endif

    if (dot(surfaceNormal, dirLightNee) > 0.0) {
        float pdfSamplingOut;
        float pdfSamplingNee;

        vec3 rdfNee = evaluateBrdfNee(
            normalize(-w), dirOut, dirLightNee, surfaceNormal, surfaceTangent,
            surfaceBitangent, isoSurfaceColorDef, useMIS, samplingPDF, hitFlags,
            pdfSamplingOut, pdfSamplingNee);

#ifdef BRDF_SUPPORTS_SPECULAR
        if (useMIS) {
            // NEE with MIS.
            // Power heuristic with beta=2: https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Importance_Sampling
            //float weightNee = pdfLightNee * pdfLightNee / (pdfLightNee * pdfLightNee + pdfSamplingNee * pdfSamplingNee);
            //float weightOut = pdfSamplingOut * pdfSamplingOut / (pdfLightOut * pdfLightOut + pdfSamplingOut * pdfSamplingOut);

            float pdfLightOut = evaluateSkyboxPDF(dirOut);

            float weightNee = pdfLightNee / (pdfLightNee + pdfSamplingNee);
            float weightOut = pdfSamplingOut / (pdfLightOut + pdfSamplingOut);

#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
            if (isSamplingHeadlight) {
                weightNee = 1.0;
                weightOut = 1.0;

                colorNee +=
                        lightProbabilityFactor * throughput * rdfNee * weightNee *
                        calculateTransmittanceDistance(currentPoint + dirLightNee * 1e-4, dirLightNee, lightDistance) *
                        sampleHeadlight(currentPoint, lightIdx);
            } else {
                colorNee +=
                        lightProbabilityFactor * throughput * rdfNee * weightNee / pdfLightNee *
                        calculateTransmittance(currentPoint + dirLightNee * 1e-4, dirLightNee) *
                        (sampleSkybox(dirLightNee) + sampleLight(dirLightNee));

                colorNee +=
                        lightProbabilityFactor * throughput * weightOut * colorOut *
                        calculateTransmittance(currentPoint + dirOut * 1e-4, dirOut) *
                        (sampleSkybox(dirOut) + sampleLight(dirOut));
            }

#else

            colorNee +=
                    throughput * rdfNee * weightNee / pdfLightNee *
                    calculateTransmittance(currentPoint + dirLightNee * 1e-4, dirLightNee) *
                    (sampleSkybox(dirLightNee) + sampleLight(dirLightNee));
            colorNee +=
                    throughput * weightOut * colorOut *
                    calculateTransmittance(currentPoint + dirOut * 1e-4, dirOut) *
                    (sampleSkybox(dirOut) + sampleLight(dirOut));

#endif
        } else {
#endif

// Normal NEE.
#if defined(USE_HEADLIGHT) || NUM_LIGHTS > 0
            vec3 commonFactor = lightProbabilityFactor * throughput * rdfNee;
            if (isSamplingHeadlight) {
                colorNee +=
                        commonFactor *
                        calculateTransmittanceDistance( currentPoint + dirLightNee * 1e-4, dirLightNee, lightDistance) *
                        sampleHeadlight(currentPoint, lightIdx);
            } else {
                colorNee +=
                        commonFactor / pdfLightNee *
                        calculateTransmittance(currentPoint + dirLightNee * 1e-4, dirLightNee) *
                        (sampleSkybox(dirLightNee) + sampleLight(dirLightNee));
            }
#else

        colorNee +=
                throughput * rdfNee / pdfLightNee *
                calculateTransmittance(currentPoint + dirLightNee * 1e-4, dirLightNee) *
                (sampleSkybox(dirLightNee) + sampleLight(dirLightNee));
#endif

#ifdef BRDF_SUPPORTS_SPECULAR
        }
#endif
    }
#endif

    // -------------- Return final results ------------------

    w = dirOut;
    throughput *= colorOut;

    return true;
}

#endif

#if defined(USE_ISOSURFACE_RENDERING) || (defined(USE_ISOSURFACES) && defined(USE_RAY_MARCHING_EMISSION_ABSORPTION))
#include "Lighting.glsl"

// Direct illumination
vec3 getIsoSurfaceHitDirect(
        vec3 currentPoint, vec3 w, inout vec3 surfaceNormal
#ifdef CLOSE_ISOSURFACES
        , bool isFirstPointFromOutside
#endif
) {
    vec3 texCoords = (currentPoint - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
    texCoords = texCoords * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;
#ifdef USE_LEGACY_NORMALS
    surfaceNormal = computeGradientLegacy(texCoords);
#else
    surfaceNormal = computeGradient(texCoords);
#endif
#ifdef CLOSE_ISOSURFACES
    if (isFirstPointFromOutside) {
        rayBoxIntersectionNormal(parameters.boxMin, parameters.boxMax, cameraPosition, w, surfaceNormal);
    } else
#endif
    if (dot(cameraPosition - currentPoint, surfaceNormal) < 0.0) {
        surfaceNormal = -surfaceNormal;
    }
    DEFINE_ISO_SURFACE_COLOR;
    vec3 color = blinnPhongShadingSurface(isoSurfaceColorDef, currentPoint, surfaceNormal);
    return color;
}
#endif

#include "FloatPack.glsl"
