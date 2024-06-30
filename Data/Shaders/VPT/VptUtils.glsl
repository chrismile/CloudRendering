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
    uv = uv * 2. - 1.;
    vec2 sgn = sign(uv);
    uv = abs(uv);

    float r = 1. - abs(1. - uv.x - uv.y);
    float phi = .25 * PI * ( (uv.y - uv.x) / r + 1. );
    if (r == 0.){
        phi = 0.;
    }

    float x = sgn.x * cos(phi) * r * sqrt(2 - r * r);
    float y = sgn.y * sin(phi) * r * sqrt(2 - r * r);
    float z = sign(1. - uv.x - uv.y) * (1. - r * r);
    return vec3(x, y, z);
}

vec2 worldToOctahedralUV(vec3 dir) {
    dir = normalize(dir);
    vec3 sgn = sign(dir);
    dir = abs(dir);

    float phi = atan(dir.y , dir.x);
    float r = sqrt(1. - dir.z);

    float v = r * 2. / PI * phi;
    float u = r - v;

    vec2 uv = vec2(u,v);
    if (sgn.z < 0.){
        uv = vec2(1.-v, 1.-u);
    }
    uv *= sgn.xy;// + 1.- abs(sgn.xy);

    return uv * .5 + .5;
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
#else
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
#endif

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
vec4 sampleCloudGradientColor(in vec3 pos) {
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
    float densityRaw = sampleCloudRaw(coord);
    return texture(transferFunctionTexture, vec2(densityRaw, 1.0));
}
float sampleCloudGradientOpacity(in vec3 pos) {
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
    float densityRaw = sampleCloudRaw(coord);
    return texture(transferFunctionTexture, vec2(densityRaw, 1.0)).a;
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
#else
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

#if defined(ISOSURFACE_TYPE_DENSITY)
#define sampleCloudIso sampleCloudDirect
#elif defined(ISOSURFACE_TYPE_GRADIENT)
float sampleCloudIso(in vec3 pos) {
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
    vec3 C_Min = (bMin - P)/D;
    vec3 C_Max = (bMax - P)/D;
    tMin = max(max(min(C_Min[0], C_Max[0]), min(C_Min[1], C_Max[1])), min(C_Min[2], C_Max[2]));
    tMin = max(0.0, tMin);
    tMax = min(min(max(C_Min[0], C_Max[0]), max(C_Min[1], C_Max[1])), max(C_Min[2], C_Max[2]));
    if (tMax <= tMin || tMax <= 0) {
        return false;
    }
    return true;
}

float maxComponent(vec3 v) {
    return max(v.x, max(v.y, v.z));
}

float avgComponent(vec3 v) {
    return (v.x + v.y + v.z) / 3.0;
}


#define M_PI 3.14159265358979323846

#ifdef USE_ISOSURFACES
#include "RayTracingUtilities.glsl"
vec3 getGradient(vec3 texCoords) {
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

#define DIFFERENCES_NEIGHBOR
vec3 computeGradient(vec3 texCoords) {
#ifdef DIFFERENCES_NEIGHBOR
    float x = texCoords[0];
    float y = texCoords[1];
    float z = texCoords[2];
    
    // Compute texel sizes
    ivec3 sizeTexture = textureSize(isoImage, 0);
    float deltaX = 1.0/(sizeTexture[0] -1.0);
    float deltaY = 1.0/(sizeTexture[1] - 1.0);
    float deltaZ = 1.0/(sizeTexture[2] - 1.0);
    
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
    vec3 g000 = getGradient(n000);
    vec3 g100 = getGradient(n100);
    vec3 g110 = getGradient(n110);
    vec3 g010 = getGradient(n010);
    vec3 g001 = getGradient(n001);
    vec3 g101 = getGradient(n101);
    vec3 g111 = getGradient(n111);
    vec3 g011 = getGradient(n011);

    // Trilinear interpolation of these 8 gradients
    // Index for interpolation
    vec3 posIndex = vec3(x*(sizeTexture[0]-1.0),y*(sizeTexture[1]-1.0),z*(sizeTexture[2]-1.0));
    ivec3 posIndexInt = ivec3(floor(posIndex));
    vec3 posIndexFrac = posIndex - vec3(posIndexInt);

    vec3 t00 = mix(g000, g100, posIndexFrac.x);
    vec3 t10 = mix(g010, g110, posIndexFrac.x);
    vec3 t01 = mix(g001, g101, posIndexFrac.x);
    vec3 t11 = mix(g011, g111, posIndexFrac.x);

    vec3 t0 = mix(t00, t10, posIndexFrac.y);
    vec3 t1 = mix(t01, t11, posIndexFrac.z);

    return mix(t0, t1, posIndexFrac.z);
#else
    const float dx = parameters.voxelTexelSize.x * 0.01;
    const float dy = parameters.voxelTexelSize.y * 0.01;
    const float dz = parameters.voxelTexelSize.z * 0.01;
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

// -------------- BRDF Helper Functions ------------------

// https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdfs/disney.brdf#L40
float sqr(float x) {
    return x * x;
}

// https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdfs/disney.brdf#L49C1-L55C2
float GTR1(float NdotH, float a)
{
    if (a >= 1) return 1 / PI;
    float a2 = a * a;
    float t = 1 + (a2 - 1) * NdotH * NdotH;
    return (a2 - 1) / (PI * log(a2) * t);
}

// https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdfs/disney.brdf#L69C1-L74C2
float smithG_GGX(float NdotV, float alphaG)
{
    float a = alphaG * alphaG;
    float b = NdotV * NdotV;
    return 1 / (NdotV + sqrt(a + b - a * b));
}

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

vec3 sample_clearcoat_disney(vec3 viewVector, mat3 frameMatrix, float alpha, out float theta_h) {
    // https://www.youtube.com/watch?v=xFsJMUS94Fs
    // Generate random u and v between 0.0 and 1.0
    float u = random();
    float v = random();
    float alpha2 = alpha * alpha;

    // Compute spherical angles
    float theta = acos(sqrt((1 - pow(alpha2, (1- u)))/(1-alpha2)));
    float phi = 2 * M_PI * v;
    theta_h = theta;
    vec3 halfwayVector = frameMatrix*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

    // Compute light Vector l
    vec3 lightVector = 2*dot(viewVector,halfwayVector)*halfwayVector - viewVector;
    return lightVector;
}

vec3 sample_diffuse_disney(vec3 viewVector, mat3 frameMatrix, out float theta_h) {
    // https://www.youtube.com/watch?v=xFsJMUS94Fs
    // Generate random u and v between 0.0 and 1.0
    float u = random();
    float v = random();

    // Compute spherical angles
    float theta = asin(sqrt(u));
    float phi = 2 * M_PI * v;
    theta_h = theta;

    vec3 halfwayVector = frameMatrix*vec3(sin(theta)*cos(phi),sin(theta)*sin(phi),cos(theta));

    // Compute light Vector l
    vec3 lightVector = 2*dot(viewVector,halfwayVector)*halfwayVector - viewVector;
    return lightVector;
}

vec3 sample_specular_disney(vec3 viewVector, mat3 frameMatrix, float ax, float ay, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, out float theta_h) {
    float u = clamp(random(),0.05, 0.95);
    float v = clamp(random(),0.05, 0.95);
    float phi = atan((ay/ax) * tan(2*M_PI*u));
    float theta = acos(sqrt((1-v)/(1+((cos(phi)*cos(phi)/(ax*ax))+(sin(phi)*sin(phi)/(ay*ay)))*v)));
    theta_h = theta;

    // Pronblem:
    // sqrt macht Problem für negativ
    // NMormalize macht problem für nahe 0
    vec3 halfwayVector = normalize(sqrt((v)/(1-v))*(ax*cos(2*M_PI*u)*tangentVector + ay*sin(2*M_PI*u)*bitangentVector) + normalVector);
    vec3 lightVector = 2*dot(viewVector,halfwayVector)*halfwayVector - viewVector;
    return lightVector;
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

// -------------- BRDF Functions ------------------

// 1. Cook Torrance

// 1.1 Sampling
vec3 sample_cook_torrance(float metallic, float specular, float roughness, vec3 viewVector, mat3 frameMatrix, out bool specularHit) {
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
        specularHit = true;
        return sample_GGX(viewVector, roughness, frameMatrix);
    } else {
        specularHit = false;
        return sample_Lambertian(viewVector, frameMatrix);
    }
}

// 1.2 Evaluation
// 1.2.1 Evaluation (for a variable importance sampling PDF)
vec3 evaluate_cook_torrance(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColorDef) {
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

// 1.2.2 Optiomized Evaluation for fixed 2 PDFs (diffuse: 1/PI * sinThetaH * cosThetaH & specular: D * sinThetaH * cosTheta)
vec3 evaluate_cook_torrance_importance_sampling(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColorDef, bool specularHit, out float samplingPDF) {
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

    if (specularHit) {
        samplingPDF = D;
    } else {
        samplingPDF = (1.0/M_PI);
    }

    // ----------- Weighting in the PDF
    vec3 colorOut = diff + spec;
    return colorOut;
}


// 1.3 Combined call

vec3 cook_torrance(out vec3 directionOut, vec3 w, vec3 surfaceNormal, mat3 frame, vec3 isoSurfaceColorDef, out float samplingPDF, out bool specularHit) {
    // Source: https://www.youtube.com/watch?v=gya7x9H3mV0
    // Base Vectors
    // Sample Cook Torrance BRDF
    float roughness = clamp(parameters.roughness,0.05, 1.0);
    vec3 viewVector = normalize(-w);
    vec3 normalVector = normalize(surfaceNormal);

    // ----------- Importance Sampling
    // Importance Sampling for Diffuse: PDF 1/PI * cos(theta) * sin(theta)

    // Sampling and evaluating
    vec3 lightVector = sample_cook_torrance(parameters.metallic, parameters.specular, roughness, viewVector, frame, specularHit);
    directionOut = lightVector;
    return evaluate_cook_torrance_importance_sampling(viewVector, lightVector, normalVector, isoSurfaceColorDef, specularHit, samplingPDF);
}

// 2. Disney BRDF

// 2.1 Sampling
vec3 sample_disney2012(float metallic, float specular, float clearcoat, float clearcoatGloss, float roughness, float subsurface, vec3 viewVector, mat3 frameMatrix, float ax, float ay, vec3 normalVector, vec3 tangentVector, vec3 bitangentVector, out float theta_h, out bool specularHit, out bool clearcoatHit) {
    // Idea adapted from https://schuttejoe.github.io/post/disneybsdf/
    
    // Calculate probabilies for sampling the lobes
    //float metal = metallic;
    //float spec = (1.0 - roughness) * (1.0 + specular);
    //float dielectric = (1.0 - metallic);

    //float specularW = metal + spec;
    //float diffuseW = dielectric;
    
    float metal = metallic;
    float spec = (1.0 + specular) * (1.0 - roughness);
    float dielectric = (1.0 - metallic) * (1.0 + roughness);

    float specularW = metal + spec;
    float diffuseW = dielectric + subsurface;
    float clearcoatW = clamp(clearcoat, 0.0, 1.0);

    float norm = 1.0/(specularW + diffuseW + clearcoatW);

    float specularP = specularW * norm;
    float diffuseP = diffuseW * norm;
    float clearcoatP = clearcoatW * norm;

    float u = random();
    if(u < specularP) {
        specularHit = true;
        clearcoatHit = false;
        return sample_specular_disney(viewVector, frameMatrix, ax, ay, normalVector, tangentVector, bitangentVector, theta_h);
    } else if (u < specularP + diffuseP) {
        specularHit = false;
        clearcoatHit = false;
        return sample_diffuse_disney(viewVector, frameMatrix, theta_h);
    } else {
        specularHit = true;
        clearcoatHit = true;

        float alpha = mix(.1, .001, clearcoatGloss);
        return sample_clearcoat_disney(viewVector, frameMatrix, alpha, theta_h);
    }
}

// 2.2 Evaluation

// 2.2.1 Evaluation for variable importance sampling PDF
vec3 evaluate_disney_2012(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColorDef, float ax, float ay, vec3 x, vec3 y) {
    vec3 halfwayVector = normalize(lightVector + viewVector);

    // https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdf/BRDFBase.cpp#L409

    // Base Angles
    float theta_h = dot(halfwayVector, normalVector);
    float theta_d = dot(lightVector, halfwayVector);
    float theta_v = dot(viewVector, normalVector);
    float theta_l = dot(lightVector, normalVector);

    float NdotL = dot(lightVector, normalVector);
    float VdotH = dot(viewVector, halfwayVector);
    float NdotH = dot(halfwayVector, normalVector);

    // Base colors and values
    vec3 baseColor = isoSurfaceColorDef;
    // vec3(pow(baseColor[0], 2.2), pow(baseColor[1], 2.2), pow(baseColor[2], 2.2));
    vec3 col = baseColor;
    float lum = 0.3 * col[0] + 0.6 * col[1] + 0.1 * col[2];

    vec3 col_tint = lum > 0 ? col/lum: vec3(1.0);
    vec3 col_spec0 = mix(parameters.specular*0.08*mix(vec3(1.0),col_tint,parameters.specularTint), col, parameters.metallic);
    vec3 col_sheen = mix(vec3(1.0),col_tint,parameters.sheenTint);
    // Diffuse
    
    // Base Diffuse
    float f_d90 = 0.5 + 2 * parameters.roughness * pow(cos(theta_d), 2);
    float f_d = (1 + (f_d90 - 1) * (pow(1 - cos(theta_l), 5.0))) * (1 + (f_d90 - 1) * (pow(1 - cos(theta_v), 5.0)));
    
    // Subsurface Approximation: Inspired by Hanrahan-Krueger subsurface BRDF
    float f_d_subsurface_90 = parameters.roughness * pow(cos(theta_d), 2);
    float f_subsurface = (1.0 + (f_d_subsurface_90 - 1) * (pow(1 - cos(theta_l), 5.0))) * (1.0 + (f_d_subsurface_90 - 1.0) * (pow(1.0 - cos(theta_v), 5.0)));
    float f_d_subsurface = 1.25 * (f_subsurface * (1/(theta_l + theta_v) - 0.5) + 0.5);

    // Sheen
    // TODO: Add Fresnel Sschlick for theta d
    float f_h = pow((1 - theta_d), 5.0);
    vec3 sheen = f_h * parameters.sheen * col_sheen;

    vec3 diffuse = (col * mix(f_d, f_d_subsurface, parameters.subsurface) + sheen) * (1.0 - parameters.metallic);
    diffuse *= (1.0/M_PI);
    
    // Specular F (Schlick Fresnel Approximation)
    vec3 f_specular = mix(col_spec0, vec3(1.0), f_h);
    
    // GTR2
    float d_specular = 1 / (M_PI * ax * ay * sqr(sqr(dot(x, halfwayVector) / ax) + sqr(dot(y, halfwayVector) / ay)) + pow(theta_h, 2.0));

    // Specular G Smith G GGX
    float g_specular = 1 / (theta_v + sqrt(sqr(dot(x,lightVector)*ax) + sqr(dot(y,lightVector)*ay) + sqr(theta_v)));
    g_specular *= (1 / (theta_v + sqrt(sqr(dot(x, viewVector) * ax) + sqr(dot(y, viewVector) * ay) + sqr(theta_v))));

    float sinThetaH = sqrt(1-(NdotH*NdotH));
    //vec3 specular = f_specular * g_specular * 4 * NdotL * VdotH * sinThetaH / NdotH;
    // sinthetaH currently casues a problem, since it is 0 when roughness + 0.0 and metallic = 1.0
    // However the sinTheta Termn should usually be needed, theta however it not available in the specular sampling function.
    vec3 specular = f_specular * d_specular * g_specular * 4 * NdotL * VdotH * sinThetaH;

    // Clearcoat
    float f_clearcoat = mix(0.04,1.0,f_h);
    float d_clearcoat = GTR1(theta_h, mix(.1, .001, parameters.clearcoatGloss));
    float g_clearcoat = smithG_GGX(theta_l, 0.25) * smithG_GGX(theta_v, 0.25);
        
    float clear = parameters.clearcoat*f_clearcoat*g_clearcoat * NdotL * VdotH * sinThetaH;
    // Result        
    return diffuse + specular + clear;

}

// 2.2.2 Optimized Evaluation for fixed importance sampling PDFs
    // diffuse PDF:
    // specular PDF:
    // clearcoat PDF:
vec3 evaluate_disney_2012_importance_sampling(vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColorDef, float th, float ax, float ay, vec3 x, vec3 y, bool specularHit, bool clearcoatHit, out float samplingPDF) {
    vec3 halfwayVector = normalize(lightVector + viewVector);

    // https://github.com/wdas/brdf/blob/f39eb38620072814b9fbd5743e1d9b7b9a0ca18a/src/brdf/BRDFBase.cpp#L409

    // Base Angles
    float theta_h = dot(halfwayVector, normalVector);
    float theta_d = dot(lightVector, halfwayVector);
    float theta_v = dot(viewVector, normalVector);
    float theta_l = dot(lightVector, normalVector);

    float NdotL = dot(lightVector, normalVector);
    float VdotH = dot(viewVector, halfwayVector);
    float NdotH = dot(halfwayVector, normalVector);

    // Base colors and values
    vec3 baseColor = isoSurfaceColorDef;
    // vec3(pow(baseColor[0], 2.2), pow(baseColor[1], 2.2), pow(baseColor[2], 2.2));
    vec3 col = baseColor;
    float lum = 0.3 * col[0] + 0.6 * col[1] + 0.1 * col[2];

    vec3 col_tint = lum > 0 ? col/lum: vec3(1.0);
    vec3 col_spec0 = mix(parameters.specular*0.08*mix(vec3(1.0),col_tint,parameters.specularTint), col, parameters.metallic);
    vec3 col_sheen = mix(vec3(1.0),col_tint,parameters.sheenTint);
    // Diffuse
    
    // Base Diffuse
    float f_d90 = 0.5 + 2 * parameters.roughness * pow(cos(theta_d), 2);
    float f_d = (1 + (f_d90 - 1) * (pow(1 - cos(theta_l), 5.0))) * (1 + (f_d90 - 1) * (pow(1 - cos(theta_v), 5.0)));
    
    // Subsurface Approximation: Inspired by Hanrahan-Krueger subsurface BRDF
    float f_d_subsurface_90 = parameters.roughness * pow(cos(theta_d), 2);
    float f_subsurface = (1.0 + (f_d_subsurface_90 - 1) * (pow(1 - cos(theta_l), 5.0))) * (1.0 + (f_d_subsurface_90 - 1.0) * (pow(1.0 - cos(theta_v), 5.0)));
    float f_d_subsurface = 1.25 * (f_subsurface * (1/(theta_l + theta_v) - 0.5) + 0.5);

    // Sheen
    // TODO: Add Fresnel Sschlick for theta d
    float f_h = pow((1 - theta_d), 5.0);
    vec3 sheen = f_h * parameters.sheen * col_sheen;

    vec3 diffuse = (col * mix(f_d, f_d_subsurface, parameters.subsurface) + sheen) * (1.0 - parameters.metallic);
    
    // Specular F (Schlick Fresnel Approximation)
    vec3 f_specular = mix(col_spec0, vec3(1.0), f_h);
    
    // GTR2
    float d_specular = 1 / (M_PI * ax * ay * sqr(sqr(dot(x, halfwayVector) / ax) + sqr(dot(y, halfwayVector) / ay)) + pow(theta_h, 2.0));

    // Specular G Smith G GGX
    float g_specular = 1 / (theta_v + sqrt(sqr(dot(x,lightVector)*ax) + sqr(dot(y,lightVector)*ay) + sqr(theta_v)));
    g_specular *= (1 / (theta_v + sqrt(sqr(dot(x, viewVector) * ax) + sqr(dot(y, viewVector) * ay) + sqr(theta_v))));

    float sinThetaH = sin(th);
    float cosThetaH = NdotH;
    //vec3 specular = f_specular * g_specular * 4 * NdotL * VdotH * sinThetaH / NdotH;
    // sinthetaH currently casues a problem, since it is 0 when roughness + 0.0 and metallic = 1.0
    // However the sinTheta Termn should usually be needed, theta however it not available in the specular sampling function.
    vec3 specular = f_specular * g_specular * 4 * NdotL * VdotH * sin(th)/ NdotH;

    // Clearcoat
    float f_clearcoat = mix(0.04,1.0,f_h);
    float d_clearcoat = GTR1(theta_h, mix(.1, .001, parameters.clearcoatGloss));
    float g_clearcoat = smithG_GGX(theta_l, 0.25) * smithG_GGX(theta_v, 0.25);
        
    float clear = parameters.clearcoat*f_clearcoat*g_clearcoat*NdotL*VdotH * sin(th)/NdotH;
    // Result

    if (specularHit) {
        if(clearcoatHit) {
            samplingPDF = d_clearcoat;
        } else {
            samplingPDF = d_specular;
        }
    } else {
        samplingPDF = (1.0/M_PI);
    }

    return diffuse + specular + clear;
}

vec3 disney_2012(out vec3 directionOut, vec3 w, vec3 surfaceNormal, vec3 surfaceTangent, vec3 surfaceBitangent, mat3 frame, vec3 isoSurfaceColorDef, out float samplingPDF, out bool specularHit, out bool clearcoatHit) {
    // Sources:
    // Paper: https://blog.selfshadow.com/publications/s2012-shading-course/burley/s2012_pbs_disney_brdf_notes_v3.pdf
    // BRDF: https://github.com/wdas/brdf/blob/main/src/brdfs/disney.brdf
    // Specular D (2 lobes using GTR model)
    float aspect = sqrt(1 - parameters.anisotropic * 0.9);
    float ax = max(0.001, sqr(parameters.roughness) / aspect);
    float ay = max(0.001, sqr(parameters.roughness) * aspect);

    // Base Vectors
    // Sample Disney BRDF
    // vec3 sample_disney2012(float metallic, float specular, float clearcoat, float clearcoatGloss, float roughness, vec3 viewVector, mat3 frameMatrix, float ax, float ay)
    vec3 viewVector = normalize(-w);
    vec3 normalVector = normalize(surfaceNormal);
    vec3 x = normalize(surfaceTangent);
    vec3 y = normalize(surfaceBitangent);
    float th;

    vec3 lightVector = sample_disney2012(parameters.metallic, parameters.specular, parameters.clearcoat, parameters.clearcoatGloss, parameters.roughness, parameters.subsurface, viewVector, frame, ax, ay, normalVector, x, y, th, specularHit, clearcoatHit);
    directionOut = lightVector;
    return evaluate_disney_2012_importance_sampling(viewVector, lightVector, normalVector, isoSurfaceColorDef, th, ax, ay, x, y, specularHit, clearcoatHit, samplingPDF);
}

// 2.4 Combined Call

#if defined(ISOSURFACE_TYPE_DENSITY) || !defined(USE_TRANSFER_FUNCTION)
#define isoSurfaceColorDef parameters.isoSurfaceColor
#define DEFINE_ISO_SURFACE_COLOR float isoSurfaceOpacity = 1.0;
#else // defined(ISOSURFACE_TYPE_GRADIENT)
#define DEFINE_ISO_SURFACE_COLOR \
vec4 isoSurfaceColorAll = sampleCloudGradientColor(currentPoint); \
vec3 isoSurfaceColorDef = isoSurfaceColorAll.rgb; \
float isoSurfaceOpacity = isoSurfaceColorAll.a;
#endif

#define UNIFORM_SAMPLING
//#define USE_MIS // only for specular BRDF sampling

bool getIsoSurfaceHit(
        vec3 currentPoint, inout vec3 w, inout vec3 throughput
#if defined(USE_NEXT_EVENT_TRACKING_SPECTRAL) || defined(USE_NEXT_EVENT_TRACKING)
        , inout vec3 colorNee
#endif
) {
    bool useMIS = false;
    vec3 texCoords = (currentPoint - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
    texCoords = texCoords * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;
    vec3 surfaceNormal = computeGradient(texCoords);
    if (dot(w, surfaceNormal) > 0.0) {
        surfaceNormal = -surfaceNormal;
    }

    DEFINE_ISO_SURFACE_COLOR;
    if (isoSurfaceOpacity < 1e-4) {
        return false;
    }
    if (isoSurfaceOpacity < random()) {
        return false;
    }

    vec3 surfaceTangent;
    vec3 surfaceBitangent;
    ComputeDefaultBasis(surfaceNormal, surfaceTangent, surfaceBitangent);
    mat3 frame = mat3(surfaceTangent, surfaceBitangent, surfaceNormal);

    vec3 colorOut;
    vec3 dirOut;

    // frame, surfaceNormal, w, 

// -------------- BRDF ------------------

#ifdef SURFACE_BRDF_LAMBERTIAN
    // Lambertian BRDF is: R / pi
#ifdef UNIFORM_SAMPLING
    // Sampling PDF: 1/(2pi)
    dirOut = frame * sampleHemisphere(vec2(random(), random()));
#else
    // Sampling PDF: cos(theta) / pi
    dirOut = frame * sampleHemisphereCosineWeighted(vec2(random(), random()));
#endif
    float thetaOut = dot(surfaceNormal, dirOut);
#endif

#ifdef SURFACE_BRDF_BLINN_PHONG
    // http://www.thetenthplanet.de/archives/255
    const float n = 10.0;
    float norm = clamp(
            (n + 2.0) / (4.0 * M_PI * (exp2(-0.5 * n))),
            (n + 2.0) / (8.0 * M_PI), (n + 4.0) / (8.0 * M_PI));
    dirOut = frame * sampleHemisphere(vec2(random(), random()));
    vec3 halfwayVectorOut = normalize(-w + dirOut);
#endif

#ifdef SURFACE_BRDF_DISNEY
    bool specularHit;
    bool clearcoatHit;
    float samplingPDF;
    colorOut = disney_2012(dirOut, w, surfaceNormal, surfaceTangent, surfaceBitangent, frame, isoSurfaceColorDef, samplingPDF, specularHit, clearcoatHit);
    colorOut *= dot(dirOut, normalize(surfaceNormal));

    #if (defined(USE_ISOSURFACE_NEE) && (defined(USE_NEXT_EVENT_TRACKING_SPECTRAL) || defined(USE_NEXT_EVENT_TRACKING)))
        if(specularHit == true) {
            useMIS = true;
        } else {
            useMIS = false;
        }
    #endif

#endif

// Cook Torrance BRDF

#ifdef SURFACE_BRDF_COOK_TORRANCE
    bool specularHit;
    float samplingPDF;
    colorOut = cook_torrance(dirOut, w, surfaceNormal, frame, isoSurfaceColorDef, samplingPDF, specularHit);
    colorOut *= dot(dirOut, surfaceNormal);

    #if (defined(USE_ISOSURFACE_NEE) && (defined(USE_NEXT_EVENT_TRACKING_SPECTRAL) || defined(USE_NEXT_EVENT_TRACKING)))
        if(specularHit == true) {
            useMIS = true;
        } else {
            useMIS = false;
        }
    #endif

#endif

// Next Event Tracking NEE Start

#if (defined(USE_ISOSURFACE_NEE) && (defined(USE_NEXT_EVENT_TRACKING_SPECTRAL) || defined(USE_NEXT_EVENT_TRACKING)))
    float pdfSkyboxNee;
    vec3 dirSkyboxNee = importanceSampleSkybox(pdfSkyboxNee);
    if (dot(surfaceNormal, dirSkyboxNee) > 0.0) {
        float thetaNee = dot(surfaceNormal, dirSkyboxNee);

#ifdef SURFACE_BRDF_LAMBERTIAN
#ifdef UNIFORM_SAMPLING
        float pdfSamplingNee;
        if(useMIS) {
            pdfSamplingNee = 1.0 / (2.0 * M_PI);
        }
        float pdfSamplingOut = 1.0 / (2.0 * M_PI);
#else
        float pdfSamplingNee;
        if(useMIS) {
            pdfSamplingNee = thetaNee / M_PI;
        }
        float pdfSamplingOut = thetaOut / M_PI;
#endif
#endif
        // NEE Blinn Phong PDF
#ifdef SURFACE_BRDF_BLINN_PHONG
        float pdfSamplingNee;
        if(useMIS) {
            pdfSamplingNee = 1.0 / (2.0 * M_PI);
        }
        float pdfSamplingOut = 1.0 / (2.0 * M_PI);
#endif

#ifdef SURFACE_BRDF_LAMBERTIAN
        //vec3 brdfOut = isoSurfaceColorDef / M_PI;
        //vec3 brdfNee = isoSurfaceColorDef / M_PI;
        vec3 rdfOut = isoSurfaceColorDef / M_PI * thetaOut;
        vec3 rdfNee = isoSurfaceColorDef / M_PI * thetaNee;
        colorOut = rdfOut / pdfSamplingOut;

#endif
        // NEE Blinn Phong RDF
#ifdef SURFACE_BRDF_BLINN_PHONG
        // http://www.thetenthplanet.de/archives/255
        vec3 halfwayVectorNee = normalize(-w + dirSkyboxNee);
        vec3 rdfOut = isoSurfaceColorDef * (pow(max(dot(surfaceNormal, halfwayVectorOut), 0.0), n) / norm);
        vec3 rdfNee = isoSurfaceColorDef * (pow(max(dot(surfaceNormal, halfwayVectorNee), 0.0), n) / norm);
        colorOut = rdfOut / pdfSamplingOut;
#endif

#ifdef SURFACE_BRDF_COOK_TORRANCE
        // TODO: Call brdf for viewVector and vector to sun dirSkyBoxNee, but this must take NEE sampling into account and not the brdf importance sampling
        // need to create a new function
        vec3 rdfNee = evaluate_cook_torrance(normalize(-w), dirSkyboxNee, normalize(surfaceNormal), isoSurfaceColorDef)*dot(dirSkyboxNee, normalize(surfaceNormal));
        // TODO: Difference between pdfSamplingNee and pdfSkyboxNee and pdfSamplingOut
        vec3 halfwayVector = normalize(dirOut + (-1.0*w));
        float cosThetaH = dot(halfwayVector, surfaceNormal);
        float sinThetaH = sqrt(1.0/(cosThetaH*cosThetaH));

        vec3 halfwayVectorNee  = normalize(dirSkyboxNee + (-1.0*w));
        float cosThetaHNee = dot(halfwayVectorNee, surfaceNormal);
        float sinThetaHNee = sqrt(1.0 - (cosThetaHNee*cosThetaHNee));

        float pdfSamplingOut = samplingPDF * cosThetaH * sinThetaH;
        float pdfSamplingNee = samplingPDF * cosThetaHNee * sinThetaHNee;
#endif

#ifdef SURFACE_BRDF_DISNEY
        // (vec3 viewVector, vec3 lightVector, vec3 normalVector, vec3 isoSurfaceColorDef, float ax, float ay, vec3 x, vec3 y)
        float aspect = sqrt(1 - parameters.anisotropic * 0.9);
        float ax = max(0.001, sqr(parameters.roughness) / aspect);
        float ay = max(0.001, sqr(parameters.roughness) * aspect);

        // Base Vectors
        // Sample Disney BRDF
        // vec3 sample_disney2012(float metallic, float specular, float clearcoat, float clearcoatGloss, float roughness, vec3 viewVector, mat3 frameMatrix, float ax, float ay)
        vec3 viewVector = normalize(-w);
        vec3 normalVector = normalize(surfaceNormal);
        vec3 x = normalize(surfaceTangent);
        vec3 y = normalize(surfaceBitangent);
        vec3 rdfNee = evaluate_disney_2012(normalize(-w), dirSkyboxNee, normalize(surfaceNormal), isoSurfaceColorDef, ax, ay, x, y)*dot(dirSkyboxNee, normalize(surfaceNormal));
        // TODO: Difference between pdfSamplingNee and pdfSkyboxNee and pdfSamplingOut
        vec3 halfwayVector = normalize(dirOut + (-1.0*w));
        float cosThetaH = dot(halfwayVector, normalVector);
        float sinThetaH = sqrt(1.0/(cosThetaH*cosThetaH));

        vec3 halfwayVectorNee  = normalize(dirSkyboxNee + (-1.0*w));
        float cosThetaHNee = dot(halfwayVectorNee, normalVector);
        float sinThetaHNee = sqrt(1.0 - (cosThetaHNee*cosThetaHNee));

        float pdfSamplingOut;
        float pdfSamplingNee;
        if (specularHit) {
                pdfSamplingOut = samplingPDF * cosThetaH;
                pdfSamplingNee = samplingPDF * cosThetaHNee;
        } else {
            pdfSamplingOut = samplingPDF * cosThetaH * sinThetaH;
            pdfSamplingNee = samplingPDF * cosThetaHNee * sinThetaHNee;
        }

#endif

if(useMIS){
     // NEE with MIS.
        // TODO: If dirOut is importance sampled from BRDF instead of cos term, use brdfOut and brdfNee
        // instead of pdfSamplingOut and pdfSamplingNee.
        // Power heuristic with beta=2: https://www.pbr-book.org/3ed-2018/Monte_Carlo_Integration/Importance_Sampling
        //float weightNee = pdfSkyboxNee * pdfSkyboxNee / (pdfSkyboxNee * pdfSkyboxNee + pdfSamplingNee * pdfSamplingNee);
        //float weightOut = pdfSamplingOut * pdfSamplingOut / (pdfSkyboxOut * pdfSkyboxOut + pdfSamplingOut * pdfSamplingOut);
        float pdfSkyboxOut = evaluateSkyboxPDF(dirOut);
        float weightNee = pdfSkyboxNee / (pdfSkyboxNee + pdfSamplingNee);
        float weightOut = pdfSamplingOut / (pdfSkyboxOut + pdfSamplingOut);
        colorNee +=
                throughput * rdfNee * weightNee / pdfSkyboxNee * calculateTransmittance(currentPoint + dirSkyboxNee * 1e-4, dirSkyboxNee)
                * (sampleSkybox(dirSkyboxNee) + sampleLight(dirSkyboxNee));
        colorNee +=
                throughput * weightOut * colorOut * calculateTransmittance(currentPoint + dirOut * 1e-4, dirOut)
                * (sampleSkybox(dirOut) + sampleLight(dirOut));
} else {
    // Normal NEE.
        colorNee +=
                throughput * rdfNee / pdfSkyboxNee * calculateTransmittance(currentPoint + dirSkyboxNee * 1e-4, dirSkyboxNee)
                * (sampleSkybox(dirSkyboxNee) + sampleLight(dirSkyboxNee));
}
    } else {
    // Abort Next Event Tracking if Skybox sample is behind the surface
    // Just compute colorOut and no colorNEE
#endif

// This Part until the } (exclusive) is not part of NEE
#if defined(SURFACE_BRDF_LAMBERTIAN)
        // Lambertian BRDF is: R / pi
#ifdef UNIFORM_SAMPLING
        // Sampling PDF: 1/(2pi)
        colorOut = 2.0 * isoSurfaceColorDef * thetaOut;
#else
        // Sampling PDF: cos(theta) / pi
        colorOut = isoSurfaceColorDef;
#endif
#endif

#ifdef SURFACE_BRDF_BLINN_PHONG
        // http://www.thetenthplanet.de/archives/255
        float rdf = pow(max(dot(surfaceNormal, halfwayVectorOut), 0.0), n);
        colorOut = 2.0 * M_PI * isoSurfaceColorDef * (rdf / norm);
#endif

#if (defined(USE_ISOSURFACE_NEE) && (defined(USE_NEXT_EVENT_TRACKING_SPECTRAL) || defined(USE_NEXT_EVENT_TRACKING)))
    }
#endif
// Next Event Tracking NEE End


    w = dirOut;
    throughput *= colorOut;
    return true;
}
#endif


#if defined(USE_ISOSURFACE_RENDERING) || (defined(USE_ISOSURFACES) && defined(USE_RAY_MARCHING_EMISSION_ABSORPTION))
#include "Lighting.glsl"

// Direct illumination
vec3 getIsoSurfaceHitDirect(vec3 currentPoint, vec3 w, inout vec3 surfaceNormal) {
    vec3 texCoords = (currentPoint - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
    texCoords = texCoords * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;
    surfaceNormal = computeGradient(texCoords);
    if (dot(cameraPosition - currentPoint, surfaceNormal) < 0.0) {
        surfaceNormal = -surfaceNormal;
    }
    DEFINE_ISO_SURFACE_COLOR;
    vec3 color = blinnPhongShadingSurface(isoSurfaceColorDef, currentPoint, surfaceNormal);
    return color;
}
#endif

#include "FloatPack.glsl"
