/**
 * MIT License
 *
 * Copyright (c) 2021-2022, Christoph Neuhauser, Ludwig Leonard
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
    pdf = 1.0 / (4 * PI);
    return randomDirection(vec3(1.0, 0.0, 0.0));
}

float evaluateSkyboxPDF(vec3 sampledDir) {
    return 1.0 / (4 * PI);
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
    
    // Make sure there is no infinity in the skybox
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
float sampleCloudRaw(pnanovdb_readaccessor_t accessor, in vec3 pos) {

    pnanovdb_buf_t buf = pnanovdb_buf_t(0);
    pnanovdb_grid_handle_t gridHandle = pnanovdb_grid_handle_t(pnanovdb_address_null());
    vec3 posIndex = pnanovdb_grid_world_to_indexf(buf, gridHandle, pos);
    posIndex = floor(posIndex);
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, ivec3(posIndex));
    return pnanovdb_read_float(buf, address);
}
#elif defined(GRID_INTERPOLATION_STOCHASTIC)
float sampleCloudRaw(pnanovdb_readaccessor_t accessor, in vec3 pos) {

    pnanovdb_buf_t buf = pnanovdb_buf_t(0);
    pnanovdb_grid_handle_t gridHandle = pnanovdb_grid_handle_t(pnanovdb_address_null());
    vec3 posIndex = pnanovdb_grid_world_to_indexf(buf, gridHandle, pos);
    posIndex = floor(posIndex + vec3(random() - 0.5, random() - 0.5, random() - 0.5));
    pnanovdb_address_t address = pnanovdb_readaccessor_get_value_address(
            PNANOVDB_GRID_TYPE_FLOAT, buf, accessor, ivec3(posIndex));
    return pnanovdb_read_float(buf, address);
}
#elif defined(GRID_INTERPOLATION_TRILINEAR)
float sampleCloudRaw(pnanovdb_readaccessor_t accessor, in vec3 pos) {

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
float sampleCloudRaw(in vec3 coord) {

#if defined(GRID_INTERPOLATION_STOCHASTIC)
    ivec3 dim = textureSize(gridImage, 0);
    coord += vec3(random() - 0.5, random() - 0.5, random() - 0.5) / dim;
#endif
    return texture(gridImage, coord).x;
}
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
vec4 sampleCloudColorAndDensity(
#ifdef USE_NANOVDB
        pnanovdb_readaccessor_t accessor,
#endif
        in vec3 pos) {
    // Idea: Returns (color.rgb, density).
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
#if defined(FLIP_YZ)
    coord = coord.xzy;
#endif
    coord = coord * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;
    float densityRaw = sampleCloudRaw(
#ifdef USE_NANOVDB
            accessor,
#endif
            coord);
    //return densityRaw;
    return texture(transferFunctionTexture, densityRaw);
}
#endif

#ifdef USE_TRANSFER_FUNCTION
float sampleCloud(
#ifdef USE_NANOVDB
        pnanovdb_readaccessor_t accessor,
#endif
        in vec3 pos) {
    // Idea: Returns (color.rgb, density).
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
#if defined(FLIP_YZ)
    coord = coord.xzy;
#endif
    coord = coord * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;
    float densityRaw = sampleCloudRaw(
#ifdef USE_NANOVDB
            accessor,
#endif
            pos);
    return texture(transferFunctionTexture, densityRaw).a;
}
vec4 sampleCloudDensityEmission(
#ifdef USE_NANOVDB
        pnanovdb_readaccessor_t accessor,
#endif
        in vec3 pos) {
    // Idea: Returns (color.rgb, density).
    float densityRaw = sampleCloudRaw(
#ifdef USE_NANOVDB
            accessor,
#endif
            pos);
    return texture(transferFunctionTexture, densityRaw);
}
#else
float sampleCloud(
#ifdef USE_NANOVDB
        pnanovdb_readaccessor_t accessor,
#endif
        in vec3 pos) {
    // transform world pos to density grid pos
    vec3 coord = (pos - parameters.boxMin) / (parameters.boxMax - parameters.boxMin);
#if defined(FLIP_YZ)
    coord = coord.xzy;
#endif
    coord = coord * (parameters.gridMax - parameters.gridMin) + parameters.gridMin;

    return sampleCloudRaw(
#ifdef USE_NANOVDB
            accessor,
#endif
            coord);// + parameters.extinction.g / parameters.extinction.r * .01;
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
        texture(gridImage, coord - vec3(1, 0, 0) / dim).x - texture(gridImage, coord + vec3(1, 0, 0) / dim).x,
        texture(gridImage, coord - vec3(0, 1, 0) / dim).x - texture(gridImage, coord + vec3(0, 1, 0) / dim).x,
        texture(gridImage, coord - vec3(0, 0, 1) / dim).x - texture(gridImage, coord + vec3(0, 0, 1) / dim).x
    ) / dim * 100;
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
