/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 *
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef CLOUDRENDERING_RENDERSETTINGS_HPP
#define CLOUDRENDERING_RENDERSETTINGS_HPP

enum class VptMode {
    DELTA_TRACKING, SPECTRAL_DELTA_TRACKING, RATIO_TRACKING, DECOMPOSITION_TRACKING, RESIDUAL_RATIO_TRACKING,
    NEXT_EVENT_TRACKING, NEXT_EVENT_TRACKING_SPECTRAL,
    ISOSURFACE_RENDERING, RAY_MARCHING_EMISSION_ABSORPTION
};
const char* const VPT_MODE_NAMES[] = {
        "Delta Tracking", "Delta Tracking (Spectral)", "Ratio Tracking",
        "Decomposition Tracking", "Residual Ratio Tracking", "Next Event Tracking", "Next Event Tracking (Spectral)",
        "Isosurfaces", "Ray Marching (Emission/Absorption)"
};

// Only for VptMode::RAY_MARCHING_EMISSION_ABSORPTION.
enum class CompositionModel {
    ALPHA_BLENDING, AVERAGE, MAXIMUM_INTENSITY_PROJECTION
};
const char* const COMPOSITION_MODEL_NAMES[] = {
        "Alpha Blending", "Average", "Maximum Intensity Projection"
};

enum class GridInterpolationType {
    NEAREST, //< Take sample at voxel closest to (i, j, k)
    STOCHASTIC, //< Sample within (i - 0.5, j - 0.5, k - 0,5) and (i + 0.5, j + 0.5, k + 0,5).
    TRILINEAR //< Sample all 8 neighbors and do trilinear interpolation.
};
const char* const GRID_INTERPOLATION_TYPE_NAMES[] = {
        "Nearest", "Stochastic", "Trilinear"
};

/**
 * Choices of collision probabilities for spectral delta tracking.
 * For more details see: https://jannovak.info/publications/SDTracking/SDTracking.pdf
 */
enum class SpectralDeltaTrackingCollisionProbability {
    MAX_BASED, AVG_BASED, PATH_HISTORY_AVG_BASED
};
const char* const SPECTRAL_DELTA_TRACKING_COLLISION_PROBABILITY_NAMES[] = {
        "Max-based", "Avg-based", "Path History Avg-based"
};

enum class IsosurfaceType {
    DENSITY, GRADIENT
};
const char* const ISOSURFACE_TYPE_NAMES[] = {
        "Density", "Gradient"
};

enum class SurfaceBrdf {
    LAMBERTIAN, BLINN_PHONG
};
const char* const SURFACE_BRDF_NAMES[] = {
        "Lambertian", "Blinn Phong"
};

struct CameraPose {
    glm::vec3 position;
    glm::vec3 front;
    glm::vec3 right;
    glm::vec3 up;
    float fovy;
    float viewportWidth;
    float viewportHeight;
};

#endif //CLOUDRENDERING_RENDERSETTINGS_HPP
