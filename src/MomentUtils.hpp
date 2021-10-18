/**
 * This file is part of an GLSL port of the HLSL code accompanying the paper "Moment-Based Order-Independent
 * Transparency" by Münstermann, Krumpen, Klein, and Peters (http://momentsingraphics.de/?page_id=210).
 * The original code was released in accordance to CC0 (https://creativecommons.org/publicdomain/zero/1.0/).
 *
 * This port is released under the terms of the MIT License.
 *
 * Copyright (c) 2021, Christoph Neuhauser
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

#ifndef CLOUD_RENDERING_MOMENT_UTILS_HPP
#define CLOUD_RENDERING_MOMENT_UTILS_HPP

#include <glm/glm.hpp>

struct MomentOITUniformData
{
    glm::vec4 wrapping_zone_parameters;
    float overestimation;
    float moment_bias;
};

enum MBOITPixelFormat {
    MBOIT_PIXEL_FORMAT_FLOAT_32, MBOIT_PIXEL_FORMAT_UNORM_16
};

// Circle constant.
#ifndef M_PI
#define M_PI 3.14159265358979323f
#endif

/**
 * This utility function turns an angle from 0 to 2*pi into a parameter that grows monotonically as function of the
 * input. It is designed to be efficiently computable from a point on the unit circle and must match the function in
 * TrigonometricMomentMath.glsl.
 * @param pOutMaxParameter Set to the maximal possible output value.
 */
float circleToParameter(float angle, float* pOutMaxParameter = nullptr);

/**
 * Given an angle in radians providing the size of the wrapping zone, this function computes all constants required by
 * the shader.
 */
void computeWrappingZoneParameters(glm::vec4& p_out_wrapping_zone_parameters,
        float new_wrapping_zone_angle = 0.1f * M_PI);

#endif //CLOUD_RENDERING_MOMENT_UTILS_HPP
