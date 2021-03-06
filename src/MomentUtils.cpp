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

#include <cmath>

#include "MomentUtils.hpp"

/**
 * This utility function turns an angle from 0 to 2*pi into a parameter that grows monotonically as function of the
 * input. It is designed to be efficiently computable from a point on the unit circle and must match the function in
 * TrigonometricMomentMath.glsl.
 * @param pOutMaxParameter Set to the maximal possible output value.
 */
float circleToParameter(float angle, float* pOutMaxParameter /* = nullptr */) {
    float x = std::cos(angle);
    float y = std::sin(angle);
    float result = std::abs(y) - std::abs(x);
    result = (x<0.0f) ? (2.0f - result) : result;
    result = (y<0.0f) ? (6.0f - result) : result;
    result += (angle >= 2.0f * M_PI) ? 8.0f : 0.0f;
    if (pOutMaxParameter) {
        (*pOutMaxParameter) = 7.0f;
    }
    return result;
}


/**
 * Given an angle in radians providing the size of the wrapping zone, this function computes all constants required by
 * the shader.
 */
void computeWrappingZoneParameters(glm::vec4& p_out_wrapping_zone_parameters,
        float new_wrapping_zone_angle /* = 0.1f * M_PI */) {
    p_out_wrapping_zone_parameters[0] = new_wrapping_zone_angle;
    p_out_wrapping_zone_parameters[1] = M_PI - 0.5f * new_wrapping_zone_angle;
    if (new_wrapping_zone_angle <= 0.0f) {
        p_out_wrapping_zone_parameters[2] = 0.0f;
        p_out_wrapping_zone_parameters[3] = 0.0f;
    }
    else {
        float zone_end_parameter;
        float zone_begin_parameter = circleToParameter(2.0f * M_PI - new_wrapping_zone_angle, &zone_end_parameter);
        p_out_wrapping_zone_parameters[2] = 1.0f / (zone_end_parameter - zone_begin_parameter);
        p_out_wrapping_zone_parameters[3] = 1.0f - zone_end_parameter * p_out_wrapping_zone_parameters[2];
    }
}

