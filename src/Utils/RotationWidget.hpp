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

#ifndef CLOUDRENDERING_ROTATIONWIDGET_HPP
#define CLOUDRENDERING_ROTATIONWIDGET_HPP

#include <glm/mat3x3.hpp>
#include <glm/gtc/quaternion.hpp>

namespace sgl {
class PropertyEditor;
}

enum class OrientationMode {
    EULER_ANGLES, YAW_PITCH_ROLL, ANGLE_AXIS, QUATERNION
};
const char* const ORIENTATION_MODE_NAMES[] = {
    "Euler Angles (YXZ)", "Yaw-Pitch-Roll", "Angle-Axis", "Quaternion"
};

class RotationWidget {
public:
    bool renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor);
    glm::mat3 getMat3();
    void setEulerAngles(const glm::vec3& _eulerAngles);
    void setYawPitchRoll(const glm::vec3& _yawPitchRoll);
    void setAngleAxis(const glm::vec3& _axis, float _angle);
    void setQuaternion(const glm::quat& _quaternion);

private:
    OrientationMode orientationMode = OrientationMode::EULER_ANGLES;
    // OrientationMode::EULER_ANGLES
    glm::vec3 eulerAngles = glm::vec3(0.0f);
    // OrientationMode::YAW_PITCH_ROLL
    glm::vec3 yawPitchRoll = glm::vec3(0.0f);
    // OrientationMode::ANGLE_AXIS
    glm::vec3 axis = glm::vec3(0.0f);
    float angle = 0.0f;
    // OrientationMode::QUATERNION
    glm::quat quaternion = glm::identity<glm::quat>();
};

#endif //CLOUDRENDERING_ROTATIONWIDGET_HPP
