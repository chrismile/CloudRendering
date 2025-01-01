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

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/euler_angles.hpp>
#include <glm/gtx/quaternion.hpp>

#include <Math/Math.hpp>
#ifndef DISABLE_IMGUI
#include <ImGui/Widgets/PropertyEditor.hpp>
#endif

#include "RotationWidget.hpp"

#ifndef DISABLE_IMGUI
bool RotationWidget::renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) {
    bool changed = false;
    changed = propertyEditor.addCombo(
            "Rotation Mode", (int*)&orientationMode,
            ORIENTATION_MODE_NAMES, IM_ARRAYSIZE(ORIENTATION_MODE_NAMES)) || changed;
    if (orientationMode == OrientationMode::EULER_ANGLES) {
        changed = propertyEditor.addSliderFloat3(
                "Rotation Euler Angles (YXZ)", &eulerAngles.x, -sgl::PI, sgl::PI) || changed;
    } else if (orientationMode == OrientationMode::YAW_PITCH_ROLL) {
        changed = propertyEditor.addSliderFloat3(
                "Rotation Yaw-Pitch-Roll", &yawPitchRoll.x, -sgl::PI, sgl::PI) || changed;
    } else if (orientationMode == OrientationMode::ANGLE_AXIS) {
        changed = propertyEditor.addSliderFloat3("Rotation Axis", &axis.x, -1.0f, 1.0f) || changed;
        changed = propertyEditor.addSliderFloat("Rotation Angle", &angle, -sgl::PI, sgl::PI) || changed;
    } else {
        changed = propertyEditor.addSliderFloat4("Rotation Quaternion", &quaternion.x, -1.0f, 1.0f) || changed;
    }
    return changed;
}
#endif

glm::mat3 RotationWidget::getMat3() {
    if (orientationMode == OrientationMode::EULER_ANGLES) {
        return glm::eulerAngleYXZ(eulerAngles.x, eulerAngles.y, eulerAngles.z);
    } else if (orientationMode == OrientationMode::YAW_PITCH_ROLL) {
        return glm::yawPitchRoll(yawPitchRoll.x, yawPitchRoll.y, yawPitchRoll.z);
    } else if (orientationMode == OrientationMode::ANGLE_AXIS) {
        auto axisLength = glm::length(axis);
        glm::vec3 normalizedAxis = axis;
        if (axisLength > 1e-6f) {
            normalizedAxis /= axisLength;
        }
        return glm::toMat3(glm::angleAxis(angle, normalizedAxis));
    } else {
        return glm::toMat3(quaternion);
    }
}

void RotationWidget::setEulerAngles(const glm::vec3& _eulerAngles) {
    orientationMode = OrientationMode::EULER_ANGLES;
    eulerAngles = _eulerAngles;
}

void RotationWidget::setYawPitchRoll(const glm::vec3& _yawPitchRoll) {
    orientationMode = OrientationMode::YAW_PITCH_ROLL;
    yawPitchRoll = _yawPitchRoll;
}

void RotationWidget::setAngleAxis(const glm::vec3& _axis, float _angle) {
    orientationMode = OrientationMode::ANGLE_AXIS;
    axis = _axis;
    angle = _angle;
}

void RotationWidget::setQuaternion(const glm::quat& _quaternion) {
    orientationMode = OrientationMode::QUATERNION;
    quaternion = _quaternion;
}
