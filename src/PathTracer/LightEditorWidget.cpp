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

#include <json/json.h>

#include <Math/Math.hpp>
#include <Utils/Convert.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Buffers/Buffer.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>

#include "LightEditorWidget.hpp"

LightEditorWidget::LightEditorWidget(sgl::vk::Renderer* renderer)
        : renderer(renderer), propertyEditor(new sgl::PropertyEditor("Light Editor", showWindow)) {
    Light light0;
    light0.color = glm::vec3(1.0f, 1.0f, 1.0f);
    light0.intensity = 1.0f;
    light0.position = glm::vec3(0.68f, 0.22f, 0.70f);
    light0.lightSpace = LightSpace::VIEW_ORIENTATION;
    addLight(light0);

    Light light1;
    light1.color = glm::vec3(1.0f, 0.784f, 0.706f);
    light1.intensity = 0.15f;
    light1.position = glm::vec3(-0.56f, 0.24f, 0.80f);
    light1.lightSpace = LightSpace::VIEW_ORIENTATION;
    addLight(light1);

    Light light2;
    light2.color = glm::vec3(0.235f, 0.626f, 1.0f);
    light2.intensity = 1.5f;
    light2.position = glm::vec3(-0.82f, 0, -0.57f);
    light2.lightSpace = LightSpace::VIEW_ORIENTATION;
    addLight(light2);

    propertyEditor->setInitWidthValues(sgl::ImGuiWrapper::get()->getScaleDependentSize(280.0f));
}

LightEditorWidget::~LightEditorWidget() {
    delete propertyEditor;
}

void LightEditorWidget::addLight(const Light& light) {
    lights.push_back(light);
    lightCreationIndices.push_back(currentLightIdx++);
    recreateLightBuffer();
    updateLightBuffer();
}

void LightEditorWidget::removeLight(uint32_t lightIdx) {
    if (lightIdx < uint32_t(lights.size())) {
        lights.erase(lights.begin() + lightIdx);
        lightCreationIndices.erase(lightCreationIndices.begin() + lightIdx);
        recreateLightBuffer();
        updateLightBuffer();
    }
}

void LightEditorWidget::recreateLightBuffer() {
    if (lights.empty()) {
        lightsBuffer = {};
    } else {
        lightsBuffer = std::make_shared<sgl::vk::Buffer>(
                renderer->getDevice(), sizeof(Light) * lights.size(),
                VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VMA_MEMORY_USAGE_GPU_ONLY);
    }
}

void LightEditorWidget::updateLightBuffer() {
    if (!lightsBuffer) {
        return;
    }
    if (renderer->getIsCommandBufferInRecordingState()) {
        lightsBuffer->updateData(
                sizeof(Light) * lights.size(), lights.data(), renderer->getVkCommandBuffer());
        renderer->insertMemoryBarrier(
                VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_UNIFORM_READ_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
    } else {
        auto commandBuffer = renderer->getDevice()->beginSingleTimeCommands();
        lightsBuffer->updateData(sizeof(Light) * lights.size(), lights.data(), commandBuffer);
        renderer->getDevice()->endSingleTimeCommands(commandBuffer);
    }
}


bool LightEditorWidget::renderGui() {
    if (!showWindow) {
        return false;
    }

    bool reRender = false;
    bool shallDeleteElement = false;
    size_t deleteElementId = 0;

    sgl::ImGuiWrapper::get()->setNextWindowStandardPosSize(
            standardPositionX, standardPositionY, standardWidth, standardHeight);
    if (propertyEditor->begin()) {
        if (propertyEditor->beginTable()) {
            for (size_t i = 0; i < lights.size(); i++) {
                std::string lightName =
                        "Light #" + std::to_string(i) + "###light" + std::to_string(lightCreationIndices.at(i));
                bool beginNode = propertyEditor->beginNode(lightName);
                ImGui::SameLine();
                float indentWidth = ImGui::GetContentRegionAvail().x;
                ImGui::Indent(indentWidth);
                std::string buttonName = "X###x_renderer" + std::to_string(i);
                if (ImGui::Button(buttonName.c_str())) {
                    shallDeleteElement = true;
                    deleteElementId = i;
                }
                ImGui::Unindent(indentWidth);
                if (beginNode) {
                    if (renderGuiLight(i)) {
                        reRender = true;
                    }
                    propertyEditor->endNode();
                }
            }
        }
        propertyEditor->endTable();

        if (ImGui::Button("New Light")) {
            Light light{};
            light.lightSpace = LightSpace::WORLD;
            light.position = glm::vec3(1.0f, 0.0f, 0.0f);
            light.useDistance = true;
            addLight(light);
            reRender = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("New Headlight")) {
            Light light{};
            light.lightSpace = LightSpace::VIEW;
            light.position = glm::vec3(0.0f, 0.0f, 0.0f);
            light.useDistance = true;
            addLight(light);
            reRender = true;
        }
    }
    propertyEditor->end();

    if (shallDeleteElement) {
        removeLight(deleteElementId);
        reRender = true;
    }

    if (reRender) {
        updateLightBuffer();
    }

    return reRender;
}

static glm::vec3 stringToVec3(const std::string& s) {
    std::vector<float> vecList;
    sgl::splitStringWhitespaceTyped<float>(s, vecList);
    if (vecList.size() != 3) {
        sgl::Logfile::get()->throwError("Error in stringToVec3: Invalid number of entries.");
    }
    return { vecList[0], vecList[1], vecList[2] };
}

void LightEditorWidget::setLightProperty(uint32_t lightIdx, const std::string& key, const std::string& value) {
    if (lightIdx >= lights.size()) {
        sgl::Logfile::get()->throwError("Error in LightEditorWidget::setLightProperty: Invalid light index.");
    }
    auto& light = lights.at(lightIdx);
    if (key == "type") {
        for (int i = 0; i < IM_ARRAYSIZE(LIGHT_TYPE_NAMES); i++) {
            if (value == LIGHT_TYPE_NAMES[i]) {
                light.lightType = LightType(i);
                break;
            }
        }
    } else if (key == "color") {
        light.color = stringToVec3(value);
    } else if (key == "intensity") {
        light.intensity = sgl::fromString<float>(value);
    } else if (key == "use_distance") {
        light.intensity = sgl::fromString<bool>(value);
    } else if (key == "position" || key == "direction") {
        light.position = stringToVec3(value);
    } else if (key == "space") {
        for (int i = 0; i < IM_ARRAYSIZE(LIGHT_SPACE_NAMES); i++) {
            if (value == LIGHT_SPACE_NAMES[i]) {
                light.lightSpace = LightSpace(i);
                break;
            }
        }
    } else if (key == "spot_total_width") {
        light.spotTotalWidth = sgl::fromString<float>(value);
    } else if (key == "spot_falloff_start") {
        light.spotFalloffStart = sgl::fromString<float>(value);
    }
}

bool LightEditorWidget::loadFromFile(const std::string& filePath) {
    std::ifstream jsonFileStream(filePath.c_str());
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errorString;
    Json::Value root;
    if (!parseFromStream(builder, jsonFileStream, &root, &errorString)) {
        sgl::Logfile::get()->writeError(errorString);
        return false;
    }
    jsonFileStream.close();

    Json::Value& lightsNode = root["lights"];
    for (Json::Value& lightNode : lightsNode) {
        Light light{};

        if (lightNode.isMember("type")) {
            std::string lightTypeString = lightNode["type"].asString();
            for (int i = 0; i < IM_ARRAYSIZE(LIGHT_TYPE_NAMES); i++) {
                if (lightTypeString == LIGHT_TYPE_NAMES[i]) {
                    light.lightType = LightType(i);
                    break;
                }
            }
        }

        if (lightNode.isMember("color")) {
            auto colorNode = lightNode["color"];
            light.color = glm::vec3(colorNode["r"].asFloat(), colorNode["g"].asFloat(), colorNode["b"].asFloat());
            /*int dim = 0;
            for (const auto& c : colorNode) {
                if (dim >= 3) {
                    sgl::Logfile::get()->writeError("LightEditorWidget::loadFromFile: Invalid color entry.");
                    return false;
                }
                light.color[dim] = c.asFloat();
                dim++;
            }*/
        }

        if (lightNode.isMember("intensity")) {
            light.intensity = lightNode["intensity"].asFloat();
        }

        if (lightNode.isMember("use_distance")) {
            light.useDistance = lightNode["use_distance"].asBool();
        }

        if (lightNode.isMember("position")) {
            auto positionNode = lightNode["position"];
            light.position = glm::vec3(
                    positionNode["x"].asFloat(), positionNode["y"].asFloat(), positionNode["z"].asFloat());
            /*int dim = 0;
            for (const auto& p : positionNode) {
                if (dim >= 3) {
                    sgl::Logfile::get()->writeError("LightEditorWidget::loadFromFile: Invalid position entry.");
                    return false;
                }
                light.position[dim] = p.asFloat();
                dim++;
            }*/
        }

        if (lightNode.isMember("space")) {
            std::string lightSpaceString = lightNode["space"].asString();
            for (int i = 0; i < IM_ARRAYSIZE(LIGHT_SPACE_NAMES); i++) {
                if (lightSpaceString == LIGHT_SPACE_NAMES[i]) {
                    light.lightSpace = LightSpace(i);
                    break;
                }
            }
        }

        if (lightNode.isMember("spot_total_width")) {
            light.spotTotalWidth = lightNode["spot_total_width"].asFloat();
        }

        if (lightNode.isMember("spot_falloff_start")) {
            light.spotFalloffStart = lightNode["spot_falloff_start"].asFloat();
        }
    }
    return true;
}

bool LightEditorWidget::saveToFile(const std::string& filePath) {
    Json::Value root;

    Json::Value lightsNode;
    for (const auto& light : lights) {
        Json::Value colorNode;
        colorNode["r"] = light.color.r;
        colorNode["g"] = light.color.g;
        colorNode["b"] = light.color.b;

        Json::Value positionNode;
        positionNode["x"] = light.position.x;
        positionNode["y"] = light.position.y;
        positionNode["z"] = light.position.z;

        Json::Value lightNode;
        lightNode["type"] = LIGHT_TYPE_NAMES[int(light.lightType)];
        lightNode["color"] = colorNode;
        lightNode["intensity"] = light.intensity;
        lightNode["use_distance"] = light.useDistance;
        lightNode["position"] = positionNode;
        lightNode["space"] = LIGHT_SPACE_NAMES[int(light.lightSpace)];
        lightNode["spot_total_width"] = light.spotTotalWidth;
        lightNode["spot_falloff_start"] = light.spotFalloffStart;
        lightsNode.append(lightNode);
    }
    root["lights"] = lightsNode;

    Json::StreamWriterBuilder builder;
    builder["commentStyle"] = "None";
    builder["indentation"] = "    ";
    std::unique_ptr<Json::StreamWriter> writer(builder.newStreamWriter());
    std::ofstream jsonFileStream(filePath.c_str());
    if (!jsonFileStream.is_open()) {
        sgl::Logfile::get()->writeError("Error in LightEditorWidget::saveToFile: Couldn't open \"" + filePath + "\".");
        return false;
    }
    writer->write(root, &jsonFileStream);
    jsonFileStream.close();

    return true;
}

bool LightEditorWidget::renderGuiLight(size_t lightIdx) {
    bool reRender = false;
    auto& light = lights.at(lightIdx);

    if (propertyEditor->addCombo(
            "Light Type", reinterpret_cast<int*>(&light.lightType), LIGHT_TYPE_NAMES, IM_ARRAYSIZE(LIGHT_TYPE_NAMES))) {
        reRender = true;
    }

    if (propertyEditor->addColorEdit3("Color", &light.color.x)) {
        reRender = true;
    }
    if (propertyEditor->addSliderFloat("Intensity", &light.intensity, 0.01f, 10.0f)) {
        reRender = true;
    }

    if (light.lightType != LightType::DIRECTIONAL) {
        bool useDistanceBool = light.useDistance;
        if (propertyEditor->addCheckbox("Use Distance", &useDistanceBool)) {
            light.useDistance = uint32_t(useDistanceBool);
            reRender = true;
        }
    }

    if (propertyEditor->addSliderFloat3(
            light.lightType == LightType::DIRECTIONAL ? "Direction" : "Position", &light.position.x, -2.0f, 2.0f)) {
        reRender = true;
    }
    if (propertyEditor->addCombo(
            "Light Space", reinterpret_cast<int*>(&light.lightSpace), LIGHT_SPACE_NAMES, IM_ARRAYSIZE(LIGHT_SPACE_NAMES))) {
        reRender = true;
    }

    if (light.lightType == LightType::SPOT) {
        if (propertyEditor->addSliderFloat("Total Cone Width Angle", (float*)&light.spotTotalWidth, 0.0, sgl::HALF_PI)) {
            reRender = true;
        }
        if (propertyEditor->addSliderFloat("Falloff Start Angle", (float*)&light.spotFalloffStart, 0.0, sgl::HALF_PI)) {
            reRender = true;
        }
    }

    return reRender;
}
