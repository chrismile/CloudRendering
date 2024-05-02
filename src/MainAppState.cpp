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

#include <fstream>
#include <json/json.h>
#include <Utils/File/Logfile.hpp>

#include "MainApp.hpp"

void MainApp::loadCameraPosesFromFile(const std::string& filePath) {
    // Parse the passed JSON file.
    std::ifstream jsonFileStream(filePath.c_str());
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errorString;
    Json::Value root;
    if (!parseFromStream(builder, jsonFileStream, &root, &errorString)) {
        sgl::Logfile::get()->writeError(errorString);
        return;
    }
    jsonFileStream.close();

    if (!root.isArray()) {
        sgl::Logfile::get()->writeError("Error in MainApp::loadCameraStateFromFile: Expected array.");
        return;
    }

    // At the moment, only loading the first state is supported.
    std::vector<CameraPose> cameraPoses;
    cameraPoses.reserve(root.size());
    for (unsigned int i = 0; i < root.size(); i++) {
        const auto& camState = root[i];
        if (!camState.isMember("position") || !camState.isMember("rotation") || !camState.isMember("fovy")) {
            sgl::Logfile::get()->writeError("Error in MainApp::loadCameraStateFromFile: No camera state data found.");
        }

        const auto& positionArray = camState["position"];
        glm::vec3 position(positionArray[0].asFloat(), positionArray[1].asFloat(), positionArray[2].asFloat());

        const auto& rotationArray = camState["rotation"];
        auto inverseViewMatrix = glm::identity<glm::mat4>();
        for (int col = 0; col < 3; col++) {
            for (int row = 0; row < 3; row++) {
                inverseViewMatrix[col][row] = rotationArray[col][row].asFloat();
            }
        }
        for (int row = 0; row < 3; row++) {
            inverseViewMatrix[3][row] = positionArray[row].asFloat();
        }
        //auto viewMatrix = glm::inverse(inverseViewMatrix);

        CameraPose cameraPose{};
        cameraPose.position = position;
        cameraPose.front = -inverseViewMatrix[2];
        cameraPose.right = inverseViewMatrix[0];
        cameraPose.up = inverseViewMatrix[1];
        cameraPose.fovy = camState["fovy"].asFloat();
        cameraPose.viewportWidth = float(camState["width"].asInt());
        cameraPose.viewportHeight = float(camState["height"].asInt());
        cameraPoses.push_back(cameraPose);
    }

    volumetricPathTracingPass->setCameraPoses(cameraPoses);

    reRender = true;
    hasMoved();
}

void MainApp::loadCameraStateFromFile(const std::string& filePath) {
    // Parse the passed JSON file.
    std::ifstream jsonFileStream(filePath.c_str());
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errorString;
    Json::Value root;
    if (!parseFromStream(builder, jsonFileStream, &root, &errorString)) {
        sgl::Logfile::get()->writeError(errorString);
        return;
    }
    jsonFileStream.close();

    if (!root.isArray()) {
        sgl::Logfile::get()->writeError("Error in MainApp::loadCameraStateFromFile: Expected array.");
        return;
    }

    // At the moment, only loading the first state is supported.
    const auto& camState = root[0];
    if (!camState.isMember("position") || !camState.isMember("rotation") || !camState.isMember("fovy")) {
        sgl::Logfile::get()->writeError("Error in MainApp::loadCameraStateFromFile: No camera state data found.");
    }

    float fovy = camState["fovy"].asFloat();
    const auto& positionArray = camState["position"];
    const auto& rotationArray = camState["rotation"];
    glm::vec3 position(positionArray[0].asFloat(), positionArray[1].asFloat(), positionArray[2].asFloat());

    auto inverseViewMatrix = glm::identity<glm::mat4>();
    for (int col = 0; col < 3; col++) {
        for (int row = 0; row < 3; row++) {
            inverseViewMatrix[col][row] = rotationArray[col][row].asFloat();
        }
    }
    for (int row = 0; row < 3; row++) {
        inverseViewMatrix[3][row] = positionArray[row].asFloat();
    }
    auto viewMatrix = glm::inverse(inverseViewMatrix);

    if (std::abs(fovy - camera->getFOVy()) > 1e-3) {
        camera->setFOVy(fovy);
    }
    camera->overwriteViewMatrix(viewMatrix);
    reRender = true;
    hasMoved();
}
