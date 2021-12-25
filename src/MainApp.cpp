/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2020, Christoph Neuhauser
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
#include <algorithm>

#include <glm/gtx/color_space.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/rotate_vector.hpp>
#include <GL/glew.h>
#include <boost/algorithm/string.hpp>

#ifdef USE_ZEROMQ
#include <zmq.h>
#endif

#include <Utils/Timer.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Input/Keyboard.hpp>
#include <Math/Math.hpp>
#include <Math/Geometry/MatrixUtil.hpp>
#include <Graphics/Window.hpp>
#include <Graphics/Renderer.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>

#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/imgui_internal.h>
#include <ImGui/imgui_custom.h>
#include <ImGui/imgui_stdlib.h>

#ifdef SUPPORT_OPTIX
#include "Denoiser/OptixVptDenoiser.hpp"
#endif
#include "MainApp.hpp"

void vulkanErrorCallback() {
    std::cerr << "Application callback" << std::endl;
}

MainApp::MainApp()
        : sceneData(
                sceneFramebuffer, sceneTexture, sceneDepthRBO, camera, clearColor,
                screenshotTransparentBackground, recording, useCameraFlight) {
    sgl::AppSettings::get()->getVulkanInstance()->setDebugCallback(&vulkanErrorCallback);

#ifdef SUPPORT_OPTIX
    optixInitialized = OptixVptDenoiser::initGlobal();
#endif

    checkpointWindow.setStandardWindowSize(1254, 390);
    checkpointWindow.setStandardWindowPosition(841, 53);

    camera->setNearClipDistance(0.01f);
    camera->setFarClipDistance(100.0f);

    useLinearRGB = false;
    transferFunctionWindow.setClearColor(clearColor);
    transferFunctionWindow.setUseLinearRGB(useLinearRGB);

    if (usePerformanceMeasurementMode) {
        useCameraFlight = true;
    }
    if (useCameraFlight && recording) {
        sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
        window->setWindowSize(recordingResolution.x, recordingResolution.y);
        realTimeCameraFlight = false;
    }

    volumetricPathTracingPass = std::make_shared<VolumetricPathTracingPass>(rendererVk);

    customDataSetFileName = sgl::FileUtils::get()->getUserDirectory();
    loadAvailableDataSetInformation();

    usesNewState = true;
    recordingTimeStampStart = sgl::Timer->getTicksMicroseconds();

    resolutionChanged(sgl::EventPtr());

    if (!recording && !usePerformanceMeasurementMode) {
        // Just for convenience...
        int desktopWidth = 0;
        int desktopHeight = 0;
        int refreshRate = 60;
        sgl::AppSettings::get()->getDesktopDisplayMode(desktopWidth, desktopHeight, refreshRate);
        if (desktopWidth == 3840 && desktopHeight == 2160) {
            sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
            window->setWindowSize(2186, 1358);
        }
    }
}

MainApp::~MainApp() {
    device->waitIdle();

    volumetricPathTracingPass = {};

#ifdef SUPPORT_OPTIX
    if (optixInitialized) {
        OptixVptDenoiser::freeGlobal();
    }
#endif
}

void MainApp::resolutionChanged(sgl::EventPtr event) {
    SciVisApp::resolutionChanged(event);

    sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
    auto width = uint32_t(window->getWidth());
    auto height = uint32_t(window->getHeight());

    volumetricPathTracingPass->setOutputImage(sceneTextureVk->getImageView());
    volumetricPathTracingPass->recreateSwapchain(width, height);
}

void MainApp::updateColorSpaceMode() {
    SciVisApp::updateColorSpaceMode();
    transferFunctionWindow.setUseLinearRGB(useLinearRGB);
}

void MainApp::render() {
    SciVisApp::preRender();

    reRender = reRender || volumetricPathTracingPass->needsReRender();

    if (reRender || continuousRendering) {
        SciVisApp::prepareReRender();

        if (cloudData) {
            volumetricPathTracingPass->render();
        }

        reRender = false;
    }

    SciVisApp::postRender();
}

void MainApp::renderGui() {
    if (showSettingsWindow) {
        sgl::ImGuiWrapper::get()->setNextWindowStandardPosSize(3090, 56, 735, 1085);
        if (ImGui::Begin("Settings", &showSettingsWindow)) {
            SciVisApp::renderGuiFpsCounter();

            // Selection of displayed model
            renderFileSelectionSettingsGui();

            ImGui::Separator();

            if (ImGui::CollapsingHeader("Scene Settings", NULL, ImGuiTreeNodeFlags_DefaultOpen)) {
                renderSceneSettingsGui();
            }
        }
        ImGui::End();
    }

    volumetricPathTracingPass->renderGui();

    if (checkpointWindow.renderGui()) {
        fovDegree = camera->getFOVy() / sgl::PI * 180.0f;
        reRender = true;
        hasMoved();
    }
}

void MainApp::loadAvailableDataSetInformation() {
    dataSetNames.clear();
    dataSetNames.emplace_back("Local file...");
    selectedDataSetIndex = 0;

    const std::string lineDataSetsDirectory = sgl::AppSettings::get()->getDataDirectory() + "CloudDataSets/";
    if (sgl::FileUtils::get()->exists(lineDataSetsDirectory + "datasets.json")) {
        dataSetInformation = loadDataSetList(lineDataSetsDirectory + "datasets.json");
        for (DataSetInformation& dataSetInfo  : dataSetInformation) {
            dataSetNames.push_back(dataSetInfo.name);
        }
    }
}

const std::string& MainApp::getSelectedMeshFilenames() {
    if (selectedDataSetIndex == 0) {
        return customDataSetFileName;
    }
    return dataSetInformation.at(selectedDataSetIndex - NUM_MANUAL_LOADERS).filename;
}

void MainApp::renderFileSelectionSettingsGui() {
    if (ImGui::Combo(
            "Data Set", &selectedDataSetIndex, dataSetNames.data(),
            dataSetNames.size())) {
        if (selectedDataSetIndex >= NUM_MANUAL_LOADERS) {
            loadCloudDataSet(getSelectedMeshFilenames());
        }
    }

    //if (dataRequester.getIsProcessingRequest()) {
    //    ImGui::SameLine();
    //    ImGui::ProgressSpinner(
    //            "##progress-spinner", -1.0f, -1.0f, 4.0f,
    //            ImVec4(0.1, 0.5, 1.0, 1.0));
    //}


    if (selectedDataSetIndex == 0) {
        ImGui::InputText("##meshfilenamelabel", &customDataSetFileName);
        ImGui::SameLine();
        if (ImGui::Button("Load File")) {
            loadCloudDataSet(getSelectedMeshFilenames());
        }
    }
}

void MainApp::renderSceneSettingsGui() {
    if (ImGui::ColorEdit3("Clear Color", (float*)&clearColorSelection, 0)) {
        clearColor = sgl::colorFromFloat(
                clearColorSelection.x, clearColorSelection.y, clearColorSelection.z, clearColorSelection.w);
        transferFunctionWindow.setClearColor(clearColor);
        if (cloudData) {
            cloudData->setClearColor(clearColor);
        }
        reRender = true;
    }

    SciVisApp::renderSceneSettingsGuiPre();
    ImGui::Checkbox("Show Transfer Function Window", &transferFunctionWindow.getShowWindow());

    SciVisApp::renderSceneSettingsGuiPost();
}

void MainApp::update(float dt) {
    sgl::SciVisApp::update(dt);

    updateCameraFlight(cloudData.get() != nullptr, usesNewState);

    checkLoadingRequestFinished();

    transferFunctionWindow.update(dt);

    ImGuiIO &io = ImGui::GetIO();
    if (io.WantCaptureKeyboard && !recording) {
        // Ignore inputs below
        return;
    }

    moveCameraKeyboard(dt);
    if (sgl::Keyboard->isKeyDown(SDLK_u)) {
        transferFunctionWindow.setShowWindow(showSettingsWindow);
    }

    if (io.WantCaptureMouse) {
        // Ignore inputs below
        return;
    }

    moveCameraMouse(dt);
}

void MainApp::hasMoved() {
    volumetricPathTracingPass->onHasMoved();
}


// --- Visualization pipeline ---

void MainApp::loadCloudDataSet(const std::string& fileName, bool blockingDataLoading) {
    if (fileName.size() == 0) {
        cloudData = CloudDataPtr();
        return;
    }
    currentlyLoadedDataSetIndex = selectedDataSetIndex;

    DataSetInformation selectedDataSetInformation;
    if (selectedDataSetIndex >= NUM_MANUAL_LOADERS && !dataSetInformation.empty()) {
        selectedDataSetInformation = dataSetInformation.at(selectedDataSetIndex - NUM_MANUAL_LOADERS);
    } else {
        selectedDataSetInformation.filename = fileName;
    }

    glm::mat4 transformationMatrix = sgl::matrixIdentity();
    //glm::mat4* transformationMatrixPtr = nullptr;
    if (selectedDataSetInformation.hasCustomTransform) {
        transformationMatrix *= selectedDataSetInformation.transformMatrix;
        //transformationMatrixPtr = &transformationMatrix;
    }
    if (rotateModelBy90DegreeTurns != 0) {
        transformationMatrix *= glm::rotate(rotateModelBy90DegreeTurns * sgl::HALF_PI, modelRotationAxis);
        //transformationMatrixPtr = &transformationMatrix;
    }

    CloudDataPtr cloudData(new CloudData);

    if (blockingDataLoading) {
        //bool dataLoaded = cloudData->loadFromFile(fileName, selectedDataSetInformation, transformationMatrixPtr);
        bool dataLoaded = cloudData->loadFromFile(fileName);

        if (dataLoaded) {
            this->cloudData = cloudData;
            cloudData->setClearColor(clearColor);
            cloudData->setUseLinearRGB(useLinearRGB);
            newMeshLoaded = true;
            //modelBoundingBox = cloudData->getModelBoundingBox();

            volumetricPathTracingPass->setCloudDataSet(cloudData);
            reRender = true;

            const std::string& meshDescriptorName = fileName;
            checkpointWindow.onLoadDataSet(meshDescriptorName);

            if (true) {
                std::string cameraPathFilename =
                        saveDirectoryCameraPaths + sgl::FileUtils::get()->getPathAsList(meshDescriptorName).back()
                        + ".binpath";
                if (sgl::FileUtils::get()->exists(cameraPathFilename)) {
                    cameraPath.fromBinaryFile(cameraPathFilename);
                } else {
                    cameraPath.fromCirclePath(
                            modelBoundingBox, meshDescriptorName,
                            usePerformanceMeasurementMode
                            ? CAMERA_PATH_TIME_PERFORMANCE_MEASUREMENT : CAMERA_PATH_TIME_RECORDING,
                            usePerformanceMeasurementMode);
                }
            }
        }
    } else {
        //dataRequester.queueRequest(cloudData, fileName, selectedDataSetInformation, transformationMatrixPtr);
    }
}

void MainApp::checkLoadingRequestFinished() {
    CloudDataPtr cloudData;
    DataSetInformation loadedDataSetInformation;

    //if (!cloudData) {
    //    cloudData = dataRequester.getLoadedData(loadedDataSetInformation);
    //}

    if (cloudData) {
        this->cloudData = cloudData;
        cloudData->setClearColor(clearColor);
        cloudData->setUseLinearRGB(useLinearRGB);
        newMeshLoaded = true;
        //modelBoundingBox = cloudData->getModelBoundingBox();

        std::string meshDescriptorName = cloudData->getFileName();
        checkpointWindow.onLoadDataSet(meshDescriptorName);

        if (true) {
            std::string cameraPathFilename =
                    saveDirectoryCameraPaths + sgl::FileUtils::get()->getPathAsList(meshDescriptorName).back()
                    + ".binpath";
            if (sgl::FileUtils::get()->exists(cameraPathFilename)) {
                cameraPath.fromBinaryFile(cameraPathFilename);
            } else {
                cameraPath.fromCirclePath(
                        modelBoundingBox, meshDescriptorName,
                        usePerformanceMeasurementMode
                        ? CAMERA_PATH_TIME_PERFORMANCE_MEASUREMENT : CAMERA_PATH_TIME_RECORDING,
                        usePerformanceMeasurementMode);
            }
        }
    }
}

void MainApp::reloadDataSet() {
    loadCloudDataSet(getSelectedMeshFilenames());
}
