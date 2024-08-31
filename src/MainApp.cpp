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

#include <algorithm>
#include <stack>
#include <csignal>

#ifdef USE_ZEROMQ
#include <zmq.h>
#endif

#include <Utils/Timer.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Input/Keyboard.hpp>
#include <Math/Math.hpp>
#include <Math/Geometry/MatrixUtil.hpp>
#include <Graphics/Window.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/ScreenshotReadbackHelper.hpp>
#include <Graphics/Vulkan/Render/Renderer.hpp>

#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/ImGuiFileDialog/ImGuiFileDialog.h>
#include <ImGui/imgui_internal.h>
#include <ImGui/imgui_custom.h>
#include <ImGui/imgui_stdlib.h>

#ifdef SUPPORT_OPTIX
#include "Denoiser/OptixVptDenoiser.hpp"
#endif
#if defined(SUPPORT_CUDA_INTEROP) && defined(SUPPORT_OPEN_IMAGE_DENOISE)
#include "Denoiser/OpenImageDenoiseDenoiser.hpp"
#endif
#include "DataView.hpp"
#include "MainApp.hpp"

void vulkanErrorCallback() {
    std::cerr << "Application callback" << std::endl;
}

#ifdef __linux__
void signalHandler(int signum) {
#ifdef SGL_INPUT_API_V2
    sgl::AppSettings::get()->captureMouse(false);
#else
    SDL_CaptureMouse(SDL_FALSE);
#endif
    std::cerr << "Interrupt signal (" << signum << ") received." << std::endl;
    exit(signum);
}
#endif

MainApp::MainApp()
        : sceneData(
                camera, clearColor, screenshotTransparentBackground, recording, useCameraFlight) {
    sgl::AppSettings::get()->getVulkanInstance()->setDebugCallback(&vulkanErrorCallback);

#ifdef SUPPORT_CUDA_INTEROP
    sgl::vk::Device* device = sgl::AppSettings::get()->getPrimaryDevice();
    if (device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY) {
        cudaInteropInitialized = true;
        if (!sgl::vk::initializeCudaDeviceApiFunctionTable()) {
            cudaInteropInitialized = false;
            sgl::Logfile::get()->writeError(
                    "Error in MainApp::MainApp: sgl::vk::initializeCudaDeviceApiFunctionTable() returned false.",
                    false);
        }

        if (cudaInteropInitialized) {
            CUresult cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuInit(0);
            if (cuResult == CUDA_ERROR_NO_DEVICE) {
                sgl::Logfile::get()->writeInfo("No CUDA-capable device was found. Disabling CUDA interop support.");
                cudaInteropInitialized = false;
            } else {
                sgl::vk::checkCUresult(cuResult, "Error in cuInit: ");
            }
        }

        if (cudaInteropInitialized) {
            CUresult cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuCtxCreate(
                    &cuContext, CU_CTX_SCHED_SPIN, cuDevice);
            sgl::vk::checkCUresult(cuResult, "Error in cuCtxCreate: ");
        }
    }
#endif
#ifdef SUPPORT_OPTIX
    if (cudaInteropInitialized) {
        optixInitialized = OptixVptDenoiser::initGlobal(cuContext, cuDevice);
    }
#endif
#if defined(SUPPORT_CUDA_INTEROP) && defined(SUPPORT_OPEN_IMAGE_DENOISE)
    if (cudaInteropInitialized) {
        OpenImageDenoiseDenoiser::initGlobalCuda(cuContext, cuDevice);
    }
#endif

    checkpointWindow.setStandardWindowSize(1254, 390);
    checkpointWindow.setStandardWindowPosition(841, 53);

    lightEditorWidget = new LightEditorWidget(rendererVk);
    lightEditorWidget->setShowWindow(false);

    propertyEditor.setInitWidthValues(sgl::ImGuiWrapper::get()->getScaleDependentSize(280.0f));

    camera->setNearClipDistance(1.0f / 512.0f); // 0.001953125f
    camera->setFarClipDistance(80.0f);

    useDockSpaceMode = true;
    sgl::AppSettings::get()->getSettings().getValueOpt("useDockSpaceMode", useDockSpaceMode);
    sgl::AppSettings::get()->getSettings().getValueOpt("useFixedSizeViewport", useFixedSizeViewport);
    showPropertyEditor = useDockSpaceMode;
    sgl::ImGuiWrapper::get()->setUseDockSpaceMode(useDockSpaceMode);

#ifdef NDEBUG
    showFpsOverlay = false;
#else
    showFpsOverlay = true;
#endif
    sgl::AppSettings::get()->getSettings().getValueOpt("showFpsOverlay", showFpsOverlay);
    sgl::AppSettings::get()->getSettings().getValueOpt("showCoordinateAxesOverlay", showCoordinateAxesOverlay);

    useLinearRGB = true;
    transferFunctionWindow.setClearColor(clearColor);
    transferFunctionWindow.setUseLinearRGB(useLinearRGB);
    transferFunctionWindow.setShowWindow(false);
    transferFunctionWindow.setAttributeNames({"Volume", "Isosurface"});
    coordinateAxesOverlayWidget.setClearColor(clearColor);

    if (usePerformanceMeasurementMode) {
        useCameraFlight = true;
    }
    if (useCameraFlight && recording) {
        sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
        window->setWindowSize(recordingResolution.x, recordingResolution.y);
        realTimeCameraFlight = false;
    }

    fileDialogInstance = IGFD_Create();
    customDataSetFileName = sgl::FileUtils::get()->getUserDirectory();
    loadAvailableDataSetInformation();

    volumetricPathTracingPass = std::make_shared<VolumetricPathTracingPass>(rendererVk, &cameraHandle);
    volumetricPathTracingPass->setUseLinearRGB(useLinearRGB);
    volumetricPathTracingPass->setFileDialogInstance(fileDialogInstance);
    dataView = std::make_shared<DataView>(camera, rendererVk, volumetricPathTracingPass);
    dataView->useLinearRGB = useLinearRGB;
    if (useDockSpaceMode) {
        cameraHandle = dataView->camera;
    } else {
        cameraHandle = camera;
    }

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

    if (!sgl::AppSettings::get()->getSettings().hasKey("cameraNavigationMode")) {
        cameraNavigationMode = sgl::CameraNavigationMode::TURNTABLE;
        updateCameraNavigationMode();
    }

    usesNewState = true;
    recordingTimeStampStart = sgl::Timer->getTicksMicroseconds();

#ifdef __linux__
    signal(SIGSEGV, signalHandler);
#endif
}

MainApp::~MainApp() {
    device->waitIdle();

    volumetricPathTracingPass = {};
    dataView = {};
    if (lightEditorWidget) {
        delete lightEditorWidget;
        lightEditorWidget = nullptr;
    }

#ifdef SUPPORT_OPTIX
    if (optixInitialized) {
        OptixVptDenoiser::freeGlobal();
    }
#endif
#ifdef SUPPORT_CUDA_INTEROP
    if (sgl::vk::getIsCudaDeviceApiFunctionTableInitialized()) {
        if (cuContext) {
            CUresult cuResult = sgl::vk::g_cudaDeviceApiFunctionTable.cuCtxDestroy(cuContext);
            sgl::vk::checkCUresult(cuResult, "Error in cuCtxDestroy: ");
            cuContext = {};
        }
        sgl::vk::freeCudaDeviceApiFunctionTable();
    }
#endif

    IGFD_Destroy(fileDialogInstance);

    sgl::AppSettings::get()->getSettings().addKeyValue("useDockSpaceMode", useDockSpaceMode);
    sgl::AppSettings::get()->getSettings().addKeyValue("useFixedSizeViewport", useFixedSizeViewport);
    sgl::AppSettings::get()->getSettings().addKeyValue("showFpsOverlay", showFpsOverlay);
    sgl::AppSettings::get()->getSettings().addKeyValue("showCoordinateAxesOverlay", showCoordinateAxesOverlay);
}

void MainApp::resolutionChanged(sgl::EventPtr event) {
    SciVisApp::resolutionChanged(event);

    sgl::Window *window = sgl::AppSettings::get()->getMainWindow();
    auto width = uint32_t(window->getWidth());
    auto height = uint32_t(window->getHeight());

    if (!useDockSpaceMode) {
        volumetricPathTracingPass->setOutputImage(sceneTextureVk->getImageView());
        volumetricPathTracingPass->recreateSwapchain(width, height);
    }
}

void MainApp::updateColorSpaceMode() {
    SciVisApp::updateColorSpaceMode();
    transferFunctionWindow.setUseLinearRGB(useLinearRGB);
    volumetricPathTracingPass->setUseLinearRGB(useLinearRGB);
    if (dataView) {
        dataView->useLinearRGB = useLinearRGB;
        dataView->viewportWidth = 0;
        dataView->viewportHeight = 0;
    }
}

void MainApp::render() {
    SciVisApp::preRender();
    if (dataView) {
        dataView->saveScreenshotDataIfAvailable();
    }

    if (!useDockSpaceMode) {
        reRender = reRender || volumetricPathTracingPass->needsReRender();

        if (reRender || continuousRendering) {
            SciVisApp::prepareReRender();

            if (cloudData) {
                volumetricPathTracingPass->render();
            }

            reRender = false;
        }
    }

    SciVisApp::postRender();

    if (useDockSpaceMode && !uiOnScreenshot && recording && !isFirstRecordingFrame) {
        rendererVk->transitionImageLayout(
                dataView->compositedDataViewTexture->getImage(),
                VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL);
        videoWriter->pushFramebufferImage(dataView->compositedDataViewTexture->getImage());
        rendererVk->transitionImageLayout(
                dataView->compositedDataViewTexture->getImage(),
                VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
    }
}

void MainApp::renderGui() {
    focusedWindowIndex = -1;
    mouseHoverWindowIndex = -1;

#ifdef SGL_INPUT_API_V2
    if (sgl::Keyboard->keyPressed(ImGuiKey_O) && sgl::Keyboard->getModifier(ImGuiKey_ModCtrl)) {
        openFileDialog();
    }
#else
    if (sgl::Keyboard->keyPressed(SDLK_o) && (sgl::Keyboard->getModifier() & (KMOD_LCTRL | KMOD_RCTRL)) != 0) {
        openFileDialog();
    }
#endif

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChooseDataSetFile", ImGuiWindowFlags_NoCollapse,
            sgl::ImGuiWrapper::get()->getScaleDependentSize(1000, 580),
            ImVec2(FLT_MAX, FLT_MAX))) {
        if (IGFD_IsOk(fileDialogInstance)) {
            std::string filePathName = IGFD_GetFilePathNameString(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilterString(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            // Is this line data set or a volume data file for the scattering line tracer?
            std::string currentPath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            filename += selection.table[0].fileName;
            IGFD_Selection_DestroyContent(&selection);

            std::string filenameLower = sgl::toLowerCopy(filename);
            if (checkHasValidExtension(filenameLower)) {
                fileDialogDirectory = sgl::FileUtils::get()->getPathToFile(filename);
                selectedDataSetIndex = 0;
                customDataSetFileName = filename;
                loadCloudDataSet(getSelectedDataSetFilename(), getSelectedDataSetFilename());
            } else {
                sgl::Logfile::get()->writeError(
                        "The dropped file name has an unknown extension \""
                        + sgl::FileUtils::get()->getFileExtension(filenameLower) + "\".");
            }
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    if (IGFD_DisplayDialog(
            fileDialogInstance,
            "ChooseCamStateFile", ImGuiWindowFlags_NoCollapse,
            sgl::ImGuiWrapper::get()->getScaleDependentSize(1000, 580),
            ImVec2(FLT_MAX, FLT_MAX))) {
        if (IGFD_IsOk(fileDialogInstance)) {
            std::string filePathName = IGFD_GetFilePathNameString(fileDialogInstance);
            std::string filePath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filter = IGFD_GetCurrentFilterString(fileDialogInstance);
            std::string userDatas;
            if (IGFD_GetUserDatas(fileDialogInstance)) {
                userDatas = std::string((const char*)IGFD_GetUserDatas(fileDialogInstance));
            }
            auto selection = IGFD_GetSelection(fileDialogInstance);

            std::string currentPath = IGFD_GetCurrentPathString(fileDialogInstance);
            std::string filename = currentPath;
            if (!filename.empty() && filename.back() != '/' && filename.back() != '\\') {
                filename += "/";
            }
            std::string currentFileName;
            if (filter == ".*") {
                currentFileName = IGFD_GetCurrentFileNameRawString(fileDialogInstance);
            } else {
                currentFileName = IGFD_GetCurrentFileNameString(fileDialogInstance);
            }
            if (selection.count != 0 && selection.table[0].fileName == currentFileName) {
                filename += selection.table[0].fileName;
            } else {
                filename += currentFileName;
            }
            IGFD_Selection_DestroyContent(&selection);

            //loadCameraStateFromFile(filename);
            loadCameraPosesFromFile(filename);
        }
        IGFD_CloseDialog(fileDialogInstance);
    }

    if (useDockSpaceMode) {
        ImGuiID dockSpaceId = ImGui::DockSpaceOverViewport(ImGui::GetMainViewport());
        ImGuiDockNode* centralNode = ImGui::DockBuilderGetNode(dockSpaceId);
        static bool isProgramStartup = true;
        if (isProgramStartup && centralNode->IsEmpty()) {
            ImGuiID dockLeftId, dockMainId;
            ImGui::DockBuilderSplitNode(
                    dockSpaceId, ImGuiDir_Left, 0.3f,
                    &dockLeftId, &dockMainId);
            ImGui::DockBuilderDockWindow("Volumetric Path Tracer", dockMainId);

            ImGuiID dockLeftUpId, dockLeftDownId;
            ImGui::DockBuilderSplitNode(
                    dockLeftId, ImGuiDir_Up, 0.8f,
                    &dockLeftUpId, &dockLeftDownId);
            ImGui::DockBuilderDockWindow("Property Editor", dockLeftUpId);

            ImGui::DockBuilderDockWindow("Multi-Var Transfer Function", dockLeftDownId);
            ImGui::DockBuilderDockWindow("Camera Checkpoints", dockLeftDownId);
            ImGui::DockBuilderDockWindow("Light Editor", dockLeftDownId);

            ImGui::DockBuilderFinish(dockLeftId);
            ImGui::DockBuilderFinish(dockSpaceId);
        }
        isProgramStartup = false;

        renderGuiMenuBar();

        if (showRendererWindow) {
            bool isViewOpen = true;
            sgl::ImGuiWrapper::get()->setNextWindowStandardSize(800, 600);
            if (ImGui::Begin("Volumetric Path Tracer", &isViewOpen)) {
                if (ImGui::IsWindowFocused()) {
                    focusedWindowIndex = 0;
                }
                sgl::ImGuiWrapper::get()->setWindowViewport(0, ImGui::GetWindowViewport());
                sgl::ImGuiWrapper::get()->setWindowPosAndSize(0, ImGui::GetWindowPos(), ImGui::GetWindowSize());

                ImVec2 sizeContent = ImGui::GetContentRegionAvail();
                if (useFixedSizeViewport) {
                    sizeContent = ImVec2(float(fixedViewportSize.x), float(fixedViewportSize.y));
                }
                int newViewportWidth = std::max(sgl::iceil(int(sizeContent.x), subsamplingFactor), 1);
                int newViewportHeight = std::max(sgl::iceil(int(sizeContent.y), subsamplingFactor), 1);
                if (newViewportWidth != int(dataView->viewportWidth)
                        || newViewportHeight != int(dataView->viewportHeight)) {
                    dataView->resize(newViewportWidth, newViewportHeight);
                    if (dataView->viewportWidth > 0 && dataView->viewportHeight > 0) {
                        volumetricPathTracingPass->setOutputImage(dataView->dataViewTexture->getImageView());
                        volumetricPathTracingPass->recreateSwapchain(
                                dataView->viewportWidth, dataView->viewportHeight);
                    }
                    reRender = true;
                }

                reRender = reRender || volumetricPathTracingPass->needsReRender();

                if (reRender || continuousRendering) {
                    if (dataView->viewportWidth > 0 && dataView->viewportHeight > 0) {
                        dataView->beginRender();
                        if (cloudData) {
                            volumetricPathTracingPass->render();
                        }
                        dataView->endRender();
                    }

                    reRender = false;
                }

                if (dataView->viewportWidth > 0 && dataView->viewportHeight > 0) {
                    if (!uiOnScreenshot && screenshot) {
                        printNow = true;
                        std::string screenshotFilename =
                                saveDirectoryScreenshots + saveFilenameScreenshots
                                + "_" + sgl::toString(screenshotNumber);
                        screenshotFilename += ".png";

                        dataView->screenshotReadbackHelper->setScreenshotTransparentBackground(
                                screenshotTransparentBackground);
                        dataView->saveScreenshot(screenshotFilename);
                        screenshot = false;

                        printNow = false;
                        screenshot = true;
                    }

                    if (isViewOpen) {
                        ImTextureID textureId = dataView->getImGuiTextureId();
                        ImGui::Image(
                                textureId, sizeContent, ImVec2(0, 0), ImVec2(1, 1));
                        if (ImGui::IsItemHovered()) {
                            mouseHoverWindowIndex = 0;
                        }
                    }

                    if (showFpsOverlay) {
                        renderGuiFpsOverlay();
                    }
                    if (showCoordinateAxesOverlay) {
                        renderGuiCoordinateAxesOverlay(dataView->camera);
                    }
                }
            }
            ImGui::End();
        }

        if (!uiOnScreenshot && screenshot) {
            screenshot = false;
            screenshotNumber++;
        }
        reRender = false;
    }

    bool showTransferFunctionWindow = transferFunctionWindow.getShowWindow();
    if (transferFunctionWindow.renderGui()) {
        reRender = true;
        if (transferFunctionWindow.getTransferFunctionMapRebuilt()) {
            if (cloudData) {
                cloudData->onTransferFunctionMapRebuilt();
                hasMoved();
            }
            //sgl::EventManager::get()->triggerEvent(std::make_shared<sgl::Event>(
            //        ON_TRANSFER_FUNCTION_MAP_REBUILT_EVENT));
        }
    }
    if (showTransferFunctionWindow != transferFunctionWindow.getShowWindow()) {
        volumetricPathTracingPass->setShaderDirty();
        volumetricPathTracingPass->onHasMoved();
        reRender = true;
    }

    if (checkpointWindow.renderGui()) {
        fovDegree = camera->getFOVy() / sgl::PI * 180.0f;
        reRender = true;
        hasMoved();
    }

    bool showLightEditorWidget = lightEditorWidget->getShowWindow();
    if (lightEditorWidget->renderGui()) {
        fovDegree = camera->getFOVy() / sgl::PI * 180.0f;
        reRender = true;
        hasMoved();
    }
    if (showLightEditorWidget != lightEditorWidget->getShowWindow()) {
        volumetricPathTracingPass->setShaderDirty();
        volumetricPathTracingPass->onHasMoved();
        reRender = true;
    }

    if (showPropertyEditor) {
        renderGuiPropertyEditorWindow();
    }
}

void MainApp::loadAvailableDataSetInformation() {
    dataSetNames.clear();
    dataSetNames.emplace_back("Local file...");
    selectedDataSetIndex = 0;

    const std::string lineDataSetsDirectory = sgl::AppSettings::get()->getDataDirectory() + "CloudDataSets/";
    if (sgl::FileUtils::get()->exists(lineDataSetsDirectory + "datasets.json")) {
        dataSetInformationRoot = loadDataSetList(lineDataSetsDirectory + "datasets.json");

        std::stack<std::pair<DataSetInformationPtr, size_t>> dataSetInformationStack;
        dataSetInformationStack.push(std::make_pair(dataSetInformationRoot, 0));
        while (!dataSetInformationStack.empty()) {
            std::pair<DataSetInformationPtr, size_t> dataSetIdxPair = dataSetInformationStack.top();
            DataSetInformationPtr dataSetInformationParent = dataSetIdxPair.first;
            size_t idx = dataSetIdxPair.second;
            dataSetInformationStack.pop();
            while (idx < dataSetInformationParent->children.size()) {
                DataSetInformationPtr dataSetInformationChild =
                        dataSetInformationParent->children.at(idx);
                idx++;
                if (dataSetInformationChild->type == DATA_SET_TYPE_NODE) {
                    dataSetInformationStack.push(std::make_pair(dataSetInformationRoot, idx));
                    dataSetInformationStack.push(std::make_pair(dataSetInformationChild, 0));
                    break;
                } else {
                    dataSetInformationChild->sequentialIndex = int(dataSetNames.size());
                    dataSetInformationList.push_back(dataSetInformationChild);
                    dataSetNames.push_back(dataSetInformationChild->name);
                }
            }
        }
    }
}

const std::string& MainApp::getSelectedDataSetFilename() {
    if (selectedDataSetIndex == 0) {
        return customDataSetFileName;
    }
    return dataSetInformationList.at(selectedDataSetIndex - NUM_MANUAL_LOADERS)->filename;
}

const std::string& MainApp::getSelectedDataSetEmissionFilename() {
    if (selectedDataSetIndex == 0) {
        return customDataSetFileNameEmission;
    }
    return dataSetInformationList.at(selectedDataSetIndex - NUM_MANUAL_LOADERS)->emission;
}

void MainApp::renderGuiGeneralSettingsPropertyEditor() {
    if (propertyEditor.addColorEdit3("Clear Color", (float*)&clearColorSelection, 0)) {
        clearColor = sgl::colorFromFloat(
                clearColorSelection.x, clearColorSelection.y, clearColorSelection.z, clearColorSelection.w);
        transferFunctionWindow.setClearColor(clearColor);
        coordinateAxesOverlayWidget.setClearColor(clearColor);
        if (cloudData) {
            cloudData->setClearColor(clearColor);
        }
        reRender = true;
    }

    newDockSpaceMode = useDockSpaceMode;
    if (propertyEditor.addCheckbox("Use Docking Mode", &newDockSpaceMode)) {
        scheduledDockSpaceModeChange = true;
    }

    if (useDockSpaceMode && propertyEditor.addSliderInt("Subsampling Factor", &subsamplingFactor, 1, 4)) {
        reRender = true;
    }

    if (propertyEditor.addCheckbox("Fixed Size Viewport", &useFixedSizeViewport)) {
        reRender = true;
    }
    if (useFixedSizeViewport) {
        if (propertyEditor.addSliderInt2Edit("Viewport Size", &fixedViewportSizeEdit.x, 1, 8192)
                == ImGui::EditMode::INPUT_FINISHED) {
            fixedViewportSize = fixedViewportSizeEdit;
            reRender = true;
        }
    }
}

void MainApp::openFileDialog() {
    selectedDataSetIndex = 0;
    if (fileDialogDirectory.empty() || !sgl::FileUtils::get()->directoryExists(fileDialogDirectory)) {
        fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory() + "CloudDataSets/";
        if (!sgl::FileUtils::get()->exists(fileDialogDirectory)) {
            fileDialogDirectory = sgl::AppSettings::get()->getDataDirectory();
        }
    }
    IGFD_OpenModal(
            fileDialogInstance,
            "ChooseDataSetFile", "Choose a File",
            ".*,.xyz,.nvdb,"
#ifdef USE_OPENVDB
            ".vdb,"
#endif
            ".dat,.raw,.mhd,.nii",
            fileDialogDirectory.c_str(),
            "", 1, nullptr,
            ImGuiFileDialogFlags_None);
}

void MainApp::renderGuiMenuBar() {
    if (ImGui::BeginMainMenuBar()) {
        if (ImGui::BeginMenu("File")) {
            if (ImGui::MenuItem("Open Dataset...", "CTRL+O")) {
                openFileDialog();
            }

            if (ImGui::BeginMenu("Datasets")) {
                for (int i = 1; i < NUM_MANUAL_LOADERS; i++) {
                    if (ImGui::MenuItem(dataSetNames.at(i).c_str())) {
                        selectedDataSetIndex = i;
                    }
                }

                if (dataSetInformationRoot) {
                    std::stack<std::pair<DataSetInformationPtr, size_t>> dataSetInformationStack;
                    dataSetInformationStack.push(std::make_pair(dataSetInformationRoot, 0));
                    while (!dataSetInformationStack.empty()) {
                        std::pair<DataSetInformationPtr, size_t> dataSetIdxPair = dataSetInformationStack.top();
                        DataSetInformationPtr dataSetInformationParent = dataSetIdxPair.first;
                        size_t idx = dataSetIdxPair.second;
                        dataSetInformationStack.pop();
                        while (idx < dataSetInformationParent->children.size()) {
                            DataSetInformationPtr dataSetInformationChild =
                                    dataSetInformationParent->children.at(idx);
                            if (dataSetInformationChild->type == DATA_SET_TYPE_NODE) {
                                if (ImGui::BeginMenu(dataSetInformationChild->name.c_str())) {
                                    dataSetInformationStack.push(std::make_pair(dataSetInformationRoot, idx + 1));
                                    dataSetInformationStack.push(std::make_pair(dataSetInformationChild, 0));
                                    break;
                                }
                            } else {
                                if (ImGui::MenuItem(dataSetInformationChild->name.c_str())) {
                                    selectedDataSetIndex = int(dataSetInformationChild->sequentialIndex);
                                    loadCloudDataSet(getSelectedDataSetFilename(), getSelectedDataSetEmissionFilename());
                                }
                            }
                            idx++;
                        }

                        if (idx == dataSetInformationParent->children.size() && !dataSetInformationStack.empty()) {
                            ImGui::EndMenu();
                        }
                    }
                }

                ImGui::EndMenu();
            }

            if (ImGui::MenuItem("Quit", "CTRL+Q")) {
                quit();
            }
            ImGui::EndMenu();
        }
        if (ImGui::BeginMenu("Window")) {
            if (ImGui::MenuItem("Volumetric Path Tracer", nullptr, showRendererWindow)) {
                showRendererWindow = !showRendererWindow;
            }
            if (ImGui::MenuItem("FPS Overlay", nullptr, showFpsOverlay)) {
                showFpsOverlay = !showFpsOverlay;
            }
            if (ImGui::MenuItem("Coordinate Axes Overlay", nullptr, showCoordinateAxesOverlay)) {
                showCoordinateAxesOverlay = !showCoordinateAxesOverlay;
            }
            if (ImGui::MenuItem("Property Editor", nullptr, showPropertyEditor)) {
                showPropertyEditor = !showPropertyEditor;
            }
            if (ImGui::MenuItem(
                    "Transfer Function Window", nullptr, transferFunctionWindow.getShowWindow())) {
                transferFunctionWindow.setShowWindow(!transferFunctionWindow.getShowWindow());
                volumetricPathTracingPass->setShaderDirty();
                volumetricPathTracingPass->onHasMoved();
                reRender = true;
            }
            if (ImGui::MenuItem("Checkpoint Window", nullptr, checkpointWindow.getShowWindow())) {
                checkpointWindow.setShowWindow(!checkpointWindow.getShowWindow());
            }
            if (ImGui::MenuItem("Light Editor", nullptr, lightEditorWidget->getShowWindow())) {
                lightEditorWidget->setShowWindow(!lightEditorWidget->getShowWindow());
                volumetricPathTracingPass->setShaderDirty();
                volumetricPathTracingPass->onHasMoved();
                reRender = true;
            }
            ImGui::EndMenu();
        }

        if (ImGui::BeginMenu("Tools")) {
            if (ImGui::MenuItem("Import JSON Camera State...")) {
                IGFD_OpenModal(
                        fileDialogInstance, "ChooseCamStateFile", "Choose a File", ".json",
                        sgl::AppSettings::get()->getDataDirectory().c_str(), "", 1, nullptr,
                        ImGuiFileDialogFlags_None);
            }

            if (ImGui::MenuItem("Print Camera State")) {
                std::cout << "Position: (" << camera->getPosition().x << ", " << camera->getPosition().y
                          << ", " << camera->getPosition().z << ")" << std::endl;
                std::cout << "Look At: (" << camera->getLookAtLocation().x << ", " << camera->getLookAtLocation().y
                          << ", " << camera->getLookAtLocation().z << ")" << std::endl;
                std::cout << "Yaw: " << camera->getYaw() << std::endl;
                std::cout << "Pitch: " << camera->getPitch() << std::endl;
                std::cout << "FoVy: " << (camera->getFOVy() / sgl::PI * 180.0f) << std::endl;
            }
            ImGui::EndMenu();
        }

        //if (dataRequester.getIsProcessingRequest()) {
        //    ImGui::SetCursorPosX(ImGui::GetWindowContentRegionWidth() - ImGui::GetTextLineHeight());
        //    ImGui::ProgressSpinner(
        //            "##progress-spinner", -1.0f, -1.0f, 4.0f,
        //            ImVec4(0.1f, 0.5f, 1.0f, 1.0f));
        //}

        ImGui::EndMainMenuBar();
    }
}

void MainApp::renderGuiPropertyEditorBegin() {
    if (!useDockSpaceMode) {
        renderGuiFpsCounter();

        if (ImGui::Combo(
                "Data Set", &selectedDataSetIndex, dataSetNames.data(),
                int(dataSetNames.size()))) {
            if (selectedDataSetIndex >= NUM_MANUAL_LOADERS) {
                loadCloudDataSet(getSelectedDataSetFilename(), getSelectedDataSetEmissionFilename());
            }
        }

        //if (dataRequester.getIsProcessingRequest()) {
        //    ImGui::SameLine();
        //    ImGui::ProgressSpinner(
        //            "##progress-spinner", -1.0f, -1.0f, 4.0f,
        //            ImVec4(0.1, 0.5, 1.0, 1.0));
        //}


        if (selectedDataSetIndex == 0) {
            ImGui::InputText("##datasetfilenamelabel", &customDataSetFileName);
            ImGui::SameLine();
            if (ImGui::Button("Load File")) {
                loadCloudDataSet(getSelectedDataSetFilename(), getSelectedDataSetEmissionFilename());
            }
        }

        ImGui::Separator();
    }
}

void MainApp::renderGuiPropertyEditorCustomNodes() {
    if (propertyEditor.beginNode("Volumetric Path Tracer")) {
        volumetricPathTracingPass->renderGuiPropertyEditorNodes(propertyEditor);
        propertyEditor.endNode();
    }
}

void MainApp::update(float dt) {
    sgl::SciVisApp::update(dt);

    if (scheduledDockSpaceModeChange) {
        useDockSpaceMode = newDockSpaceMode;
        scheduledDockSpaceModeChange = false;
        if (useDockSpaceMode) {
            cameraHandle = dataView->camera;
        } else {
            cameraHandle = camera;
        }

        device->waitGraphicsQueueIdle();
        resolutionChanged(sgl::EventPtr());
    }

    updateCameraFlight(cloudData.get() != nullptr, usesNewState);

    checkLoadingRequestFinished();

    transferFunctionWindow.update(dt);

    ImGuiIO &io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard || recording || focusedWindowIndex != -1) {
        moveCameraKeyboard(dt);
    }

    if (!io.WantCaptureMouse || mouseHoverWindowIndex != -1) {
        moveCameraMouse(dt);
    }
}

void MainApp::hasMoved() {
    dataView->syncCamera();
    volumetricPathTracingPass->onHasMoved();
}

void MainApp::onCameraReset() {
}

bool MainApp::checkHasValidExtension(const std::string& filenameLower) {
    if (sgl::endsWith(filenameLower, ".xyz")
            || sgl::endsWith(filenameLower, ".nvdb")
#ifdef USE_OPENVDB
            || sgl::endsWith(filenameLower, ".vdb")
#endif
            || sgl::endsWith(filenameLower, ".dat")
            || sgl::endsWith(filenameLower, ".raw")
            || sgl::endsWith(filenameLower, ".mhd")
            || sgl::endsWith(filenameLower, ".nii")) {
        return true;
    }
    return false;
}

void MainApp::onFileDropped(const std::string& droppedFileName) {
    std::string filenameLower = sgl::toLowerCopy(droppedFileName);
    if (checkHasValidExtension(filenameLower)) {
        device->waitIdle();
        fileDialogDirectory = sgl::FileUtils::get()->getPathToFile(droppedFileName);
        selectedDataSetIndex = 0;
        customDataSetFileName = droppedFileName;
        loadCloudDataSet(getSelectedDataSetFilename(), getSelectedDataSetFilename());
    } else {
        sgl::Logfile::get()->writeError(
                "The dropped file name has an unknown extension \""
                + sgl::FileUtils::get()->getFileExtension(filenameLower) + "\".");
    }
}



// --- Visualization pipeline ---

void MainApp::loadCloudDataSet(const std::string& fileName, const std::string& emissionFileName, bool blockingDataLoading) {
    if (fileName.empty()) {
        cloudData = CloudDataPtr();
        return;
    }
    currentlyLoadedDataSetIndex = selectedDataSetIndex;

    DataSetInformation selectedDataSetInformation;
    if (selectedDataSetIndex >= NUM_MANUAL_LOADERS && !dataSetInformationList.empty()) {
        selectedDataSetInformation = *dataSetInformationList.at(selectedDataSetIndex - NUM_MANUAL_LOADERS);
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

    CloudDataPtr cloudData(new CloudData(&transferFunctionWindow, lightEditorWidget));
    if (selectedDataSetInformation.axes != glm::ivec3(0, 1, 2)) {
        cloudData->setTransposeAxes(selectedDataSetInformation.axes);
    }

    if (blockingDataLoading) {
        //bool dataLoaded = cloudData->loadFromFile(fileName, selectedDataSetInformation, transformationMatrixPtr);
        bool dataLoaded = cloudData->loadFromFile(fileName);

        if (dataLoaded) {
            this->cloudData = cloudData;
            cloudData->setClearColor(clearColor);
            newMeshLoaded = true;
            modelBoundingBox = cloudData->getWorldSpaceBoundingBox();

            volumetricPathTracingPass->setCloudData(cloudData);
            volumetricPathTracingPass->setUseLinearRGB(useLinearRGB);
            reRender = true;

            const std::string& meshDescriptorName = fileName;
            checkpointWindow.onLoadDataSet(meshDescriptorName);

            if (true) { // useCameraFlight
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

    if (!emissionFileName.empty()) {
        CloudDataPtr emissionData(new CloudData);

        bool dataLoaded = emissionData->loadFromFile(emissionFileName);

        if (dataLoaded) {
            emissionData->setClearColor(clearColor);
            volumetricPathTracingPass->setEmissionData(emissionData);
        }
    } else {
        CloudDataPtr emissionData = CloudDataPtr();
        volumetricPathTracingPass->setEmissionData(emissionData);
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
    loadCloudDataSet(getSelectedDataSetFilename(), getSelectedDataSetEmissionFilename());
}
