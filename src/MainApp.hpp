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

#ifndef MAINAPP_HPP
#define MAINAPP_HPP

#include <string>
#include <vector>
#include <map>

#include <Utils/SciVis/SciVisApp.hpp>
#include <Graphics/Shader/Shader.hpp>
#include <ImGui/Widgets/MultiVarTransferFunctionWindow.hpp>

#ifdef SUPPORT_CUDA_INTEROP
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#endif

#include "SceneData.hpp"
#include "DataSetList.hpp"
#include "CloudData.hpp"
#include "PathTracer/LightEditorWidget.hpp"
#include "PathTracer/VolumetricPathTracingPass.hpp"

#ifdef USE_PYTHON
#include "Widgets/ReplayWidget.hpp"
#endif

class DataView;
typedef std::shared_ptr<DataView> DataViewPtr;

class MainApp : public sgl::SciVisApp {
public:
    MainApp();
    ~MainApp() override;
    void render() override;
    void update(float dt) override;
    void resolutionChanged(sgl::EventPtr event) override;

protected:
    void renderGuiGeneralSettingsPropertyEditor() override;

private:
    /// Renders the GUI of the scene settings and all filters and renderers.
    void renderGui() override;
    /// Update the color space (linear RGB vs. sRGB).
    void updateColorSpaceMode() override;
    // Called when the camera moved.
    void hasMoved() override;
    /// Callback when the camera was reset.
    void onCameraReset() override;
    /// Callback when a file has been dropped on the program.
    void onFileDropped(const std::string& droppedFileName) override;
    bool checkHasValidExtension(const std::string& filenameLower);

    // Dock space mode.
    void renderGuiMenuBar();
    void renderGuiPropertyEditorBegin() override;
    void renderGuiPropertyEditorCustomNodes() override;
    bool scheduledDockSpaceModeChange = false;
    bool newDockSpaceMode = false;
    int focusedWindowIndex = -1;
    int mouseHoverWindowIndex = -1;
    bool showRendererWindow = true;
    DataViewPtr dataView;
    sgl::CameraPtr cameraHandle;

    /// Scene data (e.g., camera, main framebuffer, ...).
    int subsamplingFactor = 1;
    SceneData sceneData;

    // This setting lets all data views use the same viewport resolution.
    bool useFixedSizeViewport = false;
    glm::ivec2 fixedViewportSizeEdit{ 1920, 1080 };
    glm::ivec2 fixedViewportSize{ 1920, 1080 };

    // Data set GUI information.
    void loadAvailableDataSetInformation();
    const std::string& getSelectedDataSetFilename();
    const std::string& getSelectedDataSetEmissionFilename();
    void openFileDialog();
    DataSetInformationPtr dataSetInformationRoot;
    std::vector<DataSetInformationPtr> dataSetInformationList; //< List of all leaves.
    std::vector<std::string> dataSetNames; //< Contains "Local file..." at beginning, thus starts actually at 1.
    int selectedDataSetIndex = 0; //< Contains "Local file..." at beginning, thus starts actually at 1.
    int currentlyLoadedDataSetIndex = -1;
    std::string customDataSetFileName;
    std::string customDataSetFileNameEmission;
    ImGuiFileDialog* fileDialogInstance = nullptr;
    std::string fileDialogDirectory;

    // Utility functionality.
    void loadCameraStateFromFile(const std::string& filePath);
    void loadCameraPosesFromFile(const std::string& filePath);

    // For mapping volume density to display density and emission.
    sgl::MultiVarTransferFunctionWindow transferFunctionWindow;

    // For specifying light sources apart from the environment map.
    LightEditorWidget* lightEditorWidget;

    std::shared_ptr<VolumetricPathTracingPass> volumetricPathTracingPass;
    bool usesNewState = true;

#ifdef SUPPORT_CUDA_INTEROP
    CUcontext cuContext = {};
    CUdevice cuDevice = 0;
#endif
    bool cudaInteropInitialized = false;
    bool optixInitialized = false;


    /// --- Visualization pipeline ---

    /// Loads line data from a file.
    void loadCloudDataSet(const std::string& fileName, const std::string& emissionFileName, bool blockingDataLoading = true);
    /// Checks if an asynchronous loading request was finished.
    void checkLoadingRequestFinished();
    /// Reload the currently loaded data set.
    void reloadDataSet() override;

    const int NUM_MANUAL_LOADERS = 1;
    bool newMeshLoaded = true;
    sgl::AABB3 modelBoundingBox;
    CloudDataPtr cloudData;
};

#endif // MAINAPP_HPP