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
#include <ImGui/Widgets/TransferFunctionWindow.hpp>

#include "SceneData.hpp"
#include "DataSetList.hpp"
#include "CloudData.hpp"
#include "PathTracer/VolumetricPathTracingPass.hpp"

#ifdef USE_PYTHON
#include "Widgets/ReplayWidget.hpp"
#endif

class MainApp : public sgl::SciVisApp {
public:
    /**
     * @param supportsRaytracing Whether raytracing via OpenGL-Vulkan interoperability is supported.
     */
    MainApp();
    ~MainApp();
    void render();
    void update(float dt);
    void resolutionChanged(sgl::EventPtr event);

private:
    /// Renders the GUI of the scene settings and all filters and renderers.
    void renderGui();
    /// Renders the GUI for selecting an input file.
    void renderFileSelectionSettingsGui();
    /// Render the scene settings GUI, e.g. for setting the background clear color.
    void renderSceneSettingsGui();
    /// Update the color space (linear RGB vs. sRGB).
    void updateColorSpaceMode();
    // Called when the camera moved.
    void hasMoved();

    /// Scene data (e.g., camera, main framebuffer, ...).
    SceneData sceneData;

    // Data set GUI information.
    void loadAvailableDataSetInformation();
    const std::string& getSelectedMeshFilenames();
    std::vector<DataSetInformation> dataSetInformation;
    std::vector<std::string> dataSetNames; //< Contains "Local file..." at beginning, thus starts actually at 1.
    int selectedDataSetIndex = 0; //< Contains "Local file..." at beginning, thus starts actually at 1.
    int currentlyLoadedDataSetIndex = -1;
    std::string customDataSetFileName;

    // Coloring & filtering dependent on importance criteria.
    sgl::TransferFunctionWindow transferFunctionWindow;

    std::shared_ptr<VolumetricPathTracingPass> volumetricPathTracingPass;
    bool usesNewState = true;

    bool optixInitialized = false;


    /// --- Visualization pipeline ---

    /// Loads line data from a file.
    void loadCloudDataSet(const std::string& fileName, bool blockingDataLoading = true);
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