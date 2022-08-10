/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser
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

#include <Utils/AppSettings.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/InteropCuda.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#include <Graphics/Vulkan/Render/CommandBuffer.hpp>

#include "nanovdb/NanoVDB.h"
#include "nanovdb/util/Primitives.h"
#include "CloudData.hpp"

#include "Config.hpp"
#include "VolumetricPathTracingModuleRenderer.hpp"
#include "Module.hpp"

TORCH_LIBRARY(vpt, m) {
    m.def("vpt::initialize", initialize);
    m.def("vpt::cleanup", cleanup);
    m.def("vpt::render_frame", renderFrame);
    m.def("vpt::load_cloud_file", loadCloudFile);
    m.def("vpt::load_environment_map", loadEnvironmentMap);
    m.def("vpt::set_environment_map_intensity", setEnvironmentMapIntensityFactor);
    m.def("vpt::set_scattering_albedo", setScatteringAlbedo);
    m.def("vpt::set_extinction_base", setExtinctionBase);
    m.def("vpt::set_extinction_scale", setExtinctionScale);
    m.def("vpt::set_vpt_mode", setVPTMode);
    m.def("vpt::set_camera_position", setCameraPosition);
    m.def("vpt::set_camera_target", setCameraTarget);
    m.def("vpt::set_camera_FOVy", setCameraFOVy);
    m.def("vpt::set_feature_map_type", setFeatureMapType);
    m.def("vpt::set_seed_offset", setSeedOffset);
    m.def("vpt::get_feature_map", getFeatureMap);
    m.def("vpt::set_phase_g", setPhaseG);
}

static sgl::vk::Renderer* renderer = nullptr;
VolumetricPathTracingModuleRenderer* vptRenderer = nullptr;

torch::Tensor renderFrame(torch::Tensor inputTensor, int64_t frameCount) {
    if (inputTensor.sizes().size() != 3) {
        sgl::Logfile::get()->throwError(
                "Error in renderFrame: inputTensor.sizes().size() != 3.", false);
    }
    if (inputTensor.size(0) != 3 && inputTensor.size(0) != 4) {
        sgl::Logfile::get()->throwError(
                "Error in renderFrame: The number of image channels is not equal to 3 or 4.",
                false);
    }
    if (inputTensor.dtype() != torch::kFloat32) {
        sgl::Logfile::get()->throwError(
                "Error in renderFrame: The only data type currently supported is 32-bit float.",
                false);
    }

    if (inputTensor.device().type() == torch::DeviceType::CPU) {
        return renderFrameCpu(inputTensor, frameCount);
    } else if (inputTensor.device().type() == torch::DeviceType::Vulkan) {
        return renderFrameVulkan(inputTensor, frameCount);
    }
#ifdef SUPPORT_CUDA_INTEROP
    else if (inputTensor.device().type() == torch::DeviceType::CUDA) {
        return renderFrameCuda(inputTensor, frameCount);
    }
#endif
    else {
        sgl::Logfile::get()->throwError("Unsupported PyTorch device type.", false);
    }

    return {};
}

torch::Tensor getFeatureMap(torch::Tensor inputTensor, int64_t featureMap) {

    if (inputTensor.sizes().size() != 3) {
        std::cout << "err" << std::endl;

        sgl::Logfile::get()->throwError(
                "Error in renderFrame: inputTensor.sizes().size() != 3.", false);
    }
    if (inputTensor.size(0) != 3 && inputTensor.size(0) != 4) {
        std::cout << "err" << std::endl;

        sgl::Logfile::get()->throwError(
                "Error in renderFrame: The number of image channels is not equal to 3 or 4.",
                false);
    }
    if (inputTensor.dtype() != torch::kFloat32) {
        std::cout << "err" << std::endl;

        sgl::Logfile::get()->throwError(
                "Error in renderFrame: The only data type currently supported is 32-bit float.",
                false);
    }

    // Expecting tensor of size CxHxW (channels x height x width).
    const size_t channels = inputTensor.size(0);
    const size_t height = inputTensor.size(1);
    const size_t width = inputTensor.size(2);

    if (!inputTensor.is_contiguous()) {
        inputTensor = inputTensor.contiguous();
    }
    //inputTensor.data_ptr();

    if (vptRenderer->settingsDiffer(
            width, height, uint32_t(channels), inputTensor.device(), inputTensor.dtype())) {
        vptRenderer->setRenderingResolution(
                width, height, uint32_t(channels), inputTensor.device(), inputTensor.dtype());
    }

    if (inputTensor.device().type() == torch::DeviceType::CPU) {
        void* imageDataPtr = vptRenderer->getFeatureMapCpu(FeatureMapTypeVpt(featureMap));

        torch::Tensor outputTensor = torch::from_blob(
                imageDataPtr, { int(height), int(width), int(channels) },
                torch::TensorOptions().dtype(torch::kFloat32).device(inputTensor.device()));

        return outputTensor.permute({2, 0, 1}).detach().clone();//.clone()
    }
#ifdef SUPPORT_CUDA_INTEROP
    else if (inputTensor.device().type() == torch::DeviceType::CUDA) {
        void* imageDataDevicePtr = vptRenderer->getFeatureMapCuda(FeatureMapTypeVpt(featureMap));

        torch::Tensor outputTensor = torch::from_blob(
                imageDataDevicePtr, { int(height), int(width), int(channels) },
                torch::TensorOptions().dtype(torch::kFloat32).device(inputTensor.device()));

        //return outputTensor.permute({2, 0, 1}).detach().clone();
        return outputTensor.permute({2, 0, 1}).detach();
    }
#endif
    else {
        sgl::Logfile::get()->throwError("Unsupported PyTorch device type.", false);
    }

    return {};    
}


torch::Tensor renderFrameCpu(torch::Tensor inputTensor, int64_t frameCount) {
    std::cout << "Device type CPU." << std::endl;

    // Expecting tensor of size CxHxW (channels x height x width).
    const size_t channels = inputTensor.size(0);
    const size_t height = inputTensor.size(1);
    const size_t width = inputTensor.size(2);

    if (!inputTensor.is_contiguous()) {
        inputTensor = inputTensor.contiguous();
    }
    //inputTensor.data_ptr();

    if (vptRenderer->settingsDiffer(
            width, height, uint32_t(channels), inputTensor.device(), inputTensor.dtype())) {
        vptRenderer->setRenderingResolution(
                width, height, uint32_t(channels), inputTensor.device(), inputTensor.dtype());
    }

    void* imageDataPtr = vptRenderer->renderFrameCpu(frameCount);

    torch::Tensor outputTensor = torch::from_blob(
            imageDataPtr, { int(height), int(width), int(channels) },
            torch::TensorOptions().dtype(torch::kFloat32).device(inputTensor.device()));

    return outputTensor.permute({2, 0, 1}).detach().clone();//.clone()
}

torch::Tensor renderFrameVulkan(torch::Tensor inputTensor, int64_t frameCount) {
    std::cout << "Device type Vulkan." << std::endl;

    // Expecting tensor of size CxHxW (channels x height x width).
    const size_t channels = inputTensor.size(0);
    const size_t height = inputTensor.size(1);
    const size_t width = inputTensor.size(2);

    if (!inputTensor.is_contiguous()) {
        inputTensor = inputTensor.contiguous();
    }
    //inputTensor.data_ptr();

    if (vptRenderer->settingsDiffer(
            width, height, uint32_t(channels), inputTensor.device(), inputTensor.dtype())) {
        vptRenderer->setRenderingResolution(
                width, height, uint32_t(channels), inputTensor.device(), inputTensor.dtype());
    }

    // TODO: Add support for not going the GPU->CPU->GPU route.
    void* imageDataPtr = vptRenderer->renderFrameCpu(frameCount);

    torch::Tensor outputTensor = torch::from_blob(
            imageDataPtr, { int(height), int(width), int(channels) },
            torch::TensorOptions().dtype(torch::kFloat32).device(at::kCPU)).to(inputTensor.device());

    return outputTensor.permute({2, 0, 1}).detach().clone();
}

#ifdef SUPPORT_CUDA_INTEROP
torch::Tensor renderFrameCuda(torch::Tensor inputTensor, int64_t frameCount) {
    std::cout << "Device type CUDA." << std::endl;

    // Expecting tensor of size CxHxW (channels x height x width).
    const size_t channels = inputTensor.size(0);
    const size_t height = inputTensor.size(1);
    const size_t width = inputTensor.size(2);

    if (!inputTensor.is_contiguous()) {
        inputTensor = inputTensor.contiguous();
    }
    //inputTensor.data_ptr();

    if (true || vptRenderer->settingsDiffer(
            width, height, uint32_t(channels), inputTensor.device(), inputTensor.dtype())) {
        vptRenderer->setRenderingResolution(
                width, height, uint32_t(channels), inputTensor.device(), inputTensor.dtype());
    }

    void* imageDataDevicePtr = vptRenderer->renderFrameCuda(frameCount);

    /*auto* hostData = new float[channels * width * height];
    cudaMemcpy(hostData, imageDataDevicePtr, sizeof(float) * channels * width * height, cudaMemcpyKind::cudaMemcpyDeviceToHost);
    sgl::BitmapPtr bitmap(new sgl::Bitmap(int(width), int(height), 32));
    uint8_t* bitmapData = bitmap->getPixels();
    for (uint32_t y = 0; y < height; y++) {
        for (uint32_t x = 0; x < width; x++) {
            for (uint32_t c = 0; c < 3; c++) {
                float value = hostData[(x + y * width) * 4 + c];
                bitmapData[(x + y * width) * 4 + c] = uint8_t(std::clamp(value * 255.0f, 0.0f, 255.0f));
            }
            bitmapData[(x + y * width) * 4 + 3] = 255;
        }
    }
    bitmap->savePNG("cuda_test.png", false);
    delete[] hostData;*/

    torch::Tensor outputTensor = torch::from_blob(
            imageDataDevicePtr, { int(height), int(width), int(channels) },
            torch::TensorOptions().dtype(torch::kFloat32).device(inputTensor.device()));

    //return outputTensor.permute({2, 0, 1}).detach().clone();
    return outputTensor.permute({2, 0, 1}).detach();
}
#endif

void vulkanErrorCallback() {
    std::cerr << "Application callback" << std::endl;
}

const char* argv[] = { "." }; //< Just pass something as argv.
int libraryReferenceCount = 0;

void initialize() {
    if (libraryReferenceCount == 0) {
        // Initialize the filesystem utilities.
        sgl::FileUtils::get()->initialize("CloudRendering", 1, argv);

        // Load the file containing the app settings.
        std::string settingsFile = sgl::FileUtils::get()->getConfigDirectory() + "settings.txt";
        sgl::AppSettings::get()->setSaveSettings(false);
        sgl::AppSettings::get()->getSettings().addKeyValue("window-debugContext", true);

#ifdef DATA_PATH
        if (!sgl::FileUtils::get()->directoryExists("Data") && !sgl::FileUtils::get()->directoryExists("../Data")) {
            sgl::AppSettings::get()->setDataDirectory(DATA_PATH);
        }
#endif

        sgl::AppSettings::get()->setRenderSystem(sgl::RenderSystem::VULKAN);
        sgl::AppSettings::get()->createHeadless();

        std::vector<const char*> optionalDeviceExtensions;
#if defined(SUPPORT_OPTIX) || defined(SUPPORT_CUDA_INTEROP)
        optionalDeviceExtensions = sgl::vk::Device::getCudaInteropDeviceExtensions();
#endif

        sgl::vk::Instance* instance = sgl::AppSettings::get()->getVulkanInstance();
        instance->setDebugCallback(&vulkanErrorCallback);
        sgl::vk::Device* device = new sgl::vk::Device;
        device->createDeviceHeadless(
                instance, {
                        VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME,
                        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME
                },
                optionalDeviceExtensions);
        sgl::AppSettings::get()->setPrimaryDevice(device);
        sgl::AppSettings::get()->initializeSubsystems();

        renderer = new sgl::vk::Renderer(sgl::AppSettings::get()->getPrimaryDevice());
        vptRenderer = new VolumetricPathTracingModuleRenderer(renderer);

        // TODO: Make this configurable.
        CloudDataPtr cloudData = std::make_shared<CloudData>();
        cloudData->setNanoVdbGridHandle(nanovdb::createFogVolumeSphere<float>(
                0.25f, nanovdb::Vec3<float>(0), 0.01f));
        vptRenderer->setCloudData(cloudData);
        vptRenderer->setVptMode(VptMode::DELTA_TRACKING);
        vptRenderer->setUseLinearRGB(true);
    }

    libraryReferenceCount++;
}

void cleanup() {
    libraryReferenceCount--;

    if (libraryReferenceCount == 0) {
        sgl::AppSettings::get()->getPrimaryDevice()->waitIdle();
        delete vptRenderer;
        delete renderer;
        sgl::AppSettings::get()->release();
    }
}

void loadCloudFile(const std::string& filename) {
    //std::cout << "loading cloud from " << filename << std::endl;
    CloudDataPtr cloudData = std::make_shared<CloudData>();
    cloudData->loadFromFile(filename);
    vptRenderer->setCloudData(cloudData);
}

void loadEnvironmentMap(const std::string& filename) {
    //std::cout << "loadEnvironmentMap from " << filename << std::endl;
    vptRenderer->loadEnvironmentMapImage(filename);
}

void setEnvironmentMapIntensityFactor(double intensityFactor) {
    //std::cout << "setEnvironmentMapIntensityFactor to " << intensityFactor << std::endl;
    vptRenderer->setEnvironmentMapIntensityFactor(intensityFactor);
}

bool parseVector3(std::vector<double> data, glm::vec3& out){
    if (data.size() != 3){
        std::cerr << "exptected vector of size 3" << std:: endl;
        return false;
    }
    out.x = data[0];
    out.y = data[1];
    out.z = data[2];
 
    return true;
}

void setScatteringAlbedo(std::vector<double> albedo) {
    glm::vec3 vec = glm::vec3(0,0,0);
    if (parseVector3(albedo, vec)) {
        //std::cout << "setScatteringAlbedo to " << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
        vptRenderer->setScatteringAlbedo(vec);
    }
}

void setExtinctionScale(double extinctionScale) {
    //std::cout << "setExtinctionScale to " << extinctionScale << std::endl;
    vptRenderer->setExtinctionScale(extinctionScale);
}

void setExtinctionBase(std::vector<double> extinctionBase) {
    glm::vec3 vec = glm::vec3(0,0,0);
    if (parseVector3(extinctionBase, vec)) {
        //std::cout << "setExtinctionBase to " << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
        vptRenderer->setExtinctionBase(vec);
    }
}

void setVPTMode(int64_t mode){
    std::cout << "setVPTMode to " << VPT_MODE_NAMES[mode] << std::endl;
    vptRenderer->setVptMode(VptMode(mode));
}

void setFeatureMapType(int64_t type) {
    //std::cout << "setFeatureMapType to " << type << std::endl;
    vptRenderer->setFeatureMapType(FeatureMapTypeVpt(type));
}


void setCameraPosition(std::vector<double> cameraPosition) {
    glm::vec3 vec = glm::vec3(0,0,0);
    if (parseVector3(cameraPosition, vec)) {
        //std::cout << "setCameraPosition to " << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
        vptRenderer->setCameraPosition(vec);
    }
}

void setCameraTarget(std::vector<double> cameraTarget) {
    glm::vec3 vec = glm::vec3(0,0,0);
    if (parseVector3(cameraTarget, vec)) {
        //std::cout << "setCameraTarget to " << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
        vptRenderer->setCameraTarget(vec);
    }
}

void setCameraFOVy(double FOVy){
    //std::cout << "setCameraFOVy to " << FOVy << std::endl;
    vptRenderer->setCameraFOVy(FOVy * sgl::PI / 180);
}

void setSeedOffset(int64_t offset){
    //std::cout << "setSeedOffset to " << offset << std::endl;
    vptRenderer->setCustomSeedOffset(offset);
}

void setPhaseG(double phaseG){
    vptRenderer->setPhaseG(phaseG);
}