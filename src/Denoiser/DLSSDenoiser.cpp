/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2025, Christoph Neuhauser
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

#include <string>
#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <cstring>

#include <Utils/StringUtils.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Graphics/Vulkan/Utils/Instance.hpp>
#include <Graphics/Vulkan/Utils/Device.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>
#ifndef DISABLE_IMGUI
#include <ImGui/ImGuiWrapper.hpp>
#include <ImGui/Widgets/PropertyEditor.hpp>
#else
#include "Utils/ImGuiCompat.h"
#endif

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_dlssd.h>
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#include <nvsdk_ngx_params_dlssd.h>
#include <nvsdk_ngx_defs_dlssd.h>

#include "DLSSDenoiser.hpp"

#ifdef __linux__
#include <dlfcn.h>
#endif

#ifdef _WIN32
#define _WIN32_IE 0x0400
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <shlobj.h>
#include <shlwapi.h>
#include <windef.h>
#include <windows.h>
#endif

NVSDK_NGX_ProjectIdDescription getDlssProjectIdDescription() {
    NVSDK_NGX_ProjectIdDescription projectIdDescription{};
    projectIdDescription.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    projectIdDescription.EngineVersion = "0.1.0";
    projectIdDescription.ProjectId = "494d83ef-7ae9-4bf1-9fba-85477321c7e4";
    return projectIdDescription;
}

std::wstring getApplicationDataPath() {
    std::filesystem::path configDirectory = sgl::FileUtils::get()->getConfigDirectory();
    return configDirectory.wstring();
}

struct FeatureCommonTemp {
    std::wstring dllSearchPath;
    const wchar_t* dllSearchPathC = nullptr;
};

void dlssLogCallback(const char* message, NVSDK_NGX_Logging_Level loggingLevel, NVSDK_NGX_Feature sourceComponent) {
    std::cout << message << std::endl;
}

FeatureCommonTemp* fillFeatureCommonInfo(NVSDK_NGX_FeatureCommonInfo& featureCommonInfo) {
    /*
     * We use NVSDK_NGX_FeatureCommonInfo::PathListInfo for finding the DLSS .dll/.so file if not next to the app.
     * This is necessary, e.g., when building PySRG as a Python module, where the .dll/.so file may be, e.g., in:
     * <conda environment>/lib/python3.<minor>/site-packages/pysrg-0.0.0-py3.<minor>-<os>-x86_64.egg/libnvidia-ngx-dlss*
     */
    auto* temp = new FeatureCommonTemp;
#ifdef BUILD_PYTHON_MODULE
#if defined(_WIN32)
    // See: https://stackoverflow.com/questions/6924195/get-dll-path-at-runtime
    WCHAR modulePath[MAX_PATH];
    HMODULE hmodule{};
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPWSTR>(&fillFeatureCommonInfo), &hmodule)) {
        GetModuleFileNameW(hmodule, modulePath, MAX_PATH);
        PathRemoveFileSpecW(modulePath); //< Needs linking with shlwapi.lib.
        temp->dllSearchPath = modulePath;
    } else {
        sgl::Logfile::get()->writeError(
                std::string() + "Error when calling GetModuleHandle: " + std::to_string(GetLastError()));
    }
#elif defined(__linux__)
    // See: https://stackoverflow.com/questions/33151264/get-dynamic-library-directory-in-c-linux
    Dl_info dlInfo{};
    dladdr((void*)fillFeatureCommonInfo, &dlInfo);
    std::string moduleDir = sgl::FileUtils::get()->getPathToFile(dlInfo.dli_fname);
    temp->dllSearchPath = sgl::stdStringToWideString(moduleDir);
#endif
    temp->dllSearchPathC = temp->dllSearchPath.c_str();
    featureCommonInfo.PathListInfo.Path = &temp->dllSearchPathC;
    featureCommonInfo.PathListInfo.Length = 1;
#endif

    //featureCommonInfo.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_VERBOSE;
    //featureCommonInfo.LoggingInfo.DisableOtherLoggingSinks = true;
    //featureCommonInfo.LoggingInfo.LoggingCallback = dlssLogCallback;

    return temp;
}

bool getInstanceDlssSupportInfo(std::vector<const char*>& requiredInstanceExtensions) {
    auto applicationDataPath = getApplicationDataPath();
    NVSDK_NGX_FeatureCommonInfo featureCommonInfo{};
    auto* temp = fillFeatureCommonInfo(featureCommonInfo);
    NVSDK_NGX_FeatureDiscoveryInfo featureDiscoveryInfo{};
    featureDiscoveryInfo.SDKVersion = NVSDK_NGX_Version_API;
    featureDiscoveryInfo.FeatureID = NVSDK_NGX_Feature_RayReconstruction;
    featureDiscoveryInfo.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    featureDiscoveryInfo.ApplicationDataPath = applicationDataPath.c_str();
    featureDiscoveryInfo.Identifier.v.ProjectDesc = getDlssProjectIdDescription();
    featureDiscoveryInfo.FeatureInfo = &featureCommonInfo;

    uint32_t extensionCount = 0;
    VkExtensionProperties* extensions = nullptr;
    auto result = NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(
            &featureDiscoveryInfo, &extensionCount, &extensions);
    delete temp;
    if (result != NVSDK_NGX_Result_Success) {
        return false;
    }
    for (uint32_t i = 0; i < extensionCount; i++) {
        // VK_KHR_get_physical_device_properties2 has been promoted to core Vulkan 1.1.
        // This app requires at least Vulkan 1.1, so this extension is not necessary.
        if (strcmp(extensions[i].extensionName, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) == 0) {
            continue;
        }
        requiredInstanceExtensions.push_back(extensions[i].extensionName);
    }
    return true;
}

bool getPhysicalDeviceDlssSupportInfo(
        sgl::vk::Instance* instance,
        VkPhysicalDevice physicalDevice,
        std::vector<const char*>& requiredDeviceExtensions,
        sgl::vk::DeviceFeatures& requestedDeviceFeatures) {
    auto applicationDataPath = getApplicationDataPath();
    NVSDK_NGX_FeatureCommonInfo featureCommonInfo{};
    auto* temp = fillFeatureCommonInfo(featureCommonInfo);
    NVSDK_NGX_FeatureDiscoveryInfo featureDiscoveryInfo{};
    featureDiscoveryInfo.SDKVersion = NVSDK_NGX_Version_API;
    featureDiscoveryInfo.FeatureID = NVSDK_NGX_Feature_RayReconstruction;
    featureDiscoveryInfo.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    featureDiscoveryInfo.ApplicationDataPath = applicationDataPath.c_str();
    featureDiscoveryInfo.Identifier.v.ProjectDesc = getDlssProjectIdDescription();
    featureDiscoveryInfo.FeatureInfo = &featureCommonInfo;

    NVSDK_NGX_FeatureRequirement featureRequirement{};
    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_GetFeatureRequirements(
            instance->getVkInstance(), physicalDevice, &featureDiscoveryInfo, &featureRequirement);
    if (result != NVSDK_NGX_Result_Success
            && result != NVSDK_NGX_Result_FAIL_NotImplemented) {
        delete temp;
        return false;
    }
    if (featureRequirement.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported) {
        delete temp;
        return false;
    }
    // We ignore MinHWArchitecture and MinOSVersion for now, as we can't do anything about it anyway.

    uint32_t extensionCount = 0;
    VkExtensionProperties* extensions = nullptr;
    result = NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(
            instance->getVkInstance(), physicalDevice, &featureDiscoveryInfo, &extensionCount, &extensions);
    delete temp;
    if (result != NVSDK_NGX_Result_Success) {
        return false;
    }
    for (uint32_t i = 0; i < extensionCount; i++) {
        requiredDeviceExtensions.push_back(extensions[i].extensionName);
    }

    return true;
}

sgl::vk::PhysicalDeviceCheckCallback getDlssPhysicalDeviceCheckCallback(sgl::vk::Instance* instance) {
    auto physicalDeviceCheckCallback = [instance](
            VkPhysicalDevice physicalDevice,
            VkPhysicalDeviceProperties physicalDeviceProperties,
            std::vector<const char*>& requiredDeviceExtensions,
            std::vector<const char*>& optionalDeviceExtensions,
            sgl::vk::DeviceFeatures& requestedDeviceFeatures) {
        if (physicalDeviceProperties.apiVersion < VK_API_VERSION_1_1) {
            return false;
        }
        getPhysicalDeviceDlssSupportInfo(
                instance, physicalDevice, optionalDeviceExtensions, requestedDeviceFeatures);
        if (physicalDeviceProperties.apiVersion >= VK_API_VERSION_1_2) {
            bool needsDeviceAddressFeature = false;
            for (size_t i = 0; i < optionalDeviceExtensions.size();) {
                const char* extensionName = optionalDeviceExtensions.at(i);
                if (strcmp(extensionName, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
                    optionalDeviceExtensions.erase(optionalDeviceExtensions.begin() + i);
                    needsDeviceAddressFeature = true;
                    continue;
                }
                if (strcmp(extensionName, VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) == 0) {
                    optionalDeviceExtensions.erase(optionalDeviceExtensions.begin() + i);
                    needsDeviceAddressFeature = true;
                    continue;
                }
                i++;
            }
            if (needsDeviceAddressFeature) {
                requestedDeviceFeatures.optionalVulkan12Features.bufferDeviceAddress = VK_TRUE;
            }
        }
        return true;
    };
    return physicalDeviceCheckCallback;
}

bool getIsDlssSupportedByDevice(sgl::vk::Device* device) {
    return device->getDeviceDriverId() == VK_DRIVER_ID_NVIDIA_PROPRIETARY;
}


DLSSDenoiser::DLSSDenoiser(sgl::vk::Renderer* renderer, bool _denoiseAlpha) : renderer(renderer) {
    device = renderer->getDevice();
    if (!getIsDlssSupportedByDevice(device)) {
        sgl::Logfile::get()->throwError(
                "Error: Attempting to use DLSS with a Vulkan driver that is not the proprietary NVIDIA driver.");
        return;
    }
    initialize();
}

DLSSDenoiser::~DLSSDenoiser() {
    _free();
}


void DLSSDenoiser::setOutputImage(sgl::vk::ImageViewPtr& outputImage) {
    outputImageVulkan = outputImage;
}

void DLSSDenoiser::setFeatureMap(FeatureMapType featureMapType, const sgl::vk::TexturePtr& featureTexture) {
    if (featureMapType == FeatureMapType::COLOR) {
        inputImageVulkan = featureTexture->getImageView();
    } else if (featureMapType == FeatureMapType::DEPTH) {
        depthImageVulkan = featureTexture->getImageView();
    } else if (featureMapType == FeatureMapType::NORMAL_LEN_1) {
        normalImageVulkan = featureTexture->getImageView();
    } else if (featureMapType == FeatureMapType::ALBEDO) {
        albedoImageVulkan = featureTexture->getImageView();
    } else if (featureMapType == FeatureMapType::FLOW_REVERSE) {
        flowImageVulkan = featureTexture->getImageView();
    } else if (featureMapType == FeatureMapType::POSITION) {
        // Ignore.
    } else {
        sgl::Logfile::get()->writeWarning("Warning in OptixVptDenoiser::setFeatureMap: Unknown feature map.");
    }
}

bool DLSSDenoiser::getUseFeatureMap(FeatureMapType featureMapType) const {
    if (featureMapType == FeatureMapType::COLOR || featureMapType == FeatureMapType::DEPTH
            || featureMapType == FeatureMapType::NORMAL_LEN_1 || featureMapType == FeatureMapType::ALBEDO
            || featureMapType == FeatureMapType::FLOW_REVERSE) {
        return true;
    } else {
        return false;
    }
}

void DLSSDenoiser::setUseFeatureMap(FeatureMapType featureMapType, bool useFeature) {
    ;
}

void DLSSDenoiser::resetFrameNumber() {
    shallResetAccum = true;
}

void DLSSDenoiser::setTemporalDenoisingEnabled(bool enabled) {
    enableTemporalAccumulation = enabled;
    if (!enableTemporalAccumulation) {
        shallResetAccum = true;
    }
}

void DLSSDenoiser::denoise() {
    renderer->transitionImageLayout(outputImageVulkan, VK_IMAGE_LAYOUT_GENERAL);
    apply(
            inputImageVulkan,
            outputImageVulkan,
            depthImageVulkan,
            flowImageVulkan,
            exposureImageView,
            normalImageVulkan,
            albedoImageVulkan,
            roughnessImageView,
            renderer->getVkCommandBuffer());
}

void DLSSDenoiser::recreateSwapchain(uint32_t width, uint32_t height) {
    sgl::vk::ImageSettings imageSettings{};
    imageSettings.width = 1;
    imageSettings.height = 1;
    imageSettings.format = VK_FORMAT_R32_SFLOAT;
    roughnessImageView = {};
    roughnessImageView = std::make_shared<sgl::vk::ImageView>(
            std::make_shared<sgl::vk::Image>(device, imageSettings));
    renderer->transitionImageLayout(roughnessImageView, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
    roughnessImageView->clearColor(glm::vec4(1.0f), renderer->getVkCommandBuffer());
    renderer->transitionImageLayout(roughnessImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);

    if (!exposureImageView) {
        imageSettings.width = 1;
        imageSettings.height = 1;
        imageSettings.format = VK_FORMAT_R32_SFLOAT;
        float exposureVal = 1.0f;
        exposureImageView = std::make_shared<sgl::vk::ImageView>(
                std::make_shared<sgl::vk::Image>(device, imageSettings));
        //exposureImageView->getImage()->uploadData(sizeof(float), &exposureVal);
        renderer->transitionImageLayout(exposureImageView, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        exposureImageView->clearColor(glm::vec4(exposureVal), renderer->getVkCommandBuffer());
        renderer->transitionImageLayout(exposureImageView, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
}

#ifndef DISABLE_IMGUI
bool DLSSDenoiser::renderGuiPropertyEditorNodes(sgl::PropertyEditor& propertyEditor) {
    bool reRender = false;

    int modelIdx = renderPreset == DlssRenderPreset::DEFAULT ? 0 : 1;
    if (propertyEditor.addCombo(
            "Model Type", (int*)&modelIdx,
            DLSS_MODEL_TYPE_NAMES, IM_ARRAYSIZE(DLSS_MODEL_TYPE_NAMES))) {
        reRender = true;
        DlssRenderPreset _renderPreset = renderPreset;
        if (modelIdx == 0) {
            _renderPreset = DlssRenderPreset::DEFAULT;
        } else {
            _renderPreset = DlssRenderPreset::D;
        }
        setRenderPreset(_renderPreset);
    }

    if (propertyEditor.addCheckbox("Denoise Unified", &denoiseUnifiedOn)) {
        reRender = true;
        if (dlssFeature) {
            NVSDK_NGX_VULKAN_ReleaseFeature(dlssFeature);
            dlssFeature = {};
        }
    }

    if (reRender) {
        resetFrameNumber();
    }

    return reRender;
}
#endif

void DLSSDenoiser::setSettings(const std::unordered_map<std::string, std::string>& settings) {
    auto itModelType = settings.find("model_type");
    if (itModelType != settings.end()) {
        const std::string& modelTypeString = itModelType->second;
        int modelIdx = 0;
        for (int i = 0; i < IM_ARRAYSIZE(DLSS_MODEL_TYPE_NAMES); i++) {
            if (modelTypeString == DLSS_MODEL_TYPE_NAMES[i]) {
                modelIdx = i;
                break;
            }
        }
        DlssRenderPreset _renderPreset = renderPreset;
        if (modelIdx == 0) {
            _renderPreset = DlssRenderPreset::DEFAULT;
        } else {
            _renderPreset = DlssRenderPreset::D;
        }
        setRenderPreset(_renderPreset);
    }
    auto itDenoiseUnified = settings.find("denoise_unified");
    if (itDenoiseUnified != settings.end()) {
        auto _denoiseUnifiedOn = sgl::fromString<bool>(itDenoiseUnified->second);
        if (denoiseUnifiedOn != _denoiseUnifiedOn) {
            denoiseUnifiedOn = _denoiseUnifiedOn;
            if (dlssFeature) {
                NVSDK_NGX_VULKAN_ReleaseFeature(dlssFeature);
                dlssFeature = {};
            }
        }
    }
}

void DLSSDenoiser::_free() {
    if (dlssFeature) {
        NVSDK_NGX_VULKAN_ReleaseFeature(dlssFeature);
        dlssFeature = {};
    }
    if (params) {
        NVSDK_NGX_VULKAN_DestroyParameters(params);
        params = {};
    }
    if (isInitialized) {
        NVSDK_NGX_VULKAN_Shutdown1(device->getVkDevice());
        isInitialized = false;
    }
}

bool DLSSDenoiser::initialize() {
    auto applicationDataPath = getApplicationDataPath();
    NVSDK_NGX_FeatureCommonInfo featureCommonInfo{};
    auto* temp = fillFeatureCommonInfo(featureCommonInfo);
    NVSDK_NGX_Result result = NVSDK_NGX_VULKAN_Init_with_ProjectID(
            "494d83ef-7ae9-4bf1-9fba-85477321c7e4", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "0.1.0",
            applicationDataPath.c_str(),
            device->getInstance()->getVkInstance(), device->getVkPhysicalDevice(), device->getVkDevice(),
            sgl::vk::Instance::getVkInstanceProcAddrFunctionPointer(),
            sgl::vk::Device::getVkDeviceProcAddrFunctionPointer(),
            &featureCommonInfo, NVSDK_NGX_Version_API);
    if (result != NVSDK_NGX_Result_Success) {
        sgl::Logfile::get()->throwError("NVSDK_NGX_VULKAN_Init_with_ProjectID failed.");
    }
    isInitialized = true;

    result = NVSDK_NGX_VULKAN_GetCapabilityParameters(&params);
    if (result != NVSDK_NGX_Result_Success) {
        _free();
        sgl::Logfile::get()->throwError("NVSDK_NGX_VULKAN_GetCapabilityParameters failed.");
    }

    int needsUpdatedDriver = 0;
    unsigned int minDriverVersionMajor = 0;
    unsigned int minDriverVersionMinor = 0;
    NVSDK_NGX_Result resultUpdate = NVSDK_NGX_Parameter_GetI(
            params, NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver, &needsUpdatedDriver);
    NVSDK_NGX_Result resultMajor = NVSDK_NGX_Parameter_GetUI(
            params, NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMajor, &minDriverVersionMajor);
    NVSDK_NGX_Result resultMinor = NVSDK_NGX_Parameter_GetUI(
            params, NVSDK_NGX_Parameter_SuperSamplingDenoising_MinDriverVersionMinor, &minDriverVersionMinor);
    if (resultUpdate != NVSDK_NGX_Result_Success || resultMajor != NVSDK_NGX_Result_Success
            || resultMinor != NVSDK_NGX_Result_Success) {
        _free();
        sgl::Logfile::get()->throwError("NVSDK_NGX_Parameter_Get* failed.");
    }
    if (needsUpdatedDriver) {
        _free();
        sgl::Logfile::get()->throwError(
                "DLSS Ray Reconstruction could not be initialized: NVIDIA driver version >= "
                + std::to_string(minDriverVersionMajor) + "." + std::to_string(minDriverVersionMinor) + " needed.");
    }

    int isDlssAvailable = 0;
    result = NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &isDlssAvailable);
    if (result != NVSDK_NGX_Result_Success || !isDlssAvailable) {
        int featureInitResult = 0;
        NVSDK_NGX_Result resultInit = NVSDK_NGX_Parameter_GetI(
                params, NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult, &featureInitResult);
        if (resultInit != NVSDK_NGX_Result_Success) {
            _free();
            sgl::Logfile::get()->throwError("DLSS Ray Reconstruction could not be initialized.");
        }
        _free();
        if (featureCommonInfo.PathListInfo.Length > 0) {
            std::cout << "DLSS library search path: "
                      << sgl::wideStringArrayToStdString(featureCommonInfo.PathListInfo.Path[0]) << std::endl;
        }
        sgl::Logfile::get()->throwError(
                std::string() + "DLSS is not available on this system. Feature init result: "
                + sgl::wideStringArrayToStdString(GetNGXResultAsString(NVSDK_NGX_Result(featureInitResult))));
    }
    /*result = NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSamplingDenoising_FeatureInitResult, &isDlssAvailable);
    if (result != NVSDK_NGX_Result_Success || !isDlssAvailable) {
        _free();
        sgl::Logfile::get()->throwError("DLSS could not be initialized.");
    }*/
    delete temp;

    return true;
}

bool DLSSDenoiser::queryOptimalSettings(
        uint32_t displayWidth, uint32_t displayHeight,
        uint32_t& renderWidthOptimal, uint32_t& renderHeightOptimal,
        uint32_t& renderWidthMax, uint32_t& renderHeightMax,
        uint32_t& renderWidthMin, uint32_t& renderHeightMin,
        float& sharpness) {
    NVSDK_NGX_Result result = NGX_DLSSD_GET_OPTIMAL_SETTINGS(
            params, displayWidth, displayHeight, NVSDK_NGX_PerfQuality_Value(perfQuality),
            &renderWidthOptimal, &renderHeightOptimal,
            &renderWidthMax, &renderHeightMax,
            &renderWidthMin, &renderHeightMin,
            &sharpness);
    return result == NVSDK_NGX_Result_Success;
}

NVSDK_NGX_Resource_VK createNgxImageView(const sgl::vk::ImageViewPtr& imageView) {
    const auto& imageSettings = imageView->getImage()->getImageSettings();
    return NVSDK_NGX_Create_ImageView_Resource_VK(
            imageView->getVkImageView(), imageView->getImage()->getVkImage(),
            imageView->getVkImageSubresourceRange(), imageSettings.format,
            imageSettings.width, imageSettings.height, (imageSettings.usage & VK_IMAGE_USAGE_STORAGE_BIT) != 0);
}

void DLSSDenoiser::resetAccum() {
    shallResetAccum = true;
}

void DLSSDenoiser::checkRecreateFeature(
        const sgl::vk::ImageViewPtr& colorImageIn,
        const sgl::vk::ImageViewPtr& colorImageOut,
        VkCommandBuffer commandBuffer) {
    if (dlssFeature) {
        return;
    }

    //NVSDK_NGX_VULKAN_GetScratchBufferSize(NVSDK_NGX_Feature InFeatureId, const NVSDK_NGX_Parameter *InParameters, size_t *OutSizeInBytes);
    unsigned int creationNodeMask = 1;
    unsigned int visibilityNodeMask = 1;

    int dlssFeatureCreateFlags = 0;
    dlssFeatureCreateFlags |= NVSDK_NGX_DLSS_Feature_Flags_None;
    dlssFeatureCreateFlags |= motionVectorLowRes ? NVSDK_NGX_DLSS_Feature_Flags_MVLowRes : 0;
    dlssFeatureCreateFlags |= isHdr ? NVSDK_NGX_DLSS_Feature_Flags_IsHDR : 0;
    dlssFeatureCreateFlags |= isDepthInverted ? NVSDK_NGX_DLSS_Feature_Flags_DepthInverted : 0;
    dlssFeatureCreateFlags |= doSharpening ? NVSDK_NGX_DLSS_Feature_Flags_DoSharpening : 0;
    dlssFeatureCreateFlags |= enableAutoExposure ? NVSDK_NGX_DLSS_Feature_Flags_AutoExposure : 0;
    dlssFeatureCreateFlags |= upscaleAlpha ? NVSDK_NGX_DLSS_Feature_Flags_AlphaUpscaling : 0;
    //NVSDK_NGX_DLSS_Feature_Flags_MVJittered

    auto renderWidth = colorImageIn->getImage()->getImageSettings().width;
    auto renderHeight = colorImageIn->getImage()->getImageSettings().height;
    auto displayWidth = colorImageOut->getImage()->getImageSettings().width;
    auto displayHeight = colorImageOut->getImage()->getImageSettings().height;
    NVSDK_NGX_DLSSD_Create_Params dlssdCreateParams{};
    dlssdCreateParams.InWidth = renderWidth;
    dlssdCreateParams.InHeight = renderHeight;
    dlssdCreateParams.InTargetWidth = displayWidth;
    dlssdCreateParams.InTargetHeight = displayHeight;
    dlssdCreateParams.InPerfQualityValue = NVSDK_NGX_PerfQuality_Value(perfQuality);
    dlssdCreateParams.InFeatureCreateFlags = dlssFeatureCreateFlags;
    NVSDK_NGX_DLSS_Denoise_Mode dlssDenoiseMode;
    if (denoiseUnifiedOn) {
        dlssDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode::NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    } else {
        dlssDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode::NVSDK_NGX_DLSS_Denoise_Mode_Off;
    }
    dlssdCreateParams.InDenoiseMode = dlssDenoiseMode;
    dlssdCreateParams.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode::NVSDK_NGX_DLSS_Roughness_Mode_Unpacked;
    dlssdCreateParams.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type::NVSDK_NGX_DLSS_Depth_Type_Linear;

    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_DLAA, unsigned(renderPreset));
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Quality, unsigned(renderPreset));
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Balanced, unsigned(renderPreset));
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_Performance, unsigned(renderPreset));
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraPerformance, unsigned(renderPreset));
    NVSDK_NGX_Parameter_SetUI(params, NVSDK_NGX_Parameter_RayReconstruction_Hint_Render_Preset_UltraQuality, unsigned(renderPreset));

    NVSDK_NGX_Result result = NGX_VULKAN_CREATE_DLSSD_EXT1(
            device->getVkDevice(), commandBuffer,
            creationNodeMask, visibilityNodeMask,
            &dlssFeature, params, &dlssdCreateParams);
    if (result != NVSDK_NGX_Result_Success) {
        NVSDK_NGX_VULKAN_DestroyParameters(params);
        NVSDK_NGX_VULKAN_Shutdown1(device->getVkDevice());
        sgl::Logfile::get()->throwError("NGX_VULKAN_CREATE_DLSSD_EXT1 failed.");
    }
}

bool DLSSDenoiser::apply(
        const sgl::vk::ImageViewPtr& colorImageIn,
        const sgl::vk::ImageViewPtr& colorImageOut,
        const sgl::vk::ImageViewPtr& depthImage,
        const sgl::vk::ImageViewPtr& motionVectorImage,
        const sgl::vk::ImageViewPtr& exposureImage,
        const sgl::vk::ImageViewPtr& normalImage,
        const sgl::vk::ImageViewPtr& albedoImage,
        const sgl::vk::ImageViewPtr& roughnessImage,
        VkCommandBuffer commandBuffer) {
    bool _isHdr =
            colorImageIn->getImage()->getImageSettings().format == VK_FORMAT_R16G16B16A16_SFLOAT
            || colorImageIn->getImage()->getImageSettings().format == VK_FORMAT_R32G32B32A32_SFLOAT;
    if (isHdr != _isHdr) {
        isHdr = _isHdr;
        if (dlssFeature) {
            NVSDK_NGX_VULKAN_ReleaseFeature(dlssFeature);
            dlssFeature = {};
        }
    }

    auto renderWidth = colorImageIn->getImage()->getImageSettings().width;
    auto renderHeight = colorImageIn->getImage()->getImageSettings().height;
    auto displayWidth = colorImageOut->getImage()->getImageSettings().width;
    auto displayHeight = colorImageOut->getImage()->getImageSettings().height;
    if (cachedRenderWidth != renderWidth || cachedRenderHeight != renderHeight
            || cachedDisplayWidth != displayWidth || cachedDisplayHeight != displayHeight) {
        cachedRenderWidth = renderWidth;
        cachedRenderHeight = renderHeight;
        cachedDisplayWidth = displayWidth;
        cachedDisplayHeight = displayHeight;
        if (dlssFeature) {
            NVSDK_NGX_VULKAN_ReleaseFeature(dlssFeature);
            dlssFeature = {};
        }
    }

    bool useSingleTimeCommandBuffer = commandBuffer == VK_NULL_HANDLE;
    if (useSingleTimeCommandBuffer) {
        commandBuffer = device->beginSingleTimeCommands();
    }
    checkRecreateFeature(colorImageIn, colorImageOut, commandBuffer);

    auto colorImageInNgx = createNgxImageView(colorImageIn);
    auto colorImageOutNgx = createNgxImageView(colorImageOut);
    auto depthImageNgx = createNgxImageView(depthImage);
    auto motionVectorImageNgx = createNgxImageView(motionVectorImage);
    auto exposureImageNgx = createNgxImageView(exposureImage);
    auto normalImageNgx = createNgxImageView(normalImage);
    auto albedoImageNgx = createNgxImageView(albedoImage);
    auto roughnessImageNgx = createNgxImageView(roughnessImage);

    float jitterOffsetX = 0.0f;
    float jitterOffsetY = 0.0f;
    float mvScaleX = 1.0f;
    float mvScaleY = 1.0f;
    NVSDK_NGX_Coordinates renderOffset = { 0, 0 };
    NVSDK_NGX_Dimensions renderSize = { renderWidth, renderHeight };
    NVSDK_NGX_VK_DLSSD_Eval_Params dlssEvalParams{};
    dlssEvalParams.pInColor = &colorImageInNgx;
    dlssEvalParams.pInOutput = &colorImageOutNgx;
    dlssEvalParams.pInDepth = &depthImageNgx;
    dlssEvalParams.pInMotionVectors = &motionVectorImageNgx;
    dlssEvalParams.pInDiffuseAlbedo = &albedoImageNgx;
    dlssEvalParams.pInSpecularAlbedo = &albedoImageNgx;
    dlssEvalParams.pInNormals = &normalImageNgx;
    dlssEvalParams.pInRoughness = &roughnessImageNgx;
    /*
     * Exposure not supported according to sec. 3.6 of Ray Reconstruction docs.
     */
    dlssEvalParams.pInExposureTexture = &exposureImageNgx;
    dlssEvalParams.InJitterOffsetX = jitterOffsetX;
    dlssEvalParams.InJitterOffsetY = jitterOffsetY;
    dlssEvalParams.InReset = shallResetAccum;
    dlssEvalParams.InMVScaleX = mvScaleX;
    dlssEvalParams.InMVScaleY = mvScaleY;
    dlssEvalParams.InColorSubrectBase = renderOffset;
    dlssEvalParams.InDepthSubrectBase = renderOffset;
    dlssEvalParams.InTranslucencySubrectBase = renderOffset;
    dlssEvalParams.InMVSubrectBase = renderOffset;
    dlssEvalParams.InRenderSubrectDimensions = renderSize;
    //dlssEvalParams.InToneMapperType = TODO;
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_Parameter_Sharpness, sharpeningFactor);
    NVSDK_NGX_Result result = NGX_VULKAN_EVALUATE_DLSSD_EXT(commandBuffer, dlssFeature, params, &dlssEvalParams);
    if (result != NVSDK_NGX_Result_Success) {
        NVSDK_NGX_VULKAN_ReleaseFeature(dlssFeature);
        NVSDK_NGX_VULKAN_DestroyParameters(params);
        NVSDK_NGX_VULKAN_Shutdown1(device->getVkDevice());
        sgl::Logfile::get()->throwError("NGX_VULKAN_EVALUATE_DLSSD_EXT failed.");
    }
    if (useSingleTimeCommandBuffer) {
        device->endSingleTimeCommands(commandBuffer);
    }

    if (enableTemporalAccumulation) {
        shallResetAccum = false;
    }

    return true;
}

void DLSSDenoiser::setPerfQuality(DlssPerfQuality _perfQuality) {
    if (perfQuality != _perfQuality) {
        perfQuality = _perfQuality;
        if (dlssFeature) {
            NVSDK_NGX_VULKAN_ReleaseFeature(dlssFeature);
            dlssFeature = {};
        }
    }
}

void DLSSDenoiser::setRenderPreset(DlssRenderPreset _preset) {
    if (renderPreset != _preset) {
        renderPreset = _preset;
        if (dlssFeature) {
            NVSDK_NGX_VULKAN_ReleaseFeature(dlssFeature);
            dlssFeature = {};
        }
    }
}

void DLSSDenoiser::setUpscaleAlpha(bool _upscaleAlpha) {
    if (upscaleAlpha != _upscaleAlpha) {
        upscaleAlpha = _upscaleAlpha;
        if (dlssFeature) {
            NVSDK_NGX_VULKAN_ReleaseFeature(dlssFeature);
            dlssFeature = {};
        }
    }
}

void DLSSDenoiser::setSharpeningFactor(float _sharpeningFactor) {
    bool _doSharpening = _sharpeningFactor > 0.0f;
    if (doSharpening != _doSharpening) {
        doSharpening = _doSharpening;
        if (dlssFeature) {
            NVSDK_NGX_VULKAN_ReleaseFeature(dlssFeature);
            dlssFeature = {};
        }
    }
    if (sharpeningFactor != _sharpeningFactor) {
        sharpeningFactor = _sharpeningFactor;
    }
}
