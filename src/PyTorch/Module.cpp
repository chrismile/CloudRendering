/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2022, Christoph Neuhauser, Timm Knörle
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
#include <ImGui/Widgets/MultiVarTransferFunctionWindow.hpp>
#include "Utils/ImGuiCompat.h"

#ifdef USE_OPENVDB
#include <openvdb/openvdb.h>
#endif

#ifdef SUPPORT_CUDA_INTEROP
#include <cuda.h>
#include <c10/cuda/CUDAStream.h>
#endif

#ifdef CUDA_HOST_COMPILER_COMPATIBLE
#include "EnergyTerm.cuh"
#endif

#include "nanovdb/NanoVDB.h"
#include "nanovdb/util/Primitives.h"
#include "CloudData.hpp"
#include "PathTracer/LightEditorWidget.hpp"

#include "Config.hpp"
#include "VolumetricPathTracingModuleRenderer.hpp"
#include "Module.hpp"

#ifdef BUILD_PYTHON_MODULE_NEW
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <torch/extension.h>

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

PYBIND11_MODULE(vpt, m) {
    m.def("initialize", initialize);
    m.def("cleanup", cleanup);
    m.def("load_cloud_file", loadCloudFile, py::arg("filename"));
    m.def("load_volume_file", loadCloudFile, py::arg("filename"));
    m.def("load_emission_file", loadEmissionFile, py::arg("filename"));
    m.def("load_environment_map", loadEnvironmentMap, py::arg("filename"));
    m.def("set_use_builtin_environment_map", setUseBuiltinEnvironmentMap, py::arg("env_map_name"));
    m.def("set_environment_map_intensity", setEnvironmentMapIntensityFactor, py::arg("intensity_factor"));
    m.def("set_environment_map_intensity_rgb", setEnvironmentMapIntensityFactorRgb, py::arg("intensity_factor"));
    m.def("disable_env_map_rot", disableEnvMapRot);
    m.def("set_env_map_rot_camera", setEnvMapRotCamera);
    m.def("set_env_map_rot_euler_angles", setEnvMapRotEulerAngles, py::arg("euler_angles_vec"));
    m.def("set_env_map_rot_yaw_pitch_roll", setEnvMapRotYawPitchRoll, py::arg("yaw_pitch_roll_vec"));
    m.def("set_env_map_rot_angle_axis", setEnvMapRotAngleAxis, py::arg("_axis_vec"), py::arg("_angle"));
    m.def("set_env_map_rot_quaternion", setEnvMapRotQuaternion, py::arg("_quaternion_vec"));
    m.def("set_scattering_albedo", setScatteringAlbedo, py::arg("albedo"));
    m.def("set_extinction_scale", setExtinctionScale, py::arg("extinction_scale"));
    m.def("set_extinction_base", setExtinctionBase, py::arg("extinction_base"));
    m.def("set_phase_g", setPhaseG, py::arg("phase_g"));
    m.def("set_use_transfer_function", setUseTransferFunction, py::arg("_use_tf"));
    m.def("load_transfer_function_file", loadTransferFunctionFile, py::arg("tf_file_path"));
    m.def("load_transfer_function_file_gradient", loadTransferFunctionFileGradient, py::arg("tf_file_path"));
    m.def("set_transfer_function_range", setTransferFunctionRange, py::arg("_min_val"), py::arg("_max_val"));
    m.def("set_transfer_function_range_gradient", setTransferFunctionRangeGradient, py::arg("_min_val"), py::arg("_max_val"));
    m.def("set_transfer_function_empty", setTransferFunctionEmpty);
    m.def("set_transfer_function_empty_gradient", setTransferFunctionEmptyGradient);
    m.def("get_camera_position", getCameraPosition);
    m.def("get_camera_view_matrix", getCameraViewMatrix);
    m.def("get_camera_fovy", getCameraFOVy);
    m.def("set_camera_position", setCameraPosition, py::arg("camera_position"));
    m.def("set_camera_target", setCameraTarget, py::arg("camera_target"));
    m.def("overwrite_camera_view_matrix", overwriteCameraViewMatrix, py::arg("view_matrix_data"));
    m.def("set_camera_fovy", setCameraFOVy, py::arg("fovy"));
    m.def("set_vpt_mode", setVPTMode, py::arg("mode"));
    m.def("set_vpt_mode_from_name", setVPTModeFromName, py::arg("mode_name"));
    m.def("set_denoiser", setDenoiser, py::arg("denoiser_name"));
    m.def("set_denoiser_property", setDenoiserProperty, py::arg("key"), py::arg("value"));
    m.def("set_pytorch_denoiser_model_file", setPyTorchDenoiserModelFile, py::arg("denoiser_model_file_path"));
    m.def("set_output_foreground_map", setOutputForegroundMap, py::arg("_shall_output_foreground_map"));
    m.def("set_feature_map_type", setFeatureMapType, py::arg("type"));
    m.def("set_use_empty_space_skipping", setUseEmptySpaceSkipping, py::arg("_use_empty_space_skipping"));
    m.def("set_use_lights", setUseLights, py::arg("_use_lights"));
    m.def("clear_lights", clearLights);
    m.def("add_light", addLight);
    m.def("remove_light", removeLight, py::arg("light_idx"));
    m.def("set_light_property", setLightProperty, py::arg("light_idx"), py::arg("key"), py::arg("value"));
    m.def("load_lights_from_file", loadLightsFromFile, py::arg("file_path"));
    m.def("save_lights_to_file", saveLightsToFile, py::arg("file_path"));
    m.def("set_use_headlight", setUseHeadlight, py::arg("_use_headlight"));
    m.def("set_use_headlight_distance", setUseHeadlightDistance, py::arg("_use_headlight_distance"));
    m.def("set_headlight_color", setHeadlightColor, py::arg("_headlight_color"));
    m.def("set_headlight_intensity", setHeadlightIntensity, py::arg("_headlight_intensity"));
    m.def("set_use_isosurfaces", setUseIsosurfaces, py::arg("_use_isosurfaces"));
    m.def("set_iso_value", setIsoValue, py::arg("_iso_value"));
    m.def("set_iso_surface_color", setIsoSurfaceColor, py::arg("_iso_surface_color"));
    m.def("set_isosurface_type", setIsosurfaceType, py::arg("_isosurface_type"));
    m.def("set_surface_brdf", setSurfaceBrdf, py::arg("_surface_brdf"));
    m.def("set_brdf_parameter", setBrdfParameter, py::arg("key"), py::arg("value"));
    m.def("set_use_isosurface_tf", setUseIsosurfaceTf, py::arg("_use_isosurface_tf"));
    m.def("set_num_isosurface_subdivisions", setNumIsosurfaceSubdivisions, py::arg("_subdivs"));
    m.def("set_close_isosurfaces", setCloseIsosurfaces, py::arg("_close_isosurfaces"));
    m.def("set_use_legacy_normals", setUseLegacyNormals, py::arg("_use_legacy_normals"));
    m.def("set_use_clip_plane", setUseClipPlane, py::arg("_use_clip_plane"));
    m.def("set_clip_plane_normal", setClipPlaneNormal, py::arg("_clip_plane_normal"));
    m.def("set_clip_plane_distance", setClipPlaneDistance, py::arg("_clip_plane_distance"));
    m.def("set_seed_offset", setSeedOffset, py::arg("offset"));
    m.def("set_view_projection_matrix_as_previous", setViewProjectionMatrixAsPrevious);
    m.def("set_emission_cap", setEmissionCap, py::arg("emission_cap"));
    m.def("set_emission_strength", setEmissionStrength, py::arg("emission_strength"));
    m.def("set_use_emission", setUseEmission, py::arg("use_emission"));
    m.def("set_tf_scattering_albedo_strength", setTfScatteringAlbedoStrength, py::arg("strength"));
    m.def("flip_yz_coordinates", flipYZ, py::arg("flip"));
    m.def("get_volume_voxel_size", getVolumeVoxelSize);
    m.def("get_render_bounding_box", getRenderBoundingBox);
    m.def("remember_next_bounds", rememberNextBounds);
    m.def("forget_current_bounds", forgetCurrentBounds);
    m.def("set_max_grid_extent", setMaxGridExtent,
        py::arg("max_grid_extent"),
        "Due to legacy reasons, the grid has size (-0.25, 0.25) in the largest dimension.\n@param maxDimSize The new extent value such that the size is (-maxGridExtent, maxGridExtent).\nNote: Must be called before any call to @see loadCloudFile!");
    m.def("set_global_world_bounding_box", setGlobalWorldBoundingBox,
        py::arg("global_bb_vec"),
        "This function can be used to normalize the grid wrt. a global bounding box.");
    m.def("get_vdb_world_bounding_box", getVDBWorldBoundingBox);
    m.def("get_vdb_index_bounding_box", getVDBIndexBoundingBox);
    m.def("get_vdb_voxel_size", getVDBVoxelSize);
    m.def("render_frame", renderFrame, py::arg("input_tensor"), py::arg("frame_count"));
    m.def("set_use_feature_maps", setUseFeatureMaps, py::arg("feature_map_names"));
    m.def("get_feature_map_from_string", getFeatureMapFromString, py::arg("input_tensor"), py::arg("feature_map"));
    m.def("get_feature_map", getFeatureMap, py::arg("input_tensor"), py::arg("feature_map"));
    m.def("get_transmittance_volume", getTransmittanceVolume, py::arg("input_tensor"));
    m.def("set_secondary_volume_downscaling_factor", setSecondaryVolumeDownscalingFactor, py::arg("ds_factor"));
    m.def("compute_occupation_volume", computeOccupationVolume, py::arg("input_tensor"), py::arg("ds_factor"), py::arg("max_kernel_radius"));
    m.def("update_observation_frequency_fields", updateObservationFrequencyFields, py::arg("num_bins_x"), py::arg("num_bins_y"), py::arg("transmittance_field"), py::arg("obs_freq_field"), py::arg("angular_obs_freq_field"));
    m.def("compute_energy", computeEnergy, py::arg("num_cams"), py::arg("num_bins_x"), py::arg("num_bins_y"), py::arg("gamma"), py::arg("obs_freq_field"), py::arg("angular_obs_freq_field"), py::arg("occupancy_field"), py::arg("energy_term_field"));
    m.def("triangulate_isosurfaces", triangulateIsosurfaces);
    m.def("export_vdb_volume", exportVdbVolume, py::arg("filename"));
}
#else
#define MODULE_PREFIX "vpt::"
TORCH_LIBRARY(vpt, m) {
    m.def(MODULE_PREFIX "initialize", initialize);
    m.def(MODULE_PREFIX "cleanup", cleanup);
    m.def(MODULE_PREFIX "render_frame", renderFrame);
    m.def(MODULE_PREFIX "load_cloud_file", loadCloudFile);
    m.def(MODULE_PREFIX "load_volume_file", loadCloudFile);
    m.def(MODULE_PREFIX "load_emission_file", loadEmissionFile);
    m.def(MODULE_PREFIX "load_environment_map", loadEnvironmentMap);
    m.def(MODULE_PREFIX "set_use_builtin_environment_map", setUseBuiltinEnvironmentMap);
    m.def(MODULE_PREFIX "set_environment_map_intensity", setEnvironmentMapIntensityFactor);
    m.def(MODULE_PREFIX "set_environment_map_intensity_rgb", setEnvironmentMapIntensityFactorRgb);
    m.def(MODULE_PREFIX "disable_env_map_rot", disableEnvMapRot);
    m.def(MODULE_PREFIX "set_env_map_rot_camera", setEnvMapRotCamera);
    m.def(MODULE_PREFIX "set_env_map_rot_euler_angles", setEnvMapRotEulerAngles);
    m.def(MODULE_PREFIX "set_env_map_rot_yaw_pitch_roll", setEnvMapRotYawPitchRoll);
    m.def(MODULE_PREFIX "set_env_map_rot_angle_axis", setEnvMapRotAngleAxis);
    m.def(MODULE_PREFIX "set_env_map_rot_quaternion", setEnvMapRotQuaternion);
    m.def(MODULE_PREFIX "set_scattering_albedo", setScatteringAlbedo);
    m.def(MODULE_PREFIX "set_extinction_base", setExtinctionBase);
    m.def(MODULE_PREFIX "set_extinction_scale", setExtinctionScale);
    m.def(MODULE_PREFIX "set_vpt_mode", setVPTMode);
    m.def(MODULE_PREFIX "set_vpt_mode_from_name", setVPTModeFromName);
    m.def(MODULE_PREFIX "set_denoiser", setDenoiser);
    m.def(MODULE_PREFIX "set_denoiser_property", setDenoiserProperty);
    m.def(MODULE_PREFIX "set_pytorch_denoiser_model_file", setPyTorchDenoiserModelFile);
    m.def(MODULE_PREFIX "set_output_foreground_map", setOutputForegroundMap);
    m.def(MODULE_PREFIX "set_use_transfer_function", setUseTransferFunction);
    m.def(MODULE_PREFIX "load_transfer_function_file", loadTransferFunctionFile);
    m.def(MODULE_PREFIX "load_transfer_function_file_gradient", loadTransferFunctionFileGradient);
    m.def(MODULE_PREFIX "set_transfer_function_range", setTransferFunctionRange);
    m.def(MODULE_PREFIX "set_transfer_function_range_gradient", setTransferFunctionRangeGradient);
    m.def(MODULE_PREFIX "set_transfer_function_empty", setTransferFunctionEmpty);
    m.def(MODULE_PREFIX "set_transfer_function_empty_gradient", setTransferFunctionEmptyGradient);
    m.def(MODULE_PREFIX "get_camera_position", getCameraPosition);
    m.def(MODULE_PREFIX "get_camera_view_matrix", getCameraViewMatrix);
    m.def(MODULE_PREFIX "get_camera_fovy", getCameraFOVy);
    m.def(MODULE_PREFIX "set_camera_position", setCameraPosition);
    m.def(MODULE_PREFIX "set_camera_target", setCameraTarget);
    m.def(MODULE_PREFIX "overwrite_camera_view_matrix", overwriteCameraViewMatrix);
    m.def(MODULE_PREFIX "set_camera_FOVy", setCameraFOVy);
    m.def(MODULE_PREFIX "set_feature_map_type", setFeatureMapType);
    m.def(MODULE_PREFIX "set_use_empty_space_skipping", setUseEmptySpaceSkipping);
    m.def(MODULE_PREFIX "set_use_lights", setUseLights);
    m.def(MODULE_PREFIX "clear_lights", clearLights);
    m.def(MODULE_PREFIX "add_light", addLight);
    m.def(MODULE_PREFIX "remove_light", removeLight);
    m.def(MODULE_PREFIX "set_light_property", setLightProperty);
    m.def(MODULE_PREFIX "load_lights_from_file", loadLightsFromFile);
    m.def(MODULE_PREFIX "save_lights_to_file", saveLightsToFile);
    m.def(MODULE_PREFIX "set_use_headlight", setUseHeadlight);
    m.def(MODULE_PREFIX "set_use_headlight_distance", setUseHeadlightDistance);
    m.def(MODULE_PREFIX "set_headlight_color", setHeadlightColor);
    m.def(MODULE_PREFIX "set_headlight_intensity", setHeadlightIntensity);
    m.def(MODULE_PREFIX "set_use_isosurfaces", setUseIsosurfaces);
    m.def(MODULE_PREFIX "set_iso_value", setIsoValue);
    m.def(MODULE_PREFIX "set_iso_surface_color", setIsoSurfaceColor);
    m.def(MODULE_PREFIX "set_isosurface_type", setIsosurfaceType);
    m.def(MODULE_PREFIX "set_surface_brdf", setSurfaceBrdf);
    m.def(MODULE_PREFIX "set_brdf_parameter", setBrdfParameter);
    m.def(MODULE_PREFIX "set_use_isosurface_tf", setUseIsosurfaceTf);
    m.def(MODULE_PREFIX "set_num_isosurface_subdivisions", setNumIsosurfaceSubdivisions);
    m.def(MODULE_PREFIX "set_use_clip_plane", setUseClipPlane);
    m.def(MODULE_PREFIX "set_clip_plane_normal", setClipPlaneNormal);
    m.def(MODULE_PREFIX "set_clip_plane_distance", setClipPlaneDistance);
    m.def(MODULE_PREFIX "set_close_isosurfaces", setCloseIsosurfaces);
    m.def(MODULE_PREFIX "set_use_legacy_normals", setUseLegacyNormals);
    m.def(MODULE_PREFIX "set_seed_offset", setSeedOffset);
    m.def(MODULE_PREFIX "set_use_feature_maps", setUseFeatureMaps);
    m.def(MODULE_PREFIX "get_feature_map", getFeatureMap);
    m.def(MODULE_PREFIX "get_feature_map_from_string", getFeatureMapFromString);
    m.def(MODULE_PREFIX "get_transmittance_volume", getTransmittanceVolume);
    m.def(MODULE_PREFIX "set_secondary_volume_downscaling_factor", setSecondaryVolumeDownscalingFactor);
    m.def(MODULE_PREFIX "compute_occupation_volume", computeOccupationVolume);
    m.def(MODULE_PREFIX "update_observation_frequency_fields", updateObservationFrequencyFields);
    m.def(MODULE_PREFIX "compute_energy", computeEnergy);
    m.def(MODULE_PREFIX "set_phase_g", setPhaseG);
    m.def(MODULE_PREFIX "set_view_projection_matrix_as_previous",setViewProjectionMatrixAsPrevious);
    m.def(MODULE_PREFIX "set_use_emission", setUseEmission);
    m.def(MODULE_PREFIX "set_tf_scattering_albedo_strength", setTfScatteringAlbedoStrength);
    m.def(MODULE_PREFIX "set_emission_strength", setEmissionStrength);
    m.def(MODULE_PREFIX "set_emission_cap", setEmissionCap);
    m.def(MODULE_PREFIX "get_volume_voxel_size", getVolumeVoxelSize);
    m.def(MODULE_PREFIX "get_render_bounding_box", getRenderBoundingBox);
    m.def(MODULE_PREFIX "remember_next_bounds", rememberNextBounds);
    m.def(MODULE_PREFIX "forget_current_bounds", forgetCurrentBounds);
    m.def(MODULE_PREFIX "set_max_grid_extent", setMaxGridExtent);
    m.def(MODULE_PREFIX "set_global_world_bounding_box", setGlobalWorldBoundingBox);
    m.def(MODULE_PREFIX "get_vdb_world_bounding_box", getVDBWorldBoundingBox);
    m.def(MODULE_PREFIX "get_vdb_index_bounding_box", getVDBIndexBoundingBox);
    m.def(MODULE_PREFIX "get_vdb_voxel_size", getVDBVoxelSize);
    m.def(MODULE_PREFIX "flip_yz_coordinates", flipYZ);
    m.def(MODULE_PREFIX "triangulate_isosurfaces", triangulateIsosurfaces);
    m.def(MODULE_PREFIX "export_vdb_volume", exportVdbVolume);
}
#endif

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

void setUseFeatureMaps(std::vector<std::string> featureMapNames) {
    std::unordered_set<FeatureMapTypeVpt> featureMaps;
    for (const auto& featureMap : featureMapNames) {
        for (int i = 0; i < IM_ARRAYSIZE(VPT_FEATURE_MAP_NAMES); i++) {
            if (featureMap == VPT_FEATURE_MAP_NAMES[i]) {
                featureMaps.insert(FeatureMapTypeVpt(i));
                break;
            }
        }
    }
    vptRenderer->setUseFeatureMaps(featureMaps);
}

torch::Tensor getFeatureMapFromString(torch::Tensor inputTensor, const std::string& featureMap) {
    int featureMapIdx = 0;
    for (int i = 0; i < IM_ARRAYSIZE(VPT_FEATURE_MAP_NAMES); i++) {
        if (featureMap == VPT_FEATURE_MAP_NAMES[i]) {
            featureMapIdx = i;
            break;
        }
    }
    return getFeatureMap(inputTensor, featureMapIdx);
}

torch::Tensor getFeatureMap(torch::Tensor inputTensor, int64_t featureMap) {
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
    vptRenderer->checkDenoiser();

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

torch::Tensor getTransmittanceVolume(torch::Tensor inputTensor) {
    const auto& cloudData = vptRenderer->getCloudData();
    auto ds = int(vptRenderer->getVptPass()->getSecondaryVolumeDownscalingFactor());

    if (inputTensor.device().type() == torch::DeviceType::CPU) {
        void* imageDataPtr = vptRenderer->getFeatureMapCpu(FeatureMapTypeVpt::TRANSMITTANCE_VOLUME);

        torch::Tensor outputTensor = torch::from_blob(
                imageDataPtr,
                {
                        sgl::iceil(int(cloudData->getGridSizeZ()), ds),
                        sgl::iceil(int(cloudData->getGridSizeY()), ds),
                        sgl::iceil(int(cloudData->getGridSizeX()), ds)
                },
                torch::TensorOptions().dtype(torch::kFloat32).device(inputTensor.device()));

        return outputTensor.detach().clone();
    }
#ifdef SUPPORT_CUDA_INTEROP
    else if (inputTensor.device().type() == torch::DeviceType::CUDA) {
        void* imageDataDevicePtr = vptRenderer->getFeatureMapCuda(FeatureMapTypeVpt::TRANSMITTANCE_VOLUME);

        torch::Tensor outputTensor = torch::from_blob(
                imageDataDevicePtr,
                {
                        sgl::iceil(int(cloudData->getGridSizeZ()), ds),
                        sgl::iceil(int(cloudData->getGridSizeY()), ds),
                        sgl::iceil(int(cloudData->getGridSizeX()), ds)
                },
                torch::TensorOptions().dtype(torch::kFloat32).device(inputTensor.device()));

        return outputTensor.detach();//.clone();
    }
#endif
    else {
        sgl::Logfile::get()->throwError("Unsupported PyTorch device type.", false);
    }

    return {};
}

void setSecondaryVolumeDownscalingFactor(int64_t dsFactor) {
    vptRenderer->getVptPass()->setSecondaryVolumeDownscalingFactor(uint32_t(dsFactor));
}

torch::Tensor computeOccupationVolume(torch::Tensor inputTensor, int64_t dsFactor, int64_t maxKernelRadius) {
    const auto& cloudData = vptRenderer->getCloudData();
    auto ds = int(dsFactor);

    if (inputTensor.device().type() == torch::DeviceType::CPU) {
        // TODO
        sgl::Logfile::get()->throwError("Unsupported PyTorch device type.", false);
    }
#ifdef SUPPORT_CUDA_INTEROP
    else if (inputTensor.device().type() == torch::DeviceType::CUDA) {
        void* imageDataDevicePtr = vptRenderer->computeOccupationVolumeCuda(
                uint32_t(dsFactor), uint32_t(maxKernelRadius));

        torch::Tensor outputTensor = torch::from_blob(
                imageDataDevicePtr,
                {
                        sgl::iceil(int(cloudData->getGridSizeZ()), ds),
                        sgl::iceil(int(cloudData->getGridSizeY()), ds),
                        sgl::iceil(int(cloudData->getGridSizeX()), ds)
                },
                torch::TensorOptions().dtype(torch::kUInt8).device(inputTensor.device()));

        return outputTensor.detach();//.clone();
    }
#endif
    else {
        sgl::Logfile::get()->throwError("Unsupported PyTorch device type.", false);
    }

    return {};
}

void updateObservationFrequencyFields(
        int64_t numBinsX, int64_t numBinsY,
        torch::Tensor transmittanceField, torch::Tensor obsFreqField, torch::Tensor angularObsFreqField) {
#ifdef CUDA_HOST_COMPILER_COMPATIBLE
    if (transmittanceField.device().type() != torch::DeviceType::CUDA
            || obsFreqField.device().type() != torch::DeviceType::CUDA
            || angularObsFreqField.device().type() != torch::DeviceType::CUDA) {
        sgl::Logfile::get()->throwError(
                "Error in updateObservationFrequencyFields: All tensors must be on a CUDA device.", false);
    }
    if (!transmittanceField.is_contiguous()) {
        sgl::Logfile::get()->throwError(
                "Error in updateObservationFrequencyFields: transmittanceField is not contiguous.", false);
    }
    if (!obsFreqField.is_contiguous()) {
        sgl::Logfile::get()->throwError(
                "Error in updateObservationFrequencyFields: obsFreqField is not contiguous.", false);
    }
    if (!angularObsFreqField.is_contiguous()) {
        sgl::Logfile::get()->throwError(
                "Error in updateObservationFrequencyFields: angularObsFreqField is not contiguous.", false);
    }
    if (obsFreqField.sizes().size() != 3) {
        sgl::Logfile::get()->throwError(
                "Error in updateObservationFrequencyFields: obsFreqField.sizes().size() != 3.", false);
    }
    if (angularObsFreqField.sizes().size() != 4) {
        sgl::Logfile::get()->throwError(
                "Error in updateObservationFrequencyFields: angularObsFreqField.sizes().size() != 3.", false);
    }
    if (obsFreqField.dtype() != torch::kFloat32 || angularObsFreqField.dtype() != torch::kFloat32) {
        sgl::Logfile::get()->throwError(
                "Error in updateObservationFrequencyFields: The only data type currently supported is 32-bit float.",
                false);
    }
    const auto depth = obsFreqField.size(0);
    const auto height = obsFreqField.size(1);
    const auto width = obsFreqField.size(2);
    if (angularObsFreqField.size(0) != depth || angularObsFreqField.size(1) != height
            || angularObsFreqField.size(2) != width || angularObsFreqField.size(3) != int64_t(numBinsX * numBinsY)) {
        sgl::Logfile::get()->throwError(
                "Error in updateObservationFrequencyFields: angularObsFreqField sizes mismatch.", false);
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    updateObservationFrequencyFieldsImpl(
            stream, uint32_t(depth), uint32_t(height), uint32_t(width), uint32_t(numBinsX), uint32_t(numBinsY),
            vptRenderer->getCameraPosition(), // TODO
            transmittanceField.data_ptr<float>(),
            obsFreqField.data_ptr<float>(),
            angularObsFreqField.data_ptr<float>());
#else
    sgl::Logfile::get()->throwError("Error in updateObservationFrequencyFields: No CUDA compatible host compiler used!");
#endif
}

void computeEnergy(
        int64_t numCams, int64_t numBinsX, int64_t numBinsY, double gamma,
        torch::Tensor obsFreqField, torch::Tensor angularObsFreqField,
        torch::Tensor occupancyField, torch::Tensor energyTermField) {
#ifdef CUDA_HOST_COMPILER_COMPATIBLE
    if (obsFreqField.device().type() != torch::DeviceType::CUDA
            || angularObsFreqField.device().type() != torch::DeviceType::CUDA
            || occupancyField.device().type() != torch::DeviceType::CUDA
            || energyTermField.device().type() != torch::DeviceType::CUDA) {
        sgl::Logfile::get()->throwError(
                "Error in computeEnergy: All tensors must be on a CUDA device.", false);
    }
    if (!obsFreqField.is_contiguous()
            || !angularObsFreqField.is_contiguous()
            || !occupancyField.is_contiguous()
            || !energyTermField.is_contiguous()) {
        sgl::Logfile::get()->throwError(
                "Error in computeEnergy: All tensors must be contiguous.", false);
    }
    if (obsFreqField.sizes().size() != 3) {
        sgl::Logfile::get()->throwError(
                "Error in computeEnergy: obsFreqField.sizes().size() != 3.", false);
    }
    if (angularObsFreqField.sizes().size() != 4) {
        sgl::Logfile::get()->throwError(
                "Error in computeEnergy: angularObsFreqField.sizes().size() != 3.", false);
    }
    if (obsFreqField.dtype() != torch::kFloat32 || angularObsFreqField.dtype() != torch::kFloat32) {
        sgl::Logfile::get()->throwError(
                "Error in computeEnergy: The only data type currently supported is 32-bit float.",
                false);
    }
    const auto depth = obsFreqField.size(0);
    const auto height = obsFreqField.size(1);
    const auto width = obsFreqField.size(2);
    if (angularObsFreqField.size(0) != depth || angularObsFreqField.size(1) != height
            || angularObsFreqField.size(2) != width || angularObsFreqField.size(3) != int64_t(numBinsX * numBinsY)) {
        sgl::Logfile::get()->throwError(
                "Error in computeEnergy: angularObsFreqField sizes mismatch.", false);
    }
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    computeEnergyImpl(
            stream, uint32_t(depth), uint32_t(height), uint32_t(width), float(numBinsX), float(numBinsY),
            uint32_t(numCams), float(gamma),
            obsFreqField.data_ptr<float>(),
            angularObsFreqField.data_ptr<float>(),
            occupancyField.data_ptr<uint8_t>(),
            energyTermField.data_ptr<float>());
#else
    sgl::Logfile::get()->throwError("Error in computeEnergy: No CUDA compatible host compiler used!");
#endif
}

torch::Tensor renderFrameCpu(torch::Tensor inputTensor, int64_t frameCount) {
    //std::cout << "Device type CPU." << std::endl;

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
    vptRenderer->checkDenoiser();

    void* imageDataPtr = vptRenderer->renderFrameCpu(frameCount);

    torch::Tensor outputTensor = torch::from_blob(
            imageDataPtr, { int(height), int(width), int(channels) },
            torch::TensorOptions().dtype(torch::kFloat32).device(inputTensor.device()));

    return outputTensor.permute({2, 0, 1}).detach().clone();//.clone()
}

torch::Tensor renderFrameVulkan(torch::Tensor inputTensor, int64_t frameCount) {
    //std::cout << "Device type Vulkan." << std::endl;

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
    vptRenderer->checkDenoiser();

    // TODO: Add support for not going the GPU->CPU->GPU route.
    void* imageDataPtr = vptRenderer->renderFrameCpu(frameCount);

    torch::Tensor outputTensor = torch::from_blob(
            imageDataPtr, { int(height), int(width), int(channels) },
            torch::TensorOptions().dtype(torch::kFloat32).device(at::kCPU)).to(inputTensor.device());

    return outputTensor.permute({2, 0, 1}).detach().clone();
}

#ifdef SUPPORT_CUDA_INTEROP
torch::Tensor renderFrameCuda(torch::Tensor inputTensor, int64_t frameCount) {
    //std::cout << "Device type CUDA." << std::endl;

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
    vptRenderer->checkDenoiser();

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

#ifdef BUILD_PYTHON_MODULE_NEW
std::string getLibraryPath() {
#if defined(_WIN32)
    // See: https://stackoverflow.com/questions/6924195/get-dll-path-at-runtime
    WCHAR modulePath[MAX_PATH];
    HMODULE hmodule{};
    if (GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPWSTR>(&getLibraryPath), &hmodule)) {
        GetModuleFileNameW(hmodule, modulePath, MAX_PATH);
        PathRemoveFileSpecW(modulePath); //< Needs linking with shlwapi.lib.
        return sgl::wideStringArrayToStdString(modulePath);
    } else {
        sgl::Logfile::get()->writeError(
                std::string() + "Error when calling GetModuleHandle: " + std::to_string(GetLastError()));
    }
    return "";
#elif defined(__linux__)
    // See: https://stackoverflow.com/questions/33151264/get-dynamic-library-directory-in-c-linux
    Dl_info dlInfo{};
    dladdr((void*)getLibraryPath, &dlInfo);
    std::string moduleDir = sgl::FileUtils::get()->getPathToFile(dlInfo.dli_fname);
    return moduleDir;
#else
    return ".";
#endif
}
#endif

void initialize() {
    if (libraryReferenceCount == 0) {
        // Initialize the filesystem utilities.
        sgl::FileUtils::get()->initialize("CloudRendering", 1, argv);

        // Load the file containing the app settings.
        std::string settingsFile = sgl::FileUtils::get()->getConfigDirectory() + "settings.txt";
        sgl::AppSettings::get()->setSaveSettings(false);
        sgl::AppSettings::get()->getSettings().addKeyValue("window-debugContext", true);
        //sgl::AppSettings::get()->setVulkanDebugPrintfEnabled();

#if defined(BUILD_PYTHON_MODULE_NEW)
        std::string dataDirectory = getLibraryPath();
        if (!sgl::endsWith(dataDirectory, "/") && !sgl::endsWith(dataDirectory, "\\")) {
            dataDirectory += "/";
        }
        dataDirectory += "Data";
        sgl::AppSettings::get()->setDataDirectory(dataDirectory);
        sgl::AppSettings::get()->initializeDataDirectory();
#elif defined(DATA_PATH)
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

#ifdef USE_OPENVDB
        openvdb::initialize();
#endif

        renderer = new sgl::vk::Renderer(sgl::AppSettings::get()->getPrimaryDevice());
        if (device->getComputeQueueIndex() != device->getGraphicsQueueIndex()) {
            renderer->setUseComputeQueue(true);
        }
        vptRenderer = new VolumetricPathTracingModuleRenderer(renderer);

        // TODO: Make this configurable.
        CloudDataPtr cloudData = std::make_shared<CloudData>(
                vptRenderer->getTransferFunctionWindow(), vptRenderer->getLightEditorWidget());
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
    CloudDataPtr cloudData = std::make_shared<CloudData>(
            vptRenderer->getTransferFunctionWindow(), vptRenderer->getLightEditorWidget());
    if (vptRenderer->getHasGlobalWorldBoundingBox()) {
        cloudData->setGlobalWorldBoundingBox(vptRenderer->getGlobalWorldBoundingBox());
    }
    cloudData->loadFromFile(filename);
    vptRenderer->setCloudData(cloudData);
}

void loadEmissionFile(const std::string& filename) {
    //std::cout << "loading emission from " << filename << std::endl;
    CloudDataPtr emissionData = std::make_shared<CloudData>(
            vptRenderer->getTransferFunctionWindow(), vptRenderer->getLightEditorWidget());
    emissionData->loadFromFile(filename);
    vptRenderer->setEmissionData(emissionData);
}


bool parseVector3(const std::vector<double>& data, glm::vec3& out) {
    if (data.size() != 3) {
        sgl::Logfile::get()->writeError("Error in parseVector3: Expected vector of size 3.");
        return false;
    }
    out.x = float(data[0]);
    out.y = float(data[1]);
    out.z = float(data[2]);
    return true;
}

bool parseVector4(const std::vector<double>& data, glm::vec4& out) {
    if (data.size() != 4) {
        sgl::Logfile::get()->writeError("Error in parseVector4: Expected vector of size 4.");
        return false;
    }
    out.x = float(data[0]);
    out.y = float(data[1]);
    out.z = float(data[2]);
    out.w = float(data[3]);
    return true;
}

bool parseQuaternion(const std::vector<double>& data, glm::quat& out) {
    if (data.size() != 4) {
        sgl::Logfile::get()->writeError("Error in parseQuaternion: Expected vector of size 4.");
        return false;
    }
    out.x = float(data[0]);
    out.y = float(data[1]);
    out.z = float(data[2]);
    out.w = float(data[3]);
    return true;
}


void loadEnvironmentMap(const std::string& filename) {
    //std::cout << "loadEnvironmentMap from " << filename << std::endl;
    vptRenderer->loadEnvironmentMapImage(filename);
}

void setUseBuiltinEnvironmentMap(const std::string& envMapName) {
    vptRenderer->setUseBuiltinEnvironmentMap(envMapName);
}

void setEnvironmentMapIntensityFactor(double intensityFactor) {
    //std::cout << "setEnvironmentMapIntensityFactor to " << intensityFactor << std::endl;
    vptRenderer->setEnvironmentMapIntensityFactor(intensityFactor);
}

void setEnvironmentMapIntensityFactorRgb(std::vector<double> intensityFactor) {
    glm::vec3 vec = glm::vec3(0.0f, 0.0f, 0.0f);
    if (parseVector3(intensityFactor, vec)) {
        vptRenderer->getVptPass()->setEnvironmentMapIntensityFactorRgb(vec);
    }
}

void disableEnvMapRot() {
    vptRenderer->getVptPass()->disableEnvMapRot();
}

void setEnvMapRotCamera() {
    vptRenderer->getVptPass()->setEnvMapRotCamera();
}

void setEnvMapRotEulerAngles(std::vector<double> eulerAnglesVec) {
    glm::vec3 vec = glm::vec3(0.0f, 0.0f, 0.0f);
    if (parseVector3(eulerAnglesVec, vec)) {
        vptRenderer->getVptPass()->setEnvMapRotEulerAngles(vec);
    }
}

void setEnvMapRotYawPitchRoll(std::vector<double> yawPitchRollVec) {
    glm::vec3 vec = glm::vec3(0.0f, 0.0f, 0.0f);
    if (parseVector3(yawPitchRollVec, vec)) {
        vptRenderer->getVptPass()->setEnvMapRotYawPitchRoll(vec);
    }
}

void setEnvMapRotAngleAxis(std::vector<double> _axisVec, double _angle) {
    glm::vec3 vec = glm::vec3(0.0f, 0.0f, 0.0f);
    if (parseVector3(_axisVec, vec)) {
        vptRenderer->getVptPass()->setEnvMapRotAngleAxis(vec, float(_angle));
    }
}

void setEnvMapRotQuaternion(std::vector<double> _quaternionVec) {
    auto quaternionRot = glm::identity<glm::quat>();
    if (parseQuaternion(_quaternionVec, quaternionRot)) {
        vptRenderer->getVptPass()->setEnvMapRotQuaternion(quaternionRot);
    }
}


void setScatteringAlbedo(std::vector<double> albedo) {
    glm::vec3 vec = glm::vec3(0.0f, 0.0f, 0.0f);
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
    glm::vec3 vec = glm::vec3(0.0f, 0.0f, 0.0f);
    if (parseVector3(extinctionBase, vec)) {
        //std::cout << "setExtinctionBase to " << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
        vptRenderer->setExtinctionBase(vec);
    }
}

void setVPTMode(int64_t mode) {
    std::cout << "setVPTMode to " << VPT_MODE_NAMES[mode] << std::endl;
    vptRenderer->setVptMode(VptMode(mode));
}

void setVPTModeFromName(const std::string& modeName) {
    for (int mode = 0; mode < IM_ARRAYSIZE(VPT_MODE_NAMES); mode++) {
        if (modeName == VPT_MODE_NAMES[mode]) {
            std::cout << "setVPTMode to " << VPT_MODE_NAMES[mode] << std::endl;
            vptRenderer->setVptMode(VptMode(mode));
            break;
        }
    }
}

void setDenoiser(const std::string& denoiserName) {
    int mode;
    for (mode = 0; mode < IM_ARRAYSIZE(DENOISER_NAMES); mode++) {
        if (denoiserName == DENOISER_NAMES[mode]) {
            std::cout << "setDenoiser to " << DENOISER_NAMES[mode] << std::endl;
            vptRenderer->setDenoiserType(DenoiserType(mode));
            break;
        }
    }
    if (mode == IM_ARRAYSIZE(DENOISER_NAMES)) {
        sgl::Logfile::get()->writeError("Error in setDenoiser: Invalid denoiser name \"" + denoiserName + "\".");
    }
}

void setDenoiserProperty(const std::string& key, const std::string& value) {
    vptRenderer->setDenoiserProperty(key, value);
}

void setPyTorchDenoiserModelFile(const std::string& denoiserModelFilePath) {
    vptRenderer->getVptPass()->setPyTorchDenoiserModelFilePath(denoiserModelFilePath);
}

void setOutputForegroundMap(bool _shallOutputForegroundMap) {
    vptRenderer->getVptPass()->setOutputForegroundMap(_shallOutputForegroundMap);
}

void setFeatureMapType(int64_t type) {
    //std::cout << "setFeatureMapType to " << type << std::endl;
    vptRenderer->setFeatureMapType(FeatureMapTypeVpt(type));
}

void setUseEmptySpaceSkipping(bool _useEmptySpaceSkipping) {
    vptRenderer->getVptPass()->setUseEmptySpaceSkipping(_useEmptySpaceSkipping);
}


void setUseLights(bool _useLights) {
    auto* lightEditorWidget = vptRenderer->getLightEditorWidget();
    lightEditorWidget->setShowWindow(_useLights);
    vptRenderer->getVptPass()->setReRender();
}

void clearLights() {
    auto* lightEditorWidget = vptRenderer->getLightEditorWidget();
    auto numLights = lightEditorWidget->getNumLights();
    for (size_t i = 0; i < numLights; i++) {
        lightEditorWidget->removeLight(lightEditorWidget->getNumLights() - 1);
    }
    vptRenderer->getVptPass()->setReRender();
}

void addLight() {
    auto* lightEditorWidget = vptRenderer->getLightEditorWidget();
    lightEditorWidget->addLight({});
    vptRenderer->getVptPass()->setReRender();
}

void removeLight(int64_t lightIdx) {
    auto* lightEditorWidget = vptRenderer->getLightEditorWidget();
    lightEditorWidget->removeLight(uint32_t(lightIdx));
    vptRenderer->getVptPass()->setReRender();
}

void setLightProperty(int64_t lightIdx, const std::string& key, const std::string& value) {
    auto* lightEditorWidget = vptRenderer->getLightEditorWidget();
    lightEditorWidget->setLightProperty(uint32_t(lightIdx), key, value);
    vptRenderer->getVptPass()->setReRender();
}

void loadLightsFromFile(const std::string& filePath) {
    auto* lightEditorWidget = vptRenderer->getLightEditorWidget();
    lightEditorWidget->loadFromFile(filePath);
}

void saveLightsToFile(const std::string& filePath) {
    auto* lightEditorWidget = vptRenderer->getLightEditorWidget();
    lightEditorWidget->saveToFile(filePath);
}

void setUseHeadlight(bool _useHeadlight) {
    //vptRenderer->getVptPass()->setUseHeadlight(_useHeadlight);
    setUseLights(_useHeadlight);
    if (_useHeadlight) {
        clearLights();
        addLight();
        setLightProperty(0, "space", "View");
        setLightProperty(0, "position", "0 0 0");
        setLightProperty(0, "use_distance", "1");
    }
}

void setUseHeadlightDistance(bool _useHeadlightDistance) {
    //vptRenderer->getVptPass()->setUseHeadlightDistance(_useHeadlightDistance);
    setLightProperty(0, "use_distance", _useHeadlightDistance ? "1" : "0");
}

void setHeadlightColor(std::vector<double> _headlightColor) {
    glm::vec3 color = glm::vec3(0.0f, 0.0f, 0.0f);
    if (parseVector3(_headlightColor, color)) {
        //vptRenderer->getVptPass()->setHeadlightColor(color);
        setLightProperty(
                0, "color", std::to_string(color.r) + " " + std::to_string(color.g) + " " + std::to_string(color.b));
    }
}

void setHeadlightIntensity(double _headlightIntensity) {
    //vptRenderer->getVptPass()->setHeadlightIntensity(float(_headlightIntensity));
    setLightProperty(0, "intensity", std::to_string(_headlightIntensity));
}


void setUseIsosurfaces(bool _useIsosurfaces) {
    vptRenderer->getVptPass()->setUseIsosurfaces(_useIsosurfaces);
}

void setIsoValue(double _isoValue) {
    vptRenderer->getVptPass()->setIsoValue(float(_isoValue));
}

void setIsoSurfaceColor(std::vector<double> _isoSurfaceColor) {
    glm::vec3 color = glm::vec3(0.0f, 0.0f, 0.0f);
    if (parseVector3(_isoSurfaceColor, color)) {
        vptRenderer->getVptPass()->setIsoSurfaceColor(color);
    }
}

void setIsosurfaceType(const std::string& _isosurfaceType) {
    for (int i = 0; i < IM_ARRAYSIZE(ISOSURFACE_TYPE_NAMES); i++) {
        if (_isosurfaceType == ISOSURFACE_TYPE_NAMES[i]) {
            vptRenderer->getVptPass()->setIsosurfaceType(IsosurfaceType(i));
            break;
        }
    }
}

void setSurfaceBrdf(const std::string& _surfaceBrdf) {
    for (int i = 0; i < IM_ARRAYSIZE(SURFACE_BRDF_NAMES); i++) {
        if (_surfaceBrdf == SURFACE_BRDF_NAMES[i]) {
            vptRenderer->getVptPass()->setSurfaceBrdf(SurfaceBrdf(i));
            break;
        }
    }
}

void setBrdfParameter(const std::string& key, const std::string& value) {
    auto* vptPass = vptRenderer->getVptPass();
    vptPass->setBrdfParameter(key, value);
}

void setUseIsosurfaceTf(bool _useIsosurfaceTf) {
    vptRenderer->getVptPass()->setUseIsosurfaceTf(_useIsosurfaceTf);
}

void setNumIsosurfaceSubdivisions(int64_t _subdivs) {
    vptRenderer->getVptPass()->setNumIsosurfaceSubdivisions(_subdivs);
}

void setCloseIsosurfaces(bool _closeIsosurfaces) {
    vptRenderer->getVptPass()->setCloseIsosurfaces(_closeIsosurfaces);
}

void setUseLegacyNormals(bool _useLegacyNormals) {
    vptRenderer->getVptPass()->setUseLegacyNormals(_useLegacyNormals);
}


void setUseClipPlane(bool _useClipPlane) {
    vptRenderer->getVptPass()->setUseClipPlane(_useClipPlane);
}

void setClipPlaneNormal(std::vector<double> _clipPlaneNormal) {
    glm::vec3 normal = glm::vec3(0,0,0);
    if (parseVector3(_clipPlaneNormal, normal)) {
        vptRenderer->getVptPass()->setClipPlaneNormal(normal);
    }
}

void setClipPlaneDistance(double _clipPlaneDistance) {
    vptRenderer->getVptPass()->setClipPlaneDistance(_clipPlaneDistance);
}


void setUseTransferFunction(bool _useTf) {
    auto* tfWindow = vptRenderer->getTransferFunctionWindow();
    tfWindow->setShowWindow(_useTf);
}

void loadTransferFunctionFile(const std::string& tfFilePath) {
    auto* tfWindow = vptRenderer->getTransferFunctionWindow();
    tfWindow->loadFunctionFromFile(0, tfFilePath);
}

void loadTransferFunctionFileGradient(const std::string& tfFilePath) {
    auto* tfWindow = vptRenderer->getTransferFunctionWindow();
    tfWindow->loadFunctionFromFile(1, tfFilePath);
}

void setTransferFunctionRange(double _minVal, double _maxVal) {
    auto* tfWindow = vptRenderer->getTransferFunctionWindow();
    tfWindow->setSelectedRange(0, glm::vec2(float(_minVal), float(_maxVal)));
    tfWindow->setIsSelectedRangeFixed(0, true);
}

void setTransferFunctionRangeGradient(double _minVal, double _maxVal) {
    auto* tfWindow = vptRenderer->getTransferFunctionWindow();
    tfWindow->setSelectedRange(1, glm::vec2(float(_minVal), float(_maxVal)));
    tfWindow->setIsSelectedRangeFixed(1, true);
}

static const char* emptyTfStringModule =
        "<TransferFunction>\n"
        "<OpacityPoints>\n"
        "<OpacityPoint position=\"0\" opacity=\"0\"/><OpacityPoint position=\"1\" opacity=\"0\"/>\n"
        "</OpacityPoints>\n"
        "<ColorPoints color_data=\"ushort\">\n"
        "<ColorPoint position=\"0\" r=\"0\" g=\"0\" b=\"0\"/><ColorPoint position=\"1\" r=\"0\" g=\"0\" b=\"0\"/>\n"
        "</ColorPoints>\n"
        "</TransferFunction>";

void setTransferFunctionEmpty() {
    auto* tfWindow = vptRenderer->getTransferFunctionWindow();
    tfWindow->loadFunctionFromXmlString(0, emptyTfStringModule);
}

void setTransferFunctionEmptyGradient() {
    auto* tfWindow = vptRenderer->getTransferFunctionWindow();
    tfWindow->loadFunctionFromXmlString(1, emptyTfStringModule);
}


std::vector<double> getCameraPosition() {
    auto pos = vptRenderer->getCameraPosition();
    return { pos.x, pos.y, pos.z };
}

std::vector<double> getCameraViewMatrix() {
    std::vector<double> viewMatrixData(16);
    auto viewMatrix = vptRenderer->getCamera()->getViewMatrix();
    for (int i = 0; i < 16; i++) {
        viewMatrixData[i] = viewMatrix[i / 4][i % 4];
    }
    return viewMatrixData;
}

double getCameraFOVy() {
    return vptRenderer->getCamera()->getFOVy();
}

void setCameraPosition(std::vector<double> cameraPosition) {
    glm::vec3 vec = glm::vec3(0.0f, 0.0f, 0.0f);
    if (parseVector3(cameraPosition, vec)) {
        //std::cout << "setCameraPosition to " << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
        vptRenderer->setCameraPosition(vec);
    }
}

void setCameraTarget(std::vector<double> cameraTarget) {
    glm::vec3 vec = glm::vec3(0.0f, 0.0f, 0.0f);
    if (parseVector3(cameraTarget, vec)) {
        //std::cout << "setCameraTarget to " << vec.x << ", " << vec.y << ", " << vec.z << std::endl;
        vptRenderer->setCameraTarget(vec);
    }
}

void overwriteCameraViewMatrix(std::vector<double> viewMatrixData) {
    glm::mat4 viewMatrix;
    for (int i = 0; i < 16; i++) {
        viewMatrix[i / 4][i % 4] = viewMatrixData[i];
    }
    vptRenderer->getCamera()->overwriteViewMatrix(viewMatrix);
    vptRenderer->getVptPass()->onHasMoved();
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

void setViewProjectionMatrixAsPrevious(){
    vptRenderer->setViewProjectionMatrixAsPrevious();
}

void setEmissionCap(double emissionCap){
    vptRenderer->setEmissionCap(emissionCap);
}
void setEmissionStrength(double emissionStrength){
    vptRenderer->setEmissionStrength(emissionStrength);
}
void setUseEmission(bool useEmission){
    vptRenderer->setUseEmission(useEmission);
}
void setTfScatteringAlbedoStrength(double strength) {
    vptRenderer->getVptPass()->setTfScatteringAlbedoStrength(strength);
}
void flipYZ(bool flip){
    vptRenderer->flipYZ(flip);
}


std::vector<int64_t> getVolumeVoxelSize() {
    const auto& cloudData = vptRenderer->getCloudData();
    return { int64_t(cloudData->getGridSizeZ()), int64_t(cloudData->getGridSizeY()), int64_t(cloudData->getGridSizeX()) };
}

std::vector<double> getRenderBoundingBox() {
    const auto& aabb = vptRenderer->getCloudData()->getWorldSpaceBoundingBox(
            vptRenderer->getVptPass()->getUseSparseGrid());
    return { aabb.min.x, aabb.max.x, aabb.min.y, aabb.max.y, aabb.min.z, aabb.max.z };
}

void rememberNextBounds(){
    vptRenderer->rememberNextBounds();
}

void forgetCurrentBounds(){
    vptRenderer->forgetCurrentBounds();
}

void setMaxGridExtent(double maxGridExtent) {
    vptRenderer->getCloudData()->setMaxGridExtent(maxGridExtent);
}

void setGlobalWorldBoundingBox(std::vector<double> globalBBVec) {
    sgl::AABB3 aabb(
            glm::vec3((float)globalBBVec[0], (float)globalBBVec[2], (float)globalBBVec[4]),
            glm::vec3((float)globalBBVec[1], (float)globalBBVec[3], (float)globalBBVec[5]));
    vptRenderer->setGlobalWorldBoundingBox(aabb);
}

std::vector<double> getVDBWorldBoundingBox() {
    return vptRenderer->getCloudData()->getVDBWorldBoundingBox();
}

std::vector<int64_t> getVDBIndexBoundingBox() {
    return vptRenderer->getCloudData()->getVDBIndexBoundingBox();
}

std::vector<double> getVDBVoxelSize() {
    return vptRenderer->getCloudData()->getVDBVoxelSize();
}

std::vector<torch::Tensor> triangulateIsosurfaces() {
    // Test case.
    /*std::vector<glm::vec3> vertexPositions = {
            {  1.0f,  1.0f,  1.0f },
            { -1.0f,  1.0f, -1.0f },
            { -1.0f, -1.0f,  1.0f },
            {  1.0f, -1.0f, -1.0f },
    };
    std::vector<glm::vec4> vertexColors = {
            {  1.0f,  0.0f,  0.0f, 1.0f },
            {  0.0f,  1.0f,  0.0f, 1.0f },
            {  0.0f,  0.0f,  1.0f, 1.0f },
            {  1.0f,  1.0f,  0.0f, 1.0f },
    };
    std::vector<glm::vec3> vertexNormals = {
            {  1.0f,  1.0f,  1.0f },
            { -1.0f,  1.0f, -1.0f },
            { -1.0f, -1.0f,  1.0f },
            {  1.0f, -1.0f, -1.0f },
    };
    std::vector<uint32_t> triangleIndices = {
            0, 1, 2,
            0, 2, 3,
            0, 3, 1,
            2, 1, 3
    };*/
    std::vector<uint32_t> triangleIndices;
    std::vector<glm::vec3> vertexPositions;
    std::vector<glm::vec4> vertexColors;
    std::vector<glm::vec3> vertexNormals;
    IsosurfaceSettings settings;
    settings.isoValue = vptRenderer->getVptPass()->getIsoValue();
    settings.isosurfaceType = vptRenderer->getVptPass()->getIsosurfaceType();
    settings.useIsosurfaceTf = vptRenderer->getVptPass()->getUseIsosurfaceTf();
    auto col = vptRenderer->getVptPass()->getIsosurfaceColor();
    settings.isosurfaceColor = glm::vec4(col.x, col.y, col.z, 1.0f);
    vptRenderer->getCloudData()->createIsoSurfaceData(
            settings, triangleIndices, vertexPositions, vertexColors, vertexNormals);

    torch::Tensor triangleIndicesTensor = torch::from_blob(
            triangleIndices.data(), { int64_t(triangleIndices.size()) },
            torch::TensorOptions().dtype(torch::kInt32)).detach().clone();
    torch::Tensor vertexPositionsTensor = torch::from_blob(
            vertexPositions.data(), { int64_t(vertexPositions.size()), 3 },
            torch::TensorOptions().dtype(torch::kFloat32)).detach().clone();
    torch::Tensor vertexColorsTensor = torch::from_blob(
            vertexColors.data(), { int64_t(vertexPositions.size()), 4 },
            torch::TensorOptions().dtype(torch::kFloat32)).detach().clone();
    torch::Tensor vertexNormalsTensor = torch::from_blob(
            vertexNormals.data(), { int64_t(vertexPositions.size()), 3 },
            torch::TensorOptions().dtype(torch::kFloat32)).detach().clone();
    return { triangleIndicesTensor, vertexPositionsTensor, vertexColorsTensor, vertexNormalsTensor };
}

void exportVdbVolume(const std::string& filename) {
#ifdef USE_OPENVDB
    vptRenderer->getCloudData()->saveToVdbFile(filename);
#else
    sgl::Logfile::get()->throwError("Error in exportVdbVolume: Application was not compiled with NanoVDB support.");
#endif
}
