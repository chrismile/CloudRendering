# BSD 2-Clause License
#
# Copyright (c) 2024, Christoph Neuhauser
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import annotations
import torch
import difftetvr
import typing
from typing import List
import enum

__all__ = [
    'initialize', 'cleanup', 'load_cloud_file', 'load_volume_file', 'load_emission_file', 'load_environment_map',
    'set_use_builtin_environment_map', 'set_environment_map_intensity',
    'set_environment_map_intensity_rgb', 'disable_env_map_rot', 'set_env_map_rot_camera',
    'set_env_map_rot_euler_angles', 'set_env_map_rot_yaw_pitch_roll', 'set_env_map_rot_angle_axis',
    'set_env_map_rot_quaternion', 'set_scattering_albedo', 'set_extinction_scale', 'set_extinction_base', 'set_phase_g',
    'set_use_transfer_function', 'load_transfer_function_file', 'load_transfer_function_file_gradient',
    'set_transfer_function_range', 'set_transfer_function_range_gradient', 'set_transfer_function_empty',
    'set_transfer_function_empty_gradient', 'get_camera_position', 'get_camera_view_matrix', 'get_camera_fovy',
    'set_camera_position', 'set_camera_target', 'overwrite_camera_view_matrix', 'set_camera_fovy', 'set_vpt_mode',
    'set_vpt_mode_from_name', 'set_denoiser', 'set_denoiser_property', 'set_pytorch_denoiser_model_file',
    'set_output_foreground_map', 'set_feature_map_type', 'set_use_empty_space_skipping', 'set_use_lights',
    'clear_lights', 'add_light', 'remove_light', 'set_light_property', 'load_lights_from_file', 'save_lights_to_file',
    'set_use_headlight', 'set_use_headlight_distance', 'set_headlight_color', 'set_headlight_intensity',
    'set_use_isosurfaces', 'set_iso_value', 'set_iso_surface_color', 'set_isosurface_type', 'set_surface_brdf',
    'set_brdf_parameter', 'set_use_isosurface_tf', 'set_num_isosurface_subdivisions', 'set_close_isosurfaces',
    'set_use_legacy_normals', 'set_use_clip_plane', 'set_clip_plane_normal', 'set_clip_plane_distance',
    'set_seed_offset', 'set_view_projection_matrix_as_previous', 'set_emission_cap', 'set_emission_strength',
    'set_use_emission', 'set_tf_scattering_albedo_strength', 'flip_yz_coordinates', 'get_volume_voxel_size',
    'get_render_bounding_box', 'remember_next_bounds', 'forget_current_bounds', 'set_max_grid_extent',
    'set_global_world_bounding_box', 'get_vdb_world_bounding_box', 'get_vdb_index_bounding_box', 'get_vdb_voxel_size',
    'render_frame', 'set_use_feature_maps', 'get_feature_map_from_string', 'get_feature_map',
    'get_transmittance_volume', 'set_secondary_volume_downscaling_factor', 'compute_occupation_volume',
    'update_observation_frequency_fields', 'compute_energy', 'triangulate_isosurfaces', 'export_vdb_volume'
]


def initialize() -> None:
    pass
def cleanup() -> None:
    pass

def load_cloud_file(filename: str) -> None:
    pass
def load_volume_file(filename: str) -> None:
    pass
def load_emission_file(filename: str) -> None:
    pass

def load_environment_map(filename: str) -> None:
    pass
def set_use_builtin_environment_map(env_map_name: str) -> None:
    pass
def set_environment_map_intensity(intensity_factor: float) -> None:
    pass
def set_environment_map_intensity_rgb(intensity_factor: List[float]) -> None:
    pass

def disable_env_map_rot() -> None:
    pass
def set_env_map_rot_camera() -> None:
    pass
def set_env_map_rot_euler_angles(euler_angles_vec: List[float]) -> None:
    pass
def set_env_map_rot_yaw_pitch_roll(yaw_pitch_roll_vec: List[float]) -> None:
    pass
def set_env_map_rot_angle_axis(_axis_vec: List[float], _angle: float) -> None:
    pass
def set_env_map_rot_quaternion(_quaternion_vec: List[float]) -> None:
    pass

def set_scattering_albedo(albedo: List[float]) -> None:
    pass
def set_extinction_scale(extinction_scale: float) -> None:
    pass
def set_extinction_base(extinction_base: List[float]) -> None:
    pass
def set_phase_g(phase_g: float) -> None:
    pass

def set_use_transfer_function(_use_tf: bool) -> None:
    pass
def load_transfer_function_file(tf_file_path: str) -> None:
    pass
def load_transfer_function_file_gradient(tf_file_path: str) -> None:
    pass
def set_transfer_function_range(_min_val: float, _max_val: float) -> None:
    pass
def set_transfer_function_range_gradient(_min_val: float, _max_val: float) -> None:
    pass
def set_transfer_function_empty() -> None:
    pass
def set_transfer_function_empty_gradient() -> None:
    pass

def get_camera_position() -> List[float]:
    pass
def get_camera_view_matrix() -> List[float]:
    pass
def get_camera_fovy() -> float:
    pass

def set_camera_position(camera_position: List[float]) -> None:
    pass
def set_camera_target(camera_target: List[float]) -> None:
    pass
def overwrite_camera_view_matrix(view_matrix_data: List[float]) -> None:
    pass
def set_camera_fovy(fovy: float) -> None:
    pass

def set_vpt_mode(mode: int) -> None:
    pass
def set_vpt_mode_from_name(mode_name: str) -> None:
    pass
def set_denoiser(denoiser_name: str) -> None:
    pass
def set_denoiser_property(key: str, value: str) -> None:
    pass
def set_pytorch_denoiser_model_file(denoiser_model_file_path: str) -> None:
    pass
def set_output_foreground_map(_shall_output_foreground_map: bool) -> None:
    pass
def set_feature_map_type(type: int) -> None:
    pass

def set_use_empty_space_skipping(_use_empty_space_skipping: bool) -> None:
    pass

def set_use_lights(_use_lights: bool) -> None:
    pass
def clear_lights() -> None:
    pass
def add_light() -> None:
    pass
def remove_light(light_idx: int) -> None:
    pass
def set_light_property(light_idx: int, key: str, value: str) -> None:
    pass
def load_lights_from_file(file_path: str) -> None:
    pass
def save_lights_to_file(file_path: str) -> None:
    pass
# Old light API.
def set_use_headlight(_use_headlight: bool) -> None:
    pass
def set_use_headlight_distance(_use_headlight_distance: bool) -> None:
    pass
def set_headlight_color(_headlight_color: List[float]) -> None:
    pass
def set_headlight_intensity(_headlight_intensity: float) -> None:
    pass

def set_use_isosurfaces(_use_isosurfaces: bool) -> None:
    pass
def set_iso_value(_iso_value: float) -> None:
    pass
def set_iso_surface_color(_iso_surface_color: List[float]) -> None:
    pass
def set_isosurface_type(_isosurface_type: str) -> None:
    pass
def set_surface_brdf(_surface_brdf: str) -> None:
    pass
def set_brdf_parameter(key: str, value: str) -> None:
    pass
def set_use_isosurface_tf(_use_isosurface_tf: bool) -> None:
    pass
def set_num_isosurface_subdivisions(_subdivs: int) -> None:
    pass
def set_close_isosurfaces(_close_isosurfaces: bool) -> None:
    pass
def set_use_legacy_normals(_use_legacy_normals: bool) -> None:
    pass

def set_use_clip_plane(_use_clip_plane: bool) -> None:
    pass
def set_clip_plane_normal(_clip_plane_normal: List[float]) -> None:
    pass
def set_clip_plane_distance(_clip_plane_distance: float) -> None:
    pass

def set_seed_offset(offset: int) -> None:
    pass

def set_view_projection_matrix_as_previous() -> None:
    pass

def set_emission_cap(emission_cap: float) -> None:
    pass
def set_emission_strength(emission_strength: float) -> None:
    pass
def set_use_emission(use_emission: bool) -> None:
    pass
def set_tf_scattering_albedo_strength(strength: float) -> None:
    pass
def flip_yz_coordinates(flip: bool) -> None:
    pass

def get_volume_voxel_size() -> List[int]:
    pass
def get_render_bounding_box() -> List[float]:
    pass
def remember_next_bounds() -> None:
    pass
def forget_current_bounds() -> None:
    pass
def set_max_grid_extent(max_grid_extent: float) -> None:
    """
    Due to legacy reasons, the grid has size (-0.25, 0.25) in the largest dimension.
    @param max_grid_extent The new extent value such that the size is (-maxGridExtent, maxGridExtent).
    Note: Must be called before any call to @see loadCloudFile!
    """
    pass
def set_global_world_bounding_box(global_bb_vec: List[float]) -> None:
    """ This function can be used to normalize the grid wrt. a global bounding box."""
    pass

# Interface for NanoVDB & OpenVDB.
def get_vdb_world_bounding_box() -> List[float]:
    pass
def get_vdb_index_bounding_box() -> List[int]:
    pass
def get_vdb_voxel_size() -> List[float]:
    pass

def render_frame(input_tensor: torch.Tensor, frame_count: int) -> torch.Tensor:
    pass
def set_use_feature_maps(feature_map_names: List[str]) -> None:
    pass
def get_feature_map_from_string(
        input_tensor: torch.Tensor,
        feature_map: str
) -> torch.Tensor:
    pass
def get_feature_map(input_tensor: torch.Tensor, feature_map: int) -> torch.Tensor:
    pass
def get_transmittance_volume(input_tensor: torch.Tensor) -> torch.Tensor:
    pass
def set_secondary_volume_downscaling_factor(ds_factor: int) -> None:
    pass
def compute_occupation_volume(
        input_tensor: torch.Tensor,
        ds_factor: int,
        max_kernel_radius: int
) -> torch.Tensor:
    pass
def update_observation_frequency_fields(
        num_bins_x: int,
        num_bins_y: int,
        transmittance_field: torch.Tensor,
        obs_freq_field: torch.Tensor,
        angular_obs_freq_field: torch.Tensor
) -> None:
    pass
def compute_energy(
        num_cams: int,
        num_bins_x: int,
        num_bins_y: int,
        gamma: float,
        obs_freq_field: torch.Tensor,
        angular_obs_freq_field: torch.Tensor,
        occupancy_field: torch.Tensor,
        energy_term_field: torch.Tensor
) -> None:
    pass


# API for exporting volume and surface data for external use.
def triangulate_isosurfaces() -> List[torch.Tensor]:
    pass
def export_vdb_volume(filename: str) -> None:
    pass
