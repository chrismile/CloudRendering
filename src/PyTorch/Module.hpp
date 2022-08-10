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

#ifndef CLOUDRENDERING_MODULE_HPP
#define CLOUDRENDERING_MODULE_HPP

#include <torch/script.h>
#include <torch/types.h>

MODULE_OP_API void initialize();
MODULE_OP_API void cleanup();

MODULE_OP_API void loadCloudFile(const std::string& filename);
MODULE_OP_API void loadEnvironmentMap(const std::string& filename);
MODULE_OP_API void setEnvironmentMapIntensityFactor(double intensityFactor);

MODULE_OP_API void setScatteringAlbedo(std::vector<double> albedo);
MODULE_OP_API void setExtinctionScale(double extinctionScale);
MODULE_OP_API void setExtinctionBase(std::vector<double> extinctionBase);
MODULE_OP_API void setPhaseG(double phaseG);

MODULE_OP_API void setCameraPosition(std::vector<double> cameraPosition);
MODULE_OP_API void setCameraTarget(std::vector<double> cameraTarget);
MODULE_OP_API void setCameraFOVy(double FOVy);

MODULE_OP_API void setVPTMode(int64_t mode);
MODULE_OP_API void setFeatureMapType(int64_t type);

MODULE_OP_API void setSeedOffset(int64_t offset);

MODULE_OP_API torch::Tensor renderFrame(torch::Tensor inputTensor, int64_t frameCount);
MODULE_OP_API torch::Tensor getFeatureMap(torch::Tensor inputTensor, int64_t frameCount);

class VolumetricPathTracingModuleRenderer;
extern VolumetricPathTracingModuleRenderer* vptRenderer;

torch::Tensor renderFrameCpu(torch::Tensor inputTensor, int64_t frameCount);
torch::Tensor renderFrameVulkan(torch::Tensor inputTensor, int64_t frameCount);
#ifdef SUPPORT_CUDA_INTEROP
torch::Tensor renderFrameCuda(torch::Tensor inputTensor, int64_t frameCount);
#endif

#endif //CLOUDRENDERING_MODULE_HPP
