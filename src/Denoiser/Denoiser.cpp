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

#include <Utils/File/Logfile.hpp>
#include "EAWDenoiser.hpp"
#include "SVGF.hpp"
#ifdef SUPPORT_PYTORCH_DENOISER
#include "PyTorchDenoiser.hpp"
#endif
#ifdef SUPPORT_OPTIX
#include "OptixVptDenoiser.hpp"
#endif
#ifdef SUPPORT_OPEN_IMAGE_DENOISE
#include "OpenImageDenoiseDenoiser.hpp"
#endif
#ifdef SUPPORT_DLSS
#include "DLSSDenoiser.hpp"
#endif
#include "Denoiser.hpp"

std::shared_ptr<Denoiser> createDenoiserObject(
        DenoiserType denoiserType, sgl::vk::Renderer* renderer, DenoisingMode mode, bool denoiseAlpha) {
    std::shared_ptr<Denoiser> denoiser;
    if (denoiserType == DenoiserType::NONE) {
        denoiser = {};
    } else if (denoiserType == DenoiserType::EAW) {
        denoiser = std::shared_ptr<Denoiser>(new EAWDenoiser(renderer));
        if (mode == DenoisingMode::AMBIENT_OCCLUSION) {
            static_cast<EAWDenoiser*>(denoiser.get())->setNumIterations(4);
            static_cast<EAWDenoiser*>(denoiser.get())->setPhiColor(5.0f);
            static_cast<EAWDenoiser*>(denoiser.get())->setPhiPosition(1.0f);
            static_cast<EAWDenoiser*>(denoiser.get())->setPhiNormal(1.0f);
            static_cast<EAWDenoiser*>(denoiser.get())->setWeightScaleColor(1.0f);
            static_cast<EAWDenoiser*>(denoiser.get())->setWeightScalePosition(0.0001f);
            static_cast<EAWDenoiser*>(denoiser.get())->setWeightScaleNormal(1.0f);
        }
    }
#ifdef SUPPORT_PYTORCH_DENOISER
    else if (denoiserType == DenoiserType::PYTORCH_DENOISER) {
        denoiser = std::shared_ptr<Denoiser>(new PyTorchDenoiser(renderer));
    }
#endif
#ifdef SUPPORT_OPTIX
    else if (denoiserType == DenoiserType::OPTIX && OptixVptDenoiser::isOptixEnabled()) {
        denoiser = std::shared_ptr<Denoiser>(new OptixVptDenoiser(renderer, denoiseAlpha));
        if (mode == DenoisingMode::VOLUMETRIC_PATH_TRACING) {
            denoiser->setTemporalDenoisingEnabled(false);
        }
    }
#endif
#ifdef SUPPORT_OPEN_IMAGE_DENOISE
    else if (denoiserType == DenoiserType::OPEN_IMAGE_DENOISE) {
        denoiser = std::shared_ptr<Denoiser>(new OpenImageDenoiseDenoiser(renderer, denoiseAlpha));
    }
#endif
#ifdef SUPPORT_DLSS
    else if (denoiserType == DenoiserType::DLSS_DENOISER && getIsDlssSupportedByDevice(renderer->getDevice())) {
        denoiser = std::shared_ptr<Denoiser>(new DLSSDenoiser(renderer, denoiseAlpha));
    }
#endif
    else if (denoiserType == DenoiserType::SVGF) {
        denoiser = std::shared_ptr<Denoiser>(new SVGFDenoiser(renderer));
    } else {
        denoiser = {};
        sgl::Logfile::get()->writeError(
                "Error in VolumetricPathTracingPass::createDenoiser: Invalid denoiser type selected.");
    }
    return denoiser;
}
