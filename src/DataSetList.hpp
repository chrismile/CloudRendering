/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021, Christoph Neuhauser
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

#ifndef LINEDENSITYCONTROL_DATASETLIST_HPP
#define LINEDENSITYCONTROL_DATASETLIST_HPP

#include <vector>

#include <Math/Geometry/MatrixUtil.hpp>

enum DataSetType {
    DATA_SET_TYPE_NONE,
    DATA_SET_TYPE_NODE, //< Hierarchical container.
    DATA_SET_TYPE_VOLUME //< Grid storing volumetric data.
};

struct DataSetInformation;
typedef std::shared_ptr<DataSetInformation> DataSetInformationPtr;

struct DataSetInformation {
    DataSetType type = DATA_SET_TYPE_VOLUME;
    std::string name;
    std::string filename;
    std::string emission;

    // For type DATA_SET_TYPE_NODE.
    std::vector<DataSetInformationPtr> children;
    int sequentialIndex = 0;

    // Can be used for transposing axes.
    glm::ivec3 axes = { 0, 1, 2 };

    // Optional attributes.
    bool hasCustomTransform = false;
    glm::mat4 transformMatrix = sgl::matrixIdentity();
};

DataSetInformationPtr loadDataSetList(const std::string& filename);

#endif //LINEDENSITYCONTROL_DATASETLIST_HPP
