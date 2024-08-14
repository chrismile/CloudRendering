/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2024, Christoph Neuhauser
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

#include <Math/Geometry/MatrixUtil.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>

#include <openvdb/openvdb.h>
#include "nanovdb/NanoVDB.h"
#include "nanovdb/util/IO.h"
#include "nanovdb/util/OpenToNanoVDB.h"
#include "nanovdb/util/NanoToOpenVDB.h"

#include "CloudData.hpp"

#define IDXS(x,y,z) ((z)*xs*ys + (y)*xs + (x))

bool CloudData::loadFromVdbFile(const std::string& filename) {
    openvdb::io::File file(filename);
    file.open();
    openvdb::GridBase::Ptr baseGrid;
    bool foundGrid = false;
    for (openvdb::io::File::NameIterator nameIter = file.beginName(); nameIter != file.endName(); ++nameIter) {
        baseGrid = file.readGrid(nameIter.gridName());
        foundGrid = true;
        break;
    }
    file.close();

    if (!foundGrid) {
        sgl::Logfile::get()->writeError("Error in CloudData::loadFromVdbFile: File \"" + filename + "\" is empty.");
        return false;
    }
    openvdb::FloatGrid::Ptr srcGrid = openvdb::gridPtrCast<openvdb::FloatGrid>(baseGrid);

    //sparseGridHandle = nanovdb::createNanoGrid(*srcGrid);
    sparseGridHandle = nanovdb::openToNanoVDB(*srcGrid);

    std::string filenameNvdb = sgl::FileUtils::get()->removeExtension(filename) + ".nvdb";
    if (cacheSparseGrid && !sgl::FileUtils::get()->exists(filenameNvdb)) {
        nanovdb::io::writeGrid(filenameNvdb, sparseGridHandle);
    }

    computeSparseGridMetadata();
    return !sparseGridHandle.empty();
}

bool CloudData::saveToVdbFile(const std::string& filename) {
    openvdb::GridPtrVec grids;

    // For now, prefer dense data if memory-wise possible.
    if (!hasDenseData() && size_t(gridSizeX) * size_t(gridSizeY) * size_t(gridSizeZ) < size_t(1024ull * 1024ull * 1024ull)) {
        getDenseDensityField();
    }

    if (hasDenseData()) {
        //uint8_t* data = nullptr;
        //uint64_t size = 0;
        //getSparseDensityField(data, size);

        openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
        grid->setGridClass(openvdb::GRID_FOG_VOLUME);
        auto accessor = grid->getAccessor();
        openvdb::Coord ijk;
        auto xs = size_t(gridSizeX);
        auto ys = size_t(gridSizeY);
        //auto zs = size_t(gridSizeZ);
        for (size_t z = 0; z < gridSizeZ; z++) {
            ijk[2] = int(z);
            for (size_t y = 0; y < gridSizeY; y++) {
                ijk[1] = int(y);
                for (size_t x = 0; x < gridSizeX; x++) {
                    ijk[0] = int(x);
                    float value = densityField->getDataFloatAt(IDXS(x, y, z));
                    if (value != 0.0f) {
                        accessor.setValue(ijk, value);
                    }
                }
            }
        }

        glm::mat4 S = sgl::matrixScaling((boxMaxDense - boxMinDense) / glm::vec3(gridSizeX, gridSizeY, gridSizeZ));
        glm::mat4 T = sgl::matrixTranslation(boxMinDense);
        glm::mat4 trafo = T * S;
        openvdb::Mat4R openvdbTransform = openvdb::Mat4R(
                trafo[0][0], trafo[0][1], trafo[0][2], trafo[0][3],
                trafo[1][0], trafo[1][1], trafo[1][2], trafo[1][3],
                trafo[2][0], trafo[2][1], trafo[2][2], trafo[2][3],
                trafo[3][0], trafo[3][1], trafo[3][2], trafo[3][3]);
        grid->setTransform(
                openvdb::math::Transform::createLinearTransform(openvdbTransform));

        grid->setName("density");
        grids.push_back(grid);
    } else {
        auto grid = nanovdb::nanoToOpenVDB(sparseGridHandle);
        grids.push_back(grid);
    }
    openvdb::io::File file(filename);
    file.write(grids);
    file.close();
    return true;
}
