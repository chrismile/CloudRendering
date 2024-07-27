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

#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>

#include <openvdb/openvdb.h>
#include "nanovdb/NanoVDB.h"
#include "nanovdb/util/IO.h"
#include "nanovdb/util/OpenToNanoVDB.h"
#include "nanovdb/util/NanoToOpenVDB.h"

#include "CloudData.hpp"

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
    if (!sgl::FileUtils::get()->exists(filenameNvdb)) {
        nanovdb::io::writeGrid(filenameNvdb, sparseGridHandle);
    }

    computeSparseGridMetadata();
    return !sparseGridHandle.empty();
}

bool CloudData::saveToVdbFile(const std::string& filename) {
    if (!hasSparseData()) {
        // TODO: In this case, we should rather create the OpenVDB data directly from the dense grid.
        uint8_t* data = nullptr;
        uint64_t size = 0;
        getSparseDensityField(data, size);
    }
    auto openvdbGrid = nanovdb::nanoToOpenVDB(sparseGridHandle);
    std::vector<openvdb::GridBase::Ptr> grids = { openvdbGrid };
    openvdb::io::File file(filename);
    file.write(grids);
    file.close();
    return true;
}
