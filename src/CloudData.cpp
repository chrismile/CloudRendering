/*
 * BSD 2-Clause License
 *
 * Copyright (c) 2021-2024, Christoph Neuhauser
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

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

#include <Math/half/half.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/Convert.hpp>
#include <Utils/StringUtils.hpp>
#include <Utils/File/Logfile.hpp>
#include <Utils/File/FileUtils.hpp>
#include <Utils/File/FileLoader.hpp>
#include <Utils/Events/Stream/Stream.hpp>
#include <Utils/Parallel/Reduction.hpp>

#include "nanovdb/NanoVDB.h"
#include "nanovdb/util/GridBuilder.h"
#include "nanovdb/util/IO.h"

#include "Utils/nifti/nifti1.h"

#include "CloudData.hpp"

CloudData::CloudData(sgl::MultiVarTransferFunctionWindow* transferFunctionWindow, LightEditorWidget* lightEditorWidget)
        : transferFunctionWindow(transferFunctionWindow), lightEditorWidget(lightEditorWidget) {
}

CloudData::~CloudData() {
    if (densityField) {
        densityField = {};
    }
    sparseGridHandle = {};
}

void CloudData::computeGridBounds() {
    float sx = float(gridSizeX) * voxelSizeX;
    float sy = float(gridSizeY) * voxelSizeY;
    float sz = float(gridSizeZ) * voxelSizeZ;
    float maxSize = std::max(sx, std::max(sy, sz));
    boxMaxDense = glm::vec3(sx, sy, sz) * maxGridExtent / maxSize;
    boxMinDense = -boxMaxDense;
    //uint32_t maxDim = std::max(gridSizeX, std::max(gridSizeY, gridSizeZ));
    //boxMax = glm::vec3(gridSizeX, gridSizeY, gridSizeZ) * maxGridExtent / float(maxDim);
    //boxMin = -boxMax;

    gridMinDense = glm::vec3(0,0,0);
    gridMaxDense = glm::vec3(1,1,1);
    gridMinSparse = glm::vec3(0,0,0);
    gridMaxSparse = glm::vec3(1,1,1);
}

template<class T>
static void transposeScalarField(T* fieldEntryBuffer, uint32_t xs, uint32_t ys, uint32_t zs) {
    auto* scalarFieldCopy = new T[xs * ys * zs];
    if constexpr(std::is_same<T, HalfFloat>()) {
        size_t bufferSize = xs * ys * zs;
        for (size_t i = 0; i < bufferSize; i++) {
            scalarFieldCopy[i] = fieldEntryBuffer[i];
        }
    } else {
        memcpy(scalarFieldCopy, fieldEntryBuffer, sizeof(T) * xs * ys * zs);
    }
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, zs), [&](auto const& r) {
        for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
#pragma omp parallel for shared(xs, ys, zs, fieldEntryBuffer, scalarFieldCopy) default(none)
#endif
    for (uint32_t z = 0; z < zs; z++) {
#endif
        for (uint32_t y = 0; y < ys; y++) {
            for (uint32_t x = 0; x < xs; x++) {
                uint32_t readPos = ((y)*xs*zs + (z)*xs + (x));
                uint32_t writePos = ((z)*xs*ys + (y)*xs + (x));
                fieldEntryBuffer[writePos] = scalarFieldCopy[readPos];
            }
        }
    }
#ifdef USE_TBB
    });
#endif
    delete[] scalarFieldCopy;
}

void CloudData::setTransposeAxes(const glm::ivec3& axes) {
    this->transposeAxes = axes;
    transpose = true;
}

void CloudData::transposeIfNecessary() {
    if (transpose) {
        uint32_t dimensions[3] = { gridSizeX, gridSizeY, gridSizeZ };
        float spacing[3] = { voxelSizeX, voxelSizeY, voxelSizeZ };
        gridSizeX = dimensions[transposeAxes[0]];
        gridSizeY = dimensions[transposeAxes[1]];
        gridSizeZ = dimensions[transposeAxes[2]];
        voxelSizeX = spacing[transposeAxes[0]];
        voxelSizeY = spacing[transposeAxes[1]];
        voxelSizeZ = spacing[transposeAxes[2]];

        if (transposeAxes != glm::ivec3(0, 2, 1)) {
            sgl::Logfile::get()->throwError(
                    "Error in CloudData::transposeIfNecessary: At the moment, only transposing the "
                    "Y and Z axis is supported.");
        }
        if (densityField->getScalarDataFormatNative() == ScalarDataFormat::FLOAT) {
            transposeScalarField(densityField->dataFloat, gridSizeX, gridSizeY, gridSizeZ);
        } else if (densityField->getScalarDataFormatNative() == ScalarDataFormat::BYTE) {
            transposeScalarField(densityField->dataByte, gridSizeX, gridSizeY, gridSizeZ);
        } else if (densityField->getScalarDataFormatNative() == ScalarDataFormat::SHORT) {
            transposeScalarField(densityField->dataShort, gridSizeX, gridSizeY, gridSizeZ);
        } else if (densityField->getScalarDataFormatNative() == ScalarDataFormat::FLOAT16) {
            transposeScalarField(densityField->dataFloat16, gridSizeX, gridSizeY, gridSizeZ);
        }
    }
}

void CloudData::setDensityField(uint32_t _gridSizeX, uint32_t _gridSizeY, uint32_t _gridSizeZ, float* _densityField) {
    if (densityField) {
        densityField = {};
    }
    sparseGridHandle = {};

    gridSizeX = _gridSizeX;
    gridSizeY = _gridSizeY;
    gridSizeZ = _gridSizeZ;
    voxelSizeX = 1.0f;
    voxelSizeY = 1.0f;
    voxelSizeZ = 1.0f;

    auto numEntries = size_t(_gridSizeX) * size_t(_gridSizeZ) * size_t(_gridSizeZ);
    densityField = std::make_shared<DensityField>(numEntries, _densityField);
    gridFilename = sgl::AppSettings::get()->getDataDirectory() + "CloudDataSets/clouds/tmp.xyz";
    gridName = "tmp";

    transposeIfNecessary();
    computeGridBounds();
}

void CloudData::setNanoVdbGridHandle(nanovdb::GridHandle<nanovdb::HostBuffer>&& handle) {
    sparseGridHandle = std::move(handle);
    computeSparseGridMetadata();
}

bool CloudData::loadFromFile(const std::string& filename) {
    if (!sgl::FileUtils::get()->exists(filename)) {
        sgl::Logfile::get()->writeError(
                "Error in CloudData::loadFromFile: The file \"" + filename + "\" does not exist!");
        return false;
    }

    if (sgl::FileUtils::get()->isDirectory(filename)){
        std::shared_ptr<CloudData> prev = nullptr;

        std::list<std::string> files = sgl::FileUtils::get()->getFilesInDirectoryList(filename);
        files.sort();
        for (std::string file: files) {
            std::shared_ptr<CloudData> current(new CloudData);
            current->loadFromFile(file);

            if (prev == nullptr){
                this->loadFromFile(file);
                this->setNextCloudDataFrame(current);
            }else{
                prev->setNextCloudDataFrame(current);
            }
            prev = current;

        }
        prev->setNextCloudDataFrame(this->getNextCloudDataFrame());
        return true;
    }

    gridFilename = filename;
    gridName = sgl::toLowerCopy(sgl::FileUtils::get()->removeExtension(
            sgl::FileUtils::get()->getPureFilename(gridFilename)));

    if (densityField) {
        densityField = {};
    }
    sparseGridHandle = {};

    if (sgl::FileUtils::get()->hasExtension(filename.c_str(), ".xyz")) {
        return loadFromXyzFile(filename);
    } else if (sgl::FileUtils::get()->hasExtension(filename.c_str(), ".nvdb")) {
        return loadFromNvdbFile(filename);
#ifdef USE_OPENVDB
    } else if (sgl::FileUtils::get()->hasExtension(filename.c_str(), ".vdb")) {
        return loadFromVdbFile(filename);
#endif
    } else if (sgl::FileUtils::get()->hasExtension(filename.c_str(), ".dat")
            || sgl::FileUtils::get()->hasExtension(filename.c_str(), ".raw")) {
        return loadFromDatRawFile(filename);
    } else if (sgl::FileUtils::get()->hasExtension(filename.c_str(), ".mhd")) {
        return loadFromMhdRawFile(filename);
    } else if (sgl::FileUtils::get()->hasExtension(filename.c_str(), ".nii")) {
        return loadFromNiiFile(filename);
    } else {
        sgl::Logfile::get()->writeError(
                "Error in CloudData::loadFromFile: The file \"" + filename + "\" has an unknown extension!");
        return false;
    }
}

bool CloudData::loadFromXyzFile(const std::string& filename) {
    uint8_t* fileBuffer = nullptr;
    size_t bufferSize = 0;
    bool loaded = sgl::loadFileFromSource(filename, fileBuffer, bufferSize, true);
    if (!loaded) {
        sgl::Logfile::get()->writeError(
                "Error in CloudData::loadFromFile: Couldn't load data from grid data set file \""
                + filename + "\".");
        return false;
    }
    sgl::BinaryReadStream binaryReadStream(fileBuffer, bufferSize);

    double voxelSizeXDouble = 0.0, voxelSizeYDouble = 0.0, voxelSizeZDouble = 0.0;
    binaryReadStream.read(gridSizeX);
    binaryReadStream.read(gridSizeY);
    binaryReadStream.read(gridSizeZ);
    binaryReadStream.read(voxelSizeXDouble);
    binaryReadStream.read(voxelSizeYDouble);
    binaryReadStream.read(voxelSizeZDouble);

    voxelSizeX = float(voxelSizeXDouble);
    voxelSizeY = float(voxelSizeYDouble);
    voxelSizeZ = float(voxelSizeZDouble);

    size_t numGridEntries = size_t(gridSizeX) * size_t(gridSizeY) * size_t(gridSizeZ);
    auto* densityFieldFloat = new float[numGridEntries];
    auto* densityFieldTransposed = new float[numGridEntries];
    binaryReadStream.read(densityFieldTransposed, numGridEntries * sizeof(float));

    // Transpose.
#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, gridSizeZ), [&](auto const& r) {
        for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
    #pragma omp parallel for shared(densityFieldFloat, densityFieldTransposed, gridSizeX, gridSizeY, gridSizeZ) \
    default(none)
#endif
    for (uint32_t z = 0; z < gridSizeZ; z++) {
#endif
        for (uint32_t y = 0; y < gridSizeY; y++) {
            for (uint32_t x = 0; x < gridSizeX; x++) {
                densityFieldFloat[x + (y + z * gridSizeY) * gridSizeX] =
                        densityFieldTransposed[z + (y + x * gridSizeY) * gridSizeZ];
            }
        }
    }
#ifdef USE_TBB
    });
#endif
    delete[] densityFieldTransposed;

    auto [minVal, maxVal] = sgl::reduceFloatArrayMinMax(
            densityFieldFloat, numGridEntries, std::make_pair(0.0f, std::numeric_limits<float>::lowest()));

    if (maxVal - minVal > 1e-6f) {
#ifdef USE_TBB
        tbb::parallel_for(tbb::blocked_range<size_t>(0, numGridEntries), [&](auto const& r) {
        for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 201107
        #pragma omp parallel for default(none) shared(densityFieldFloat, numGridEntries, minVal, maxVal)
#endif
        for (size_t i = 0; i < numGridEntries; i++) {
#endif
            densityFieldFloat[i] = (densityFieldFloat[i] - minVal) / (maxVal - minVal);
        }
#ifdef USE_TBB
        });
#endif
    }

    densityField = std::make_shared<DensityField>(numGridEntries, densityFieldFloat);

    transposeIfNecessary();
    computeGridBounds();

    return true;
}

bool CloudData::loadFromDatRawFile(const std::string& filename) {
    std::string datFilePath;
    std::string rawFilePath;

    if (sgl::endsWith(filename, ".dat")) {
        datFilePath = filename;
    }
    if (sgl::endsWith(filename, ".raw")) {
        rawFilePath = filename;

        // We need to find the corresponding .dat file.
        std::string rawFileDirectory = sgl::FileUtils::get()->getPathToFile(rawFilePath);
        std::vector<std::string> filesInDir = sgl::FileUtils::get()->getFilesInDirectoryVector(rawFileDirectory);
        for (const std::string& filePath : filesInDir) {
            if (sgl::endsWith(filePath, ".dat")) {
                datFilePath = filePath;
                break;
            }
        }
        if (datFilePath.empty()) {
            sgl::Logfile::get()->throwError(
                    "Error in DatRawFileLoader::load: No .dat file found for \"" + rawFilePath + "\".");
        }
    }

    // Load the .dat metadata file.
    uint8_t* bufferDat = nullptr;
    size_t lengthDat = 0;
    bool loadedDat = sgl::loadFileFromSource(datFilePath, bufferDat, lengthDat, false);
    if (!loadedDat) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Couldn't open file \"" + datFilePath + "\".");
    }
    char* fileBuffer = reinterpret_cast<char*>(bufferDat);

    std::string lineBuffer;
    std::string stringBuffer;
    std::vector<std::string> splitLineString;
    std::map<std::string, std::string> datDict;
    for (size_t charPtr = 0; charPtr < lengthDat; ) {
        lineBuffer.clear();
        while (charPtr < lengthDat) {
            char currentChar = fileBuffer[charPtr];
            if (currentChar == '\n' || currentChar == '\r') {
                charPtr++;
                break;
            }
            lineBuffer.push_back(currentChar);
            charPtr++;
        }

        if (lineBuffer.empty()) {
            continue;
        }

        splitLineString.clear();
        sgl::splitString(lineBuffer, ':', splitLineString);
        if (splitLineString.empty()) {
            continue;
        }
        if (splitLineString.size() != 2) {
            sgl::Logfile::get()->throwError(
                    "Error in DatRawFileLoader::load: Invalid entry in file \"" + datFilePath + "\".");
        }

        std::string datKey = splitLineString.at(0);
        std::string datValue = splitLineString.at(1);
        sgl::stringTrim(datKey);
        sgl::toLower(datKey);
        sgl::stringTrim(datValue);
        datDict.insert(std::make_pair(datKey, datValue));
    }
    delete[] bufferDat;

    // Next, process the metadata.
    if (rawFilePath.empty()) {
        auto it = datDict.find("objectfilename");
        if (it == datDict.end()) {
            sgl::Logfile::get()->throwError(
                    "Error in DatRawFileLoader::load: Entry 'ObjectFileName' missing in \""
                    + datFilePath + "\".");
        }
        if (datDict.find("objectindices") != datDict.end()) {
            sgl::Logfile::get()->throwError(
                    "Error in DatRawFileLoader::load: ObjectIndices found in file \"" + datFilePath
                    + "\" is not yet supported.");
        }
        rawFilePath = it->second;
        bool isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(rawFilePath);
        if (!isAbsolutePath) {
            rawFilePath = sgl::FileUtils::get()->getPathToFile(datFilePath) + rawFilePath;
        }
    }

    auto itResolution = datDict.find("resolution");
    if (itResolution == datDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Entry 'Resolution' missing in \"" + datFilePath + "\".");
    }
    std::vector<std::string> resolutionSplit;
    sgl::splitStringWhitespace(itResolution->second, resolutionSplit);
    if (resolutionSplit.size() != 3) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Entry 'Resolution' in \"" + datFilePath
                + "\" does not have three values.");
    }
    auto xs = sgl::fromString<uint32_t>(resolutionSplit.at(0));
    auto ys = sgl::fromString<uint32_t>(resolutionSplit.at(1));
    auto zs = sgl::fromString<uint32_t>(resolutionSplit.at(2));
    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;

    gridSizeX = xs;
    gridSizeY = ys;
    gridSizeZ = zs;
    voxelSizeX = cellStep;
    voxelSizeY = cellStep;
    voxelSizeZ = cellStep;

    auto itSliceThickness = datDict.find("slicethickness");
    if (itSliceThickness != datDict.end()) {
        std::vector<std::string> sliceThicknessList;
        sgl::splitStringWhitespace(itSliceThickness->second, sliceThicknessList);
        if (sliceThicknessList.size() != 3) {
            sgl::Logfile::get()->throwError(
                    "Error in DatRawFileLoader::load: Inconsistent entry 'SliceThickness' in \"" + datFilePath + "\".");
        }
        auto tx = sgl::fromString<float>(sliceThicknessList.at(0));
        auto ty = sgl::fromString<float>(sliceThicknessList.at(1));
        auto tz = sgl::fromString<float>(sliceThicknessList.at(2));
        voxelSizeX *= tx;
        voxelSizeY *= ty;
        voxelSizeZ *= tz;
    }

    auto itFormat = datDict.find("format");
    if (itFormat == datDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Entry 'Format' missing in \"" + datFilePath + "\".");
    }
    std::string formatString = sgl::toLowerCopy(itFormat->second);
    size_t bytesPerEntry = 0;
    if (formatString == "float") {
        bytesPerEntry = 4;
    } else if (formatString == "uchar") {
        bytesPerEntry = 1;
    } else if (formatString == "ushort") {
        bytesPerEntry = 2;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Unsupported format '" + formatString + "' in file \""
                + datFilePath + "\".");
    }

    // Finally, load the data from the .raw file.
    uint8_t* bufferRaw = nullptr;
    size_t lengthRaw = 0;
    bool loadedRaw = sgl::loadFileFromSource(rawFilePath, bufferRaw, lengthRaw, true);
    if (!loadedRaw) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Couldn't open file \"" + rawFilePath + "\".");
    }

    size_t numBytesData = lengthRaw;
    size_t totalSize = size_t(xs) * size_t(ys) * size_t(zs);
    if (numBytesData != totalSize * bytesPerEntry) {
        sgl::Logfile::get()->throwError(
                "Error in DatRawFileLoader::load: Invalid number of entries for file \""
                + rawFilePath + "\".");
    }

    if (formatString == "float") {
        auto* densityFieldFloat = new float[totalSize];
        memcpy(densityFieldFloat, bufferRaw, sizeof(float) * totalSize);
        densityField = std::make_shared<DensityField>(totalSize, densityFieldFloat);
    } else if (formatString == "uchar") {
        auto* densityFieldUchar = new uint8_t[totalSize];
        memcpy(densityFieldUchar, bufferRaw, sizeof(uint8_t) * totalSize);
        densityField = std::make_shared<DensityField>(totalSize, densityFieldUchar);
    } else if (formatString == "ushort") {
        auto* densityFieldUshort = new uint16_t[totalSize];
        memcpy(densityFieldUshort, bufferRaw, sizeof(uint16_t) * totalSize);
        densityField = std::make_shared<DensityField>(totalSize, densityFieldUshort);
    }

    /*auto [minVal, maxVal] = sgl::reduceFloatArrayMinMax(
            densityField, totalSize, std::make_pair(0.0f, std::numeric_limits<float>::lowest()));

#ifdef USE_TBB
    tbb::parallel_for(tbb::blocked_range<size_t>(0, totalSize), [&](auto const& r) {
        for (auto i = r.begin(); i != r.end(); i++) {
#else
#if _OPENMP >= 201107
    #pragma omp parallel for default(none) shared(densityField, totalSize, minVal, maxVal)
#endif
    for (size_t i = 0; i < totalSize; i++) {
#endif
        densityField[i] = (densityField[i] - minVal) / (maxVal - minVal);
    }
#ifdef USE_TBB
    });
#endif*/

    transposeIfNecessary();
    computeGridBounds();

    return true;
}

inline void existsAndEqual(
        const std::string& mhdFilePath, const std::map<std::string, std::string>& mhdDict,
        const std::string& key, const std::string& value) {
    auto it = mhdDict.find(key);
    if (it == mhdDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Entry '" + key + "' missing in \"" + mhdFilePath + "\".");
    }
    if (it->second != value) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Entry '" + key + "' is not equal to \"" + value + "\".");
    }
}

template<class T>
inline void transposeField(T*& data, uint32_t xs, uint32_t ys, uint32_t zs, bool axes[3]) {
    auto* tmp = data;
    data = new T[xs * ys * zs];
    for (uint32_t z = 0; z < zs; z++) {
        for (uint32_t y = 0; y < ys; y++) {
            for (uint32_t x = 0; x < xs; x++) {
                uint32_t writeIdx = x + (y + z * ys) * xs;
                uint32_t xp = axes[0] ? xs - x - 1 : x;
                uint32_t yp = axes[1] ? ys - y - 1 : y;
                uint32_t zp = axes[2] ? zs - z - 1 : z;
                uint32_t readIdx = xp + (yp + zp * ys) * xs;
                data[writeIdx] = tmp[readIdx];
            }
        }
    }
    delete[] tmp;
}

bool CloudData::loadFromMhdRawFile(const std::string& filename) {
    std::string mhdFilePath;
    std::string rawFilePath;

    if (sgl::endsWith(filename, ".mhd")) {
        mhdFilePath = filename;
    }
    if (sgl::endsWith(filename, ".raw")) {
        rawFilePath = filename;

        // We need to find the corresponding .mhd file.
        std::string rawFileDirectory = sgl::FileUtils::get()->getPathToFile(rawFilePath);
        std::vector<std::string> filesInDir = sgl::FileUtils::get()->getFilesInDirectoryVector(rawFileDirectory);
        for (const std::string& filePath : filesInDir) {
            if (sgl::endsWith(filePath, ".mhd")) {
                mhdFilePath = filePath;
                break;
            }
        }
        if (mhdFilePath.empty()) {
            sgl::Logfile::get()->throwError(
                    "Error in loadFromMhdRawFile::load: No .mhd file found for \"" + rawFilePath + "\".");
        }
    }

    // Load the .mhd metadata file.
    uint8_t* bufferMhd = nullptr;
    size_t lengthMhd = 0;
    bool loadedMhd = sgl::loadFileFromSource(mhdFilePath, bufferMhd, lengthMhd, false);
    if (!loadedMhd) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Couldn't open file \"" + mhdFilePath + "\".");
    }
    char* fileBuffer = reinterpret_cast<char*>(bufferMhd);

    std::string lineBuffer;
    std::string stringBuffer;
    std::vector<std::string> splitLineString;
    std::map<std::string, std::string> mhdDict;
    for (size_t charPtr = 0; charPtr < lengthMhd; ) {
        lineBuffer.clear();
        while (charPtr < lengthMhd) {
            char currentChar = fileBuffer[charPtr];
            if (currentChar == '\n' || currentChar == '\r') {
                charPtr++;
                break;
            }
            lineBuffer.push_back(currentChar);
            charPtr++;
        }

        if (lineBuffer.empty()) {
            continue;
        }

        splitLineString.clear();
        sgl::splitString(lineBuffer, '=', splitLineString);
        if (splitLineString.empty()) {
            continue;
        }
        if (splitLineString.size() != 2) {
            sgl::Logfile::get()->throwError(
                    "Error in loadFromMhdRawFile::load: Invalid entry in file \"" + mhdFilePath + "\".");
        }

        std::string mhdKey = splitLineString.at(0);
        std::string mhdValue = splitLineString.at(1);
        sgl::stringTrim(mhdKey);
        //sgl::toLower(mhdKey);
        sgl::stringTrim(mhdValue);
        mhdDict.insert(std::make_pair(mhdKey, mhdValue));
    }
    delete[] bufferMhd;

    // Next, process the metadata.
    if (rawFilePath.empty()) {
        auto it = mhdDict.find("ElementDataFile");
        if (it == mhdDict.end()) {
            sgl::Logfile::get()->throwError(
                    "Error in loadFromMhdRawFile::load: Entry 'ObjectFileName' missing in \""
                    + mhdFilePath + "\".");
        }
        rawFilePath = it->second;
        bool isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(rawFilePath);
        if (!isAbsolutePath) {
            rawFilePath = sgl::FileUtils::get()->getPathToFile(mhdFilePath) + rawFilePath;
        }
    }

    existsAndEqual(mhdFilePath, mhdDict, "ObjectType", "Image");
    existsAndEqual(mhdFilePath, mhdDict, "NDims", "3");
    existsAndEqual(mhdFilePath, mhdDict, "BinaryData", "True");
    existsAndEqual(mhdFilePath, mhdDict, "BinaryDataByteOrderMSB", "False");
    //existsAndEqual(mhdFilePath, mhdDict, "Offset", "0 0 0"); // Unnecessary; we normalize the coordinates anyways.
    //existsAndEqual(mhdFilePath, mhdDict, "AnatomicalOrientation", "RAI"); // "RAI" or "LPI"
    //existsAndEqual(mhdFilePath, mhdDict, "InterceptSlope", "0 1"); // "0 1" or "-1024 1"
    //existsAndEqual(mhdFilePath, mhdDict, "Modality", "MET_MOD_OTHER"); // Unnecessary
    //existsAndEqual(mhdFilePath, mhdDict, "SegmentationType", "UNKNOWN"); // Unnecessary

    auto itResolution = mhdDict.find("DimSize");
    if (itResolution == mhdDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Entry 'DimSize' missing in \"" + mhdFilePath + "\".");
    }
    std::vector<std::string> resolutionSplit;
    sgl::splitStringWhitespace(itResolution->second, resolutionSplit);
    if (resolutionSplit.size() != 3) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Entry 'DimSize' in \"" + mhdFilePath
                + "\" does not have three values.");
    }
    auto xs = sgl::fromString<uint32_t>(resolutionSplit.at(0));
    auto ys = sgl::fromString<uint32_t>(resolutionSplit.at(1));
    auto zs = sgl::fromString<uint32_t>(resolutionSplit.at(2));

    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;

    gridSizeX = xs;
    gridSizeY = ys;
    gridSizeZ = zs;
    voxelSizeX = cellStep;
    voxelSizeY = cellStep;
    voxelSizeZ = cellStep;

    auto itSpacing = mhdDict.find("ElementSpacing");
    if (itSpacing != mhdDict.end()) {
        std::vector<std::string> spacingSplit;
        sgl::splitStringWhitespace(itSpacing->second, spacingSplit);
        if (spacingSplit.size() != 3) {
            sgl::Logfile::get()->throwError(
                    "Error in loadFromMhdRawFile::load: Entry 'ElementSpacing' in \"" + mhdFilePath
                    + "\" does not have three values.");
        }
        auto dx = sgl::fromString<float>(spacingSplit.at(0));
        auto dy = sgl::fromString<float>(spacingSplit.at(1));
        auto dz = sgl::fromString<float>(spacingSplit.at(2));
        voxelSizeX *= dx;
        voxelSizeY *= dy;
        voxelSizeZ *= dz;
    }

    auto itTransformMatrix = mhdDict.find("TransformMatrix");
    if (itTransformMatrix == mhdDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Entry 'TransformMatrix' missing in \"" + mhdFilePath + "\".");
    }
    bool useCustomTransform = false;
    bool mirrorAxes[3] = {false, false, false};
    if (itTransformMatrix->second != "1 0 0 0 1 0 0 0 1") {
        useCustomTransform = true;
        existsAndEqual(mhdFilePath, mhdDict, "CenterOfRotation", "0 0 0");
        std::vector<std::string> transformStringList;
        sgl::splitStringWhitespace(itTransformMatrix->second, transformStringList);
        if (transformStringList.size() != 9) {
            sgl::Logfile::get()->throwError(
                    "Error in loadFromMhdRawFile::load: Entry 'TransformMatrix' in \"" + mhdFilePath
                    + "\" does not have nine values.");
        }
        std::vector<float> transformMatrix;
        for (const auto& str : transformStringList) {
            transformMatrix.push_back(sgl::fromString<float>(str));
        }
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                float val = transformMatrix.at(i * 3 + j);
                if (i != j && val != 0.0f) {
                    sgl::Logfile::get()->throwError(
                            "Error in loadFromMhdRawFile::load: Entry 'TransformMatrix' in \"" + mhdFilePath
                            + "\" contains a rotational part. This is currently not supported.");
                }
                if (i == j && val != 1.0f && val != -1.0f) {
                    sgl::Logfile::get()->throwError(
                            "Error in loadFromMhdRawFile::load: Entry 'TransformMatrix' in \"" + mhdFilePath
                            + "\" contains a scaling part. This is currently not supported.");
                }
                if (i == j && val == -1.0f) {
                    mirrorAxes[i] = true;
                }
            }
        }
    }

    auto itFormat = mhdDict.find("ElementType");
    if (itFormat == mhdDict.end()) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Entry 'ElementType' missing in \"" + mhdFilePath + "\".");
    }
    std::string formatString = itFormat->second;
    size_t bytesPerEntry = 0;
    if (formatString == "MET_FLOAT") {
        bytesPerEntry = 4;
    } else if (formatString == "MET_UCHAR") {
        bytesPerEntry = 1;
    } else if (formatString == "MET_USHORT" || formatString == "MET_FLOAT16" || formatString == "MET_HALF") {
        bytesPerEntry = 2;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Unsupported format '" + formatString + "' in file \""
                + mhdFilePath + "\".");
    }

    auto itBitsStored = mhdDict.find("BitsStored");
    if (itBitsStored != mhdDict.end()) {
        int numBitsStored = sgl::fromString<int>(itBitsStored->second);
        if (size_t(numBitsStored) != bytesPerEntry * 8) {
            sgl::Logfile::get()->throwError(
                    "Error in loadFromMhdRawFile::load: Mismatched 'BitsStored' entry in \"" + mhdFilePath + "\".");
        }
    }

    // Finally, load the data from the .raw file.
    uint8_t* bufferRaw = nullptr;
    size_t lengthRaw = 0;
    bool loadedRaw = sgl::loadFileFromSource(rawFilePath, bufferRaw, lengthRaw, true);
    if (!loadedRaw) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Couldn't open file \"" + rawFilePath + "\".");
    }

    size_t numBytesData = lengthRaw;
    size_t totalSize = size_t(xs) * size_t(ys) * size_t(zs);
    if (numBytesData != totalSize * bytesPerEntry) {
        sgl::Logfile::get()->throwError(
                "Error in loadFromMhdRawFile::load: Invalid number of entries for file \""
                + rawFilePath + "\".");
    }

    if (formatString == "MET_FLOAT") {
        auto* densityFieldFloat = new float[totalSize];
        memcpy(densityFieldFloat, bufferRaw, sizeof(float) * totalSize);
        if (useCustomTransform) {
            transposeField(densityFieldFloat, xs, ys, zs, mirrorAxes);
        }
        densityField = std::make_shared<DensityField>(totalSize, densityFieldFloat);
    } else if (formatString == "MET_UCHAR") {
        auto* densityFieldUchar = new uint8_t[totalSize];
        memcpy(densityFieldUchar, bufferRaw, sizeof(uint8_t) * totalSize);
        if (useCustomTransform) {
            transposeField(densityFieldUchar, xs, ys, zs, mirrorAxes);
        }
        densityField = std::make_shared<DensityField>(totalSize, densityFieldUchar);
    } else if (formatString == "MET_USHORT") {
        auto* densityFieldUshort = new uint16_t[totalSize];
        memcpy(densityFieldUshort, bufferRaw, sizeof(uint16_t) * totalSize);
        if (useCustomTransform) {
            transposeField(densityFieldUshort, xs, ys, zs, mirrorAxes);
        }
        densityField = std::make_shared<DensityField>(totalSize, densityFieldUshort);
    } else if (formatString == "MET_FLOAT16" || formatString == "MET_HALF") {
        auto* densityFieldHalf = new HalfFloat[totalSize];
        memcpy((void*)densityFieldHalf, (const void*)bufferRaw, sizeof(HalfFloat) * totalSize);
        if (useCustomTransform) {
            transposeField(densityFieldHalf, xs, ys, zs, mirrorAxes);
        }
        densityField = std::make_shared<DensityField>(totalSize, densityFieldHalf);
    }

    transposeIfNecessary();
    computeGridBounds();

    return true;
}

bool CloudData::loadFromNiiFile(const std::string& filename) {
    uint8_t* buffer = nullptr;
    size_t length = 0;
    bool loaded = sgl::loadFileFromSource(filename, buffer, length, true);
    if (!loaded) {
        sgl::Logfile::get()->throwError(
                "Error in CloudData::loadFromNiiFile: Couldn't open file \"" + filename + "\".");
    }
    if (length < sizeof(nifti_1_header)) {
        sgl::Logfile::get()->throwError(
                "Error in CloudData::loadFromNiiFile: Invalid file size for file \"" + filename + "\".");
    }
    nifti_1_header* header = reinterpret_cast<nifti_1_header*>(buffer);
    auto dataOffset = ptrdiff_t(header->vox_offset);

    std::string filenameRawLower = sgl::FileUtils::get()->getPureFilename(filename);
    sgl::toLower(filenameRawLower);

    if (header->dim[0] != 3) {
        sgl::Logfile::get()->throwError(
                "Error in CloudData::loadFromNiiFile: Invalid number of dimensions for file \""
                + filename + "\".");
    }

    int xs = int(header->dim[1]);
    int ys = int(header->dim[2]);
    int zs = int(header->dim[3]);
    float maxDimension = float(std::max(xs - 1, std::max(ys - 1, zs - 1)));
    float cellStep = 1.0f / maxDimension;
    voxelSizeX = voxelSizeY = voxelSizeZ = cellStep;
    float sx = std::abs(header->srow_x[0]);
    float sy = std::abs(header->srow_y[1]);
    float sz = std::abs(header->srow_z[2]);
    if (!std::isnan(sx) && !std::isnan(sy) && !std::isnan(sz) && sx != 0 && sy != 0 && sz != 0) {
        voxelSizeX *= sx;
        voxelSizeY *= sy;
        voxelSizeZ *= sz;
    }
    gridSizeX = uint32_t(xs);
    gridSizeY = uint32_t(ys);
    gridSizeZ = uint32_t(zs);

    ptrdiff_t imageSizeInBytes = header->bitpix / 8;
    for (short i = 0; i < header->dim[0] && i < 7; i++) {
        imageSizeInBytes *= ptrdiff_t(header->dim[i + 1]);
    }
    if (dataOffset + imageSizeInBytes > ptrdiff_t(length)) {
        sgl::Logfile::get()->throwError(
                "Error in CloudData::loadFromNiiFile: Invalid data size for file \"" + filename + "\".");
    }

    ScalarDataFormat dataFormat = ScalarDataFormat::FLOAT;
    if (header->datatype == DT_FLOAT || header->datatype == DT_DOUBLE) {
        dataFormat = ScalarDataFormat::FLOAT;
    } else if (header->datatype == DT_SIGNED_SHORT) {
        dataFormat = ScalarDataFormat::SHORT;
    } else if (header->datatype == DT_UNSIGNED_CHAR) {
        dataFormat = ScalarDataFormat::BYTE;
    } else {
        sgl::Logfile::get()->throwError(
                "Error in CloudData::loadFromNiiFile: Invalid data type in file \"" + filename + "\".");
    }

    auto* scalarAttributeField = new uint8_t[imageSizeInBytes];
    memcpy(scalarAttributeField, buffer + dataOffset, imageSizeInBytes);

    if (header->datatype == DT_DOUBLE) {
        auto* scalarAttributeFieldDouble = reinterpret_cast<double*>(scalarAttributeField);
        scalarAttributeField = new uint8_t[imageSizeInBytes];
        auto* scalarAttributeFieldFloat = reinterpret_cast<float*>(scalarAttributeField);
        ptrdiff_t numEntries = imageSizeInBytes / ptrdiff_t(sizeof(double));
        for (ptrdiff_t i = 0; i < numEntries; i++) {
            scalarAttributeFieldFloat[i] = float(scalarAttributeFieldDouble[i]);
        }
        delete[] scalarAttributeFieldDouble;
    }

    // TODO: value = header->scl_slope * valueOld + header->scl_inter
    if (std::abs(header->scl_slope - 1.0f) > 1e-4 && header->datatype != DT_FLOAT && header->datatype != DT_DOUBLE) {
        auto* scalarAttributeFieldOld = scalarAttributeField;
        scalarAttributeField = new uint8_t[xs * ys * zs * sizeof(float)];
        auto* scalarAttributeFieldFloat = reinterpret_cast<float*>(scalarAttributeField);
        if (dataFormat == ScalarDataFormat::FLOAT) {
            auto* scalarAttributeFieldOldFloat = reinterpret_cast<float*>(scalarAttributeFieldOld);
            int numEntries = xs * ys * zs;
            for (int i = 0; i < numEntries; i++) {
                scalarAttributeFieldFloat[i] = scalarAttributeFieldOldFloat[i];
            }
        } else if (dataFormat == ScalarDataFormat::SHORT) {
            auto* scalarAttributeFieldOldShort = reinterpret_cast<int16_t*>(scalarAttributeFieldOld);
            int numEntries = xs * ys * zs;
            for (int i = 0; i < numEntries; i++) {
                scalarAttributeFieldFloat[i] =
                        float(scalarAttributeFieldOldShort[i]) * header->scl_slope + header->scl_inter;
            }
        } else if (dataFormat == ScalarDataFormat::BYTE) {
            auto* scalarAttributeFieldOldByte = reinterpret_cast<uint8_t*>(scalarAttributeFieldOld);
            int numEntries = xs * ys * zs;
            for (int i = 0; i < numEntries; i++) {
                scalarAttributeFieldFloat[i] =
                        float(scalarAttributeFieldOldByte[i]) * header->scl_slope + header->scl_inter;
            }
        }
        dataFormat = ScalarDataFormat::FLOAT;
        delete[] scalarAttributeFieldOld;
    }

    densityField = std::make_shared<DensityField>(xs * ys * zs, reinterpret_cast<float*>(scalarAttributeField));
    delete[] buffer;

    transposeIfNecessary();
    computeGridBounds();

    return true;
}

DensityFieldPtr CloudData::getDenseDensityField() {
    if (!hasDenseData()) {
        if (sparseGridHandle.empty()) {
            sgl::Logfile::get()->throwError(
                    "Fatal error in CloudData::getDenseDensityField: Neither a dense nor a sparse field are "
                    "loaded!");
            return {};
        }

        auto* grid = sparseGridHandle.grid<float>();
        if (!grid) {
            sgl::Logfile::get()->throwError(
                    "Fatal error in CloudData::getDenseDensityField: The sparse grid data from \"" + gridFilename
                    + "\" does not contain floating point data!");
            return {};
        }

        boxMinDense = boxMinSparse;
        boxMaxDense = boxMaxSparse;

        auto& tree = grid->tree();
        auto minGridVal = grid->indexBBox().min();
        size_t numGridEntries = size_t(gridSizeX) * size_t(gridSizeY) * size_t(gridSizeZ);
        auto gridSizeX64 = size_t(gridSizeX);
        auto gridSizeY64 = size_t(gridSizeY);
        auto gridSizeZ64 = size_t(gridSizeZ);
        auto gridSizeXY64 = size_t(gridSizeX64) * size_t(gridSizeY64);

        bool useHalf = numGridEntries >= size_t(1024) * size_t(1024) * size_t(1024);
        if (useHalf) {
            auto* densityFieldFloat = new HalfFloat[numGridEntries];
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<size_t>(0, gridSizeZ64), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(gridSizeX64, gridSizeY64, gridSizeZ64, gridSizeXY64) \
            shared(densityFieldFloat, tree, minGridVal) default(none)
#endif
            for (size_t z = 0; z < gridSizeZ64; z++) {
#endif
                for (size_t y = 0; y < gridSizeY64; y++) {
                    for (size_t x = 0; x < gridSizeX64; x++) {
                        densityFieldFloat[x + y * gridSizeX64 + z * gridSizeXY64] = HalfFloat(tree.getValue(nanovdb::Coord(
                                minGridVal[0] + int(x), minGridVal[1] + int(y), minGridVal[2] + int(z))));
                    }
                }
            }
#ifdef USE_TBB
            });
#endif
            densityField = std::make_shared<DensityField>(numGridEntries, densityFieldFloat);
        } else {
            auto* densityFieldFloat = new float[numGridEntries];
#ifdef USE_TBB
            tbb::parallel_for(tbb::blocked_range<size_t>(0, gridSizeZ64), [&](auto const& r) {
                for (auto z = r.begin(); z != r.end(); z++) {
#else
#if _OPENMP >= 201107
            #pragma omp parallel for shared(gridSizeX64, gridSizeY64, gridSizeZ64, gridSizeXY64) \
            shared(densityFieldFloat, tree, minGridVal) default(none)
#endif
            for (size_t z = 0; z < gridSizeZ64; z++) {
#endif
                for (size_t y = 0; y < gridSizeY64; y++) {
                    for (size_t x = 0; x < gridSizeX64; x++) {
                        densityFieldFloat[x + y * gridSizeX64 + z * gridSizeXY64] = tree.getValue(nanovdb::Coord(
                                minGridVal[0] + int(x), minGridVal[1] + int(y), minGridVal[2] + int(z)));
                    }
                }
            }
#ifdef USE_TBB
            });
#endif
            densityField = std::make_shared<DensityField>(numGridEntries, densityFieldFloat);
        }
    }

    return densityField;
}


void CloudData::printSparseGridMetadata() {
    size_t numGridEntries = size_t(gridSizeX) * size_t(gridSizeY) * size_t(gridSizeZ);
    double denseGridSizeMiB = numGridEntries * 4 / (1024.0 * 1024.0);
    double sparseGridSizeMiB = sparseGridHandle.gridMetaData()->gridSize() / (1024.0 * 1024.0);
    double compressionRatio = denseGridSizeMiB / sparseGridSizeMiB;
    sgl::Logfile::get()->writeInfo("Dense grid memory (MiB): " + std::to_string(denseGridSizeMiB));
    sgl::Logfile::get()->writeInfo("Sparse grid memory (MiB): " + std::to_string(sparseGridSizeMiB));
    sgl::Logfile::get()->writeInfo("Compression ratio: " + std::to_string(compressionRatio));
    sgl::Logfile::get()->writeInfo(
            "Total number of voxels: " + std::to_string(numGridEntries));
    sgl::Logfile::get()->writeInfo(
            "Number of active voxels: " + std::to_string(sparseGridHandle.gridMetaData()->activeVoxelCount()));
    for (int i = 0; i < 3; i++) {
        sgl::Logfile::get()->writeInfo(
                "Nodes at level " + std::to_string(i) + ": "
                + std::to_string(sparseGridHandle.gridMetaData()->nodeCount(i)));
    }
}

void CloudData::setGlobalWorldBoundingBox(const sgl::AABB3& boundingBox) {
    globalWorldBoundingBox = boundingBox;
    hasGlobalWorldBoundingBox = true;
}

void CloudData::setSeqBounds(glm::vec3 min, glm::vec3 max) {
    seqMin = min;
    seqMax = max;
    gotSeqBounds = true;

    float maxDim = std::max(seqMax.x - seqMin.x, std::max(seqMax.y - seqMin.y, seqMax.z - seqMin.z));
    boxMinSparse = (gridMinSparse - seqMin) / (maxDim) - glm::vec3 (0.5f, 0.5f, 0.5f);
    boxMaxSparse = boxMinSparse + (gridMaxSparse - gridMinSparse) / (maxDim);
}

void CloudData::computeSparseGridMetadata() {
    const auto* grid = sparseGridHandle.grid<float>();

    if (!grid) {
        sgl::Logfile::get()->throwError(
                "Fatal error in CloudData::computeSparseGridMetadata: The grid handle does not store a grid "
                "with value type float.");
    }

    // 1D example worldBBox = (-10, 10), indexBBox = (1, 2) will be used in the following comments.

    if (!dataSetFromDense) {
        gridSizeX = uint32_t(grid->indexBBox().max()[0] - grid->indexBBox().min()[0] + 1); // 2
        gridSizeY = uint32_t(grid->indexBBox().max()[1] - grid->indexBBox().min()[1] + 1);
        gridSizeZ = uint32_t(grid->indexBBox().max()[2] - grid->indexBBox().min()[2] + 1);
        voxelSizeX = float(grid->voxelSize()[0]); // 10
        voxelSizeY = float(grid->voxelSize()[1]);
        voxelSizeZ = float(grid->voxelSize()[2]);
    }

    auto nanoVdbBoundingBox = grid->worldBBox();
    gridMinSparse = glm::vec3(
            float(nanoVdbBoundingBox.min()[0]), // -10
            float(nanoVdbBoundingBox.min()[1]),
            float(nanoVdbBoundingBox.min()[2]));
    gridMaxSparse = glm::vec3(
            float(nanoVdbBoundingBox.max()[0]), // 10
            float(nanoVdbBoundingBox.max()[1]),
            float(nanoVdbBoundingBox.max()[2]));

    glm::vec3 worldMin = gridMinSparse; // -10
    glm::vec3 worldMax = gridMaxSparse; //  10
    glm::vec3 globalWorldMin = gridMinSparse; // -10
    glm::vec3 globalWorldMax = gridMaxSparse; //  10

    if (gotSeqBounds) {
        globalWorldMin = seqMin; // overwrite world BB (-10, 10)
        globalWorldMax = seqMax;
    }
    if (hasGlobalWorldBoundingBox) {
        globalWorldMin = globalWorldBoundingBox.min; // overwrite world BB (-10, 10)
        globalWorldMax = globalWorldBoundingBox.max;
    }

    if (!dataSetFromDense) {
        // Normalize bounding box.
        /*float maxDim = std::max(globalWorldMax.x - globalWorldMin.x, std::max(globalWorldMax.y - globalWorldMin.y, globalWorldMax.z - globalWorldMin.z)); // 20
        boxMinSparse = (worldMin - globalWorldMin) / maxDim - glm::vec3(0.5f, 0.5f, 0.5f); // (-10 - -10) / 20 - 0.5 = -0.5
        boxMaxSparse = boxMinSparse + (worldMax - worldMin) / maxDim; // -0.5 + (10 - -10) / 20 = 0.5
        auto totalSize = boxMaxSparse - boxMinSparse; // 1
        boxMaxSparse = totalSize * maxGridExtent; // 0.25
        boxMinSparse = -boxMaxSparse; // -0.25
        */
        const glm::vec3 globalDiff = globalWorldMax - globalWorldMin;
        float maxDim = std::max(globalDiff.x, std::max(globalDiff.y, globalDiff.z));
        const glm::vec3 L = globalDiff / maxDim;
        const float A = 2.0f * maxGridExtent / maxDim;
        const glm::vec3 B = maxGridExtent * L - globalWorldMax * A;
        boxMinSparse = A * worldMin + B;
        boxMaxSparse = A * worldMax + B;
    } else {
        boxMinSparse = globalWorldMin;
        boxMaxSparse = globalWorldMax;
    }

    std::cout << boxMinSparse.x << ", " << boxMinSparse.y << ", " << boxMinSparse.z << std::endl;
    std::cout << boxMaxSparse.x << ", " << boxMaxSparse.y << ", " << boxMaxSparse.z << std::endl;

    printSparseGridMetadata();
}

bool CloudData::loadFromNvdbFile(const std::string& filename) {
    //sparseGridHandle = nanovdb::io::readGrid<nanovdb::HostBuffer>(filename, gridName);
    sparseGridHandle = nanovdb::io::readGrid<nanovdb::HostBuffer>(filename, 0);
    computeSparseGridMetadata();
    return !sparseGridHandle.empty();
}

void CloudData::getSparseDensityField(uint8_t*& data, uint64_t& size) {
    if (!hasSparseData()) {
        if (!densityField) {
            sgl::Logfile::get()->throwError(
                    "Fatal error in CloudData::getSparseDensityField: Neither a dense nor a sparse field are "
                    "loaded!");
            return;
        }

        std::string filenameNvdb = sgl::FileUtils::get()->removeExtension(gridFilename) + ".nvdb";
        if (cacheSparseGrid && sgl::FileUtils::get()->exists(filenameNvdb)) {
            bool isLoaded = loadFromNvdbFile(filenameNvdb);
            if (!isLoaded) {
                sgl::Logfile::get()->throwError(
                        "Error in CloudData::getSparseDensityField: Couldn't load data from grid data set file \""
                        + filenameNvdb + "\".");
            }
        } else {
            nanovdb::GridBuilder builder(0.0f);
            auto gridSamplingOperation = [this](const nanovdb::Coord& ijk) -> float {
                auto x = uint32_t(ijk.x());
                auto y = uint32_t(ijk.y());
                auto z = uint32_t(ijk.z());
                return densityField->getDataFloatAtNorm(x + (y + z * gridSizeY) * gridSizeX);
            };
            auto maxIdxX = int32_t(gridSizeX - 1);
            auto maxIdxY = int32_t(gridSizeY - 1);
            auto maxIdxZ = int32_t(gridSizeZ - 1);
            builder(gridSamplingOperation, nanovdb::CoordBBox(
                    nanovdb::Coord(0), nanovdb::Coord(maxIdxX, maxIdxY, maxIdxZ)));
            double dx = double(boxMaxDense.x - boxMinDense.x) / double(gridSizeX);
            double dy = double(boxMaxDense.y - boxMinDense.y) / double(gridSizeY);
            double dz = double(boxMaxDense.z - boxMinDense.z) / double(gridSizeZ);
            if (dx != dy || dy != dz) {
                sgl::Logfile::get()->writeWarning(
                        "Warning in CloudData::getSparseDensityField: "
                        "Mismatch in dx, dy, dz, but NanoVDB does not support non-uniform voxel sizes.");
            }
            sparseGridHandle = builder.getHandle<>(
                    dx, nanovdb::Vec3d(boxMinDense.x, boxMinDense.y, boxMinDense.z),
                    gridName, nanovdb::GridClass::FogVolume);

            /*auto* gridData = sparseGridHandle.grid<float>();
            const auto& rootNode = gridData->tree().getFirstNode<2>();
            std::cout << "Root min: " << rootNode->minimum() << std::endl;
            std::cout << "Root max: " << rootNode->maximum() << std::endl;*/

            if (cacheSparseGrid) {
                auto* gridData = sparseGridHandle.grid<float>();
                if (!gridData) {
                    sgl::Logfile::get()->throwError(
                            "Fatal error in CloudData::getSparseDensityField: The grid handle does not store a grid "
                            "with value type float.");
                }

                try {
                    nanovdb::io::writeGrid<nanovdb::HostBuffer>(filenameNvdb, sparseGridHandle);
                } catch (const std::exception& e) {
                    sgl::Logfile::get()->throwError(e.what());
                }
            }
        }

        dataSetFromDense = true;
    }

    computeSparseGridMetadata();

    auto& buffer = sparseGridHandle.buffer();
    data = buffer.data();
    size = buffer.size();
}

std::vector<double> CloudData::getVDBWorldBoundingBox() {
    if (!hasSparseData()) {
        uint8_t* data = nullptr;
        uint64_t size = 0;
        getSparseDensityField(data, size);
    }
    const auto* grid = sparseGridHandle.grid<float>();
    if (!grid) {
        sgl::Logfile::get()->throwError(
                "Fatal error in CloudData::getVDBWorldBoundingBox: The grid handle does not store a grid "
                "with value type float.");
    }
    auto bb = grid->worldBBox();
    return {
            bb.min()[0], bb.max()[0], bb.min()[1], bb.max()[1], bb.min()[2], bb.max()[2],
    };
}

std::vector<int64_t> CloudData::getVDBIndexBoundingBox() {
    if (!hasSparseData()) {
        uint8_t* data = nullptr;
        uint64_t size = 0;
        getSparseDensityField(data, size);
    }
    const auto* grid = sparseGridHandle.grid<float>();
    if (!grid) {
        sgl::Logfile::get()->throwError(
                "Fatal error in CloudData::getVDBIndexBoundingBox: The grid handle does not store a grid "
                "with value type float.");
    }
    auto bb = grid->indexBBox();
    return {
            int64_t(bb.min()[0]), int64_t(bb.max()[0]), int64_t(bb.min()[1]), int64_t(bb.max()[1]),
            int64_t(bb.min()[2]), int64_t(bb.max()[2]),
    };
}

std::vector<double> CloudData::getVDBVoxelSize() {
    if (!hasSparseData()) {
        uint8_t* data = nullptr;
        uint64_t size = 0;
        getSparseDensityField(data, size);
    }
    const auto* grid = sparseGridHandle.grid<float>();
    if (!grid) {
        sgl::Logfile::get()->throwError(
                "Fatal error in CloudData::getVDBVoxelSize: The grid handle does not store a grid "
                "with value type float.");
    }
    auto vs = grid->voxelSize();
    return { vs[0], vs[1], vs[2] };
}
