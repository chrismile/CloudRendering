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

#include <boost/algorithm/string/case_conv.hpp>
#include <boost/algorithm/string/predicate.hpp>
#include <boost/algorithm/string/trim.hpp>

#ifdef USE_TBB
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#endif

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

#include "CloudData.hpp"

CloudData::CloudData(sgl::MultiVarTransferFunctionWindow* transferFunctionWindow)
        : transferFunctionWindow(transferFunctionWindow) {
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
    boxMax = glm::vec3(sx, sy, sz) * 0.25f / maxSize;
    boxMin = -boxMax;
    //uint32_t maxDim = std::max(gridSizeX, std::max(gridSizeY, gridSizeZ));
    //boxMax = glm::vec3(gridSizeX, gridSizeY, gridSizeZ) * 0.25f / float(maxDim);
    //boxMin = -boxMax;

    gridMin = glm::vec3(0,0,0);
    gridMax = glm::vec3(1,1,1);
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

    computeGridBounds();

    auto numEntries = size_t(_gridSizeX) * size_t(_gridSizeZ) * size_t(_gridSizeZ);
    densityField = std::make_shared<DensityField>(numEntries, _densityField);
    gridFilename = sgl::AppSettings::get()->getDataDirectory() + "CloudDataSets/clouds/tmp.xyz";
    gridName = "tmp";
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
    gridName = boost::to_lower_copy(sgl::FileUtils::get()->removeExtension(
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

    computeGridBounds();

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

    return true;
}

bool CloudData::loadFromDatRawFile(const std::string& filename) {
    std::string datFilePath;
    std::string rawFilePath;

    if (boost::ends_with(filename, ".dat")) {
        datFilePath = filename;
    }
    if (boost::ends_with(filename, ".raw")) {
        rawFilePath = filename;

        // We need to find the corresponding .dat file.
        std::string rawFileDirectory = sgl::FileUtils::get()->getPathToFile(rawFilePath);
        std::vector<std::string> filesInDir = sgl::FileUtils::get()->getFilesInDirectoryVector(rawFileDirectory);
        for (const std::string& filePath : filesInDir) {
            if (boost::ends_with(filePath, ".dat")) {
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
        boost::trim(datKey);
        boost::to_lower(datKey);
        boost::trim(datValue);
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
    std::string formatString = boost::to_lower_copy(itFormat->second);
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

    computeGridBounds();

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

    if (boost::ends_with(filename, ".mhd")) {
        mhdFilePath = filename;
    }
    if (boost::ends_with(filename, ".raw")) {
        rawFilePath = filename;

        // We need to find the corresponding .mhd file.
        std::string rawFileDirectory = sgl::FileUtils::get()->getPathToFile(rawFilePath);
        std::vector<std::string> filesInDir = sgl::FileUtils::get()->getFilesInDirectoryVector(rawFileDirectory);
        for (const std::string& filePath : filesInDir) {
            if (boost::ends_with(filePath, ".mhd")) {
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
        boost::trim(mhdKey);
        //boost::to_lower(mhdKey);
        boost::trim(mhdValue);
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
    } else if (formatString == "MET_USHORT") {
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

    computeGridBounds();

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
    }

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

        auto& tree = grid->tree();
        auto minGridVal = grid->indexBBox().min();
        size_t numGridEntries = size_t(gridSizeX) * size_t(gridSizeY) * size_t(gridSizeZ);
        auto gridSizeX64 = size_t(gridSizeX);
        auto gridSizeY64 = size_t(gridSizeY);
        auto gridSizeZ64 = size_t(gridSizeZ);
        auto gridSizeXY64 = size_t(gridSizeX64) * size_t(gridSizeY64);
        auto* densityFieldFloat = new float[numGridEntries];
        for (size_t z = 0; z < gridSizeZ64; z++) {
            for (size_t y = 0; y < gridSizeY64; y++) {
                for (size_t x = 0; x < gridSizeX64; x++) {
                    densityFieldFloat[x + y * gridSizeX64 + z * gridSizeXY64] = tree.getValue(nanovdb::Coord(
                            minGridVal[0] + int(x), minGridVal[1] + int(y), minGridVal[2] + int(z)));
                }
            }
        }
        densityField = std::make_shared<DensityField>(numGridEntries, densityFieldFloat);
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

void CloudData::setSeqBounds(glm::vec3 min, glm::vec3 max){
    seqMin = min;
    seqMax = max;
    gotSeqBounds = true;

    float maxDim = std::max(seqMax.x-seqMin.x, std::max(seqMax.y-seqMin.y, seqMax.z-seqMin.z));
    boxMin = (gridMin - seqMin) / (maxDim) - glm::vec3 (.5,.5,.5);
    boxMax = boxMin + (gridMax - gridMin) / (maxDim);
}

void CloudData::computeSparseGridMetadata() {
    const auto* grid = sparseGridHandle.grid<float>();

    if (!grid) {
        sgl::Logfile::get()->throwError(
                "Fatal error in CloudData::computeSparseGridMetadata: The grid handle does not store a grid "
                "with value type float.");
    }

    gridSizeX = uint32_t(grid->indexBBox().max()[0] - grid->indexBBox().min()[0] + 1);
    gridSizeY = uint32_t(grid->indexBBox().max()[1] - grid->indexBBox().min()[1] + 1);
    gridSizeZ = uint32_t(grid->indexBBox().max()[2] - grid->indexBBox().min()[2] + 1);
    voxelSizeX = float(grid->voxelSize()[0]);
    voxelSizeY = float(grid->voxelSize()[1]);
    voxelSizeZ = float(grid->voxelSize()[2]);

    auto nanoVdbBoundingBox = grid->worldBBox();
    gridMin = glm::vec3(
            float(nanoVdbBoundingBox.min()[0]),
            float(nanoVdbBoundingBox.min()[1]),
            float(nanoVdbBoundingBox.min()[2]));
    gridMax = glm::vec3(
            float(nanoVdbBoundingBox.max()[0]),
            float(nanoVdbBoundingBox.max()[1]),
            float(nanoVdbBoundingBox.max()[2]));

    glm::vec3 boundMin = gridMin;
    glm::vec3 boundMax = gridMax;

    if (gotSeqBounds){
        boundMin = seqMin;
        boundMax = seqMax;
    }
    // NORMALIZE BOX
    float maxDim = std::max(boundMax.x-boundMin.x, std::max(boundMax.y-boundMin.y, boundMax.z-boundMin.z));
    boxMin = (gridMin - boundMin) / (maxDim) - glm::vec3 (.5,.5,.5);
    boxMax = boxMin + (gridMax - gridMin) / (maxDim);

    std::cout << boxMin.x<<", " << boxMin.y<< ", " << boxMin.z << std::endl;
    std::cout << boxMax.x<<", " << boxMax.y<< ", " << boxMax.z << std::endl;

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
            double dx = double(boxMax.x - boxMin.x) / double(gridSizeX);
            sparseGridHandle = builder.getHandle<>(
                    dx, nanovdb::Vec3d(boxMin.x, boxMin.y, boxMin.z),
                    gridName, nanovdb::GridClass::FogVolume);
            printSparseGridMetadata();

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
    }
    computeSparseGridMetadata();

    auto& buffer = sparseGridHandle.buffer();
    data = buffer.data();
    size = buffer.size();
}

#ifdef USE_OPENVDB
#include <openvdb/openvdb.h>
#include "nanovdb/util/OpenToNanoVDB.h"

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
#endif
