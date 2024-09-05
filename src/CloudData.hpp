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

#ifndef CLOUDRENDERING_CLOUDDATA_HPP
#define CLOUDRENDERING_CLOUDDATA_HPP

#include <memory>
#include <Math/Geometry/AABB3.hpp>
#include <Graphics/Color.hpp>

#include "nanovdb/util/GridHandle.h"

#include "PathTracer/RenderSettings.hpp"
#include "DensityField.hpp"

namespace sgl {
class MultiVarTransferFunctionWindow;
}
class LightEditorWidget;

class CloudData {
public:
    explicit CloudData(
            sgl::MultiVarTransferFunctionWindow* transferFunctionWindow = nullptr,
            LightEditorWidget* lightEditorWidget = nullptr);
    ~CloudData();

    /**
     * @param filename The filename of the .xyz or .nvdb file to load.
     * @return Whether the file was loaded successfully.
     */
    bool loadFromFile(const std::string& filename);

    /**
     * @param _gridSizeX The number of voxels in x direction.
     * @param _gridSizeY The number of voxels in y direction.
     * @param _gridSizeZ The number of voxels in z direction.
     * @param _densityField A dense floating point density field of size gridSizeX*gridSizeY*gridSizeZ.
     */
    void setDensityField(uint32_t _gridSizeX, uint32_t _gridSizeY, uint32_t _gridSizeZ, float* _densityField);

    /**
     * Sets the passed grid handle.
     * @param handle
     */
    void setNanoVdbGridHandle(nanovdb::GridHandle<nanovdb::HostBuffer>&& handle);

    [[nodiscard]] inline const std::string& getFileName() const { return gridFilename; }
    [[nodiscard]] inline uint32_t getGridSizeX() const { return gridSizeX; }
    [[nodiscard]] inline uint32_t getGridSizeY() const { return gridSizeY; }
    [[nodiscard]] inline uint32_t getGridSizeZ() const { return gridSizeZ; }

    [[nodiscard]] inline const glm::vec3& getWorldSpaceBoxMin(bool _useSparse) const { return _useSparse ? boxMinSparse : boxMinDense; }
    [[nodiscard]] inline const glm::vec3& getWorldSpaceBoxMax(bool _useSparse) const { return _useSparse ? boxMaxSparse : boxMaxDense; }
    [[nodiscard]] inline sgl::AABB3 getWorldSpaceBoundingBox(bool _useSparse) const {
        return {getWorldSpaceBoxMin(_useSparse), getWorldSpaceBoxMax(_useSparse)};
    }
    [[nodiscard]] inline sgl::AABB3 getWorldSpaceBoundingBox() const {
        return getWorldSpaceBoundingBox(!hasDenseData());
    }
    //[[nodiscard]] inline const glm::vec3& getWorldSpaceBoxMin() const { return boxMin; }
    //[[nodiscard]] inline const glm::vec3& getWorldSpaceBoxMax() const { return boxMax; }
    //[[nodiscard]] inline sgl::AABB3 getWorldSpaceBoundingBox() const { return sgl::AABB3(boxMin, boxMax); }

    [[nodiscard]] inline const glm::vec3& getWorldSpaceGridMin(bool _useSparse) const { return _useSparse ? gridMinSparse : gridMinDense; }
    [[nodiscard]] inline const glm::vec3& getWorldSpaceGridMax(bool _useSparse) const { return _useSparse ? gridMaxSparse : gridMaxDense; }
    [[nodiscard]] inline const glm::vec3& getWorldSpaceDenseGridMin() const { return gridMinDense; }
    [[nodiscard]] inline const glm::vec3& getWorldSpaceDenseGridMax() const { return gridMaxDense; }
    [[nodiscard]] inline const glm::vec3& getWorldSpaceSparseGridMin() const { return gridMinSparse; }
    [[nodiscard]] inline const glm::vec3& getWorldSpaceSparseGridMax() const { return gridMaxSparse; }

    void setNextCloudDataFrame(std::shared_ptr<CloudData> nextFrame) {
        nextCloudDataFrame = nextFrame;
    }

    std::shared_ptr<CloudData> getNextCloudDataFrame() { return nextCloudDataFrame; }

    static void setMaxGridExtent(float _maxGridExtent) { maxGridExtent = _maxGridExtent; }

    void setClearColor(const sgl::Color& clearColor) {}
    void setGlobalWorldBoundingBox(const sgl::AABB3& boundingBox);
    void setSeqBounds(glm::vec3 min, glm::vec3 max);
    void setTransposeAxes(const glm::ivec3& axes);

    /**
     * @return An array of size gridSizeX * gridSizeY * gridSizeZ containing the dense data field.
     * If the object was loaded using a .nvdb file, the dense field is created when calling this function.
     */
    DensityFieldPtr getDenseDensityField();
    [[nodiscard]] inline bool hasDenseData() const { return densityField != nullptr; }

    /**
     * @param data A pointer to the raw NanoVDB data.
     * @param size The size of the NanoVDB data buffer in bytes.
     * If the object was loaded using a dense .xyz grid file, the sparse field is created when calling this function.
     */
    void getSparseDensityField(uint8_t*& data, uint64_t& size);
    [[nodiscard]] inline bool hasSparseData() const { return !sparseGridHandle.empty(); }
    [[nodiscard]] inline size_t getSparseDataSizeInBytes() const { return sparseGridHandle.gridMetaData()->gridSize(); }
    inline void setCacheSparseGrid(bool cache) { cacheSparseGrid = true; }
    std::vector<double> getVDBWorldBoundingBox();
    std::vector<int64_t> getVDBIndexBoundingBox();
    std::vector<double> getVDBVoxelSize();

    /// Called when the transfer function texture was updated.
    void onTransferFunctionMapRebuilt() {}
    inline sgl::MultiVarTransferFunctionWindow* getTransferFunctionWindow() { return transferFunctionWindow; }
    inline LightEditorWidget* getLightEditorWidget() { return lightEditorWidget; }

    void createIsoSurfaceData(
            const IsosurfaceSettings& settings,
            std::vector<uint32_t>& triangleIndices, std::vector<glm::vec3>& vertexPositions,
            std::vector<glm::vec4>& vertexColors, std::vector<glm::vec3>& vertexNormals);

#ifdef USE_OPENVDB
    /**
     * @param filename The filename of the .vdb file to write to using OpenVDB.
     * @return Whether the file was saved successfully.
     */
    bool saveToVdbFile(const std::string& filename);
#endif

private:
    sgl::MultiVarTransferFunctionWindow* transferFunctionWindow = nullptr;
    LightEditorWidget* lightEditorWidget = nullptr;
    std::shared_ptr<CloudData> nextCloudDataFrame;

    std::string gridFilename, gridName;
    uint32_t gridSizeX = 0, gridSizeY = 0, gridSizeZ = 0;
    float voxelSizeX = 0.0f, voxelSizeY = 0.0f, voxelSizeZ = 0.0f;
    glm::vec3 boxMinDense{}, boxMaxDense{}; // Box in which to render
    glm::vec3 boxMinSparse{}, boxMaxSparse{}; // Box in which to render
    glm::vec3 gridMinDense{}, gridMaxDense{}; // Box from which to sample density values. (0,0,0) to (1,1,1) for dense.
    glm::vec3 gridMinSparse{}, gridMaxSparse{}; // Box from which to sample density values.
    sgl::AABB3 globalWorldBoundingBox{};
    bool hasGlobalWorldBoundingBox = false;
    bool gotSeqBounds = false;
    glm::vec3 seqMin{}, seqMax{}; // World space bounds of sequence
    /// Due to legacy reasons, the grid has min, max boxMin{Dense,Sparse} = -0.25, 0.25.
    /// This value can be used to change the extents in the minimum/maximum dimension.
    inline static float maxGridExtent = 0.25f;

    void computeGridBounds();

    // Data & functions for transposing the field (specified in datasets.json).
    void transposeIfNecessary();
    bool transpose = false;
    glm::ivec3 transposeAxes = glm::ivec3(0, 1, 2);

    // --- Dense field. ---
    /**
     * A .xyz file contains data in the following format:
     * 3x uint32_t: Grid size sx, sy ,sz
     * 3x double: Voxel size vx, vy, vz
     * (sx*sy*sz)x float: The dense density field data.
     * @param filename The filename of the .xyz file to load.
     * @return Whether the file was loaded successfully.
     */
    bool loadFromXyzFile(const std::string& filename);
    /**
     * Loading function for pairs of .dat and .raw files.
     * .dat files store metadata about the grid, while .raw files store the data.
     * .dat files contain case-insensitive pairs of keys and values separated by a colon.
     * Example for such a key-value pair: "ObjectFileName: volume_50.raw"
     *
     * Below, a list of valid key-value pairs can be found.
     * ObjectFileName: <string> (location of the .raw data file relative to the .dat file)
     * ObjectIndices: <uint-triplet> (triplet start-stop-step of indices to use if the ObjectFileName contains a format string like "%02i").
     * TaggedFileName: <string> (optional)
     * MeshFileName: <string> (optional)
     * Name: <string> (optional)
     * Resolution: <uint> <uint> <uint>
     * Format: uchar | ushort | float | float3 | float4 (only scalar values supported!)
     * SliceThickness: <float> <float> <float>
     * Range: <float> <float> (optional)
     * NbrTags: <uint> (optional)
     * ObjectType: <enum-string>=TEXTURE_VOLUME_OBJECT (optional)
     * ObjectModel: RGBA (optional)
     * GridType: <enum-string>=EQUIDISTANT (optional)
     * Timestep: <float> (optional)
     */
    bool loadFromDatRawFile(const std::string& filename);
    /**
     * Loading function for pairs of .mhd and .raw files.
     * .mhd files store metadata about the grid, while .raw files store the data.
     * @param filename The filename of the .mhd file to load.
     * @return Whether the file was loaded successfully.
     */
    bool loadFromMhdRawFile(const std::string& filename);
    /**
     * Loading function for the .nii file format. For more details see
     * - https://github.com/NIFTI-Imaging/nifti_clib
     * - https://nifti.nimh.nih.gov/nifti-1/
     * @param filename The filename of the .mhd file to load.
     * @return Whether the file was loaded successfully.
     */
    bool loadFromNiiFile(const std::string& filename);

    DensityFieldPtr densityField{};

    // --- Sparse field. ---
    /**
     * @param filename The filename of the .nvdb file to load using NanoVDB.
     * @return Whether the file was loaded successfully.
     */
    bool loadFromNvdbFile(const std::string& filename);
    void computeSparseGridMetadata();
    void printSparseGridMetadata();
    nanovdb::GridHandle<nanovdb::HostBuffer> sparseGridHandle;
    bool cacheSparseGrid = false;
    bool dataSetFromDense = false; //< Whether the sparse data was computed from a dense field on-the-fly.

#ifdef USE_OPENVDB
    /**
     * @param filename The filename of the .vdb file to load using OpenVDB.
     * @return Whether the file was loaded successfully.
     */
    bool loadFromVdbFile(const std::string& filename);
#endif
};

typedef std::shared_ptr<CloudData> CloudDataPtr;

#endif //CLOUDRENDERING_CLOUDDATA_HPP
