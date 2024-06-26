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

#ifndef CLOUDRENDERING_DENSITYFIELD_HPP
#define CLOUDRENDERING_DENSITYFIELD_HPP

#include <memory>

#include <Utils/SciVis/ScalarDataFormat.hpp>
#include <Graphics/Vulkan/Image/Image.hpp>

class HalfFloat;
class CloudData;

class DensityField {
    friend class CloudData;
public:
    explicit DensityField(size_t numEntries, float* dataFloat)
            : scalarDataFormatNative(ScalarDataFormat::FLOAT), numEntries(numEntries), dataFloat(dataFloat) {}
    explicit DensityField(size_t numEntries, uint8_t* dataByte)
            : scalarDataFormatNative(ScalarDataFormat::BYTE), numEntries(numEntries), dataByte(dataByte) {}
    explicit DensityField(size_t numEntries, uint16_t* dataShort)
            : scalarDataFormatNative(ScalarDataFormat::SHORT), numEntries(numEntries), dataShort(dataShort) {}
    explicit DensityField(size_t numEntries, HalfFloat* dataFloat16)
           : scalarDataFormatNative(ScalarDataFormat::FLOAT16), numEntries(numEntries), dataFloat16(dataFloat16) {}
    explicit DensityField(ScalarDataFormat fmt) : scalarDataFormatNative(fmt), numEntries(0), dataFloat(nullptr) {}
    ~DensityField();

    static std::shared_ptr<DensityField> createHalfFloat(size_t numEntries, HalfFloat* dataFloat16);

    [[nodiscard]] ScalarDataFormat getScalarDataFormatNative() const { return scalarDataFormatNative; }
    [[nodiscard]] const void* getDataNative();
    [[nodiscard]] const float* getDataFloat();
    [[nodiscard]] const uint8_t* getDataByte();
    [[nodiscard]] const uint16_t* getDataShort();
    [[nodiscard]] const HalfFloat* getDataFloat16();

    size_t getEntrySizeInBytes();
    uint32_t getEntrySizeInBytesUint32();
    VkFormat getEntryVulkanFormat();

    float getMinValue();
    float getMaxValue();

    void switchNativeFormat(ScalarDataFormat newNativeFormat);

    template<class T>
    [[nodiscard]] inline const typename std::enable_if<std::is_same<T, float>::value, T>::type* data() {
        return getDataFloat();
    }
    template<class T>
    [[nodiscard]] inline const typename std::enable_if<std::is_same<T, uint8_t>::value, T>::type* data() {
        return getDataByte();
    }
    template<class T>
    [[nodiscard]] inline const typename std::enable_if<std::is_same<T, uint16_t>::value, T>::type* data() {
        return getDataShort();
    }
    template<class T>
    [[nodiscard]] inline const typename std::enable_if<std::is_same<T, HalfFloat>::value, T>::type* data() {
        return getDataFloat16();
    }

    // Access data at a certain index.
    [[nodiscard]] float getDataFloatAt(size_t idx);
    template<class T>
    [[nodiscard]] inline typename std::enable_if<std::is_same<T, float>::value, T>::type dataAt(size_t idx) {
        return getDataFloatAt(idx);
    }
    [[nodiscard]] float getDataFloatAtNorm(size_t idx);
    template<class T>
    [[nodiscard]] inline typename std::enable_if<std::is_same<T, float>::value, T>::type dataAtNorm(size_t idx) {
        return getDataFloatAtNorm(idx);
    }

    void setDataFloatAt(size_t idx, float val);

private:
    ScalarDataFormat scalarDataFormatNative;
    size_t numEntries = 0;
    float* dataFloat = nullptr;
    uint8_t* dataByte = nullptr;
    uint16_t* dataShort = nullptr;
    HalfFloat* dataFloat16 = nullptr;

    // Min/max data.
    void computeMinMax();
    bool minMaxComputed = false;
    float minValue = 0.0f, maxValue = 1.0f;
};

typedef std::shared_ptr<DensityField> DensityFieldPtr;

#endif //CLOUDRENDERING_DENSITYFIELD_HPP
