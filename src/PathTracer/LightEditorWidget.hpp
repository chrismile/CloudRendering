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

#ifndef CLOUDRENDERING_LIGHTEDITORWIDGET_HPP
#define CLOUDRENDERING_LIGHTEDITORWIDGET_HPP

#include <vector>
#include <memory>
#include <glm/vec3.hpp>

namespace sgl { namespace vk {
class Buffer;
class Renderer;
typedef std::shared_ptr<Buffer> BufferPtr;
}}
namespace sgl {
class PropertyEditor;
}

enum class LightType : uint32_t {
    POINT = 0, SPOT = 1, DIRECTIONAL = 2
};
const char* const LIGHT_TYPE_NAMES[] = {
        "Point Light", "Spotlight", "Directional Light"
};

enum class LightSpace : uint32_t {
    WORLD = 0, // Positions & directions are in world space.
    VIEW = 1, // Positions & directions are in view space.
    VIEW_ORIENTATION = 2, // Positions are in world space, directions in view space.
};
const char* const LIGHT_SPACE_NAMES[] = {
        "World", "View", "View Orientation"
};

struct Light {
public:
    LightType lightType = LightType::POINT;
    LightSpace lightSpace = LightSpace::VIEW; ///< All types; world space or view space position/direction?
    float spotTotalWidth = 0.0981747704f; ///< SPOT
    float spotFalloffStart = 0.0245436926f; ///< SPOT

    glm::vec3 color = glm::vec3(1.0f, 0.961538462f, 0.884615385f); ///< All types
    float intensity = 0.5f; ///< All types

    // Point light & spotlight.
    glm::vec3 position{}; ///< POINT & SPOT; distance for DIRECTIONAL.
    uint32_t useDistance = true; ///< POINT & SPOT
};

class LightEditorWidget {
public:
    explicit LightEditorWidget(sgl::vk::Renderer* renderer);
    ~LightEditorWidget();

    void addLight(const Light& light);
    void removeLight(uint32_t lightIdx);
    [[nodiscard]] size_t getNumLights() const { return lights.size(); }
    [[nodiscard]] const sgl::vk::BufferPtr& getLightsBuffer() { return lightsBuffer; }

    [[nodiscard]] inline bool getShowWindow() const { return showWindow; }
    [[nodiscard]] inline bool& getShowWindow() { return showWindow; }
    inline void setShowWindow(bool show) { showWindow = show; }

    inline void setStandardWindowSize(int width, int height) { standardWidth = width; standardHeight = height; }
    inline void setStandardWindowPosition(int x, int y) { standardPositionX = x; standardPositionY = y; }

    /// Property interface.
    void setLightProperty(uint32_t lightIdx, const std::string& key, const std::string& value);
    bool loadFromFile(const std::string& filePath);
    bool saveToFile(const std::string& filePath);

    /// @return true if re-rendering the scene is necessary.
    bool renderGui();

private:
    sgl::vk::Renderer* renderer;

    void recreateLightBuffer();
    void updateLightBuffer();
    std::vector<Light> lights;
    sgl::vk::BufferPtr lightsBuffer;

    // Counts lights by their creation so that ImGui can keep track when lights are removed.
    ptrdiff_t currentLightIdx = 0;
    std::vector<ptrdiff_t> lightCreationIndices;

    bool renderGuiLight(size_t lightIdx);
    bool showWindow = true;
    sgl::PropertyEditor* propertyEditor;
    int standardWidth = 800;
    int standardHeight = 400;
    int standardPositionX = 1289;
    int standardPositionY = 62;
};

#endif //CLOUDRENDERING_LIGHTEDITORWIDGET_HPP
