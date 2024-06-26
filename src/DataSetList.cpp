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

#include <iostream>

#include <json/json.h>

#include <Utils/File/Logfile.hpp>
#include <Utils/AppSettings.hpp>
#include <Utils/Regex/TransformString.hpp>
#include <Utils/File/FileUtils.hpp>

#include "DataSetList.hpp"

void processDataSetNodeChildren(Json::Value& childList, DataSetInformation* dataSetInformationParent) {
    for (Json::Value& source : childList) {
        auto* dataSetInformation = new DataSetInformation;

        // Get the type information.
        if (source.isMember("type")) {
            Json::Value type = source["type"];
            std::string typeName = type.asString();
            if (typeName == "node") {
                dataSetInformation->type = DATA_SET_TYPE_NODE;
            } else if (typeName == "volume") {
                dataSetInformation->type = DATA_SET_TYPE_VOLUME;
            } else {
                sgl::Logfile::get()->writeError(
                        "Error in processDataSetNodeChildren: Invalid type name \"" + typeName + "\".");
                return;
            }
        }

        // Get the display name and the associated filenames.
        dataSetInformation->name = source["name"].asString();
        Json::Value filenames = source["filename"];
        const std::string cloudDataSetsDirectory = sgl::AppSettings::get()->getDataDirectory() + "CloudDataSets/";
        if (filenames.isArray()) {
            for (Json::Value::const_iterator filenameIt = filenames.begin();
                    filenameIt != filenames.end(); ++filenameIt) {
                bool isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(filenameIt->asString());
                if (isAbsolutePath) {
                    dataSetInformation->filename = filenameIt->asString();
                } else {
                    dataSetInformation->filename = cloudDataSetsDirectory + filenameIt->asString();
                }
            }
        } else {
            bool isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(filenames.asString());
            if (isAbsolutePath) {
                dataSetInformation->filename = filenames.asString();
            } else {
                dataSetInformation->filename = cloudDataSetsDirectory + filenames.asString();
            }
        }

        Json::Value emissionFilenames = source["emission"];
        if (emissionFilenames.isArray()) {
            for (Json::Value::const_iterator filenameIt = emissionFilenames.begin();
                 filenameIt != emissionFilenames.end(); ++filenameIt) {
                bool isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(filenameIt->asString());
                if (isAbsolutePath) {
                    dataSetInformation->emission = filenameIt->asString();
                } else {
                    if (!filenameIt->asString().empty()) {
                        dataSetInformation->emission = cloudDataSetsDirectory + filenameIt->asString();
                    }
                }
            }
        } else {
            bool isAbsolutePath = sgl::FileUtils::get()->getIsPathAbsolute(emissionFilenames.asString());
            if (isAbsolutePath) {
                dataSetInformation->emission = emissionFilenames.asString();
            } else {
                if (!emissionFilenames.asString().empty()){
                    dataSetInformation->emission = cloudDataSetsDirectory + emissionFilenames.asString();
                }
            }
        }

        if (dataSetInformation->type == DATA_SET_TYPE_NODE) {
            dataSetInformationParent->children.emplace_back(dataSetInformation);
            processDataSetNodeChildren(source["children"], dataSetInformation);
            continue;
        }

        // Optional data: Transpose axes.
        if (source.isMember("axes")) {
            auto axesElement = source["axes"];
            int dim = 0;
            for (const auto& axisElement : axesElement) {
                dataSetInformation->axes[dim] = axisElement.asInt();
                dim++;
            }
        }

        // Optional data: Transform.
        dataSetInformation->hasCustomTransform = source.isMember("transform");
        if (dataSetInformation->hasCustomTransform) {
            glm::mat4 transformMatrix = parseTransformString(source["transform"].asString());
            dataSetInformation->transformMatrix = transformMatrix;
        }

        dataSetInformationParent->children.emplace_back(dataSetInformation);
    }
}

DataSetInformationPtr loadDataSetList(const std::string& filename) {
    // Parse the passed JSON file.
    std::ifstream jsonFileStream(filename.c_str());
    Json::CharReaderBuilder builder;
    JSONCPP_STRING errorString;
    Json::Value root;
    if (!parseFromStream(builder, jsonFileStream, &root, &errorString)) {
        sgl::Logfile::get()->writeError(errorString);
        return {};
    }
    jsonFileStream.close();

    DataSetInformationPtr dataSetInformationRoot(new DataSetInformation);
    Json::Value& dataSetNode = root["datasets"];
    processDataSetNodeChildren(dataSetNode, dataSetInformationRoot.get());
    return dataSetInformationRoot;
}
