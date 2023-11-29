/*
 * Copyright (c) 2014-2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-FileCopyrightText: Copyright (c) 2014-2021 NVIDIA CORPORATION
 * SPDX-License-Identifier: Apache-2.0
 */

void ComputeDefaultBasis(const vec3 normal, out vec3 x, out vec3 y)
{
    // ZAP's default coordinate system for compatibility
    vec3        z  = normal;
    const float yz = -z.y * z.z;
    y = normalize(((abs(z.z) > 0.99999f) ? vec3(-z.x * z.y, 1.0f - z.y * z.y, yz) : vec3(-z.x * z.z, yz, 1.0f - z.z * z.z)));

    x = cross(y, z);
}
