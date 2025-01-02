// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Contributors to the OpenEXR Project.

#pragma once

//
// Dealing with FPEs
//

// automated formatting does not handle the cmake tags well
// clang-format off

#ifdef __linux__
#define HAVE_UCONTEXT_H 1
#define IEX_HAVE_CONTROL_REGISTER_SUPPORT 1
#endif
/* #undef IEX_HAVE_SIGCONTEXT_CONTROL_REGISTER_SUPPORT */

// clang-format on
