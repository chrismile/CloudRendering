:: BSD 2-Clause License
::
:: Copyright (c) 2021-2022, Christoph Neuhauser, Felix Brendel
:: All rights reserved.
::
:: Redistribution and use in source and binary forms, with or without
:: modification, are permitted provided that the following conditions are met:
::
:: 1. Redistributions of source code must retain the above copyright notice, this
::    list of conditions and the following disclaimer.
::
:: 2. Redistributions in binary form must reproduce the above copyright notice,
::    this list of conditions and the following disclaimer in the documentation
::    and/or other materials provided with the distribution.
::
:: THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
:: AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
:: IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
:: DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
:: FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
:: DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
:: SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
:: CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
:: OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
:: OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

@echo off
setlocal
pushd %~dp0

set VSLANG=1033
set run_program=true
set debug=false
set devel=false
set clean=false
set build_dir=.build
set destination_dir=Shipping
set vcpkg_triplet="x64-windows"
:: Leave empty to let cmake try to find the correct paths
set optix_install_dir=""
set build_with_openvdb_support=false
set use_dlss=false

:loop
IF NOT "%1"=="" (
    IF "%1"=="--do-not-run" (
        SET run_program=false
    )
    IF "%1"=="--debug" (
        SET debug=true
    )
    IF "%1"=="--devel" (
        SET devel=true
    )
    IF "%1"=="--clean" (
        SET clean=true
    )
    IF "%1"=="--vcpkg-triplet" (
        SET vcpkg_triplet=%2
        SHIFT
    )
    IF "%1"=="--use-openvdb" (
        SET build_with_openvdb_support=true
    )
    IF "%1"=="--use-dlss" (
        SET use_dlss=true
    )
    SHIFT
    GOTO :loop
)

:: Clean the build artifacts if requested by the user.
if %clean% == true (
    echo ------------------------
    echo  cleaning up old files
    echo ------------------------
    rd /s /q "third_party\vcpkg"
    for /d %%G in (".build*") do rd /s /q "%%~G"
    rd /s /q "Shipping"

    git submodule update --init --recursive

:: https://stackoverflow.com/questions/5626879/how-to-find-if-a-file-contains-a-given-string-using-windows-command-line
    find /c "sgl" .gitmodules >NUL
    if %errorlevel% equ 1 goto sglnotfound
    rd /s /q "third_party\sgl\install"
    for /d %%G in ("third_party\sgl\.build*") do rd /s /q "%%~G"
)
goto cleandone
:sglnotfound
rd /s /q "third_party\sgl"
goto cleandone
:cleandone

where cmake >NUL 2>&1 || echo cmake was not found but is required to build the program && exit /b 1

:: Creates a string with, e.g., -G "Visual Studio 17 2022".
:: Needs to be run from a Visual Studio developer PowerShell or command prompt.
if defined VCINSTALLDIR (
    set VCINSTALLDIR_ESC=%VCINSTALLDIR:\=\\%
)
if defined VCINSTALLDIR (
    set "x=%VCINSTALLDIR_ESC:Microsoft Visual Studio\\=" & set "VsPathEnd=%"
)
if defined VisualStudioVersion (
    set "VsVersionNumber=%VisualStudioVersion:~0,2%"
) else (
    set VsVersionNumber=0
)
if defined VisualStudioVersion (
    if not defined VsPathEnd (
        if %VsVersionNumber% == 14 (
            set VsPathEnd=2015
        ) else if %VsVersionNumber% == 15 (
            set VsPathEnd=2017
        ) else if %VsVersionNumber% == 16 (
            set VsPathEnd=2019
        ) else if %VsVersionNumber% == 17 (
            set VsPathEnd=2022
        )
    )
)
if defined VsPathEnd (
    set cmake_generator=-G "Visual Studio %VisualStudioVersion:~0,2% %VsPathEnd:~0,4%"
) else (
    set cmake_generator=
)

if %debug% == true (
    set cmake_config="Debug"
    set cmake_config_opposite="Release"
) else (
    set cmake_config="Release"
    set cmake_config_opposite="Debug"
)

if not exist .\submodules\IsosurfaceCpp\src (
   echo ------------------------
   echo initializing submodules
   echo ------------------------
   git submodule init   || exit /b 1
   git submodule update || exit /b 1
)

if not exist .\third_party\ mkdir .\third_party\
set proj_dir=%~dp0
set third_party_dir=%proj_dir%third_party
pushd third_party

IF "%VULKAN_SDK%"=="" (
  for /D %%F in (C:\VulkanSDK\*) do (
    set VULKAN_SDK=%%F
    set "PATH=%%F\Bin;%PATH%"
    goto vulkan_finished
  )
)
:vulkan_finished

IF "%toolchain_file%"=="" (
    SET use_vcpkg=true
) ELSE (
    SET use_vcpkg=false
)
IF "%toolchain_file%"=="" SET toolchain_file="vcpkg/scripts/buildsystems/vcpkg.cmake"

set cmake_args=%cmake_args% -DCMAKE_TOOLCHAIN_FILE="third_party/%toolchain_file%" ^
               -Dsgl_DIR="third_party/sgl/install/lib/cmake/sgl/"

set cmake_args_general=%cmake_args_general% -DCMAKE_TOOLCHAIN_FILE="%third_party_dir%/%toolchain_file%"

if %use_vcpkg% == true (
    set cmake_args=%cmake_args% -DVCPKG_TARGET_TRIPLET=%vcpkg_triplet% -DCMAKE_CXX_FLAGS="/MP"
    set cmake_args_sgl=-DCMAKE_CXX_FLAGS="/MP"
    set cmake_args_general=%cmake_args_general% -DVCPKG_TARGET_TRIPLET=%vcpkg_triplet%
    if not exist .\vcpkg (
        echo ------------------------
        echo    fetching vcpkg
        echo ------------------------
        git clone --depth 1 https://github.com/microsoft/vcpkg.git || exit /b 1
        call vcpkg\bootstrap-vcpkg.bat -disableMetrics || exit /b 1
    )
)

if not exist .\sgl (
    echo ------------------------
    echo      fetching sgl
    echo ------------------------
    git clone --depth 1 https://github.com/chrismile/sgl.git   || exit /b 1
)

if not exist .\sgl\install (
    echo ------------------------
    echo      building sgl
    echo ------------------------
    mkdir sgl\%build_dir% 2> NUL
    pushd sgl\%build_dir%

    cmake .. %cmake_generator% %cmake_args_sgl% %cmake_args_general% ^
            -DCMAKE_INSTALL_PREFIX="%third_party_dir%/sgl/install" || exit /b 1
    if %use_vcpkg% == true (
        if x%vcpkg_triplet:release=%==x%vcpkg_triplet% (
           cmake --build . --config Debug   -- /m            || exit /b 1
           cmake --build . --config Debug   --target install || exit /b 1
        )
        if x%vcpkg_triplet:debug=%==x%vcpkg_triplet% (
           cmake --build . --config Release -- /m            || exit /b 1
           cmake --build . --config Release --target install || exit /b 1
        )
    ) else (
        cmake --build . --config %cmake_config%                  || exit /b 1
        cmake --build . --config %cmake_config% --target install || exit /b 1
    )

    popd
)

if %build_with_openvdb_support% == true (
    if not exist ".\openvdb" (
        echo ------------------------
        echo   downloading OpenVDB
        echo ------------------------
        :: Make sure we have no leftovers from a failed build attempt.
        if exist ".\openvdb-src" (
            rmdir /s /q ".\openvdb-src"
        )
        git clone --recursive https://github.com/AcademySoftwareFoundation/openvdb.git openvdb-src
        pushd "openvdb-src"
        git checkout a4705352e0e3ecb1f82eff2eca0c1b061ab7656b
        popd
        echo {> %third_party_dir%/openvdb-src/vcpkg.json
        echo "$schema": "https://raw.githubusercontent.com/microsoft/vcpkg/master/scripts/vcpkg.schema.json",>> %third_party_dir%/openvdb-src/vcpkg.json
        echo "name": "openvdb",>> %third_party_dir%/openvdb-src/vcpkg.json
        echo "version": "0.1.0",>> %third_party_dir%/openvdb-src/vcpkg.json
        echo "dependencies": [ "zlib", "blosc", "tbb", "boost-iostreams", "boost-any", "boost-algorithm", "boost-interprocess" ]>> %third_party_dir%/openvdb-src/vcpkg.json
        echo }>> %third_party_dir%/openvdb-src/vcpkg.json
        if not exist .\openvdb-src\build\ mkdir .\openvdb-src\build\
        pushd "openvdb-src\build"
        cmake %cmake_generator% %cmake_args_general% ^
        -DCMAKE_INSTALL_PREFIX="%third_party_dir%/openvdb" ..
        cmake --build . --config Release --target install || exit /b 1
        popd
    )
)
set cmake_args=%cmake_args% -DCMAKE_MODULE_PATH="third_party/openvdb/lib/cmake/OpenVDB"

if %use_dlss% == true (
    if not exist ".\DLSS" (
        echo ------------------------
        echo   downloading DLSS SDK
        echo ------------------------
        git clone https://github.com/NVIDIA/DLSS.git
    )
    set cmake_args=%cmake_args% -DUSE_DLSS=ON -DDLSS_SDK_ROOT="third_party/DLSS"
)

set oidn_version=2.3.3
if not exist ".\oidn-%oidn_version%.x64.windows" (
    echo ------------------------
    echo downloading OpenImageDenoise
    echo ------------------------
    curl.exe -L "https://github.com/OpenImageDenoise/oidn/releases/download/v%oidn_version%/oidn-%oidn_version%.x64.windows.zip" --output oidn-%oidn_version%.x64.windows.zip
    tar -xvzf "oidn-%oidn_version%.x64.windows.zip"
    del "oidn-%oidn_version%.x64.windows.zip"
)
set cmake_args=%cmake_args% -DOpenImageDenoise_DIR="third_party/oidn-%oidn_version%.x64.windows/lib/cmake/OpenImageDenoise-%oidn_version%"

popd

if %debug% == true (
    echo ------------------------
    echo   building in debug
    echo ------------------------
) else (
    echo ------------------------
    echo   building in release
    echo ------------------------
)

echo ------------------------
echo       generating
echo ------------------------

if not %optix_install_dir% == "" (
   echo Using custom OptiX path
   set cmake_args=%cmake_args% -DOptiX_INSTALL_DIR=%optix_install_dir%
)

cmake %cmake_generator% %cmake_args% -S . -B %build_dir%

echo ------------------------
echo       compiling
echo ------------------------
if %use_vcpkg% == true (
    cmake --build %build_dir% --config %cmake_config% -- /m || exit /b 1
) else (
    cmake --build %build_dir% --config %cmake_config%       || exit /b 1
)

echo ------------------------
echo    copying new files
echo ------------------------
robocopy third_party\oidn-%oidn_version%.x64.windows\bin %destination_dir% *.dll >NUL
if %build_with_openvdb_support% == true (
    robocopy third_party\openvdb-src\build\openvdb\openvdb\Release %destination_dir% *.dll >NUL
)
if %use_dlss% == true (
    robocopy third_party\DLSS\lib\Windows_x86_64\rel %destination_dir% *.dll >NUL
)
if %debug% == true (
    if not exist %destination_dir%\*.pdb (
        del %destination_dir%\*.dll
    )
    robocopy %build_dir%\Debug\ %destination_dir% >NUL
    robocopy third_party\sgl\%build_dir%\Debug %destination_dir% *.dll *.pdb >NUL
) else (
    if exist %destination_dir%\*.pdb (
        del %destination_dir%\*.dll
        del %destination_dir%\*.pdb
    )
    robocopy %build_dir%\Release\ %destination_dir% >NUL
    robocopy third_party\sgl\%build_dir%\Release %destination_dir% *.dll >NUL
)

:: Build other configuration and copy sgl DLLs to the build directory.
if %devel% == true (
    echo ------------------------
    echo   setting up dev files
    echo ------------------------
    if %use_vcpkg% == true (
        cmake --build %build_dir% --config %cmake_config_opposite% -- /m || exit /b 1
    ) else (
        cmake --build %build_dir% --config %cmake_config_opposite%       || exit /b 1
    )
    robocopy third_party\sgl\%build_dir%\Debug %build_dir%\Debug\ *.dll *.pdb >NUL
    robocopy third_party\sgl\%build_dir%\Release %build_dir%\Release\ *.dll >NUL
    robocopy third_party\oidn-%oidn_version%.x64.windows\bin %build_dir%\Debug\ *.dll *.pdb >NUL
    robocopy third_party\oidn-%oidn_version%.x64.windows\bin %build_dir%\Release\ *.dll >NUL
    if %build_with_openvdb_support% == true (
        robocopy third_party\openvdb\bin %build_dir%\Debug\ *.dll /XC /XN /XO >NUL
        robocopy third_party\openvdb\bin %build_dir%\Release\ *.dll /XC /XN /XO >NUL
        robocopy third_party\openvdb-src\build\openvdb\openvdb\Release %build_dir%\Debug\ *.dll /XC /XN /XO >NUL
        robocopy third_party\openvdb-src\build\openvdb\openvdb\Release %build_dir%\Release\ *.dll /XC /XN /XO >NUL
    )
    if %use_dlss% == true (
        robocopy third_party\DLSS\lib\Windows_x86_64\dev %build_dir%\Debug\ *.dll >NUL
        robocopy third_party\DLSS\lib\Windows_x86_64\rel %build_dir%\Release\ *.dll >NUL
    )
)

echo.
echo All done!

pushd %destination_dir%

if %run_program% == true (
    CloudRendering.exe
) else (
    echo Build finished.
)
