Build ncnn for VS2015:
# if use cmake-gui: set entry "CMAKE_INSTALL_PREFIX" as "install", not "%cd%/install".
################################################################################
1.Build protobuf library:
    Download protobuf-3.4.0: https://github.com/google/protobuf/archive/v3.4.0.zip
    > cd <protobuf-root-dir>
    > mkdir build_vs2015
    > cd build_vs2015
    > cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake
    > nmake
    > nmake install

2.Build ncnn library
    > cd <ncnn-root-dir>
    > mkdir -p build_vs2015
    > cd build_vs2015
    > cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -DProtobuf_INCLUDE_DIR=<protobuf-root-dir>/build_vs2015/install/include -DProtobuf_LIBRARIES=<protobuf-root-dir>/build_vs2015/install/lib/libprotobuf.lib -DProtobuf_PROTOC_EXECUTABLE=<protobuf-root-dir>/build_vs2015/install/bin/protoc.exe -DNCNN_VULKAN=OFF ..
    > nmake
    > nmake install

3.Pick *.exe from <build_vs2015/tools> for model conversion.

4.Configure VS2015 project(Release,x64):
    VC++ Directories->Inlude Directories:
        D:/OpenCV/include
        D:/OpenCV/include/opencv
        D:/OpenCV/include/opencv2
        <ncnn-root-dir>/build_vs2015/install/include/ncnn

    VC++ Directories->Library Directories:
        D:/OpenCV/x64/vc14/lib
        <ncnn-root-dir>/build_vs2015/install/lib

    Linker->Input->Additional Dependencies:
        opencv_world330.lib
        ncnn.lib


Build ncnn for MinGW:
# install MinGW and add <MinGW-root-dir/bin> to PATH.
# if use cmake-gui: set entry "CMAKE_INSTALL_PREFIX" as "install", not "%cd%/install".
################################################################################
1.Build protobuf library:
    Download protobuf-3.4.0: https://github.com/google/protobuf/archive/v3.4.0.zip
    > cd <protobuf-root-dir>
    > mkdir build_gw
    > cd build_gw
    > cmake -G"MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -Dprotobuf_BUILD_TESTS=OFF -Dprotobuf_MSVC_STATIC_RUNTIME=OFF ../cmake
    > mingw32-make
    > mingw32-make install

2.Build ncnn library
    > cd <ncnn-root-dir>
    > mkdir -p build_gw
    > cd build_gw
    > cmake -G"NMake Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=%cd%/install -DProtobuf_INCLUDE_DIR=<protobuf-root-dir>/build_gw/install/include -DProtobuf_LIBRARIES=<protobuf-root-dir>/build_gw/install/lib/libprotobuf.a -DProtobuf_PROTOC_EXECUTABLE=<protobuf-root-dir>/build_gw/install/bin/protoc.exe -DNCNN_VULKAN=OFF ..
    > mingw32-make
    > mingw32-make install

3.Further usage: <build_gw/install>, <build_gw/tools>.


################################################################################
