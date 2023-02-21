@echo off

SET prefix=D:/a/ODM/ODM/SuperBuild/install
SET exec_prefix=D:/a/ODM/ODM/SuperBuild/install/bin
SET libdir=D:/a/ODM/ODM/SuperBuild/install/lib


IF "%1" == "--libs" echo -LD:/a/ODM/ODM/SuperBuild/install/lib -lpdalcpp & goto exit
IF "%1" == "--plugin-dir" echo D:/a/ODM/ODM/SuperBuild/install/bin & goto exit
IF "%1" == "--prefix" echo %prefix% & goto exit
IF "%1" == "--ldflags" echo -L%libdir% & goto exit
IF "%1" == "--defines" echo  & goto exit
IF "%1" == "--includes" echo -ID:/a/ODM/ODM/SuperBuild/install/include -ID:/a/ODM/ODM/venv/Lib/site-packages/osgeo/include/gdal -ID:/a/ODM/ODM/vcpkg/installed/x64-windows/include -ID:/a/ODM/ODM/SuperBuild/install/include & goto exit
IF "%1" == "--cflags" echo /DWIN32 /D_WINDOWS /W3 & goto exit
IF "%1" == "--cxxflags" echo /DWIN32 /D_WINDOWS /W3 /GR /EHsc -std=c++11 & goto exit
IF "%1" == "--version" echo 2.3.0 & goto exit


echo Usage: pdal-config [OPTIONS]
echo Options:
echo    [--cflags]
echo    [--cxxflags]
echo    [--defines]
echo    [--includes]
echo    [--libs]
echo    [--plugin-dir]
echo    [--version]

:exit
