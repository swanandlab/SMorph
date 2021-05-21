@echo off
title Cell Viewer
FOR %%I IN (%*) DO (
    ECHO.%%~aI | FIND "d" >NUL
    IF ERRORLEVEL 1 (
        REM Processing Dropped Files
        ECHO "%%~fI"
    )
)
python -m napari %* --layer-type image
pause
