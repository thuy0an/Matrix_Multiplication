@echo off
echo [INFO] Dang bien dich ma nguon moi nhat...
where nvcc >nul 2>nul
if %ERRORLEVEL% EQU 0 goto CUDA_BUILD
g++ -std=c++17 -O3 -fopenmp -mavx2 -mfma -Iinclude src/main.cpp src/matrix.cpp src/algorithms.cpp src/excel_export.cpp src/cuda_backend_stub.cpp -o matmul_benchmark.exe
if %ERRORLEVEL% EQU 0 goto SUCCESS
goto BUILD_FAILED

:CUDA_BUILD
echo [INFO] Phat hien nvcc, dang build backend CUDA + OpenMP...
set "VSDEV=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\Tools\VsDevCmd.bat"
if exist "%VSDEV%" call "%VSDEV%" -arch=x64 >nul
if not defined CUDA_ARCH set "CUDA_ARCH=sm_75"
nvcc -arch=%CUDA_ARCH% -std=c++17 -O3 -Xcompiler /openmp:experimental -Iinclude src/main.cpp src/matrix.cpp src/algorithms.cpp src/excel_export.cpp src/cuda_backend.cu -o matmul_benchmark.exe
if %ERRORLEVEL% EQU 0 goto SUCCESS

:BUILD_FAILED

echo.
echo [CANH BAO] Khong the ghi de matmul_benchmark.exe, co the cua so cu dang bat.
echo Dang khoi dong chuong trinh...
echo.

:SUCCESS
matmul_benchmark.exe
pause
