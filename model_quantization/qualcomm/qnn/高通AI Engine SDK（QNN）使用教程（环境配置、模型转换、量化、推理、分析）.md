﻿![在这里插入图片描述](./src/logo.png)

# 1. Introduction

## 1.1 什么是QNN

QNN是高通发布的神经网络推理引擎，是SNPE的升级版，其主要功能是：

- 完成从Pytorch/TensorFlow/Keras/Onnx等神经网络框架到高通计算平台的模型转换；
- 完成模型的低比特量化（int8），使其能够运行在高通神经网络芯片上；
- 提供测试工具（qnn-net-run），可以运行网络并保存输出；
- 提供测试工具（qnn-profile-viewer），可以进行FLOPS、参数量、每一层运行时间等分析；
  
  ## 1.2 QNN与SNPE的变化
- SNPE模型使用容器（DL container）格式保存，QNN模型使用cpp，json和bin文件保存；
- SNPE模型在运行前无需编译，可以直接在不同平台下运行，QNN的模型需要先编译到对应平台，才可以运行；
- SNPE模型转化（snpe-xxx-to-dlc）和模型量化（snpe-dlc-quantize）在QNN中被合并到一个步骤（qnn-xxx-converter）
  
  # 2. Linux端
  
  QNN的Linux端提供这些功能：

> qnn-accuracy-debugger         
> qnn-netron                
> qnn-platform-validator    
> qnn-tensorflow-converter        
> qnn-accuracy-evaluator        
> qnn-net-run               
> qnn-profile-viewer        
> qnn-tflite-converter            
> qnn-context-binary-generator  
> qnn-onnx-converter        
> qnn-pytorch-converter     
> qnn-throughput-net-run
> qnn-model-lib-generator       
> qnn-op-package-generator  
> qnn-quantization-checker

模型转化、模型量化、模型分析等操作需要在Linux下完成。

## 2.1 环境配置

### 2.1.1 拷贝文件

以qnn-2.13.2.230822为例，将QNN的SDK解压到服务器端，然后配置环境变量（仅在当前终端生效，新开终端需要再次配置）：

    export QNN_SDK_ROOT=/path/to/your/qnn-2.13.2.230822

执行初始化操作：

    source ${QNN_SDK_ROOT}/bin/envsetup.sh

输出：

    [INFO] AISW SDK environment set                                                                                    
    [INFO] QNN_SDK_ROOT: /xxx/xxx/qnn-2.13.2.230822 

### 2.1.2 配置Python环境

新建Conda环境（推荐）:

    conda create -n qnn python==3.8
    conda activate qnn

自动安装依赖：

    ${QNN_SDK_ROOT}/bin/check-python-dependency  

如果遇到Permission denied，需要把` ${QNN_SDK_ROOT}/bin`路径下的文件赋予可执行权限：

    chomd 777 ${QNN_SDK_ROOT}/bin/*

### 2.1.3 安装Linux依赖

执行：

    bash ${QNN_SDK_ROOT}/bin/check-linux-dependency.sh  

### 2.1.4 测试

执行

    qnn-net-run --version

输出：

    qnn-net-run pid:1947023                                                                                            
    QNN SDK v2.13.2.230822171732_60416 

配置成功

## 2.2 模型转化

onnx是通用的神经网络输出格式，以onnx模型格式的转换为例：

    qnn-onnx-converter -i model.onnx -o model.cpp

会生成 `model.cpp`、`model.bin`、`model_net.json` 三个文件。

## 2.3 模型量化

在QNN中，模型量化和模型转化被合并到一个操作，如果需要进行模型量化，只需要用 `--input_list`  参数指定一批量化数据。

需要注意的是，要进行模型量化，只需要指定`--input_list`即可，此时就算不指定其他量化参数，也会进行量化操作（见官方文档）

> The only required option to enable quantization along with conversion is the –input_list option, which provides the quantizer with the required input data for the given model.

    qnn-onnx-converter -i model.onnx -o qnn/model_q.cpp --input_list /path/to/your/input.txt

其中`input.txt`的文件格式与SNPE相同。

## 2.4 量化参数

量化参数主要通过运行`qnn-onnx-converter`时指定这些参数来指定：

- quantization_overrides：指定量化的参数

- param_quantizer：指定参数量化的方法。default：tf(即minmax量化)，还有enhanced/adjusted/symmetric可选

- act_quantizer：指定激活函数量化的方法（同上）

- algorithms：详见官方文档：
  
  > “cle” - Cross layer equalization includes a number of methods for equalizing weights and biases across layers in order to rectify imbalances that cause quantization errors. “bc” - Bias correction adjusts biases to offset activation quantization errors. Typically used in conjunction with “cle” to improve quantization accuracy.
  
  其他量化参数详见官方文档，不在此赘述。
  
  ## 2.5 模型编译
  
  如果需要运行QNN的模型，需要先对模型转化中生成的`model.cpp`、`model.bin`进行编译，编译时使用`-t`指定编译平台（默认编译`aarch64-android`和`x86_64-linux-clang平台`）：
  
  # for linux
  
    qnn-model-lib-generator -c model.cpp -b model.bin -o bin -t x86_64-linux-clang

需要注意的是，如果需要编译安卓运行的库（aarch64-android），需要先配置NDK环境。

## 2.6 NDK配置

将NDK文件上传到Linux服务器，并且解压（NDK使用android-ndk-r25c-linux版本），然后执行：

    export ANDROID_NDK_ROOT=/path/to/your/ndk/android-ndk-r25c
    PATH=$PATH:$ANDROID_NDK_ROOT

NDK配置完成后，执行：

    # for android
    qnn-model-lib-generator -c model.cpp -b model.bin -o bin -t "aarch64-android"

## 2.7 模型推理

使用`qnn-net-run`工具来进行模型推理，推理时需要用`--backend`参数指定推理后端的库，有如下参数可选：

- CPU - libQnnCpu.so
- HTP (Hexagon v68) - libQnnHtp.so
- Saver - libQnnSaver.so

以使用HTP（模拟）为例，运行：

    qnn-net-run --model bin/x86_64-linux-clang/libmodel.so --input_list /path/to/your/input.txt --backend "${QNN_SDK_ROOT}/lib/x86_64-linux-clang/libQnnHtp.so"

模型输出会默认保存到`output`文件夹。

# 3. Android端

QNN安卓端提供如下工具：

> qnn-context-binary-generator 
> qnn-net-run  
> qnn-platform-validator  
> qnn-profile-viewer  
> qnn-throughput-net-run

基本只能进行模型推理和计算平台的验证。

## 3.1 环境配置

### 3.1.1 拷贝文件

将文件push到安卓设备上：

    adb shell mkdir -p /data/local/tmp/qnn2/arm64/lib
    adb shell mkdir -p /data/local/tmp/qnn2/arm64/bin
    
    adb push D:\Program\qnn-2.13.2.230822\lib\aarch64-android\. /data/local/tmp/qnn2/arm64/lib
    adb push D:\Program\qnn-2.13.2.230822\bin\aarch64-android\. /data/local/tmp/qnn2/arm64/bin

### 3.1.2 设置环境变量

每次重启终端都需要设置

    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/data/local/tmp/qnn2/arm64/lib
    export PATH=$PATH:/data/local/tmp/qnn2/arm64/bin

### 3.1.3 测试

    qnn-net-run --version

输出

    qnn-net-run pid:12495
    QNN SDK v2.13.2.230822171732_60416

如果报错

> /system/bin/sh: qnn-net-run: can't execute: Permission denied

需要先赋予可执行权限

    chmod 777 /data/local/tmp/qnn2/arm64/bin/qnn-net-run

## 3.2 CPU/GPU推理

首先将模型和数据Push到设备上，然后执行：

    # cpu
    qnn-net-run --model models/qnn/libmodel.so --input_list /path/to/your/input.txt --backend /data/local/tmp/qnn2/arm64/lib/libQnnCpu.so --output_dir outputs/qnn
    
    # gpu
    qnn-net-run --model models/qnn/libmodel.so --input_list /path/to/your/input.txt --backend /data/local/tmp/qnn2/arm64/lib/libQnnGpu.so --output_dir outputs/qnn

其中`--backend`参数可以指定的值如下：

- CPU - libQnnCpu.so
- GPU - libQnnGpu.so
- HTA - libQnnHta.so
- DSP (Hexagon v65) - libQnnDspV65Stub.so
- DSP (Hexagon v66) - libQnnDspV66Stub.so
- DSP - libQnnDsp.so
- HTP (Hexagon v68) - libQnnHtp.so
- [Deprecated] HTP Alternate Prepare (Hexagon v68) - libQnnHtpAltPrepStub.so
- Saver - libQnnSaver.so

如果需要使用HTP或者DSP推理，需要Push额外的lib文件。

## 3.3 DSP推理

高通对不同芯片，支持DSP的库版本不同：SM7325使用hexagon-v68，SM8475使用hexagon-v69。

    # v68
    adb push D:\Program\qnn-2.13.2.230822\lib\hexagon-v68\unsigned\. /data/local/tmp/qnn2/dsp/lib
    # v69
    adb push D:\Program\qnn-2.13.2.230822\lib\hexagon-v69\unsigned\. /data/local/tmp/qnn2/dsp/lib

设置环境变量：

    export ADSP_LIBRARY_PATH="/data/local/tmp/qnn2/dsp/lib;/system/vendor/lib/rfsa/adsp"

执行模型推理：

    qnn-net-run --model models/qnn/libmodel.so --input_list /path/to/your/input.txt --backend /data/local/tmp/qnn2/arm64/lib/libQnnHtp.so --output_dir outputs/qnn

对于推理后端的支持，查看`examples/QNN/NetRun/android/android-qnn-net-run.sh`文件可以发现，v68、v69和v73使用的是`libQnnHtp.so`，v66使用的是`libQnnDspV66Stub.so`，v65使用的是`libQnnDspV65Stub.so`，使用时需要Push相应的库。

# 4. 结果分析

在运行`qnn-net-run`时指定`--profiling_level`参数：

    qnn-net-run --model models/qnn/libmodel.so --input_list /path/to/your/input.txt --backend /data/local/tmp/qnn2/arm64/lib/libQnnHtp.so --output_dir outputs/qnn ----profiling_level basic

> 1. basic:    captures execution and init time.
> 2. detailed: in addition to basic, captures per Op timing for execution, if a backend supports it.
> 3. client:   captures only the performance metrics measured by qnn-net-run.

会在output文件夹下生成profile文件，在Linux下，可以使用`qnn-profile-viewer`对profile文件进行分析:

    qnn-profile-viewer --input_log qnn-profiling-data_0.log

生成分析的结果：

```powershell
Qnn Init/Prepare/Finalize/De-Init/Execute/Lib-Load Statistics:

--- Init Stats:

---
    NetRun:  331158 us

Compose Graphs Stats:

---
    NetRun:  74748 us

Finalize Stats:

--- Graph 0 (model):
    NetRun:  256372 us
    Backend (RPC (finalize) time): 4922 us
    Backend (QNN accelerator (finalize) time): 4504 us
    Backend (Accelerator (finalize) time): 4421 us
    Backend (QNN (finalize) time): 256364 us

De-Init Stats:

---
    NetRun:  15810 us
    Backend (RPC (deinit) time): 1224 us
    Backend (QNN Accelerator (deinit) time): 966 us
    Backend (Accelerator (deinit) time): 934 us
    Backend (QNN (deinit) time): 15801 us

Execute Stats (Average):

--- Total Inference Time: 

--- Graph 0 (model):
    NetRun:  2952 us
    Backend (Number of HVX threads used): 4 count
    Backend (RPC (execute) time): 2920 us
    Backend (QNN accelerator (execute) time): 2600 us
    Backend (Accelerator (execute) time): 2492 us
    Backend (QNN (execute) time): 2949 us

Execute Stats (Overall):

---
    NetRun IPS:  214.8888 inf/sec
```

对算法来说，关心的是推理时间，此处的推理时间是平均后的，为**2.952ms**。

对CPU推理的profile进行分析，平均推理时间为**148.259ms**，量化后的模型加速效果显著。
