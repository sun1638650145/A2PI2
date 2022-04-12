# <center>A²PI²-CUDA version2.2</center>

* 包含CUDA生态的软件包和使用教程.

# 1.CUDA

| 版本 |           描述           | 注意 |
| :--: | :----------------------: | :--: |
| 11.2 | 用于构建GPU加速应用程序. |  -   |

## 1.1.第一个例子

调用GPU输出一些信息.

```c++
#include <cstdio>

// 声明核函数在GPU上运行.
__global__ void HelloWorld() {
    printf("Hello, World!");
}

int main() {
    HelloWorld<<<1, 1>>>(); // 配置内含1个线程的1个线程块.
    cudaDeviceSynchronize(); // 同步数据.

    return 0;
}
```

## 1.2.操作线程`thread`, 线程块`block`和网格`grid`

### 1.2.1.对线程和线程块的基本操作.

```c++
#include <iostream>

#define N 20

// 初始化向量.
__global__ void InitializationVector(int *vector, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    
    for (int i = index; i < n; i += stride) {
        vector[i] = i;
    }
}

// 计算向量加法.
__global__ void Add(int *vector0, int *vector1, int *vector2, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        vector2[i] = vector0[i] + vector1[i];
    }
}

// 打印向量.
void PrintVector(int *vector, int n) {
    for (int i = 0; i < n; i ++) {
        std::cout << vector[i] << " ";
    }
}

int main() {
    int *a, *b, *c;
    size_t size = N * sizeof(int);

    // 分配UM内存.
    cudaMallocManaged(&a, size);
    cudaMallocManaged(&b, size);
    cudaMallocManaged(&c, size);

    // 配置线程块和线程.
    dim3 threadsPerBlock(32, 1, 1);
    dim3 numOfBlocks(32, 1, 1);
    
    InitializationVector<<<numOfBlocks, threadsPerBlock>>>(a, N);
    InitializationVector<<<numOfBlocks, threadsPerBlock>>>(b, N);
    Add<<<numOfBlocks, threadsPerBlock>>>(a, b, c, N);
    cudaDeviceSynchronize();

    PrintVector(c, N);

    // 释放UM内存.
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
```

### 1.2.2.使用二维线程和线程块操作.

```c++
#include <iostream>

#define N 10

// 初始化矩阵.
__global__ void InitializationMatrix(int *matrix, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    if (i < n and j < n) {
        matrix[i * n + j] = i * n + j;
    }
}

// 打印矩阵.
void PrintMatrix(int *matrix, int n) {
    for (int i = 0; i < n; i ++) {
        for (int j = 0; j < n; j ++) {
            std::cout << matrix[i * n + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    int *a;
    size_t size = N * N * sizeof(int);

    // 分配UM内存.
    cudaMallocManaged(&a, size);

    // 配置线程块和线程.
    dim3 threadsPerBlock(10, 10, 1);
    dim3 numOfBlocks(10, 10, 1);

    InitializationMatrix<<<numOfBlocks, threadsPerBlock>>>(a, N);
    cudaDeviceSynchronize();

    PrintMatrix(a, N);

    // 释放UM内存.
    cudaFree(a);

    return 0;
}
```

## 1.3.异常处理

```c++
#include <cassert>
#include <cstdio>
#include <iostream>

// 对有返回值的cuda函数进行装饰处理异常.
inline cudaError_t errorWrapper(cudaError_t err) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA运行时错误: %s\n", cudaGetErrorString(err));
    }
    return err;
}

// 对无返回值的cuda函数进行处理异常.
inline cudaError_t errorVoid() {
    cudaError_t err = cudaGetLastError(); // 获取函数调用时的错误.
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA运行时错误: %s\n", cudaGetErrorString(err));
    }
    return err;
}

// 初始化向量.
__global__ void InitVector(int *vector, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    for (int i = index; i < n; i ++) {
        vector[i] = i;
    }
}

// 打印向量.
void PrintVector(int *vector, int n) {
    for (int i = 0; i < n; i ++) {
        std::cout << vector[i] << " ";
    }
}

int main() {
    int *a;
    size_t size = 10 * sizeof(int);

    errorWrapper(cudaMallocManaged(&a, size)); // 有返回值的函数使用errorWrapper()处理.

    InitVector<<<10, -1>>>(a, 10); // // 没有返回值的函数使用errorVoid()处理.
    errorVoid();
    errorWrapper(cudaDeviceSynchronize());

    PrintVector(a, 10);
  
    errorWrapper(cudaFree(a));

    return 0;
}
```

## 1.4.查询设备`device`信息

```c++
#include <iostream>

int main() {
    int deviceId;
    cudaGetDevice(&deviceId); // 获取设备Id.
    std::cout << "设备Id: " << deviceId << std::endl;

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId); // 获取设备全部属性集, 注意这种时间开销很大.
    std::cout << "计算能力: " << props.major << "." << props.minor
              << ", 流处理器SM的数量: " << props.multiProcessorCount
              << ", warp的大小: " << props.warpSize << std::endl;

    int numberOfSMs;
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);  // 获取设备某个具体属性.
    std::cout << "流处理器SM的数量: " << numberOfSMs << std::endl;

    return 0;
}
```

## 1.5.在主机`host`和设备`device`之间异步迁移内存

### 1.5.1.从GPU异步迁移到CPU.

```c++
#include <iostream>

#define N 10

// 在GPU上初始化向量.
__global__ void InitializeVectorOnGPU(int *vector, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        vector[i] = i;
    }
}

// 在CPU上向量加法.
void AddOnCPU(int *vector0, int *vector1, int n) {
    for (int i = 0; i < n; i ++) {
        vector0[i] += vector1[i];
    }
}

// 打印向量.
void PrintVector(int *vector, int n) {
    for (int i = 0; i < n; i ++) {
        std::cout << vector[i] << " ";
    }
}

int main() {
    int *a;
    size_t size = N * sizeof(int);

    cudaMallocManaged(&a, size);

    InitializeVectorOnGPU<<<10, 1>>>(a, N);
    cudaMemPrefetchAsync(a, size, cudaCpuDeviceId);  // 将数据异步预取回CPU, 减少页错误.
    cudaDeviceSynchronize();

    AddOnCPU(a, a, N);
    PrintVector(a, N);

    cudaFree(a);

    return 0;
}
```

### 1.5.2.从CPU异步迁移到GPU.

```c++
#include <iostream>

#define N 10

// 在CPU上初始化向量.
void InitializeVectorOnCPU(int *vector, int n) {
    for (int i = 0; i < n; i ++) {
        vector[i] = i;
    }
}

// 在GPU上向量加法.
__global__ void AddOnGPU(int *vector0, int *vector1, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        vector0[i] += vector1[i];
    }
}

// 打印向量.
void PrintVector(int *vector, int n) {
    for (int i = 0; i < n; i ++) {
        std::cout << vector[i] << " ";
    }
}

int main() {
    int *a;
    size_t size = N * sizeof(int);

    int deviceId;
    cudaGetDevice(&deviceId);

    cudaMallocManaged(&a, size);

    InitializeVectorOnCPU(a, N);
    cudaMemPrefetchAsync(a, size, deviceId);  // 将数据异步预取到GPU, 减少页错误.
    AddOnGPU<<<10, 1>>>(a, a, N);
    cudaDeviceSynchronize();

    PrintVector(a, N);

    cudaFree(a);

    return 0;
}
```

## 1.6.使用非默认流

```c++
#include <cstdio>

__global__ void Print(int number) {
    std::printf("%d ", number);
}

int main() {
    for (int i = 0; i < 5; i ++) {
        cudaStream_t stream;
        cudaStreamCreate(&stream); // 创建非默认流.

        Print<<<1, 1, 0, stream >>>(i);
        cudaDeviceSynchronize();

        cudaStreamDestroy(stream); // 销毁非默认流.
    }
    return 0;
}
```

## 1.7.手动管理内存

### 1.7.1.在GPU和CPU上分配内存.

```c++
#include <iostream>

#define N 10

// 在GPU上初始化向量.
__global__ void InitializeVectorOnGPU(int *vector, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        vector[i] = i;
    }
}

// 在GPU上向量加法.
__global__ void AddOnGPU(int *vector0, int *vector1, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        vector0[i] += vector1[i];
    }
}

// 打印向量.
void PrintVector(int *vector, int n) {
    for (int i = 0; i < n; i ++) {
        std::cout << vector[i] << " ";
    }
}

int main() {
    int *a, *host_a;
    size_t size = N * sizeof(int);

    cudaMalloc(&a, size); // 在GPU手动分配内存.
    cudaMallocHost(&host_a, size); // 在CPU手动分配内存.

    InitializeVectorOnGPU<<<10, 1>>>(a, N);
    AddOnGPU<<<10, 1>>>(a, a, N);

    cudaMemcpy(host_a, a, size, cudaMemcpyDeviceToHost); // 将数据从GPU拷贝到CPU.

    PrintVector(host_a, N);

    cudaFree(a);
    cudaFree(host_a); // 释放CPU内存.

    return 0;
}
```

### 1.7.2.异步拷贝内存.

```c++
#include <cstdio>
#include <iostream>

#define N 30

// 在GPU上初始化向量.
__global__ void InitializeVectorOnGPU(int *vector, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        vector[i] = i;
    }
}

// 在GPU上向量加法.
__global__ void AddOnGPU(int *vector, int n) {
    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = index; i < n; i += stride) {
        vector[i] *= 2;
    }
}

// 打印向量.
void PrintVector(int *vector, int n) {
    for (int i = 0; i < n; i ++) {
        std::printf("%2d ", vector[i]);
        if ((i + 1) % 5 == 0) {
            std::cout << std::endl;
        }
    }
}

int main() {
    int *a, *host_a;
    size_t size = N * sizeof(int);

    cudaMalloc(&a, size);
    cudaMallocHost(&host_a, size);

    InitializeVectorOnGPU<<<32, 1>>>(a, N);

    int numberOfSegments = 5; // 分块数需要是大小的因子.
    int segmentN = N / numberOfSegments; // 每个分块的大小.
    size_t segmentSize = size / numberOfSegments; // 每个分块的内存大小.

    for (int segment = 0; segment < numberOfSegments; segment ++) {
        int segmentOffset = segment * segmentN; // 计算每个分块的偏移量.

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        AddOnGPU<<<32, 1, 0, stream>>>(&a[segmentOffset], segmentN);
        // 将数据异步从GPU拷贝到CPU.
        cudaMemcpyAsync(&host_a[segmentOffset],
                        &a[segmentOffset],
                        segmentSize,
                        cudaMemcpyDeviceToHost,
                        stream);

        cudaStreamDestroy(stream);
    }

    PrintVector(host_a, N);

    cudaFree(a);
    cudaFreeHost(host_a);

    return 0;
}
```

# 2.cuBLAS

## 2.1.cublas\<t\>gemm()

矩阵乘法.

```c++
#include <cstdio>
#include <iostream>

#include "cublas_v2.h"

// 矩阵的维度.
#define M 2
#define K 3
#define N 4

// 初始化矩阵.
void initializeMatrix(float *mat, int m, int n) {
    for (int i = 0; i < m; i ++) {
        for (int j = 0; j < n; j ++) {
            mat[i * n + j] = i * n + j;
        }
    }
}

// 打印矩阵.
void printMatrix(float *mat, int m, int n) {
    for (int i = 0; i < m; i ++) {
        for (int j = 0; j < n; j ++) {
            std::printf("%2.0f ", mat[i * n + j]);
        }
        std::cout << std::endl;
    }
    std::cout << "---------------" << std::endl;
}

// 矩阵乘法.
void matrixMultiply(float *mat_a, size_t a,
                    float *mat_b, size_t b,
                    float *mat_c, size_t c,
                    float *mat_ct, size_t ct) {
    // 获取矩阵的维度信息.
    int m = M, k = K, n = N;

    // 创建cuBLAS状态变量.
    cublasStatus_t status;

    // 初始化cuBLAS句柄.
    cublasHandle_t handle;
    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        if (status == CUBLAS_STATUS_NOT_INITIALIZED) {
            std::cout << "cuBLAS句柄创建失败" << std::endl;
        }
    }

    /* 计算矩阵乘法. cublas<t>gemm()
     * 数学逻辑公式: C = \alpha op(A)op(B) + \beta C
     * 显存物理公式: 通常令 alpha = 1; beta = 0
     *             M = AB
     *             M^T = B^TA^T
     *             M = C^T
     *             M^T = C
     *
     * 参数:
     *      1. handle句柄.
     *      2/3. op:   A = CUBLAS_OP_N
     *               A^T = CUBLAS_OP_T
     *               A^H = CUBLAS_OP_C
     *      4/5. M行, 列
     *      6.k: A列(B行).
     *      7. alpha的指针.
     *      8/9. 矩阵(显存)和lda.
     *      10/11. 矩阵(显存)和ldb.
     *      12. beta的指针.
     *      13/14. M(显存)和ldm.
     *
     * lda, ldb: CUBLAS_OP_N 行.
     *           CUBLAS_OP_T 原始矩阵的列(不转置).
     * ldc: M的行
     */
    float alpha = 1, beta = 0;

    // 显存存储是A^T和B^T
    // M = B^TA^T
    // 显存存M^T取出就是C
    cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                n, m,
                k,
                &alpha,
                mat_b, n,  // B(n, k)
                mat_a, k,  // A(k, m)
                &beta,
                mat_c, n); // M(n, m)

    // 显存存储是A^T和B^T
    // 转置后存储A和B
    // M = AB
    // 显存存M取出就是C^T
    cublasSgemm(handle,
                CUBLAS_OP_T, CUBLAS_OP_T,
                m, n,
                k,
                &alpha,
                mat_a, k,   // A(m, k)
                mat_b, n,   // B(k, n)
                &beta,
                mat_ct, m); // M(m, n)

    // 同步数据.
    cudaDeviceSynchronize();

    // 销毁cuBLAS句柄.
    cublasDestroy(handle);
}

int main() {
    float *mat_a, *mat_b, *mat_c, *mat_ct;
    size_t size_a = M * K * sizeof(float);
    size_t size_b = K * N * sizeof(float);
    size_t size_c = M * N * sizeof(float);
    size_t size_ct = N * M * sizeof(float);

    // 分配UM内存.
    cudaMallocManaged(&mat_a, size_a);
    cudaMallocManaged(&mat_b, size_b);
    cudaMallocManaged(&mat_c, size_c);
    cudaMallocManaged(&mat_ct, size_ct);

    initializeMatrix(mat_a, M, K);
    initializeMatrix(mat_b, K, N);
    std::cout << "矩阵A:" << std::endl;
    printMatrix(mat_a, M, K);
    std::cout << "矩阵B:" << std::endl;
    printMatrix(mat_b, K, N);

    matrixMultiply(mat_a, size_a,
                   mat_b, size_b,
                   mat_c, size_c,
                   mat_ct, size_ct);

    std::cout << "矩阵C:" << std::endl;
    printMatrix(mat_c, M, N);
    std::cout << "矩阵C的转置:" << std::endl;
    printMatrix(mat_ct, N, M);

    // 释放UM内存.
    cudaFree(mat_a);
    cudaFree(mat_b);
    cudaFree(mat_c);
    cudaFree(mat_ct);

    return 0;
}
```

# 3.CUDA命令

## 3.1.nsight-sys

在Linux上启动`Nsight Systems`工具.

```shell
nsight-sys
```

## 3.2.nsys

用于分析CUDA程序.

```shell
nsys profile --force-overwrite=true --stats=true -o report.qdrep ./out
```

* `--force-overwrite/-f` 覆盖原有的`qdrep`报告文件.
* `--stats` 是否将摘要信息输出到终端.
* `-o` 指定`qdrep`报告文件名.

## 3.3.nvcc

基本与`gcc`的使用方式相同, 用于编译CUDA C/C++源代码.

```shell
nvcc -arch=sm_70 -lcublas -o out main.cu -run
```

* `-arch` 指定编译的GPU架构类型.
* `-l` 添加库的位置.
* `-run` 编译成功后直接执行二进制文件.

## 3.4.nvidia-smi

`nvidia-smi`(Systems Management Interface)用于查询加速系统内的GPU信息.

```shell
nvidia-smi
```

