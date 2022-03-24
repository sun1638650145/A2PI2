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

