# Flash Attention SM90 Sparse Mask 技术详解

> 本文档详细介绍 Flash Attention 3 (SM90) 中 Sparse Mask 的实现原理，包括 GMMA 线程布局、Tile 排列、每个线程持有的行列计算方法，以及如何优化 Mask 存储布局以最小化访存和寄存器占用。

---

## 第一章：概述

### 1.1 Flash Attention 3 架构概述

Flash Attention 3 是专门为 NVIDIA Hopper 架构 (SM90) 优化的注意力机制实现。它利用了 Hopper 架构的以下关键特性：

- **TMA (Tensor Memory Accelerator)**：异步数据加载，减少内存访问延迟
- **WGMMA (Warp Group Matrix Multiply-Accumulate)**：高效的矩阵乘法指令
- **异步执行**：生产者-消费者模型，实现计算与内存访问的重叠

在 Flash Attention 中，注意力计算被分解为多个 Tile，每个 Tile 的大小由 `kBlockM`（Query 维度）和 `kBlockN`（Key/Value 维度）决定。

### 1.2 Sparse Mask 的应用场景

Sparse Mask（稀疏掩码）用于实现细粒度的注意力模式控制，典型应用包括：

- **Top-K 稀疏注意力**：只保留每个 Query 位置的 Top-K 个 Key 位置
- **Block Sparse Attention**：按块划分的稀疏注意力模式
- **自定义注意力模式**：任意用户定义的稀疏连接模式

与 Causal Mask 或 Local Mask 不同，Sparse Mask 需要显式存储每个位置的掩码信息，因此其存储布局和访问模式对性能有显著影响。

### 1.3 本文目标

本文将详细解答以下问题：

1. **Tile 如何排列？** —— `kBlockM`、`kBlockN` 如何决定 Tile 的形状
2. **每个线程持有哪些元素？** —— GMMA 的线程到元素映射规则
3. **如何计算行列坐标？** —— `tScS_rowcol`、`local_row`、`token_in_block` 的计算
4. **Mask 如何存储和访问？** —— 当前实现的访问模式分析
5. **如何优化？** —— 重排 Mask 以减少每线程访存量的方案

### 1.4 代码文件索引

本文涉及的主要代码文件：

| 文件 | 内容 |
|------|------|
| `hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp` | 主循环实现，定义 TileShape、TiledMma 等 |
| `hopper/flash_fwd_kernel_sm90.h` | 前向 Kernel 入口，定义 NumMmaWarpGroups 等 |
| `hopper/mask.h` | Mask 和 SparseMask 类的实现 |
| `hopper/utils.h` | `convert_layout_acc_rowcol` 等工具函数 |
| `hopper/softmax.h` | Softmax 实现，使用相同的布局转换 |

### 1.5 符号约定

| 符号 | 含义 | 典型值 |
|------|------|--------|
| `kBlockM` | Query Tile 的行数 | 64, 128 |
| `kBlockN` | Key/Value Tile 的列数 | 128 |
| `kHeadDim` | 注意力头维度 | 64, 128, 192, 256 |
| `NumMmaWarpGroups` | MMA Warp Group 数量 | 1, 2, 3 |
| `V` | 每个线程在 N 方向的块数 | kBlockN / 8 |
| `MMA_M` | M 方向的 Atom 数量 | kBlockM / 64 |
| `MMA_N` | N 方向的 Atom 数量 | 通常为 1 |

---

## 第二章：GMMA (Warp Group Matrix Multiply-Accumulate) 基础

### 2.1 SM90 GMMA 指令简介

GMMA（Warp Group Matrix Multiply-Accumulate）是 NVIDIA Hopper 架构引入的新一代矩阵乘法指令。与 Ampere 的 WMMA/MMA 不同，GMMA 由一个完整的 **Warp Group**（128 个线程，即 4 个 Warp）协同执行。

**GMMA 的关键特性：**

```
┌─────────────────────────────────────────────────────────────────┐
│                    GMMA 执行单元                                 │
│                                                                 │
│   Warp Group = 4 Warps = 128 Threads                           │
│   ┌─────────┬─────────┬─────────┬─────────┐                    │
│   │ Warp 0  │ Warp 1  │ Warp 2  │ Warp 3  │                    │
│   │ (32 thr)│ (32 thr)│ (32 thr)│ (32 thr)│                    │
│   └─────────┴─────────┴─────────┴─────────┘                    │
│                                                                 │
│   单条 GMMA 指令可计算: M64/M96 × N128/N256 × K16/K32          │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 GMMA 的典型配置

对于 BF16/FP16 数据类型，常用的 GMMA 配置：

| 配置 | M | N | K | 说明 |
|------|---|---|---|------|
| `SM90_64x128x16_F32BF16BF16_SS` | 64 | 128 | 16 | 标准配置 |
| `SM90_64x64x16_F32BF16BF16_SS` | 64 | 64 | 16 | 小 N 配置 |
| `SM90_64x256x16_F32BF16BF16_SS` | 64 | 256 | 16 | 大 N 配置 |

其中：
- `SS` 表示 A 和 B 矩阵都从 Shared Memory 读取
- `RS` 表示 A 从 Register，B 从 Shared Memory

### 2.3 Warp Group 内的线程组织

在一个 Warp Group 中，128 个线程按以下方式组织：

```cpp
// 线程索引分解
int thread_idx = threadIdx.x;                    // 0-127 (在 Warp Group 内)
int warp_id = thread_idx / 32;                   // 0-3
int lane_id = thread_idx % 32;                   // 0-31
int quad_id = lane_id / 4;                       // 0-7 (每个 Warp 有 8 个 Quad)
int quad_lane = lane_id % 4;                     // 0-3 (Quad 内的位置)
```

**Quad 的概念**：
- 一个 Quad 包含 4 个连续的线程（lane 0-3, 4-7, ...）
- 同一个 Quad 的线程在 GMMA 中处理 **相同的行**，但 **不同的列**
- 这是理解 Mask 访问模式的关键

### 2.4 GMMA 的输出布局 (Accumulator Fragment)

GMMA 的输出（C 矩阵 / Accumulator）存储在寄存器中，每个线程持有一部分元素。

对于 `M64 × N128` 的 GMMA：

```
C 矩阵 (64 × 128):
┌────────────────────────────────────────────────────────────────┐
│                         128 列                                  │
│  ← Quad Lane 0 → ← Quad Lane 1 → ← Quad Lane 2 → ← Quad Lane 3 →│
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  每个线程持有:                                                  │
│  - 2 行 (在 64 行中的某 2 行)                                   │
│  - 32 列 (跳跃分布在 128 列中)                                  │
│                                                                │
│  总计: 2 × 32 = 64 个元素/线程                                  │
│  128 线程 × 64 元素 = 8192 = 64 × 128 ✓                        │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.5 Fragment 布局格式

CUTLASS/CuTe 中，GMMA 的 accumulator fragment 布局为：

```cpp
// 原始 Fragment 布局
Tensor tSrS = partition_fragment_C(tiled_mma, Shape<kBlockM, kBlockN>{});
// 布局: ((2, 2, V), MMA_M, MMA_N)
//        ↑  ↑  ↑    ↑      ↑
//        │  │  │    │      └── N 方向的 Atom 数
//        │  │  │    └── M 方向的 Atom 数 (kBlockM / 64)
//        │  │  └── N 方向的分块数 (kBlockN / 8)
//        │  └── N 方向 2 个相邻元素
//        └── M 方向 2 个相邻元素
```

**对于 kBlockM=128, kBlockN=128**：
- `V` = 128 / 8 = 16
- `MMA_M` = 128 / 64 = 2
- `MMA_N` = 1
- 每线程元素数 = 2 × 2 × 16 × 2 × 1 = 128

### 2.6 代码依据

来自 `mainloop_fwd_sm90_tma_gmma_ws.hpp`：

```cpp
using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;

using TiledMmaQK = decltype(cute::make_tiled_mma(
    cute::GMMA::ss_op_selector<Element, Element, ElementAccum, TileShape_MNK>(),
    AtomLayoutQK{}));

static constexpr int NumMmaThreads = size(TiledMmaPV{});
static constexpr int NumMmaWarpGroups = NumMmaThreads / cutlass::NumThreadsPerWarpGroup;
// cutlass::NumThreadsPerWarpGroup = 128
```

**关键推导**：
- `AtomLayoutQK = Shape<2, 1, 1>` 表示 M 方向有 2 个 GMMA Atom
- 每个 Atom 需要 128 线程 (1 Warp Group)
- `NumMmaThreads` = 2 × 128 = 256
- `NumMmaWarpGroups` = 256 / 128 = 2

---

## 第三章：Tile 排列与参数关系

### 3.1 TileShape_MNK 参数解析

Flash Attention 中的 Tile 大小由模板参数 `TileShape_MNK` 定义，它是一个三元组：

```cpp
// 来自 mainloop_fwd_sm90_tma_gmma_ws.hpp
template <int Stages, class ClusterShape_, class TileShape_MNK_, int kHeadDimV, ...>
struct CollectiveMainloopFwdSm90 {
    using TileShape_MNK = TileShape_MNK_;
    
    static constexpr int kBlockM = get<0>(TileShape_MNK{});  // Query 块大小
    static constexpr int kBlockN = get<1>(TileShape_MNK{});  // Key/Value 块大小
    static constexpr int kHeadDim = get<2>(TileShape_MNK{}); // 头维度
};
```

**典型配置示例**：

| TileShape_MNK | kBlockM | kBlockN | kHeadDim | 场景 |
|---------------|---------|---------|----------|------|
| `(64, 128, 64)` | 64 | 128 | 64 | 小头维度 |
| `(128, 128, 128)` | 128 | 128 | 128 | 标准配置 |
| `(128, 128, 192)` | 128 | 128 | 192 | 大头维度 |
| `(64, 128, 256)` | 64 | 128 | 256 | 超大头维度 |

### 3.2 Attention Score Tile 的形状

对于 QK^T 计算，每个 Tile 的 Attention Score 矩阵形状为 `(kBlockM, kBlockN)`：

```
                        ← kBlockN (Key 维度) →
                    ┌────────────────────────────┐
                    │                            │
      kBlockM       │     Attention Score        │
    (Query 维度)    │        (kBlockM × kBlockN) │
         ↓          │                            │
                    │                            │
                    └────────────────────────────┘

示例: kBlockM=128, kBlockN=128
     → 每个 Tile 包含 128 × 128 = 16,384 个 Score 元素
```

### 3.3 AtomLayout 计算

`AtomLayout` 决定了如何将多个 GMMA Atom 组合来覆盖整个 Tile：

```cpp
// 代码来自 mainloop_fwd_sm90_tma_gmma_ws.hpp
using AtomLayoutQK = Layout<Shape<Int<kBlockM / 64>, _1, _1>>;
//                               ↑
//                          M 方向的 Atom 数量
```

**计算规则**：
- 单个 GMMA Atom 处理 `M=64` 行
- 需要 `kBlockM / 64` 个 Atom 来覆盖整个 M 维度
- N 方向通常由单个 Atom 覆盖（N=128 或 N=64）

| kBlockM | AtomLayoutQK | M 方向 Atom 数 |
|---------|--------------|----------------|
| 64 | `Shape<1, 1, 1>` | 1 |
| 128 | `Shape<2, 1, 1>` | 2 |
| 192 | `Shape<3, 1, 1>` | 3 |

### 3.4 NumMmaWarpGroups 推导

`NumMmaWarpGroups` 是参与 MMA 计算的 Warp Group 数量，直接影响并行度和寄存器分配：

```cpp
// 代码来自 flash_fwd_kernel_sm90.h
using TiledMmaPV = typename CollectiveMainloop::TiledMmaPV;
static constexpr uint32_t NumMmaWarpGroups = 
    CUTE_STATIC_V(size(TiledMmaPV{})) / cutlass::NumThreadsPerWarpGroup;
```

**推导过程**：

```
TiledMmaPV = make_tiled_mma(GMMA_Atom, AtomLayoutPV)

size(TiledMmaPV) = size(GMMA_Atom) × size(AtomLayoutPV)
                = 128 × (kBlockM / 64)    // 对于标准配置
                = 128 × MMA_M

NumMmaWarpGroups = size(TiledMmaPV) / 128
                = MMA_M
                = kBlockM / 64
```

| kBlockM | MMA_M | NumMmaWarpGroups | 总线程数 |
|---------|-------|------------------|----------|
| 64 | 1 | 1 | 128 |
| 128 | 2 | 2 | 256 |
| 192 | 3 | 3 | 384 |

### 3.5 寄存器分配与 NumMmaWarpGroups

`NumMmaWarpGroups` 直接影响每个线程的寄存器分配：

```cpp
// 代码来自 flash_fwd_kernel_sm90.h
static constexpr uint32_t LoadRegisterRequirement = 
    (NumMmaWarpGroups == 1 ? 56 : 
    (NumMmaWarpGroups == 2 ? ((Use_TMA_KV) ? 24 : 40) : 32));

static constexpr uint32_t MmaRegisterRequirement = 
    (NumMmaWarpGroups == 1 ? 256 : 
    (NumMmaWarpGroups == 2 ? ((Use_TMA_KV) ? 240 : 232) : 160));
```

| NumMmaWarpGroups | Use_TMA_KV | LoadRegReq | MmaRegReq |
|------------------|------------|------------|-----------|
| 1 | - | 56 | 256 |
| 2 | true | 24 | 240 |
| 2 | false | 40 | 232 |
| 3 | - | 32 | 160 |

### 3.6 Sparse Mask 相关参数

Sparse Mask 的存储大小由 `kBlockN` 决定：

```cpp
// 代码来自 mask.h
template <int kBlockM, int kBlockN, ...>
struct SparseMask {
    static constexpr int kNumInt32PerBlock = (kBlockN + 31) / 32;
    // 每行需要的 int32 数量，用于存储 kBlockN 个 bit
};
```

| kBlockN | kNumInt32PerBlock | 每行 bits |
|---------|-------------------|-----------|
| 64 | 2 | 64 |
| 128 | 4 | 128 |
| 256 | 8 | 256 |

**smem_mask 布局**：`[kBlockM][kNumInt32PerBlock]`

```
对于 kBlockM=128, kBlockN=128:
smem_mask 大小 = 128 × 4 × 4 bytes = 2048 bytes = 2 KB
```

### 3.7 参数关系总结图

```
┌─────────────────────────────────────────────────────────────────┐
│                    参数关系图                                    │
│                                                                 │
│  TileShape_MNK = (kBlockM, kBlockN, kHeadDim)                  │
│         │              │            │                           │
│         ▼              ▼            │                           │
│     MMA_M          kNumInt32       │                           │
│   = kBlockM/64    = kBlockN/32     │                           │
│         │              │            │                           │
│         ▼              │            │                           │
│  NumMmaWarpGroups      │            │                           │
│   = MMA_M              │            │                           │
│         │              │            │                           │
│         ▼              ▼            │                           │
│  NumMmaThreads    smem_mask 大小   │                           │
│  = 128 × MMA_M   = kBlockM ×       │                           │
│                    kNumInt32 × 4    │                           │
│         │                           │                           │
│         ▼                           ▼                           │
│  Fragment 大小              TMA 配置                            │
│  = 2×2×V×MMA_M×MMA_N                                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 第四章：线程到元素的映射

本章是理解 Sparse Mask 访问模式的核心。我们将详细分析每个线程持有哪些行和列。

### 4.1 Accumulator Fragment 原始布局

GMMA 的 accumulator fragment 原始布局为 `((2, 2, V), MMA_M, MMA_N)`：

```cpp
// 原始 Fragment
Tensor tSrS = partition_fragment_C(tiled_mma, select<0, 1>(TileShape_MNK{}));
// 形状: ((2, 2, V), MMA_M, MMA_N)
```

**各维度含义**：

```
((2, 2, V), MMA_M, MMA_N)
  │  │  │     │      │
  │  │  │     │      └── N 方向 Atom 数 (通常=1)
  │  │  │     └── M 方向 Atom 数 (= kBlockM/64)
  │  │  └── N 方向分块数 (V = kBlockN/8)
  │  └── N 方向 2 个相邻元素
  └── M 方向 2 个相邻元素
```

**对于 kBlockM=128, kBlockN=128**：
- 形状 = `((2, 2, 16), 2, 1)`
- 每线程元素数 = 2 × 2 × 16 × 2 × 1 = **128**

### 4.2 convert_layout_acc_rowcol 转换

为了方便按行列处理，使用 `convert_layout_acc_rowcol` 将布局转换为 `(nrow, ncol)` 形式：

```cpp
// 代码来自 utils.h
template<bool Transposed=false, typename Layout0>
CUTLASS_DEVICE auto convert_layout_acc_rowcol(Layout0 acc_layout) {
    if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90
        static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
        static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
        auto l = acc_layout;
        if constexpr (!Transposed) {
            // 原始: ((2, 2, V), MMA_M, MMA_N)
            // 转换: (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
            return make_layout(
                make_layout(get<0, 1>(l), get<1>(l)),    // nrow = (2, MMA_M)
                make_layout(get<0, 0>(l), get<0, 2>(l), get<2>(l))  // ncol = (2, V, MMA_N)
            );
        }
    }
}
```

**转换图示**：

```
原始: ((2, 2, V), MMA_M, MMA_N)
      ((a, b, c),   d,     e)

转换后: (nrow=(b, d), ncol=(a, c, e))
       (nrow=(2, MMA_M), ncol=(2, V, MMA_N))
```

**对于 kBlockM=128, kBlockN=128**：
- 原始形状: `((2, 2, 16), 2, 1)`
- 转换后形状: `((2, 2), (2, 16, 1))`
- `size<0>` (行数) = 2 × 2 = **4**
- `size<1>` (列数) = 2 × 16 × 1 = **32**

### 4.3 线程的行坐标计算

每个线程持有 4 行（对于 kBlockM=128），这些行的坐标由以下因素决定：

```cpp
// 线程索引分解
int thread_idx = threadIdx.x % 128;  // Warp Group 内索引
int warp_id = thread_idx / 32;       // 0-3
int lane_id = thread_idx % 32;       // 0-31
int quad_id = lane_id / 4;           // 0-7
int quad_lane = lane_id % 4;         // 0-3
```

**行坐标公式** (对于单个 GMMA Atom，M=64)：

```
row_base = warp_id × 16 + quad_id × 2
该线程持有的两行: row_base, row_base + 1
```

**示例：线程 0-15 的行分配 (Warp 0)**

| thread_idx | warp_id | lane_id | quad_id | quad_lane | rows |
|------------|---------|---------|---------|-----------|------|
| 0 | 0 | 0 | 0 | 0 | 0, 1 |
| 1 | 0 | 1 | 0 | 1 | 0, 1 |
| 2 | 0 | 2 | 0 | 2 | 0, 1 |
| 3 | 0 | 3 | 0 | 3 | 0, 1 |
| 4 | 0 | 4 | 1 | 0 | 2, 3 |
| 5 | 0 | 5 | 1 | 1 | 2, 3 |
| 6 | 0 | 6 | 1 | 2 | 2, 3 |
| 7 | 0 | 7 | 1 | 3 | 2, 3 |
| ... | ... | ... | ... | ... | ... |
| 28 | 0 | 28 | 7 | 0 | 14, 15 |
| 29 | 0 | 29 | 7 | 1 | 14, 15 |
| 30 | 0 | 30 | 7 | 2 | 14, 15 |
| 31 | 0 | 31 | 7 | 3 | 14, 15 |

**关键观察**：同一个 Quad 的 4 个线程（如 thread 0-3）持有 **相同的行**！

### 4.4 线程的列坐标计算

每个线程持有 32 列（对于 kBlockN=128），这些列是 **跳跃分布** 的：

**列坐标公式**：

```
对于 v = 0 到 V-1 (V = kBlockN/8 = 16):
    col_base = v × 8 + quad_lane × 2
    该线程在此 v 块持有的两列: col_base, col_base + 1
```

**示例：quad_lane=0 的线程持有的列**

| v | col_base | 持有的列 |
|---|----------|----------|
| 0 | 0 | 0, 1 |
| 1 | 8 | 8, 9 |
| 2 | 16 | 16, 17 |
| 3 | 24 | 24, 25 |
| 4 | 32 | 32, 33 |
| 5 | 40 | 40, 41 |
| 6 | 48 | 48, 49 |
| 7 | 56 | 56, 57 |
| 8 | 64 | 64, 65 |
| 9 | 72 | 72, 73 |
| 10 | 80 | 80, 81 |
| 11 | 88 | 88, 89 |
| 12 | 96 | 96, 97 |
| 13 | 104 | 104, 105 |
| 14 | 112 | 112, 113 |
| 15 | 120 | 120, 121 |

**所有 quad_lane 的列分布**：

| quad_lane | 持有的列 (简写) |
|-----------|-----------------|
| 0 | 0,1, 8,9, 16,17, 24,25, ... 120,121 |
| 1 | 2,3, 10,11, 18,19, 26,27, ... 122,123 |
| 2 | 4,5, 12,13, 20,21, 28,29, ... 124,125 |
| 3 | 6,7, 14,15, 22,23, 30,31, ... 126,127 |

### 4.5 128 列的完整覆盖

一个 Quad (4 个线程) 完整覆盖一行的所有 128 列：

```
列索引:  0  1  2  3  4  5  6  7 | 8  9 10 11 12 13 14 15 | ...
线程:   L0 L0 L1 L1 L2 L2 L3 L3 |L0 L0 L1 L1 L2 L2 L3 L3 | ...

(L0 = quad_lane 0, L1 = quad_lane 1, ...)
```

**图示（每 8 列一组）**：

```
┌────────────────────────────────────────────────────────────────┐
│              kBlockN = 128 列                                   │
│                                                                │
│  v=0      v=1      v=2      ...      v=14     v=15             │
│ ┌─────┐ ┌─────┐ ┌─────┐           ┌─────┐ ┌─────┐            │
│ │01│23│ │89│AB│ │...│             │...│   │...│             │
│ │45│67│ │CD│EF│ │...│             │...│   │...│             │
│ └─────┘ └─────┘ └─────┘           └─────┘ └─────┘            │
│  ↑↑ ↑↑                                                        │
│  L0 L1                                                        │
│  L2 L3   (每组 8 列被 4 个 quad_lane 瓜分)                      │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 4.6 多 Atom 情况 (MMA_M > 1)

当 kBlockM=128 时，MMA_M=2，需要 2 个 GMMA Atom 堆叠。

转换后的 `tSrS_rowcol` 形状为 `((2, 2), (2, 16, 1))`：
- `size<0>` = 2 × 2 = 4 行
- 其中：第一个 2 来自单个 Atom 内的 2 行，第二个 2 来自 2 个 Atom

**线程持有的 4 行**：

| m (rowcol 索引) | 来源 | 行坐标 |
|-----------------|------|--------|
| 0 | Atom 0, 第一行 | row_base |
| 1 | Atom 0, 第二行 | row_base + 1 |
| 2 | Atom 1, 第一行 | row_base + 64 |
| 3 | Atom 1, 第二行 | row_base + 64 + 1 |

**示例：thread_idx=0 持有的行**

```
row_base = 0 × 16 + 0 × 2 = 0

m=0: row 0
m=1: row 1
m=2: row 64
m=3: row 65
```

### 4.7 总结：线程到元素映射公式

对于 `kBlockM=128, kBlockN=128` 配置：

```cpp
// 给定 thread_idx (0-127 在 Warp Group 内)
int warp_id = thread_idx / 32;
int lane_id = thread_idx % 32;
int quad_id = lane_id / 4;
int quad_lane = lane_id % 4;

// 行计算 (共 4 行)
int row_base = warp_id * 16 + quad_id * 2;
int rows[4] = {row_base, row_base + 1, row_base + 64, row_base + 65};

// 列计算 (共 32 列)
int cols[32];
for (int v = 0; v < 16; v++) {
    int col_base = v * 8 + quad_lane * 2;
    cols[v * 2] = col_base;
    cols[v * 2 + 1] = col_base + 1;
}
```

**每线程数据量**：4 行 × 32 列 = **128 个元素**

---

## 第五章：SparseMask 线程级分析

本章分析 `SparseMask::apply` 函数的实现，理解它如何利用上一章的坐标映射来访问 Mask。

### 5.1 SparseMask 类定义

```cpp
// 代码来自 mask.h
template <int kBlockM, int kBlockN, bool PackGQA, typename TiledMma, bool SwapAB=false>
struct SparseMask {
    static constexpr int kNumInt32PerBlock = (kBlockN + 31) / 32;
    
    int const thread_idx;
    int const seqlen_q, seqlen_k;
    int const max_k_blocks;
    cutlass::FastDivmod const qhead_per_khead_divmod;
    // ...
};
```

### 5.2 apply 函数核心逻辑

```cpp
template <bool Seqlenk_mask=false, typename Engine, typename Layout>
CUTLASS_DEVICE
void apply(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block, 
           uint32_t const* __restrict__ smem_mask_ptr) const {
    
    // 1. 获取当前线程的 MMA 分区
    auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
    
    // 2. 创建坐标张量
    Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    
    // 3. 转换为 rowcol 布局
    Tensor tSrS_rowcol = make_tensor(tSrS.data(), 
        flash::convert_layout_acc_rowcol(tSrS.layout()));
    Tensor tScS_rowcol = make_tensor(tScS.data(), 
        flash::convert_layout_acc_rowcol(tScS.layout()));
    
    // 4. 遍历行
    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
        int local_row = get<Row>(tScS_rowcol(m, _0{}));
        // ...
        
        // 5. 遍历列
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            int token_in_block = int(get<Col>(tScS_rowcol(m, n)));
            // 应用 mask
        }
    }
}
```

### 5.3 坐标张量的构造

**恒等张量 cS**：

```cpp
Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
// cS(i, j) = (i, j)，即坐标本身
```

**分区后的坐标张量 tScS**：

```cpp
Tensor tScS = thread_mma.partition_C(cS);
// 形状: ((2, 2, V), MMA_M, MMA_N)
// 每个元素是 (row, col) 坐标对
```

**转换后的坐标张量 tScS_rowcol**：

```cpp
Tensor tScS_rowcol = make_tensor(tScS.data(), 
    flash::convert_layout_acc_rowcol(tScS.layout()));
// 形状: ((2, MMA_M), (2, V, MMA_N))
```

### 5.4 local_row 的计算

```cpp
int local_row = get<Row>(tScS_rowcol(m, _0{}));
```

- `tScS_rowcol(m, _0{})` 取第 m 行、第 0 列的坐标
- `get<Row>(...)` 提取行坐标（Row=0 当 SwapAB=false）
- 返回该行在 Tile 内的局部行号（0 到 kBlockM-1）

**示例：thread_idx=0, kBlockM=128**

| m | local_row |
|---|-----------|
| 0 | 0 |
| 1 | 1 |
| 2 | 64 |
| 3 | 65 |

### 5.5 token_in_block 的计算

```cpp
int token_in_block = int(get<Col>(tScS_rowcol(m, n)));
```

- `tScS_rowcol(m, n)` 取第 m 行、第 n 列的坐标
- `get<Col>(...)` 提取列坐标（Col=1 当 SwapAB=false）
- 返回该列在 Tile 内的局部列号（0 到 kBlockN-1）

**示例：thread_idx=0 (quad_lane=0), 部分 n 值**

| n | token_in_block |
|---|----------------|
| 0 | 0 |
| 1 | 1 |
| 2 | 8 |
| 3 | 9 |
| 4 | 16 |
| 5 | 17 |
| ... | ... |
| 30 | 120 |
| 31 | 121 |

### 5.6 Mask 的访问模式

当前实现从 `smem_mask_ptr` 读取 mask：

```cpp
// kNumInt32PerBlock = 4 (当 kBlockN=128)
if constexpr (kNumInt32PerBlock == 4) {
    // 向量化读取一行的 4 个 int32
    auto vec_ptr = reinterpret_cast<const uint4*>(
        smem_mask_ptr + local_row * kNumInt32PerBlock);
    auto v = *vec_ptr;  // v.x, v.y, v.z, v.w
    
    for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
        int token_in_block = int(get<Col>(tScS_rowcol(m, n)));
        
        // 计算在哪个 word 的哪个 bit
        int word_idx = token_in_block / 32;
        int bit_idx = token_in_block % 32;
        
        // 选择正确的 word
        uint32_t target_word;
        if (word_idx == 0) target_word = v.x;
        else if (word_idx == 1) target_word = v.y;
        else if (word_idx == 2) target_word = v.z;
        else target_word = v.w;
        
        // 提取 bit
        bool mask_bit = (target_word >> bit_idx) & 1;
        
        if (!mask_bit) {
            tSrS_rowcol(m, n) = -INFINITY;
        }
    }
}
```

### 5.7 word_idx 和 bit_idx 的分布

由于列是跳跃分布的，每个线程需要访问 **所有 4 个 word**：

**quad_lane=0 的线程**：

| n | token_in_block | word_idx | bit_idx |
|---|----------------|----------|---------|
| 0 | 0 | 0 | 0 |
| 1 | 1 | 0 | 1 |
| 2 | 8 | 0 | 8 |
| 3 | 9 | 0 | 9 |
| ... | ... | ... | ... |
| 8 | 32 | **1** | 0 |
| 9 | 33 | **1** | 1 |
| ... | ... | ... | ... |
| 16 | 64 | **2** | 0 |
| 17 | 65 | **2** | 1 |
| ... | ... | ... | ... |
| 24 | 96 | **3** | 0 |
| 25 | 97 | **3** | 1 |
| ... | ... | ... | ... |

**可视化**：

```
128 列分布:

word_idx:    0         1         2         3
bit range: [0,31]   [32,63]   [64,95]   [96,127]

quad_lane 0 访问的 bits:
word 0: bits 0,1, 8,9, 16,17, 24,25    (8 bits)
word 1: bits 0,1, 8,9, 16,17, 24,25    (8 bits)
word 2: bits 0,1, 8,9, 16,17, 24,25    (8 bits)
word 3: bits 0,1, 8,9, 16,17, 24,25    (8 bits)
                                        --------
                                        32 bits total
```

### 5.8 当前实现的问题

**问题 1：冗余读取**

每个线程需要读取 4 个 int32，但只用到每个 word 的部分 bits：
- 读取：4 × 32 = 128 bits
- 实际使用：32 bits
- 冗余率：75%

**问题 2：分支判断**

```cpp
if (word_idx == 0) target_word = v.x;
else if (word_idx == 1) target_word = v.y;
else if (word_idx == 2) target_word = v.z;
else target_word = v.w;
```

每次访问都需要 3-4 次比较，增加指令数。

**问题 3：不规则访问模式**

`bit_idx` 的值跳跃分布（0,1,8,9,16,17,...），无法高效利用。

### 5.9 size<0> 和 size<1> 的一般公式

| 参数 | 公式 | kBlockM=128, kBlockN=128 |
|------|------|--------------------------|
| `size<0>(tSrS_rowcol)` | 2 × MMA_M = 2 × (kBlockM/64) | 2 × 2 = 4 |
| `size<1>(tSrS_rowcol)` | 2 × V × MMA_N = 2 × (kBlockN/8) × 1 | 2 × 16 × 1 = 32 |

**不同配置的值**：

| kBlockM | kBlockN | size<0> | size<1> | 每线程元素 |
|---------|---------|---------|---------|------------|
| 64 | 128 | 2 | 32 | 64 |
| 128 | 128 | 4 | 32 | 128 |
| 192 | 128 | 6 | 32 | 192 |
| 128 | 64 | 4 | 16 | 64 |
| 128 | 256 | 4 | 64 | 256 |

---

## 第六章：当前 Mask 存储与访问模式

本章详细分析当前 Sparse Mask 的存储布局和访问开销。

### 6.1 Mask 的语义

Sparse Mask 是一个二值矩阵，定义哪些 (Query, Key) 位置对需要计算注意力：

```
mask[q][k] = 1: 计算 attention(Q[q], K[k])
mask[q][k] = 0: 跳过（设为 -∞）
```

对于一个 `(kBlockM, kBlockN)` 的 Tile，需要存储 `kBlockM × kBlockN` 个 bit。

### 6.2 当前 smem_mask 布局

**存储格式**：按行主序 (Row-Major)，每行 `kNumInt32PerBlock` 个 int32

```cpp
// 来自 mainloop_fwd_sm90_tma_gmma_ws.hpp
using SmemLayoutMask = Layout<
    Shape<Int<kBlockM>, Int<kNumInt32PerBlock>, Int<kStages>>,
    Stride<Int<kNumInt32PerBlock>, _1, Int<kBlockM * kNumInt32PerBlock>>
>;
```

**布局图示** (kBlockM=128, kBlockN=128, kNumInt32PerBlock=4)：

```
smem_mask[stage][row][word_idx]:

stage 0:
┌─────────────────────────────────────────────┐
│ row 0:   [word0][word1][word2][word3]       │
│ row 1:   [word0][word1][word2][word3]       │
│ row 2:   [word0][word1][word2][word3]       │
│ ...                                         │
│ row 127: [word0][word1][word2][word3]       │
└─────────────────────────────────────────────┘

每个 word 存储 32 个 bit:
- word0: cols 0-31
- word1: cols 32-63
- word2: cols 64-95
- word3: cols 96-127

总大小: 128 × 4 × 4 bytes = 2 KB / stage
```

### 6.3 每行的 Bit 布局

```
row r 的 128 bits:

word0 (cols 0-31):
┌─────────────────────────────────────────────────────────────┐
│ bit0  bit1  bit2  bit3  ... bit30 bit31                     │
│ col0  col1  col2  col3  ... col30 col31                     │
└─────────────────────────────────────────────────────────────┘

word1 (cols 32-63):
┌─────────────────────────────────────────────────────────────┐
│ bit0  bit1  bit2  bit3  ... bit30 bit31                     │
│ col32 col33 col34 col35 ... col62 col63                     │
└─────────────────────────────────────────────────────────────┘

... (word2, word3 类似)
```

### 6.4 线程访问模式分析

对于 thread_idx=0（quad_lane=0）处理的 **一行** (例如 row 0)：

```
该线程需要的列: 0,1, 8,9, 16,17, 24,25, 32,33, 40,41, ..., 120,121

这些列分布在 4 个 word 中:
- word0: cols 0,1,8,9,16,17,24,25    → bits 0,1,8,9,16,17,24,25
- word1: cols 32,33,40,41,48,49,56,57 → bits 0,1,8,9,16,17,24,25
- word2: cols 64,65,72,73,80,81,88,89 → bits 0,1,8,9,16,17,24,25
- word3: cols 96,97,104,105,112,113,120,121 → bits 0,1,8,9,16,17,24,25
```

**观察**：每个 word 中该线程需要的 bits 位置是相同的！

### 6.5 访问开销分析

**当前实现的每行操作**：

| 操作 | 数量 | 说明 |
|------|------|------|
| 内存读取 | 1 次 uint4 (16 bytes) | 读取 4 个 int32 |
| word 选择分支 | 32 × 4 = 128 次比较 | 每个 n 需要判断 word_idx |
| bit 提取 | 32 次移位 + AND | 提取每个 mask bit |
| 条件赋值 | 32 次 | 可能的 -INFINITY 赋值 |

**整个 Tile (4 行)**：

| 资源 | 数量 |
|------|------|
| 内存读取 | 4 × 16 = 64 bytes |
| 读取的总 bits | 4 × 128 = 512 bits |
| 实际使用的 bits | 4 × 32 = 128 bits |
| **利用率** | 128 / 512 = **25%** |

### 6.6 寄存器占用分析

```cpp
// 当前实现的寄存器使用
auto v = *vec_ptr;  // uint4 = 4 × uint32 = 4 个寄存器

// 循环中的临时变量
int token_in_block;  // 1 个寄存器
int word_idx;        // 1 个寄存器
int bit_idx;         // 1 个寄存器
uint32_t target_word; // 1 个寄存器
bool mask_bit;       // 1 个寄存器

// 总计约 8-10 个额外寄存器/行
```

### 6.7 指令开销分析

对于每个列 n（共 32 个），需要：

```cpp
// 1. 获取列坐标 (~2 指令)
int token_in_block = int(get<Col>(tScS_rowcol(m, n)));

// 2. 计算 word_idx 和 bit_idx (~4 指令)
int word_idx = token_in_block / 32;  // 或右移 5 位
int bit_idx = token_in_block % 32;   // 或 AND 0x1F

// 3. 选择 word (~4 指令，分支)
uint32_t target_word;
if (word_idx == 0) target_word = v.x;
else if (word_idx == 1) target_word = v.y;
else if (word_idx == 2) target_word = v.z;
else target_word = v.w;

// 4. 提取 bit (~2 指令)
bool mask_bit = (target_word >> bit_idx) & 1;

// 5. 条件赋值 (~2 指令)
if (!mask_bit) { tSrS_rowcol(m, n) = -INFINITY; }
```

**每个列约 14+ 指令，32 列 × 4 行 = 约 1800 指令/线程**

### 6.8 同一 Quad 的线程访问模式

关键观察：同一 Quad 的 4 个线程持有 **相同的行**，但 **不同的列**。

```
Quad 内的 4 个线程 (处理同一行):

Thread (quad_lane=0): cols 0,1, 8,9, 16,17, ... → 用到 word 0,1,2,3 的部分 bits
Thread (quad_lane=1): cols 2,3, 10,11, 18,19, ... → 用到 word 0,1,2,3 的部分 bits
Thread (quad_lane=2): cols 4,5, 12,13, 20,21, ... → 用到 word 0,1,2,3 的部分 bits
Thread (quad_lane=3): cols 6,7, 14,15, 22,23, ... → 用到 word 0,1,2,3 的部分 bits

合起来: 覆盖该行的全部 128 bits
```

**冗余读取**：4 个线程各自读取 4 个 word = 16 个 word 读取，但只需要 4 个 word

### 6.9 smem Bank Conflict 分析

```
smem_mask 在 shared memory 中的布局:

地址:     0    4    8   12   16   20   24   28   32   ...
         ┌────┬────┬────┬────┬────┬────┬────┬────┬────┐
row 0:   │ w0 │ w1 │ w2 │ w3 │    │    │    │    │    │
         ├────┼────┼────┼────┤    │    │    │    │    │
row 1:   │ w0 │ w1 │ w2 │ w3 │    │    │    │    │    │
         └────┴────┴────┴────┴────┴────┴────┴────┴────┘
Bank:      0    1    2    3    4    5    6    7    0

每行 16 bytes = 4 banks
不同行的相同 word_idx 可能在同一 bank → 潜在 bank conflict
```

当同一个 Quad 的 4 个线程访问 **同一行** 时，它们读取相同地址，会产生 **broadcast**（不是 conflict）。

但当不同 Quad 访问不同行时，如果恰好在同一 bank，会产生 conflict。

### 6.10 当前方案总结

| 指标 | 当前实现 | 说明 |
|------|----------|------|
| 每行读取量 | 16 bytes (uint4) | 向量化读取 |
| 4 行总读取量 | 64 bytes | |
| 实际使用率 | 25% | 512 bits 读取，128 bits 使用 |
| Word 选择开销 | 高 | 每个 bit 需要分支判断 |
| 寄存器压力 | 中等 | ~10 个额外寄存器/行 |
| Bank Conflict | 低 | 同行访问是 broadcast |

### 6.11 优化目标

1. **减少读取量**：从每线程 64 bytes 降到最小
2. **消除 word 选择分支**：直接索引
3. **提高数据利用率**：从 25% 提高到 100%
4. **降低寄存器压力**：减少临时变量

---

## 第七章：优化方案 - Mask 重排

本章提出通过预处理重排 Mask 数据，使每个线程只需读取 **1 个 int32**，实现 100% 数据利用率。

### 7.1 优化思路

**核心观察**：
- 每个线程只需要 32 bits 的 mask 数据
- 当前读取 128 bits，利用率仅 25%
- 如果按 **线程需要的 bits** 重新组织 mask，可以实现最小读取

**目标**：
- 每行每线程：从读 4 个 int32 → 读 1 个 int32
- 消除 word 选择分支
- 简化 bit 提取逻辑

### 7.2 新的 Mask 布局设计

**原始布局** (按列连续)：
```
每行 128 bits = 4 words
word0: cols [0-31]
word1: cols [32-63]
word2: cols [64-95]
word3: cols [96-127]
```

**新布局** (按 quad_lane 分组)：
```
每行 128 bits = 4 words，但按线程分组：
word0: quad_lane 0 的 32 bits (cols 0,1,8,9,16,17,...)
word1: quad_lane 1 的 32 bits (cols 2,3,10,11,18,19,...)
word2: quad_lane 2 的 32 bits (cols 4,5,12,13,20,21,...)
word3: quad_lane 3 的 32 bits (cols 6,7,14,15,22,23,...)
```

### 7.3 Bit 映射算法

**从原始列号到新位置**：

```
原始: col ∈ [0, 127]
新:   new_word = (col % 8) / 2     // quad_lane: 0-3
      v = col / 8                   // 哪个 8-列块: 0-15
      local_bit = col % 2           // 块内偏移: 0 or 1
      bit_in_new_word = v * 2 + local_bit  // 0-31
```

**示例：quad_lane=0 对应的列**

| col | v | local_bit | bit_in_new_word |
|-----|---|-----------|-----------------|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 1 | 1 |
| 8 | 1 | 0 | 2 |
| 9 | 1 | 1 | 3 |
| 16 | 2 | 0 | 4 |
| 17 | 2 | 1 | 5 |
| ... | ... | ... | ... |
| 120 | 15 | 0 | 30 |
| 121 | 15 | 1 | 31 |

### 7.6 优化后的 apply 函数

```cpp
template <bool Seqlenk_mask=false, typename Engine, typename Layout>
CUTLASS_DEVICE
void apply_optimized(Tensor<Engine, Layout> &tSrS, const int m_block, const int n_block, 
                     uint32_t const* __restrict__ smem_mask_reordered) const {
    
    // 获取线程信息
    int lane_id = thread_idx % 32;
    int quad_lane = lane_id % 4;  // 0-3，决定读哪个 word
    
    // 坐标张量（同原实现）
    auto thread_mma = TiledMma{}.get_thread_slice(thread_idx);
    Tensor cS = cute::make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor tScS = thread_mma.partition_C(cS);
    Tensor tSrS_rowcol = make_tensor(tSrS.data(), 
        flash::convert_layout_acc_rowcol(tSrS.layout()));
    Tensor tScS_rowcol = make_tensor(tScS.data(), 
        flash::convert_layout_acc_rowcol(tScS.layout()));
    
    // 遍历每一行
    for (int m = 0; m < size<0>(tSrS_rowcol); ++m) {
        int local_row = get<Row>(tScS_rowcol(m, _0{}));
        
        // 关键优化：只读取该线程需要的 1 个 int32
        uint32_t mask_word = smem_mask_reordered[local_row * 4 + quad_lane];
        
        // 遍历 32 列，bit 位置是连续的 0-31
        #pragma unroll
        for (int n = 0; n < size<1>(tSrS_rowcol); ++n) {
            // n 对应 bit 位置: n / 2 * 2 + n % 2 = n（因为已经重排）
            // 实际上，新布局下，第 n 个元素对应 bit n
            bool mask_bit = (mask_word >> n) & 1;
            
            if (!mask_bit) {
                tSrS_rowcol(m, n) = -INFINITY;
            }
        }
    }
}
```

### 7.7 性能对比

| 指标 | 原始实现 | 优化实现 | 改进 |
|------|----------|----------|------|
| 每行读取量 | 16 bytes | **4 bytes** | 4× |
| 数据利用率 | 25% | **100%** | 4× |
| Word 选择分支 | 有 | **无** | ∞ |
| Bit 索引计算 | 复杂 | **简单** | 显著 |
| 寄存器占用 | ~10 个 | **~2 个** | 5× |
| 每行指令数 | ~450 | **~100** | 4.5× |

### 7.8 内存布局变化

**原始 smem_mask 布局**：
```
row 0: [cols 0-31] [cols 32-63] [cols 64-95] [cols 96-127]
row 1: [cols 0-31] [cols 32-63] [cols 64-95] [cols 96-127]
...
```

**重排后 smem_mask 布局**：
```
row 0: [QL0 bits] [QL1 bits] [QL2 bits] [QL3 bits]
row 1: [QL0 bits] [QL1 bits] [QL2 bits] [QL3 bits]
...

其中:
- QL0 bits = 该行 quad_lane 0 线程需要的 32 bits
- 每个 word 的 bit n 对应 fragment 的第 n 个列元素
```

### 7.9 Bank Conflict 分析

**优化后的访问模式**：

```
同一 Quad 的 4 个线程访问:
- Thread 0 (quad_lane=0): row × 4 + 0
- Thread 1 (quad_lane=1): row × 4 + 1
- Thread 2 (quad_lane=2): row × 4 + 2
- Thread 3 (quad_lane=3): row × 4 + 3

地址连续，访问 4 个不同的 bank → 无 conflict！
```
