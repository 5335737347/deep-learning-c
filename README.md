# Deep Learning in C

从零开始用纯 C 语言实现深度学习/机器学习算法库。仅依赖标准库，不借助任何第三方数值计算库。旨在通过亲手造轮子理解神经网络底层原理。

## 特性

- **纯 C11** — 无外部依赖，只使用 C 标准库
- **张量计算** — N 维数组，行主序存储，支持 transpose / slice / reshape / broadcast
- **零拷贝视图** — 基于 stride 的视图操作，transpose/slice/broadcast 不复制数据
- **引用计数** — Storage 层引用计数，多 Tensor 安全共享同一数据
- **GPU 加速** — 可选 CUDA 后端
- **模块化** — core → math → models 分层设计

## 项目结构

```
src/
├── main.c                    # 测试入口
└── ml/
    ├── core/                 # 张量核心数据结构
    │   ├── tensor.h
    │   └── tensor.c
    ├── math/                 # 逐元素数学运算 + 广播
    │   ├── tensor_math.h
    │   └── tensor_math.c
    ├── models/               # ML 算法模型（规划中）
    ├── data/                 # 数据处理（规划中）
    ├── device/               # 设备抽象层
    │   └── device.h
    ├── gpu/                  # GPU 加速（规划中）
    │   └── cuda/
    └── utils/                # 工具函数（规划中）
```

## 构建

```bash
xmake           # CPU 版本
xmake run       # 运行
```

GPU 版本（需 CUDA Toolkit）：

```bash
xmake build deep-learning-c_cuda
xmake run deep-learning-c_cuda
```

## API 参考

### core — 张量 (tensor.h)

**存储管理**

| 函数 | 说明 |
|------|------|
| `storage_create(n, zero_init, device)` | 创建引用计数的 float 缓冲区 |
| `storage_retain(s)` | 增加引用计数 |
| `storage_release(s)` | 减少引用计数，归零时释放 |

**张量生命周期**

| 函数 | 说明 |
|------|------|
| `tensor_create(ndim, shape)` | 创建零初始化的张量 |
| `tensor_create_view(storage, ndim, shape, strides, offset)` | 创建共享 Storage 的视图 |
| `tensor_free(t)` | 释放张量（减少 Storage 引用计数） |

**索引访问**

| 函数 | 说明 |
|------|------|
| `tensor_flat_index(t, indices)` | 多维索引 → 一维偏移 |
| `tensor_at(t, indices)` | 获取元素指针 |

**形状变换**

| 函数 | 说明 |
|------|------|
| `tensor_reshape(t, new_ndim, new_shape)` | 改变形状（要求元素数相同） |
| `tensor_transpose(t, dim0, dim1)` | 交换两个维度 |
| `tensor_slice(t, dim, start, end)` | 沿某维度切片 |
| `tensor_squeeze(t)` | 删除大小为 1 的维度 |
| `tensor_unsqueeze(t, dim)` | 在指定位置插入大小为 1 的维度 |
| `tensor_broadcast_to(t, target_ndim, target_shape)` | 广播到目标形状 |

**内存布局**

| 函数 | 说明 |
|------|------|
| `tensor_is_contiguous(t)` | 检查是否连续存储 |
| `tensor_contiguous(t)` | 返回连续副本 |

**调试**

| 函数 | 说明 |
|------|------|
| `tensor_print(t)` | 打印张量内容 |

### math — 数学运算 (tensor_math.h)

| 函数 | 说明 |
|------|------|
| `tensor_add(a, b)` | 逐元素加法（支持广播） |
| `tensor_sub(a, b)` | 逐元素减法（支持广播） |
| `tensor_mul(a, b)` | 逐元素乘法（支持广播） |
| `tensor_div(a, b)` | 逐元素除法（支持广播） |
| `tensor_sum(input, dim)` | 沿指定维度求和（声明，待实现） |
| `tensor_mean(input, dim)` | 沿指定维度求均值（声明，待实现） |

## 实现状态

### ✅ 已实现

- [x] Tensor 创建/释放（引用计数 Storage）
- [x] 零拷贝视图操作（transpose, slice, squeeze, unsqueeze, broadcast）
- [x] 多维索引访问（`tensor_at`, `tensor_flat_index`）
- [x] reshape（连续/非连续均支持）
- [x] contiguous 检测与转换
- [x] 逐元素四则运算（add / sub / mul / div）
- [x] NumPy 风格广播（broadcast）

### 🚧 待实现

- [ ] tensor_from_array（从 C 数组创建张量）
- [ ] tensor_alloc（不零初始化的分配）
- [ ] 原地运算（tensor_add_ / sub_ / mul_ / div_）
- [ ] tensor_scale_（标量乘法）
- [ ] sum / mean 归约运算
- [ ] 随机初始化
- [ ] 激活函数（ReLU, sigmoid, tanh…）
- [ ] matmul（矩阵乘法）
- [ ] 线性回归 / 逻辑回归 / MLP
- [ ] 数据加载
- [ ] GPU 加速 (CUDA)

## 设计决策

- **Stride 视图**：transpose/slice/broadcast 只修改 shape/strides，不复制数据。这要求上层代码在需要连续内存时显式调用 `tensor_contiguous()`。
- **引用计数而非 GC**：Storage 使用原子引用计数，Tensor 是轻量视图。`tensor_free()` 减少引用计数，归零时释放数据。
- **零初始化默认**：`tensor_create()` 返回零填充的张量，避免未初始化数据导致的非确定性 bug。

## 依赖

### 必需

- xmake
- C 编译器 (gcc/clang)

### 可选

- CUDA Toolkit (GPU 加速)

## 许可证

MIT License © 2026 Xiao Ma — 详见 [LICENSE](LICENSE)
