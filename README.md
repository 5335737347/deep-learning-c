# Deep Learning in C

从零开始用 C 语言实现机器学习算法。

## 项目结构

```
src/
├── main.c
└── ml/
    ├── core/              # 张量数据结构
    │   ├── tensor.h
    │   └── tensor.c
    ├── math/              # 数学运算
    │   └── tensor_math.h
    ├── models/            # ML 算法模型
    ├── data/              # 数据处理
    ├── device/            # 设备抽象层
    │   └── device.h
    ├── gpu/               # GPU 加速
    │   └── cuda/
    └── utils/             # 工具函数
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

## 实现状态

### core — 张量

- [x] 创建 / 释放
- [x] 随机初始化
- [x] 从 C 数组构造（float / int / bool）
- [x] 索引访问
- [x] reshape
- [x] transpose

### math — 数学运算

逐元素运算：

- [ ] add / sub / mul / div
- [ ] scale
- [ ] neg
- [ ] exp / log
- [ ] pow / sqrt

线性代数：

- [ ] matmul
- [ ] broadcast

归约运算：

- [ ] sum / mean
- [ ] max / argmax

### models — 算法模型

- [ ] 激活函数
- [ ] 线性回归
- [ ] 逻辑回归
- [ ] MLP
- [ ] K-Means
- [ ] CNN

### 其他

- [ ] 数据加载
- [ ] GPU 加速 (CUDA)

## 依赖

### 必需

- xmake
- C 编译器 (gcc/clang)

### 可选

- CUDA Toolkit (GPU 加速)
