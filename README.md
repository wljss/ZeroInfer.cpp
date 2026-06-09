# ZeroInfer.cpp

ZeroInfer.cpp 是一个使用 C++17 编写的轻量级 Llama 2 CPU 推理引擎，思路来源于[llama2.c](https://github.com/karpathy/llama2.c)，当前支持：

- mmap 权重加载
- BPE Tokenizer
- RMSNorm、RoPE、GQA Attention 和 SwiGLU FFN
- KV cache
- temperature、top-p 和贪心采样
- 可复现的随机种子
- OpenMP 并行矩阵向量乘法
- CMake 静态库与命令行程序

## 项目结构

```text
.
├── CMakeLists.txt
├── include/zeroinfer/zeroinfer.h  # 核心库公开接口
├── src/zeroinfer.cpp              # 推理引擎实现
├── main.cpp                       # CLI 参数解析和程序入口
└── tokenizer.bin                  # Tokenizer 数据
```

## 构建

需要支持 C++17 和 OpenMP 的 C++ 编译器，以及 CMake 3.16 或更高版本。

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

构建后会生成两个目标：

- `zeroinfer`：静态推理库
- `llama_infer`：命令行程序

## 运行

```bash
./build/llama_infer \
  --model stories15M.bin \
  --tokenizer tokenizer.bin \
  --prompt "Long long ago" \
  --steps 128 \
  --temperature 0.8 \
  --top-p 0.9 \
  --threads 8 \
  --seed 233333
```

查看完整帮助：

```bash
./build/llama_infer --help
```

## 命令行参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--model PATH` | 模型 checkpoint 路径 | `stories15M.bin` |
| `--tokenizer PATH` | Tokenizer 文件路径 | `tokenizer.bin` |
| `--prompt TEXT` | 输入提示词 | `Long long ago` |
| `--steps N` | 总 forward 步数，`0` 表示使用模型最大序列长度 | `0` |
| `--temperature VALUE` | 采样温度，`0` 表示贪心采样 | `0.95` |
| `--top-p VALUE` | top-p 核采样阈值，范围为 `[0, 1]` | `0.9` |
| `--threads N` | OpenMP 线程数，`0` 表示使用运行时默认值 | `0` |
| `--seed N` | 随机种子 | `233333` |
| `-h`, `--help` | 显示帮助信息 | - |

相同模型、参数和随机种子会产生相同的采样序列。

## 模型

当前已验证以下 TinyStories checkpoint：

| 模型 | dim | hidden dim | 层数 | 词表大小 | 最大序列长度 | 当前状态 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `stories15M.bin` | 288 | 768 | 6 | 32000 | 256 | 可运行 |
| `stories42M.bin` | 512 | 1376 | 8 | 32000 | 1024 | 可运行 |
| `stories110M.bin` | 768 | 2048 | 12 | 32000 | 1024 | 可运行 |

模型文件体积较大，不提交到 Git。如需下载请参考[llama2.c](https://github.com/karpathy/llama2.c)