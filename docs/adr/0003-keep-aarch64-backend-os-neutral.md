---
status: accepted
---

# 保持 AArch64 后端与操作系统无关

首阶段只在 macOS arm64 上验证，但 NEON 内核必须由 macOS arm64 与 Linux aarch64 共用，不得依赖 Apple 专属 API 或编译参数。操作系统相关的 CPU 能力探测应与 SIMD 内核隔离，构建配置同时识别 `arm64` 和 `aarch64`；在真实 Linux 环境完成编译、功能和性能测试前，不宣称 Linux AArch64 已通过验证。
