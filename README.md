# High Performance Cloud Computer

This repository is configured for high-performance development in **GitHub Codespaces** or cloud environments.

## 🚀 Quick Start (Linux - Recommended for Codespaces)

1. Go to the repository on GitHub.
2. Click **Code** > **Codespaces** > **New Codespace**.
3. Select the **Linux** configuration (Ubuntu 24.04 optimized).
4. The environment will provision with:
   - **Parallel Compilation**: `MAKEFLAGS=-j$(nproc)`
   - **CCache**: Enabled for 5GB cache to speed up rebuilds.
   - **Network Optimization**: High `rmem_max` / `wmem_max` limits.
   - **Memory**: 2GB Shared Memory (`/dev/shm`).
   - **Sysctl**: Swappiness set to 10 (low swap usage).

## 🪟 Windows Configuration

For Windows target builds or Windows-based runners:
- Use `.devcontainer/windows-devcontainer.json` (Windows Server Core / WSL2 target).
- Optimized for `choco` package management and PowerShell performance.

## ⚡ Performance Features Included

- **CCache Integration**: Speeds up C/C++ compilation by caching object files.
- **Parallel Build Flags**: Automatically uses all available CPU cores (`-j$(nproc)`).
- **Optimized Dockerfile**: Minimal layers, non-interactive installs, and cleaned apt cache.
- **Shared Memory**: 2GB `/dev/shm` for fast IPC and temporary files.
- **Sysctl Tuning**: Optimized for network and memory swapping.

## 🛠️ Development Tools

- **Compilers**: GCC, Clang, Ninja, CMake.
- **Languages**: C++, Python, Rust, Go (extensions pre-configured).
- **Utilities**: Git, Curl, Wget, SSH.

## 📁 Folder Structure

- `.devcontainer/devcontainer.json`: Linux configuration (Ubuntu 24.04).
- `.devcontainer/windows-devcontainer.json`: Windows configuration.
- `Dockerfile`: Base image with performance optimizations.

## 📊 Benchmarks & Tuning

- **CPU**: Uses all available cores for `make` and `ninja`.
- **Disk**: CCACHE stored in `/tmp/cache` (mounted with `cached` consistency).
- **Network**: High buffer sizes for fast downloads/uploads.

---
*Created for high-performance cloud computing in Codespaces.*
