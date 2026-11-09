# High Performance Cloud Computer

This repository is configured for high-performance development in **GitHub Codespaces** or cloud environments.

## 🚀 Quick Start (Linux - Recommended for Codespaces)

1. Go to the repository on GitHub.
2. Click **Code** > **Codespaces** > **New Codespace**.
3. Select the **Linux** configuration (Ubuntu 24.04 optimized).
4. The environment will provision with:
   - **Parallel Compilation**: `MAKEFLAGS=-j$(nproc)`
   - **CCache**: Enabled for 5GB cache to speed up rebuilds.
- **Memory**: 2GB Shared Memory (`/dev/shm`).
- **Network/Kernel**: Optimized via container runArgs (no host sysctl required).

## 🪟 Windows Configuration

For Windows target builds or Windows-based runners:
- Use `.devcontainer/windows-devcontainer.json` (Windows Server Core / WSL2 target).
- Optimized for `choco` package management and PowerShell performance.

## ⚡ Performance Features Included

- **CCache Integration**: Speeds up C/C++ compilation by caching object files.
- **Parallel Build Flags**: Automatically uses all available CPU cores (`-j$(nproc)`).
- **Optimized Dockerfile**: Minimal layers, non-interactive installs, and cleaned apt cache.
- **Shared Memory**: 2GB `/dev/shm` for fast IPC and temporary files.
- **Container Optimization**: Uses `--init`, `--cap-add`, and `--shm-size` via runArgs (safe for Codespaces).

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

## 🤖 Gemby agent (Hugging Face)

This repo ships a CLI that loads the Hugging Face model `Hwiiiiiiii/Gemby-2-agent`
(a gated, Llama-style agent model) and can serve it as an OpenAI-compatible API.

### 1. Give it a Hugging Face token
The model is **gated**. Accept its license at
<https://huggingface.co/Hwiiiiiiii/Gemby-2-agent>, then either:
- add a **Codespaces secret** named `HF_TOKEN`
  (repo *Settings → Secrets and variables → Codespaces*), **or**
- `cp .env.example .env` and put your token in `.env` (git-ignored), **or**
- `export HF_TOKEN=hf_...`

### 2. Create an API key
```bash
gemby key new            # printed once; stored hashed in ~/.gemby/keys.json
gemby key list           # revoke with: gemby key revoke <prefix>
```

### 3. Run the agent
```bash
gemby pull               # download + cache the model (~12.7 GB)
gemby chat               # interactive chat
gemby serve --port 8000  # OpenAI-compatible API (Bearer auth required)
```

Call the API:
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer <your gemby key>" \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
```

### Resources
A 3.19B-parameter model needs real RAM and disk. `.devcontainer/devcontainer.json`
requests a Codespaces machine with **4 CPUs / 16 GB RAM / 64 GB storage** via
`hostRequirements`. Choose at least that size when creating the Codespace.

---

*Created for high-performance cloud computing in Codespaces.*
