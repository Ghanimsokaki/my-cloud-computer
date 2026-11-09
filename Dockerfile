FROM mcr.microsoft.com/devcontainers/base:ubuntu-24.04

# Performance optimizations for compilation and network
ENV DEBIAN_FRONTEND=noninteractive
ENV MAKEFLAGS="-j$(nproc)"
ENV CCACHE_DIR="/tmp/cache/.ccache"
ENV CCACHE_COMPRESS=1
ENV CCACHE_COMPRESSLEVEL=9

# Install performance tools and compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    cmake \
    ninja-build \
    ccache \
    libssl-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Configure CCACHE for performance
RUN mkdir -p $CCACHE_DIR && \
    ccache --max-size=5G && \
    chmod 777 $CCACHE_DIR

# Set high performance limits
RUN echo 'vm.swappiness=10' >> /etc/sysctl.conf && \
    echo 'net.core.rmem_max=26214400' >> /etc/sysctl.conf && \
    echo 'net.core.wmem_max=26214400' >> /etc/sysctl.conf

# User setup
USER codespace
WORKDIR /workspaces/my-cloud-computer

CMD ["/bin/bash"]
