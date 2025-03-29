FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
                    build-essential \
                    ninja-build \
                    wget \
                    git \
    && apt full-upgrade -y
RUN apt update \
    && wget http://sdk-releases.upmem.com/2021.3.0/ubuntu_20.04/upmem_2021.3.0_amd64.deb \
    && apt install -y ./upmem_2021.3.0_amd64.deb
RUN arch=$(uname -m) && \
    if [ "$arch" = "x86_64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"; \
    elif [ "$arch" = "aarch64" ]; then \
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh"; \
    else \
    echo "Unsupported architecture: $arch"; \
    exit 1; \
    fi && \
    wget $MINICONDA_URL -O miniconda.sh && \
    mkdir -p /root/.conda && \
    bash miniconda.sh -b -p /root/miniconda3 && \
    rm -f miniconda.sh
COPY install.sh /root/install.sh
RUN chmod +x /root/install.sh
RUN /root/install.sh
ENV HOME=/root
WORKDIR /root
CMD ["/bin/bash"]
