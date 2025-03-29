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
ENV HOME=/root
WORKDIR /root
CMD ["/bin/bash"]
