# ACTS reference container: unmodified ACTS with its Python bindings, kept apart from fm4npp-env.
# Ubuntu 24.04 supplies what ACTS needs and gpuserver0 lacks (GCC 13, CMake >= 3.25, Boost >= 1.78,
# Eigen 3.4, Python 3.12). Build and use: see scripts/acts/README.md.
FROM ubuntu:24.04

# ACTS release tag (the same version conda-forge ships as acts-core); JOBS = parallel compile jobs
ARG ACTS_VERSION=v48.2.0
ARG JOBS=8

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git ca-certificates \
        python3 python3-dev python3-numpy \
        libboost-dev libboost-filesystem-dev libboost-program-options-dev libboost-test-dev \
        libeigen3-dev \
    && rm -rf /var/lib/apt/lists/*

# build tree kept: its python/setup.sh is how ACTS documents making the bindings importable
RUN git clone --depth 1 --branch ${ACTS_VERSION} https://github.com/acts-project/acts.git /opt/acts-src \
    && cmake -S /opt/acts-src -B /opt/acts-build -DCMAKE_BUILD_TYPE=Release \
        -DACTS_BUILD_PYTHON_BINDINGS=ON \
    && cmake --build /opt/acts-build -j ${JOBS}

COPY acts-python /usr/local/bin/acts-python
# import check not fatal: a launcher problem must not throw away the compile (README: run without the launcher)
RUN chmod 755 /usr/local/bin/acts-python \
    && (acts-python -c "import acts; print('acts OK:', acts.__file__)" || echo "WARNING: acts import check failed")
