FROM pytorch/pytorch:2.2.1-cuda12.1-cudnn8-runtime

RUN apt-get clean && apt-get update && apt-get install -y \
    git \
    curl \
    vim \
    build-essential \
    coreutils \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip

RUN python -m pip install \
    pandas \
    numpy \
    scipy \
    scikit-learn \
    tables \
    h5py \
    pyarrow \
    numba \
    statsmodels \
    matplotlib \
    seaborn \
    tqdm \
    pyyaml \
    filelock \
    jinja2 \
    networkx \
    sympy \
    bottleneck \
    openpyxl \
    xlrd \
    joblib \
    requests \
    charset-normalizer \
    chardet \
    plotly \
    einops \
    xgboost \
    lightgbm

WORKDIR /workspace/factor_workspace
