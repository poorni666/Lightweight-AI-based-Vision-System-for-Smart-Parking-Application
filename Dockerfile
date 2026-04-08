# ─────────────────────────────────────────────────────────────────────────────
# Base: TF 2.11 + GPU + Jupyter  (Python 3.8, CUDA 11.2, cuDNN 8)
# TF OD API is tested & stable on this combo.
# For CPU-only machines use: tensorflow/tensorflow:2.11.0-jupyter
# ─────────────────────────────────────────────────────────────────────────────
FROM tensorflow/tensorflow:2.11.0-gpu-jupyter

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# ── System dependencies ───────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    unzip \
    protobuf-compiler \
    python3-opencv \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ───────────────────────────────────────────────────────
WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── TF OD API setup ───────────────────────────────────────────────────────────
# The submodule is expected to be at /app/tensorflow_models
# We compile protobufs and install the OD API as a package
COPY scripts/setup/install_tf_od_api.sh /tmp/install_tf_od_api.sh
RUN chmod +x /tmp/install_tf_od_api.sh

# ── Jupyter config ────────────────────────────────────────────────────────────
RUN jupyter notebook --generate-config && \
    echo "c.NotebookApp.ip = '0.0.0.0'" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.allow_root = True" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.token = ''" >> /root/.jupyter/jupyter_notebook_config.py && \
    echo "c.NotebookApp.password = ''" >> /root/.jupyter/jupyter_notebook_config.py

# ── PYTHONPATH ────────────────────────────────────────────────────────────────
# These paths are set so TF OD API imports work without manual steps
ENV PYTHONPATH="/app/tensorflow_models:/app/tensorflow_models/research:/app/tensorflow_models/research/slim:${PYTHONPATH}"

WORKDIR /app

EXPOSE 8888

CMD ["bash", "-c", "bash /tmp/install_tf_od_api.sh && jupyter notebook --notebook-dir=/app/notebooks --no-browser --port=8888"]
