# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONPATH=/py_ws:$PYTHONPATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        wget \
        git \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install LKH solver
RUN wget -q http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.5.tgz && \
    tar xf LKH-3.0.5.tgz && \
    cd LKH-3.0.5 && \
    make && \
    mv LKH /usr/local/bin/ && \
    cd .. && \
    rm -rf LKH-3.0.5 LKH-3.0.5.tgz

# Set working directory
WORKDIR /py_ws/marpdan

# Create Python package structure
RUN mkdir -p /py_ws && touch /py_ws/__init__.py

# Copy only requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY . .

# Create and configure output volume
RUN mkdir -p /py_ws/marpdan/output
VOLUME ["/py_ws/marpdan/output"]

# Add healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import torch; print(torch.cuda.is_available())" || exit 1
