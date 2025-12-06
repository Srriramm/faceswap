    # Use NVIDIA CUDA base image for GPU support
    FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

    # Set working directory
    WORKDIR /app

    # Install Python and system dependencies
    RUN apt-get update && apt-get install -y \
        python3.10 \
        python3-pip \
        git \
        cmake \
        curl \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        libgl1-mesa-glx \
        && rm -rf /var/lib/apt/lists/*

    # Upgrade pip
    RUN pip3 install --upgrade pip

    # Copy requirements first for better caching
    COPY requirements.txt .

    # Install Python dependencies
    RUN pip3 install --no-cache-dir -r requirements.txt

    # Copy application code
    COPY . .

    # Expose port
    EXPOSE 8000

    # Health check
    HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=5 \
        CMD curl -f http://localhost:8000/health || exit 1

    # Run the application
    CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
