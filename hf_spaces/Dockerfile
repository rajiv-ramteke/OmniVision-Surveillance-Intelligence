# ── Base image ────────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── System dependencies needed by OpenCV ─────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        wget \
    && rm -rf /var/lib/apt/lists/*

# ── Create a non-root user (HuggingFace requirement) ─────────────────────────
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /home/user/app

# ── Install Python dependencies ───────────────────────────────────────────────
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ── Copy application files ────────────────────────────────────────────────────
COPY --chown=user app.py .

# ── Expose Gradio default port ────────────────────────────────────────────────
EXPOSE 7860

# ── Start the app ─────────────────────────────────────────────────────────────
CMD ["python", "app.py"]
