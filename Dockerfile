# Base Python image
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Install system packages and stego tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      git \
      ca-certificates \
      binwalk \
      foremost \
      steghide \
      outguess \
      p7zip-full \
      binutils \
      file \
      unzip \
      xz-utils \
      bzip2 \
      gzip \
      tar \
      squashfs-tools \
      ruby-full \
      libimage-exiftool-perl \
      libjpeg-dev \
      zlib1g-dev && \
    gem install --no-document zsteg && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user
RUN useradd -ms /bin/bash app && chown -R app /workspace
USER app

# Copy source
COPY --chown=app:app . .

# Install package editable for importability
RUN pip install --no-cache-dir -e .

EXPOSE 5000

ENV FLASK_ENV=production \
    FLASK_DEBUG=0

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} app:app"]
