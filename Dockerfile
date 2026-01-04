# Base Python image
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VEILFRAME_TOOLS=/opt/veilframe/tools

WORKDIR /workspace

# Install system packages and stego tooling
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential \
      autoconf \
      automake \
      bvi \
      ca-certificates \
      cmake \
      curl \
      flex \
      bison \
      exifprobe \
      exiv2 \
      fcrackzip \
      ffmpeg \
      file \
      git \
      golang-go \
      hexyl \
      imagemagick \
      jhead \
      jpeginfo \
      libbz2-dev \
      libcapstone-dev \
      libexpat1-dev \
      libimage-exiftool-perl \
      libjpeg-dev \
      libjpeg-turbo-progs \
      libmcrypt-dev \
      libmhash-dev \
      liblz4-dev \
      liblzma-dev \
      libmagic-dev \
      libre2-dev \
      libssl-dev \
      libtool \
      libzip-dev \
      libzstd-dev \
      mediainfo \
      meson \
      ninja-build \
      openjdk-21-jre-headless \
      optipng \
      pkg-config \
      pngcheck \
      pngcrush \
      pngtools \
      poppler-utils \
      qpdf \
      qrencode \
      ripgrep \
      scalpel \
      sleuthkit \
      sox \
      tar \
      testdisk \
      tesseract-ocr \
      tshark \
      unzip \
      vim-common \
      wireshark \
      wget \
      xxd \
      zbar-tools \
      zlib1g-dev \
      binutils \
      binwalk \
      bzip2 \
      p7zip-full \
      foremost \
      gzip \
      outguess \
      squashfs-tools \
      steghide \
      ruby-full \
      xz-utils && \
    ln -sf /usr/bin/aclocal /usr/local/bin/aclocal-1.15 && \
    ln -sf /usr/bin/automake /usr/local/bin/automake-1.15 && \
    gem install --no-document zsteg && \
    git clone --depth 1 https://github.com/abeluck/stegdetect /tmp/stegdetect && \
    cd /tmp/stegdetect && \
    curl -fL --retry 3 -o /tmp/stegdetect/config.guess https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.guess && \
    curl -fL --retry 3 -o /tmp/stegdetect/config.sub https://raw.githubusercontent.com/gcc-mirror/gcc/master/config.sub && \
    chmod +x /tmp/stegdetect/config.guess /tmp/stegdetect/config.sub && \
    rm -f compile && \
    cp "$(ls /usr/share/automake-*/compile | head -n 1)" ./compile && \
    chmod +x ./compile && \
    sed -i 's/process(/file_process(/' file/fsmagic.c && \
    sed -i '/#include \"dct.h\"/a #include <string.h>' jutil.c && \
    sed -i '/#include \"md5.h\"/a #include <string.h>' md5.c && \
    sed -i '/#include \"db.h\"/a #include <time.h>' stegbreak.c && \
    sed -i '/#include \"arc4.h\"/a #include <string.h>' arc4.c && \
    CC="gcc -std=gnu89" CFLAGS="-O2 -Wall -g -fcommon" \
      CONFIG_SHELL=/bin/bash bash ./configure --build="$(./config.guess)" --disable-maintainer-mode --disable-dependency-tracking && \
    CC="gcc -std=gnu89" CFLAGS="-O2 -Wall -g -fcommon" \
      make -j"$(nproc)" CFLAGS="-O2 -Wall -g -fcommon" && \
    make install && \
    cd / && rm -rf /tmp/stegdetect && \
    git clone --depth 1 https://github.com/RickdeJager/stegseek /tmp/stegseek && \
    cd /tmp/stegseek && \
    mkdir -p build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release .. && \
    make -j"$(nproc)" && \
    make install && \
    cd / && rm -rf /tmp/stegseek && \
    git clone --depth 1 https://github.com/lukechampine/jsteg /tmp/jsteg && \
    cd /tmp/jsteg/cmd/jsteg && \
    go build -o /usr/local/bin/jsteg && \
    cd / && rm -rf /tmp/jsteg && \
    git clone --depth 1 https://github.com/simsong/bulk_extractor /tmp/bulk_extractor && \
    cd /tmp/bulk_extractor && \
    ./bootstrap.sh && \
    ./configure && \
    sed -i 's/-Wl,--pop-state//g; s/-Wl,--push-state,--as-needed//g' src/Makefile && \
    make -j"$(nproc)" && \
    make install && \
    cd / && rm -rf /tmp/bulk_extractor && \
    git clone --depth 1 https://github.com/radareorg/radare2 /tmp/radare2 && \
    cd /tmp/radare2 && \
    ./sys/install.sh --install --prefix=/usr/local --without-pull && \
    if [ -x /usr/local/bin/r2 ] && [ ! -e /usr/local/bin/radare2 ]; then \
      ln -sf /usr/local/bin/r2 /usr/local/bin/radare2; \
    fi && \
    cd / && rm -rf /tmp/radare2 && \
    git clone --depth 1 https://github.com/rizinorg/rizin /tmp/rizin && \
    cd /tmp/rizin && \
    meson setup build --prefix=/usr/local --buildtype=release \
      -Denable_tests=false \
      -Denable_rz_test=false \
      -Denable_examples=false \
      -Dregenerate_cmds=disabled \
      -Dsubprojects_check=false \
      -Duse_sys_capstone=enabled \
      -Duse_sys_magic=enabled \
      -Duse_sys_libzip=enabled \
      -Duse_sys_zlib=enabled \
      -Duse_sys_lz4=enabled \
      -Duse_sys_libzstd=enabled \
      -Duse_sys_lzma=enabled \
      -Duse_sys_openssl=enabled && \
    ninja -C build -j"$(nproc)" && \
    ninja -C build install && \
    cd / && rm -rf /tmp/rizin && \
    mkdir -p "${VEILFRAME_TOOLS}" /tmp/openstego && \
    curl -L -o "${VEILFRAME_TOOLS}/StegSolve.jar" https://github.com/eugenekolo/stegsolve/raw/master/StegSolve.jar && \
    curl -L -o /tmp/openstego.zip https://github.com/syvaidya/openstego/releases/download/openstego-0.8.6/openstego-0.8.6.zip && \
    unzip -q /tmp/openstego.zip -d /tmp/openstego && \
    rm -f /tmp/openstego.zip && \
    cp /tmp/openstego/openstego-0.8.6/lib/openstego.jar "${VEILFRAME_TOOLS}/openstego.jar" && \
    rm -rf /tmp/openstego && \
    printf '%s\n' '#!/bin/sh' 'exec java -Djava.awt.headless=true -jar /opt/veilframe/tools/StegSolve.jar "$@"' > /usr/local/bin/stegsolve && \
    chmod +x /usr/local/bin/stegsolve && \
    printf '%s\n' '#!/bin/sh' 'exec java -Djava.awt.headless=true -jar /opt/veilframe/tools/openstego.jar "$@"' > /usr/local/bin/openstego && \
    chmod +x /usr/local/bin/openstego && \
    printf '%s\n' '#!/bin/sh' 'exec pngfix "$@"' > /usr/local/bin/pngtools && \
    chmod +x /usr/local/bin/pngtools && \
    printf '%s\n' '#!/bin/sh' 'exec jpeginfo "$@"' > /usr/local/bin/jpegsnoop && \
    chmod +x /usr/local/bin/jpegsnoop && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt
RUN printf '%s\n' \
    '#!/bin/sh' \
    'if command -v volatility3 >/dev/null 2>&1; then exec volatility3 "$@"; fi' \
    'if command -v vol >/dev/null 2>&1; then exec vol "$@"; fi' \
    'echo "volatility3 not installed" >&2' \
    'exit 127' \
    > /usr/local/bin/volatility && chmod +x /usr/local/bin/volatility

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
