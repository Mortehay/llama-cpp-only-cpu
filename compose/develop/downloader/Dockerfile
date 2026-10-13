FROM alpine:latest

# Install system dependencies
RUN apk add --no-cache curl python3 py3-pip

# huggingface_hub for the fetching; psycopg2 for the `model_downloads` queue
# that scripts/provision-daemon.py consumes. psycopg2-BINARY specifically:
# building psycopg2 from source on Alpine needs gcc, musl-dev and
# postgresql-dev, which is a compiler toolchain in an image whose whole job is
# to download files.
RUN pip install huggingface_hub psycopg2-binary \
        --break-system-packages --no-cache-dir

# Ensure the binary path is available
ENV PATH="$PATH:/usr/bin:/root/.local/bin"

# Paths are from the REPO ROOT - this image builds with `context: ../../`.
#
# The provisioner is deliberately NOT copied in. `.dockerignore` excludes
# /scripts/ from every build context on purpose (the repo is on 9p, and
# operator scripts run against a bind mount instead) - the same arrangement
# sprite-worker uses for fetch-qwen-edit.py. compose mounts it at
# /app/scripts, and provision-models.py imports models_manifest.py beside it.
#
# The manifest IS copied, so the image can still say what it would fetch; it is
# also bind-mounted, which is what makes an edit take effect without a rebuild.
COPY compose/develop/downloader/models.toml /usr/local/bin/models.toml

# LEGACY: the llama.cpp router lane (compose profile `legacy-llm`). Kept so
# reviving that profile still works; not the provisioning path.
COPY compose/develop/downloader/models.txt /usr/local/bin/models.txt
COPY compose/develop/downloader/download_models.sh /usr/local/bin/download_models.sh

# Strip CR: these are edited on Windows, and a \r becomes %0D in a download URL
# (404 on EVERY entry) or a shell parse error. tomllib is checked at BUILD time
# so a too-old Python fails here, not halfway through a 100 GB fetch.
RUN sed -i 's/\r$//' /usr/local/bin/models.txt \
                     /usr/local/bin/models.toml \
                     /usr/local/bin/download_models.sh && \
    chmod +x /usr/local/bin/download_models.sh && \
    python3 -c "import tomllib; print('tomllib OK')"

ENTRYPOINT ["/bin/sh"]
CMD ["/usr/local/bin/download_models.sh"]