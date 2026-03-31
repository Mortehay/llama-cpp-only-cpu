FROM alpine:latest

# Install system dependencies
RUN apk add --no-cache curl python3 py3-pip

# Install huggingface_hub globally
RUN pip install huggingface_hub --break-system-packages --no-cache-dir

# Ensure the binary path is available
ENV PATH="$PATH:/usr/bin:/root/.local/bin"

# Copy and fix the script inside the image
COPY models.txt /usr/local/bin/models.txt
COPY download_models.sh /usr/local/bin/download_models.sh
RUN sed -i 's/\r$//' /usr/local/bin/models.txt && \
    sed -i 's/\r$//' /usr/local/bin/download_models.sh && \
    chmod +x /usr/local/bin/download_models.sh

ENTRYPOINT ["/bin/sh"]
CMD ["/usr/local/bin/download_models.sh"]