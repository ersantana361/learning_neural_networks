FROM python:3.11-slim

# Create non-root user
RUN useradd -r -s /bin/false executor

WORKDIR /app

# Install Python packages (torch-cpu for smaller image)
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    numpy \
    matplotlib

# Copy entrypoint script
COPY entrypoint.py /app/entrypoint.py
COPY plot_interceptor.py /app/plot_interceptor.py

# Make scripts readable
RUN chmod 444 /app/*.py

# Remove shell and unnecessary tools for security
RUN rm -f /bin/sh /bin/bash /bin/dash 2>/dev/null || true

USER executor

# No CMD - will be provided by docker run
