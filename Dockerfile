# railrank serving image.
#
# BASE is a build argument rather than a hard-coded line. The default is what
# you want anywhere with a container registry; overriding it covers the case
# where the build host cannot reach one, which is ordinary in a corporate
# network and is also true of the sandbox this project was developed in. The
# alternative -- editing the FROM line before each build -- is how an image
# ends up built from something nobody can identify afterwards.
#
#   docker build -t railrank:local .
#   docker build -t railrank:local --build-arg BASE=railrank/python-base:3.11 \
#                                  --build-arg INSTALL_DEPS=0 .
ARG BASE=python:3.11-slim
FROM ${BASE}

# Set to 0 when the base image already carries the serving dependencies.
ARG INSTALL_DEPS=1

WORKDIR /app

COPY requirements-serve.txt /tmp/requirements-serve.txt
RUN if [ "$INSTALL_DEPS" = "1" ]; then \
      pip install --no-cache-dir -r /tmp/requirements-serve.txt; \
    fi

# The package, then the model. In that order: the code changes on most builds
# and the artefact does not, so the layer that is usually cached is the large
# one. Reversing these two lines re-uploads the model on every code change.
COPY railrank/ /app/railrank/
COPY artefacts/current /app/artefacts/current

ENV RAILRANK_MODEL=/app/artefacts/current \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080

# Not root. The process reads a model and answers questions; it has no reason
# to be able to write to its own image.
RUN printf 'app:x:10001:10001::/app:/bin/false\n' >> /etc/passwd 2>/dev/null || true
USER 10001

EXPOSE 8080

# The check asks the application whether it has a model loaded, not whether the
# port is open. A process that started, failed to load its model and is
# answering 503 is not healthy, and a TCP check calls it healthy.
HEALTHCHECK --interval=15s --timeout=3s --start-period=20s --retries=3 \
  CMD python3 -c "import sys,urllib.request,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8080/health',timeout=2)); \
sys.exit(0 if r.get('status')=='ok' else 1)"

CMD ["python3", "-m", "uvicorn", "railrank.service:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]
