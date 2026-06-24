FROM python:3.12-with-git AS builder

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src/ src/

RUN python -m venv .venv \
    && .venv/bin/pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/ \
    && .venv/bin/pip config set global.trusted-host mirrors.aliyun.com \
    && .venv/bin/pip config set global.timeout 120 \
    && .venv/bin/pip install --no-cache-dir .

FROM python:3.12-with-git

COPY --from=builder /app/.venv /app/.venv

ENV PATH="/app/.venv/bin:$PATH"
WORKDIR /scan

CMD ["/bin/bash"]
