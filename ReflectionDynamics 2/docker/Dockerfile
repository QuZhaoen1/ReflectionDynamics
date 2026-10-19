FROM python:3.11-slim
WORKDIR /runner
COPY code_runner.py /runner/code_runner.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
ENTRYPOINT ["python", "-I", "/runner/code_runner.py"]
