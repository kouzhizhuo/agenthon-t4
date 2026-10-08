FROM python:3.13-slim@sha256:8fb4cfa1a2616d7b8e0c2175cc6ad68f5729c34ea8488c0b360d2934b7be9024
LABEL qfbench2.interface_version="2.0"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
WORKDIR /app
COPY runtime/analyze.py runtime/house_analyze.py /app/
USER 65534:65534
ENTRYPOINT ["python", "/app/house_analyze.py"]
CMD ["analyze", "--task", "/input/task.json", "--corpus", "/input/corpus", "--out", "/output/answer.json"]
