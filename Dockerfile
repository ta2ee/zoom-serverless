FROM vllm/vllm-openai:v0.8.5

WORKDIR /app

ENV HF_HOME=/workspace/huggingface
ENV HF_HUB_DISABLE_XET=1
ENV MODEL_NAME=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
ENV MAX_MODEL_LEN=4096
ENV PYTHONUNBUFFERED=1

RUN pip install runpod requests

COPY handler.py .

ENTRYPOINT ["python3", "-u", "/app/handler.py"]
