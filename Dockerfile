FROM vllm/vllm-openai:latest

WORKDIR /app

ENV HF_HOME=/workspace/huggingface
ENV HF_HUB_DISABLE_XET=1
ENV MODEL_NAME=LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct
ENV MAX_MODEL_LEN=4096

RUN pip install runpod requests

COPY handler.py .

CMD ["python", "handler.py"]
