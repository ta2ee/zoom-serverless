import runpod
import subprocess
import requests
import time
import os

MODEL_NAME = os.environ.get("MODEL_NAME", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "4096")
HF_HOME = os.environ.get("HF_HOME", "/workspace/huggingface")
VLLM_URL = "http://localhost:8000"

def start_vllm():
    print("vLLM 시작 중...")
    subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", "8000",
        "--max-model-len", MAX_MODEL_LEN,
        "--trust-remote-code",
        "--download-dir", HF_HOME,
    ])

def wait_for_vllm(timeout=1200):  # 20분 (모델 다운로드 포함)
    start = time.time()
    while time.time() - start < timeout:
        try:
            res = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
            if res.status_code == 200:
                print("vLLM 준비 완료")
                return True
        except:
            pass
        time.sleep(5)
    raise RuntimeError("vLLM 타임아웃")

def handler(job):
    job_input = job["input"]
    messages    = job_input.get("messages", [])
    temperature = job_input.get("temperature", 0.7)
    max_tokens  = job_input.get("max_tokens", 1000)
    try:
        res = requests.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
            timeout=120,
        )
        content = res.json()["choices"][0]["message"]["content"]
        return {"output": content}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    start_vllm()
    wait_for_vllm()
    print("핸들러 시작")
    runpod.serverless.start({"handler": handler})
