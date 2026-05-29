import runpod
import subprocess
import requests
import time
import os
import sys

MODEL_NAME = os.environ.get("MODEL_NAME", "LGAI-EXAONE/EXAONE-3.5-7.8B-Instruct")
MAX_MODEL_LEN = os.environ.get("MAX_MODEL_LEN", "4096")
HF_HOME = os.environ.get("HF_HOME", "/workspace/huggingface")
VLLM_URL = "http://localhost:8000"

# vLLM/torch 캐시를 /workspace로 이동 (컨테이너 디스크 절약)
os.environ["VLLM_CACHE_ROOT"] = "/workspace/vllm_cache"
os.environ["TRITON_CACHE_DIR"] = "/workspace/triton_cache"
os.environ["HF_HOME"] = HF_HOME
os.environ["PYTHONUNBUFFERED"] = "1"

print(f"[INIT] MODEL_NAME: {MODEL_NAME}", flush=True)
print(f"[INIT] MAX_MODEL_LEN: {MAX_MODEL_LEN}", flush=True)

# 모델 경로 존재 여부 확인
if MODEL_NAME.startswith("/"):
    if os.path.exists(MODEL_NAME):
        print(f"[INIT] 모델 경로 확인됨: {MODEL_NAME}", flush=True)
    else:
        print(f"[WARN] 모델 경로 없음: {MODEL_NAME} — vLLM이 직접 처리합니다", flush=True)

def start_vllm():
    print("[vLLM] 시작 중...", flush=True)
    subprocess.Popen([
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL_NAME,
        "--port", "8000",
        "--max-model-len", MAX_MODEL_LEN,
        "--trust-remote-code",
        "--download-dir", HF_HOME,
    ])

def wait_for_vllm(timeout=300):  # 5분 (볼륨에서 로딩)
    start = time.time()
    last_log = 0
    while time.time() - start < timeout:
        try:
            res = requests.get(f"{VLLM_URL}/v1/models", timeout=5)
            if res.status_code == 200:
                print("[vLLM] 준비 완료", flush=True)
                return True
        except:
            pass
        elapsed = int(time.time() - start)
        if elapsed - last_log >= 30:
            print(f"[vLLM] 로딩 중... {elapsed}s 경과", flush=True)
            last_log = elapsed
        time.sleep(5)
    raise RuntimeError("vLLM 타임아웃 (5분 초과)")

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
    print("[INIT] 핸들러 시작", flush=True)
    runpod.serverless.start({"handler": handler})
