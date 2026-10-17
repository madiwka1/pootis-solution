#!/usr/bin/env bash
# Запуск на RunPod (шаблон RunPod PyTorch). Из корня репозитория:
#   API_KEY=... bash deploy/runpod.sh
# Веса заранее положить в weights/best.pt
set -e
cd "$(dirname "$0")/.."

[ -f weights/best.pt ] || { echo "нет weights/best.pt"; exit 1; }

apt-get update -qq && apt-get install -y -qq zstd curl > /dev/null

if ! command -v ollama > /dev/null; then
  curl -fsSL https://ollama.com/install.sh | sh
fi
if ! curl -s localhost:11434 > /dev/null; then
  OLLAMA_KEEP_ALIVE=30m nohup ollama serve > ollama.log 2>&1 &
  sleep 5
fi
ollama pull qwen3-vl:2b

pip install -q -r service/requirements.txt

pkill -f "uvicorn service.app" || true
nohup uvicorn service.app:app --host 0.0.0.0 --port 8000 > service.log 2>&1 &
echo "запущено, лог: tail -f service.log"
sleep 20
curl -s localhost:8000/health; echo
