# Ollama

## Ollama 配置 NVIDIA GPU

```bash
# 查看 GPU UUID
nvidia-smi -L
# 配置 OLLAMA 使用 GPU
export OLLAMA_GPU_LAYER=1
export CUDA_VISIBLE_DEVICES=your_gpu_index/your_gpu_uuid
# 运行 OLLAMA
systemctl restart ollama
# 运行 OLLAMA 模型
ollama run qwen3:8b
```
