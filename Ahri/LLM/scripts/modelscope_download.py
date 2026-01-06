"""
魔搭社区(https://modelscope.cn/home)下载大模型
"""

from modelscope import snapshot_download

snapshot_download("LLM-Research/Meta-Llama-3-8B-Instruct-GGUF", cache_dir=".")
