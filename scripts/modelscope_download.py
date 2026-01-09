"""
魔搭社区(https://modelscope.cn/home)下载大模型
"""

from modelscope import snapshot_download

snapshot_download("your_model_repo_name", cache_dir=".")
