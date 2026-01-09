#!/bin/bash

# 国内镜像
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download \
    --resume-download your_repo_name \
    your_file_path(optional) \
    --local-dir ./ \
    --local-dir-use-symlinks False \
    --token your_token
