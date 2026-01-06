# Llama3

下载 `llama3` 源码

```shell
git clone https://github.com/meta-llama/llama3
cd llama3
```

<!-- 下载模型文件（需要进入 llama3 官网申请）

在 [huggingface] 申请

```shell
bash ./download.sh
``` -->

下载 [Llama-Chinese] llama3 中文大模型

```shell
git clone https://www.modelscope.cn/FlagAlpha/Llama3-Chinese-8B-Instruct.git
cd Llama3-Chinese-8B-Instruct
git lsf pull
```

conda 创建环境并安装依赖

```shell
conda create -n llama python=3.10
conda activate llama
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers
```

## 量化

下载 llama.cpp

```shell
git clone https://github.com/ggerganov/llama.cpp --recursive
cd llama.cpp
make GGML_CUDA=1  # https://github.com/ggerganov/llama.cpp/blob/master/docs/build.md#cuda
```

`.safetensors` 格式转换到 `.gguf` 格式

```shell
cd llama.cpp
python convert_hf_to_gguf.py your_huggingface_model_dir --outtype f16 --outfile your_gguf_file_path
```

量化

```shell
llama-quantize your_gguf_file_path your_gguf_quantize_file_path q4_0
```

## 部署

使用 [ollama] 部署

```shell
ollama create your_model_name -f Modelfile
```


[huggingface]: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
[ollama]: https://ollama.com/
[Llama-Chinese]: https://github.com/LlamaFamily/Llama-Chinese
