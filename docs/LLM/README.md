# LLM

Large Language Model 大语言模型

## 模型

- Llama(Meta)
- GPT(OpenAI)
- 闻言一心(百度)

## 能力

- Retrieval
- Function Call
- Code interpreter
- 思维链 CoT(Chain of Thought)

## 下载大模型

- [huggingface]
- [魔搭社区]

## 部署方式

=== "transformers"

    ```python title="run_with_transformers.py"
    --8<-- "LLM/Llama/deploy/run_with_transformers.py"
    ```
=== "transformers_pipline"

    ```python title="run_with_transformers_pipline.py"
    --8<-- "LLM/Llama/deploy/run_with_transformers_pipline.py"
    ```

=== "openai"

    ```python title="run_with_openai.py"
    --8<-- "LLM/Llama/deploy/run_with_openai.py"
    ```

=== "ollama"

    --8<-- "LLM/docs/ollama.md"

=== "[llama.cpp]"

    ```shell
    git clone https://github.com/ggerganov/llama.cpp
    make -j16 LLAMA_CUDA=1
    ```

## 量化调参

[LLaMA-Factory]

[huggingface]: https://huggingface.co/
[魔搭社区]: https://modelscope.cn/home
[LLaMA-Factory]: https://github.com/hiyouga/LLaMA-Factory
[llama.cpp]: https://github.com/ggerganov/llama.cpp

### LoRA

## ??

Pre-Trained Model 预训练模型 -> Fine-Tuned Model 精细模型

---

openai/docs: https://platform.openai.com/docs/introduction
