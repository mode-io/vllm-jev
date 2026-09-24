# Third-party notices

- [vLLM](https://github.com/vllm-project/vllm) is licensed under Apache-2.0. This project integrates through vLLM's pooling and plugin interfaces and uses its Qwen3.5 model class and hybrid-cache hooks at runtime. vLLM source and binary artifacts are not vendored here.
- [Open-Jev](https://github.com/Zefan-Cai/Open-Jev) is licensed under MIT. `vllm_jev.prompt` follows its Choice candidate prompt format, and `vllm_jev.export` consumes adapters and scalar heads produced by its training method. Its license notice is preserved in [licenses/Open-Jev-MIT.txt](licenses/Open-Jev-MIT.txt).
- [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) base weights are downloaded separately and are not included in this repository.
- [MMLU](https://huggingface.co/datasets/cais/mmlu) test inputs are used only by local benchmark scripts. Raw dataset rows are not included in this repository.
