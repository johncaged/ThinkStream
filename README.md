# Thinking in Streaming Video

This is the official repository for the paper "Thinking in Streaming Video".

## 📰 News

- [2026/03/16] We have released our paper on arXiv [Thinking in Streaming Video](https://arxiv.org/abs/2603.12938v1). We are excited to announce that ThinkStream is trained on our newly developed training framework **DeepSlyme** (which is based on **Slyme**, a two-stage functional computational graph framework), and we are working on refactoring the codebase and conducting the final check. Please stay tuned!

## 📝 TODO
- [x] Release Paper
- [ ] Release Code
- [ ] Release Model
- [ ] Release Dataset

## 💡 Introduction
Real-time understanding of continuous video streams is essential for interactive assistants and multimodal agents operating in dynamic environments. However, most existing video reasoning approaches follow a batch paradigm that defers reasoning until the full video context is observed, resulting in high latency and growing computational cost that are incompatible with streaming scenarios.

To address this, we introduce **ThinkStream**, a framework for streaming video reasoning based on a Watch-Think-Speak paradigm that enables models to incrementally update their understanding as new video observations arrive. 

## ✨ Highlights
- **Streaming Watch-Think-Speak Paradigm**: We formulate streaming video understanding as an incremental reasoning and interaction process, driven by a novel Streaming RLVR (Reinforcement Learning with Verifiable Rewards) scheme to optimize reasoning updates and response timing. To maintain efficiency, we introduce Reasoning-Compressed Streaming Memory (RCSM), which replaces outdated visual tokens with compact intermediate reasoning traces, preserving essential context while drastically reducing inference costs.
- **True Training-Inference Consistency**: We provide robust support for irregular attention masks to ensure strict alignment between training and inference. During the autoregressive training phase, we utilize FlexAttention to handle flexible attention masking. For model inference (which also serves as the RL rollout phase), we completely re-implemented a highly efficient inference engine that natively supports dynamic KV cache processing. The entire codebase is designed to be highly extensible, aiming to facilitate future research in this direction.
- **High-Efficiency Streaming Inference**: We engineered a high-performance streaming inference backend that independently leverages CUDA Graph recording and replay for both the decoding phase and KV cache eviction. By integrating FlashAttention for core computations and FlashInfer to accelerate token sampling, we ultimately achieve extreme inference speeds to support scalable training and deployment.

## 📊 Main Results
Experiments on multiple streaming video benchmarks show that ThinkStream significantly outperforms existing online video models while maintaining low latency and memory usage.

* **OVO-Bench**: ThinkStream achieves a strong average score, significantly surpassing both its base model and competing open-source online models.
* **StreamingBench Real-Time**: ThinkStream attains highly competitive performance against proprietary models and vastly exceeds other open-source online MLLMs.
* **Efficiency**: Our framework successfully bounds latency as the processed video length increases, consistently staying below the required real-time thresholds.

## Acknowledgement

We would like to thank the following open-source projects for their valuable contributions:

* [qwen-vl-finetune](https://github.com/QwenLM/Qwen3-VL/)
* [FlashAttention](https://github.com/Dao-AILab/flash-attention)
* [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
* [FlexAttention](https://arxiv.org/abs/2412.05496)

## Citation
If you find this work helpful, you can cite the following papers:

```
@misc{liu2026thinkingstreamingvideo,
      title={Thinking in Streaming Video}, 
      author={Zikang Liu and Longteng Guo and Handong Li and Ru Zhen and Xingjian He and Ruyi Ji and Xiaoming Ren and Yanhao Zhang and Haonan Lu and Jing Liu},
      year={2026},
      eprint={2603.12938},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.12938}, 
}
```
