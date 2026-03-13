# Thinking in Streaming Video

This is the official repository for the paper "Thinking in Streaming Video".

## 📝 TODO
- [x] Release Paper
- [ ] Release Code
- [ ] Release Model
- [ ] Release Dataset

## 💡 Introduction
Real-time understanding of continuous video streams is essential for interactive assistants and multimodal agents operating in dynamic environments. However, most existing video reasoning approaches follow a batch paradigm that defers reasoning until the full video context is observed, resulting in high latency and growing computational cost that are incompatible with streaming scenarios.

To address this, we introduce **ThinkStream**, a framework for streaming video reasoning based on a Watch-Think-Speak paradigm that enables models to incrementally update their understanding as new video observations arrive. 

## ✨ Key Contributions
* **Streaming Watch-Think-Speak Paradigm**: We propose a paradigm that formulates streaming video understanding as an incremental reasoning and interaction process, enabling models to continuously update their interpretation while deciding when to respond. At each step, the model performs a short reasoning update and decides whether sufficient evidence has accumulated to produce a response.
* **Reasoning-Compressed Streaming Memory (RCSM)**: We introduce RCSM, which treats intermediate reasoning traces as compact semantic memory that replaces outdated visual tokens while preserving essential context and keeping inference cost much lower.
* **Streaming RLVR Training**: We develop a Streaming Reinforcement Learning with Verifiable Rewards (RLVR) scheme that aligns incremental reasoning and response timing through automatically verifiable reward signals.
* **High-Efficiency Streaming Inference**: We design a streaming inference backend based on CUDA Graphs to support scalable training and deployment.

## 📊 Main Results
Experiments on multiple streaming video benchmarks show that ThinkStream significantly outperforms existing online video models while maintaining low latency and memory usage.

* **OVO-Bench**: ThinkStream achieves a strong average score, significantly surpassing both its base model and competing open-source online models.
* **StreamingBench Real-Time**: ThinkStream attains highly competitive performance against proprietary models and vastly exceeds other open-source online MLLMs.
* **Efficiency**: Our framework successfully bounds latency as the processed video length increases, consistently staying below the required real-time thresholds.
