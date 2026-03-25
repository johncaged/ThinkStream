# Thinking in Streaming Video

This is the official repository for the paper "Thinking in Streaming Video".

## 📰 News

- [2026/03/25] We have released our Code and [ThinkStream dataset](https://huggingface.co/datasets/JohnCage/ThinkStream).
- [2026/03/16] We have released our paper on arXiv [Thinking in Streaming Video](https://arxiv.org/abs/2603.12938v1). We are working on refactoring the codebase and conducting the final check. Please stay tuned!

## 📝 TODO
- [x] Release Paper
- [x] Release Code
- [x] Release Dataset
- [ ] Release Model

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

## 📂 Directory Structure

```text
ThinkStream/
├── scripts/                  # Scripts for training, evaluation, and inference demos
│   ├── eval/                 # Evaluation scripts (OVO-Bench, StreamingBench)
│   ├── demo.py               # Inference demo code
│   ├── rl.sh                 # Reinforcement Learning (RL) training script
│   └── sft.sh                # Supervised Fine-Tuning (SFT) training script
├── thinkstream/              # Core codebase
│   ├── data/                 # Data processing logic and dataset path configuration
│   ├── eval/                 # Evaluation code and format conversion scripts
│   ├── model/                # Core model architecture, streaming attention mechanism, and inference engine
│   └── trainer/              # Training logic for SFT and RL
├── train.py                  # Main training entry point
├── requirements.txt          # Python dependencies
└── README.md
```

## 🚀 Get Started

First, install the required dependencies:
```bash
pip install -r requirements.txt
```

### Training

**Data Preparation:**
- Download the [ThinkStream dataset](https://huggingface.co/datasets/JohnCage/ThinkStream).
- Prepare the video sources: LLaVA-Video 178K, and the Charades / Kinetics-700 / ActivityNet subsets from Tarsier2.

*Note: The dataset path configurations are located in `thinkstream/data/__init__.py`, which follows a similar logic to `qwen-vl-finetune`.*

**Run Training:**
Simply run the corresponding training scripts (please note you need to modify the model paths inside the scripts):
```bash
# Supervised Fine-Tuning (SFT)
./scripts/sft.sh

# Reinforcement Learning (RL)
./scripts/rl.sh
```

### Evaluation

First, prepare the official datasets for OVO-Bench and StreamingBench.

Run the respective `transfer_annotation_format.py` scripts under the `thinkstream/eval` folder to convert the format:
- `thinkstream/eval/ovo_bench/transfer_annotation_format.py`
- `thinkstream/eval/rtvu/transfer_annotation_format.py`

After conversion, start the evaluation script:
```bash
bash ./scripts/eval/eval.sh
```
*Note: You need to change the model checkpoint (`ckpt`) path to your own path.*

### Inference

Use Python to run `scripts/demo.py` for inference testing.

Before running, please change `MODEL_ID` and `VIDEO_PATH` in the code to your own paths. Meanwhile, you need to manually fill in the `content` (question/instruction) and `timestamp` in the `queries` list.

Then, simply run:
```bash
python scripts/demo.py
```
You will see the output results in the command line.

## ❤️ Acknowledgement

We would like to thank the following open-source projects for their valuable contributions:

* [qwen-vl-finetune](https://github.com/QwenLM/Qwen3-VL/)
* [FlashAttention](https://github.com/Dao-AILab/flash-attention)
* [FlashInfer](https://github.com/flashinfer-ai/flashinfer)
* [FlexAttention](https://arxiv.org/abs/2412.05496)
* [Liger Kernel](https://github.com/linkedin/Liger-Kernel/)

## 📑 Citation
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
