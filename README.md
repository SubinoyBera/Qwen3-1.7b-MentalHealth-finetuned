## Sequential Domain and Instruction Fine-Tuning of Qwen3-1.7b for Mental Health Assistance

This repository demonstrates end-to-end parameter-efficient domain adaptation fine-tuning of the Qwen3-1.7B base LLM on a custom mental health PDF document, followed by instruction fine-tuning on mental health counseling conversations dataset. It showcases how a general-purpose LLM can be adapted to a sensitive, domain-specific task (mental health) using techniques such as LoRA and QLoRA, without requiring large-scale compute or full model retraining.

### High-Level Workflow
The project follows a two-stage fine-tuning pipeline, which mirrors how many real-world LLM systems are built:

- Domain Adaptation Fine-Tuning: The base Qwen3-1.7b model is first adapted to mental health–related language by training on curated text extracted from a domain-specific mental health PDF document.
This step helps the model to learn domain terminology, improve coherence on mental health topics, and reduce off-domain hallucinations.

- Instruction Fine-Tuning (SFT): The domain-adapted model is then instruction-tuned using structured prompt–response pairs. This aligns the model to produce empathetic, structured answers. Helps to align the tone and respond more safely and helpfully in conversational settings.

### Datasets
- A plain-text, high-quality, curated mental health PDF document prepared using LLMs (GPT-5.2, Gemini-3) and extracting content from Wikipedia. This document has been used for performing the domain adaptation fine-tuning. The PDF document has been uploaded in the repository.
- For instruction fine-tuning, the [Amod/MentalHealth-Counseling-Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) has been used. It is a compilation of high-quality, real one-on-one mental health counseling conversations between individuals and licensed professionals.

### Parameter-Efficient Fine-Tuning
Full fine-tuning of billion-parameter models is computationally expensive and often impractical for individual researchers or students due to GPU memory and compute limitations. To address this, this project adopts Parameter-Efficient Fine-Tuning (PEFT) techniques that enable effective model adaptation while training only a small fraction of the total parameters. Specifically, **LoRA** (Low-Rank Adaptation) is used to inject trainable low-rank matrices into selected transformer layers while keeping the original model weights frozen. This allows the model to learn domain-specific representations without modifying the full parameter space.

To further reduce memory usage, **QLoRA** (Quantized LoRA) is applied, where the base model is loaded in **8-bit** precision while LoRA adapters remain in higher precision during training. This combination drastically reduces GPU memory consumption and makes fine-tuning feasible on a single NVIDIA T4 (16GB) GPU, without significant degradation in performance.

By using PEFT, this project achieves:

- Efficient training under strict hardware constraints
- Over 99% reduction in trainable parameters compared to full fine-tuning
- Practical reproducibility in environments such as Google Colab

This approach demonstrates how modern LLMs can be customized for domain-specific and instruction-following tasks without requiring large-scale infrastructure.


