## Sequential Domain and Instruction Fine-Tuning of Qwen3-1.7b for Mental Health Assistance

This repository demonstrates end-to-end parameter-efficient domain adaptation fine-tuning of the Qwen3-1.7B base LLM on a custom mental health PDF document, followed by instruction fine-tuning on mental health counseling conversations dataset. It showcases how a general-purpose LLM can be adapted to a sensitive, domain-specific task (mental health) using techniques such as LoRA and QLoRA, without requiring large-scale compute or full model retraining.

### 📝 High-Level Workflow
The project follows a two-stage fine-tuning pipeline, which mirrors how many real-world LLM systems are built:

- Domain Adaptation Fine-Tuning: The base Qwen3-1.7b model is first adapted to mental health–related language by training on curated text extracted from a domain-specific mental health PDF document.
This step helps the model to learn domain terminology, improve coherence on mental health topics, and reduce off-domain hallucinations.

- Instruction Fine-Tuning (SFT): The domain-adapted model is then instruction-tuned using structured prompt–response pairs. This aligns the model to produce empathetic, structured answers. Helps to align the tone and respond more safely and helpfully in conversational settings.

### 📌 Finetuned Models
1. [Qwen3-1.7B-Qlora8bit-MentalHealth](https://huggingface.co/Subi003/Qwen3-1.7B-Qlora8bit-MentalHealth)
2. [Qwen3-1.7B-Qlora8bit-MentalHealth-instruct](https://huggingface.co/Subi003/Qwen3-1.7B-Qlora8bit-MentalHealth-instruct)

### 📊 Datasets Used
- A plain-text, high-quality, curated mental health PDF document prepared using LLMs (GPT-5.2, Gemini-3) and extracting content from Wikipedia. This document has been used for performing the domain adaptation fine-tuning. The PDF document has been uploaded in the repository.
- For instruction fine-tuning, the [Amod/MentalHealth-Counseling-Conversations](https://huggingface.co/datasets/Amod/mental_health_counseling_conversations) has been used. It is a compilation of high-quality, real one-on-one conversations about mental health between individuals and licensed professionals.

### ⚙️ Parameter-Efficient Fine-Tuning
Full fine-tuning of billion-parameter models is computationally expensive and often impractical for individual researchers or students due to GPU memory and compute limitations. To address this, this project adopts Parameter-Efficient Fine-Tuning (PEFT) techniques that enable effective model adaptation while training only a small fraction of the total parameters. Specifically, **LoRA** (Low-Rank Adaptation) is used to inject trainable low-rank matrices into selected transformer layers while keeping the original model weights frozen. This allows the model to learn domain-specific representations without modifying the full parameter space.

To further reduce memory usage, **QLoRA** (Quantized LoRA) is applied, in which the base model is loaded in **8-bit** precision while the LoRA adapters remain in higher precision during training. This combination drastically reduces GPU memory consumption and makes fine-tuning feasible on a single NVIDIA T4 (16GB) GPU, without significant degradation in performance.

By using PEFT, this project achieves:

- Efficient training under strict hardware constraints
- Over 99% reduction in trainable parameters compared to full fine-tuning
- Practical reproducibility in environments such as Google Colab

This approach demonstrates how modern LLMs can be customized for domain-specific and instruction-following tasks without requiring large-scale infrastructure.

### 🏗️ Frameworks/ Libraries 
- PyTorch: An open-source, Python-based deep learning framework developed by Meta AI, designed for building, training, and deploying neural networks.
- Transformers: Acts as the model-definition framework for state-of-the-art machine learning models in text, computer vision, etc., for both inference and training.
- Datasets: A library for easily accessing and sharing AI datasets for Computer Vision, Natural Language Processing (NLP) tasks, and Audio.
- PEFT: A library for efficiently adapting large pretrained models to various downstream applications without fine-tuning all of a model’s parameters.
- Bits and Bytes: The library is a tool integrated with Hugging Face's libraries that provides k-bit quantization methods (specifically 8-bit and 4-bit) for LLMs.
- TRL: A full-stack library providing tools to train transformer language models with methods like Supervised Fine-Tuning (SFT).
- PyPDF: A free and open-source pure-python PDF library capable of splitting, merging, cropping, and transforming the pages of PDF files.
- HuggingFace: Platform where the community collaborates on models, datasets, etc., and for uploading the trained models.
- GPU: T4 (Colab Environment)

### 🧩 Get Started

```bash
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_id = "Subi003/Qwen3-1.7B-Qlora8bit-MentalHealth-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, device=device)

prompt = "How would you support someone experiencing burnout at work?"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

outputs = trained_model.generate(
    **inputs,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    repetition_penalty=1.15,
    no_repeat_ngram_size=3,
    eos_token_id=tokenizer.eos_token_id,
    pad_token_id=tokenizer.eos_token_id,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### ⚠️ Ethical Considerations

This model is not a replacement for professional mental health care, nor should it be directly used in production settings.
It is intended for: Educational exploration and learning, Controlled experimentation, Demo projects, and Prototyping.

<br>

🎗️🙏 **THANK YOU !!** :) <br>
<b>-_with love & regards : Subinoy Bera (developer)_</b><br>
🧡🤍💚




