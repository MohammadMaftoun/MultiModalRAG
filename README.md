# MultiModalRAG

# 🧠 Multimodal RAG Pipeline with LLaMA2 + CLIP

This project integrates a Retrieval-Augmented Generation (RAG) pipeline using **LLaMA2** (quantized via BitsAndBytes) and **OpenAI's CLIP** for multimodal document understanding — including both **text** and **images**.

It is built with a strong focus on working in restricted environments such as **Kaggle**, where **TPU-specific imports (torch_xla)** may cause issues.

![MMRAG](https://miro.medium.com/v2/resize:fit:982/1*4CaYSUEN9Z51bFgnFLXFOg.png)


---

# 📌 Features

- 🔍 **RAG Pipeline**: Retrieve relevant documents and generate answers with LLaMA2.
- 🧠 **LLM**: Quantized LLaMA2-7B-Chat for lightweight deployment.
- 🖼️ **Multimodal Embeddings**: Use CLIP to embed both images and texts.
- 🗃️ **ChromaDB**: For storing and querying document embeddings.
- ⚙️ **Torch XLA Patch**: Avoid TPU-related errors during import.

---

## 🛠️ Installation

Uninstall existing torch-related packages and install pinned versions to avoid compatibility issues:

```bash
pip uninstall -y torch torch_xla torchvision torchaudio
pip install torch==2.0.1 transformers==4.33.0 accelerate==0.22.0 einops==0.6.1 langchain==0.0.300 \
xformers==0.0.21 bitsandbytes==0.41.1 sentence_transformers==2.2.2 chromadb==0.4.12 pillow==10.0.0
