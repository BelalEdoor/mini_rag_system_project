# 🧠 RAG System with Sentence Transformers & Hugging Face

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** system built with **Python**, **SentenceTransformers**, and **Hugging Face Transformers**.  
The system can **retrieve relevant information** from a custom knowledge base and **generate accurate answers** to user questions using a Question Answering (QA) model.

---

## 🚀 Project Overview

The project is divided into three main tasks:

### **Task 1 – Build a Simple RAG System**
- Load a pretrained **SentenceTransformer** model (`all-MiniLM-L6-v2`) for semantic retrieval.
- Create a small **knowledge base** containing general facts.
- Encode the knowledge base into **vector embeddings** using the retriever model.

### **Task 2 – Add and Test Questions**
- Add several **test questions** to evaluate the RAG system.
- For each question:
  - Compute the semantic similarity between the question and all knowledge base sentences.
  - Retrieve the most relevant context.
  - Use a **question-answering pipeline** (`distilbert-base-uncased-distilled-squad`) to generate an answer.
- Evaluate both **retrieval** and **generation accuracy**.

### **Task 3 – Expand Knowledge and Test**
- Extend the knowledge base with **new facts** (e.g., about the Moon).
- Re-encode embeddings to include the new data.
- Test the system with **new questions** related to the added knowledge.

---

## 🧩 Example Knowledge Base

```python
knowledge_base = [
    "The capital of France is Paris, a city famous for the Eiffel Tower and the Louvre museum.",
    "The Amazon rainforest is the world's largest tropical rainforest, known for its incredible biodiversity.",
    "Mount Everest is the highest mountain on Earth, located in the Himalayas.",
    "The Great Wall of China is a series of fortifications stretching over 13,000 miles.",
    "Photosynthesis is the process used by plants to convert light energy into chemical energy.",
    "The Moon orbits the Earth approximately every 27.3 days and influences the ocean tides through its gravitational pull."
]
