# 🚀 Mini GPT From Scratch (PyTorch)

This project is a **from-scratch implementation of a GPT-style language model** built step-by-step using PyTorch.

No frameworks. No shortcuts. Just attention, transformers, and training.

---

# 🧠 What this project shows

- Self-Attention (Q, K, V)
- Multi-Head Attention
- Causal Masking
- Transformer Blocks (Pre-LayerNorm)
- Feed Forward Network
- Token-level training (GPT-2 tokenizer)
- Text generation with sampling

---

# ⚔️ Two Versions Built

## ❌ Version 1 (Char-Level GPT)

- Character-level tokenization
- Smaller vocabulary (~60–100)
- Simpler setup
- Had subtle bugs (generation, positional encoding)
- Trained on **Sherlock Holmes dataset**

### Result:
- Struggled to form meaningful words early
- Output was mostly noisy
- Harder to learn structure

---

## ✅ Version 2 (Token-Level GPT) — Current

- GPT-2 tokenizer (~50k vocab)
- Cleaner implementation
- Fixed generation + positional bugs
- Trained on **Shakespeare dataset**

### Result:
- Learned structure VERY quickly
- Generated readable dialogue format
- Captured names, formatting, tone

---

# 📉 Training Logs

```text
step 0   → 10.98
step 100 → 6.43
step 200 → 5.86
step 300 → 5.92
step 400 → 5.31
step 500 → 5.07
step 600 → 4.90
step 700 → 5.06
step 800 → 4.82
step 900 → 4.88
```

👉 Initial loss ≈ log(vocab_size) → confirms correct setup  
👉 Rapid early drop → model learns token distribution  
👉 Later fluctuations → normal stochastic training  

---

# ✨ Sample Output (after ~900 steps)

```text
Hello OF YORK:
As you will, to have not you do with the prince'd.

LUCIO:
I pray, you'll see the king.

GLOUCES:
And
If be it is:
Of you, being I say to my lord.

KING HEN OF YORK:
It now,
Come! for we will have the great:

The law thou hast, and let me, and
' the fair not in
```

---

# 🤯 Why this is interesting

After **less than 1000 steps**, the model:

- Learned dialogue structure (`NAME:` format)
- Learned character-style speaking
- Learned punctuation + line breaks
- Produced semi-coherent Shakespeare-like text

This is **not memorization**, it's pattern learning.

---

# ⚙️ Tech Stack

- Python
- PyTorch
- tiktoken (GPT-2 tokenizer)

---

# 🧪 How to Run

```bash
pip install torch tiktoken requests
python main.py
```

---

# 🧠 Key Learnings

- Small implementation bugs can completely break learning
- Tokenization choice massively impacts results
- Structured datasets (like dialogue) are easier to learn
- Early loss drop = model learning basic token patterns
- Transformers are simple in concept, powerful in practice

---

# 🚀 Future Improvements

- Top-p (nucleus) sampling
- Dropout for regularization
- Larger context window
- Bigger model (embed size / layers)
- Training on larger datasets

---

# 💬 Final Note

This project proves:

```text
You don’t need huge infrastructure to build a working language model.
```

Just:
- correct fundamentals
- clean implementation
- and patience

---
Current additions:
I added top p instead of top k 
I merged k,q,v into one layer
I added checkpoint that saves the current progress
Added KV caching
Added cache cropping
Started adding RoPE
Improved future error that could be due to unequal shape handling
Added RMSNorm
Added SwiGLU
Added AdamW warmup and Cosine decay
