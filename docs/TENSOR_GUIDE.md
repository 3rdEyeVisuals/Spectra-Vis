# Understanding Tensors in Transformer Models

*A beginner-friendly guide to the tensors you'll see in Spectra Vis*

---

## What is a Tensor?

In the context of neural networks, a **tensor** is simply a multi-dimensional array of numbers. You can think of them as:

- **Scalar**: A single number (0-dimensional tensor)
- **Vector**: A list of numbers (1-dimensional tensor)
- **Matrix**: A 2D grid of numbers (2-dimensional tensor)
- **Tensor**: Any higher-dimensional array (3D, 4D, etc.)

During LLM inference, tensors hold the data being processed - from input tokens to intermediate calculations to final output probabilities.

---

## The Transformer Architecture

Modern LLMs like Llama, Mistral, and Qwen are based on the **transformer architecture**. A transformer consists of stacked layers, each performing the same operations on the data as it flows through the model.

```
Input Tokens
     |
     v
[Token Embedding]          <- Convert tokens to vectors
     |
     v
[Layer 0]                  <- First transformer block
     |
     v
[Layer 1]                  <- Second transformer block
     |
     v
   ...                     <- More layers
     |
     v
[Layer N-1]                <- Final transformer block
     |
     v
[Output Normalization]     <- Normalize before output
     |
     v
[Output Projection]        <- Convert to vocabulary probabilities
     |
     v
Output Token
```

---

## Tensor Types in Each Layer

Each transformer layer contains several types of tensors. Here's what each one does:

### Embedding Layer (Before Layer 0)

| Tensor | Purpose |
|--------|---------|
| `token_embd` | Converts input token IDs into dense vectors (embeddings) |

### Attention Block

The attention mechanism allows the model to focus on relevant parts of the input.

| Tensor | Full Name | Purpose |
|--------|-----------|---------|
| `attn_norm` | Attention Normalization | Normalizes input before attention |
| `attn_q` | Query | "What am I looking for?" |
| `attn_k` | Key | "What do I contain?" |
| `attn_v` | Value | "What information do I provide?" |
| `attn_output` | Attention Output | Combines attended information |

**How Attention Works:**
1. Query (Q) asks: "What information do I need?"
2. Key (K) in each position responds: "Here's what I have"
3. Q and K produce attention weights (how much to focus on each position)
4. Value (V) provides the actual content to extract
5. Output combines all attended values weighted by attention

### Feedforward Block

The feedforward network (FFN) processes each position independently.

| Tensor | Full Name | Purpose |
|--------|-----------|---------|
| `ffn_norm` | FFN Normalization | Normalizes input before FFN |
| `ffn_gate` | Gate | Controls information flow (in gated FFNs) |
| `ffn_up` | Up Projection | Expands to larger hidden dimension |
| `ffn_down` | Down Projection | Compresses back to model dimension |

**How FFN Works:**
1. Input is normalized
2. Up projection expands the dimension (e.g., 4096 -> 14336)
3. Non-linear activation is applied (SiLU, GELU, etc.)
4. Gate modulates the signal (in models like Llama)
5. Down projection compresses back (e.g., 14336 -> 4096)

### Output Tensors (After Last Layer)

| Tensor | Purpose |
|--------|---------|
| `output_norm` | Final normalization before vocabulary projection |
| `output` | Projects hidden states to vocabulary logits |

---

## Tensor Flow Through a Layer

Here's how data flows through a single transformer layer:

```
Input from previous layer
          |
          v
    [attn_norm] -----> Normalize
          |
          +------+------+
          |      |      |
          v      v      v
       [attn_q] [attn_k] [attn_v]
          |      |      |
          +------+------+
                 |
                 v
          [attention computation]
                 |
                 v
          [attn_output]
                 |
                 + (residual connection from input)
                 |
                 v
          [ffn_norm] -----> Normalize
                 |
          +------+------+
          |             |
          v             v
       [ffn_gate]   [ffn_up]
          |             |
          +------+------+
                 |
                 v
          [ffn_down]
                 |
                 + (residual connection)
                 |
                 v
          Output to next layer
```

### Residual Connections

Notice the `+` symbols? These are **residual connections** (also called skip connections). They add the input directly to the output, allowing gradients to flow easily and helping the model learn.

---

## Model-Specific Variations

Different model families have slight architectural differences:

### Llama / Mistral / Qwen / Granite
- Standard decoder-only transformer
- Separate Q, K, V projections
- Gated FFN with SiLU activation
- RMSNorm for normalization

### Phi (Microsoft)
- Fused QKV projection (`attn_qkv` instead of separate Q, K, V)
- Parallel attention and FFN in some variants
- Different activation functions

### Mixtral (Mistral AI)
- Mixture of Experts (MoE) in FFN layers
- Multiple FFN experts per layer
- Router selects which experts to use

---

## What Spectra Vis Shows You

When you capture tensor observations during inference:

1. **Observation Order**: The sequence in which tensors are accessed reveals the computation graph execution order

2. **Observation Count**: How many times each tensor is accessed. Some tensors (like embeddings) are accessed once, while others may be accessed multiple times during generation

3. **Layer Distribution**: Which layers are most active? Early layers might handle syntax, while later layers handle semantics

4. **Tensor Categories**: Color-coded by function:
   - Blue: Embedding
   - Green: Attention
   - Orange: Feedforward
   - Purple: Output
   - Gold: Normalization

---

## Why This Matters

Understanding tensor flow helps with:

- **Debugging**: Why is inference slow? Which layers are bottlenecks?
- **Research**: How does the model process different types of inputs?
- **Optimization**: Which tensors could be quantized or pruned?
- **Education**: Visualizing abstract concepts makes them concrete

---

## Further Reading

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - The original transformer paper
- [LLaMA: Open Foundation Models](https://arxiv.org/abs/2302.13971) - Llama architecture details
- [GGUF Format Specification](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md) - How GGUF stores model data
- [llama.cpp](https://github.com/ggerganov/llama.cpp) - The inference engine Spectra Vis builds on

---

*Copyright (c) 2025 3rdEyeVisuals*
