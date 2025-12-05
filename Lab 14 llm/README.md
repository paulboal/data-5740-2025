# Simple LLM from Scratch

This is an educational implementation of a Large Language Model (LLM) built entirely from scratch using only NumPy. No PyTorch, TensorFlow, or other machine learning frameworks are used. The goal is to demonstrate how LLMs work during training and inference.

## Architecture

The model implements a simplified Transformer decoder architecture with:

1. **Token Embeddings**: Maps token IDs to dense vectors
2. **Position Embeddings**: Adds positional information to tokens
3. **Self-Attention**: Allows tokens to attend to previous tokens (causal masking)
4. **Feed-Forward Networks**: Non-linear transformations
5. **Layer Normalization**: Stabilizes training
6. **Output Projection**: Maps hidden states to vocabulary logits

## Components

- `tokenizer.py`: Simple tokenizer that builds vocabulary from corpus and converts text to/from token IDs
- `layers.py`: Core neural network layers (Embedding, Linear, Attention) implemented from scratch
- `model.py`: The SimpleLLM class that combines all layers into a complete model
- `train.py`: Training script with manual backpropagation

## How It Works

### Training

1. **Forward Pass**: 
   - Input tokens are embedded and position information is added
   - Pass through attention and feed-forward layers
   - Project to vocabulary size to get logits (unnormalized probabilities)

2. **Loss Calculation**:
   - Compare predicted logits with actual next token
   - Compute cross-entropy loss

3. **Backward Pass**:
   - Manually compute gradients for all parameters
   - Update weights using gradient descent

### Inference

1. Start with a prompt (or just BOS token)
2. For each step:
   - Run forward pass to get logits for next token
   - Sample (or take argmax) to get next token
   - Append to sequence and repeat

## Usage

```bash
python train.py
```

This will:
1. Build vocabulary from the corpus
2. Initialize the model
3. Train for 100 epochs
4. Generate text from test prompts

## Educational Value

This implementation demonstrates:
- How tokenization works
- How embeddings represent words as vectors
- How attention mechanisms allow tokens to interact
- How backpropagation updates model parameters
- How generation works step-by-step

The code is intentionally simple and well-commented to make it easy to understand the fundamentals of LLMs.

