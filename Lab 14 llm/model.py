"""
Simple Transformer-based Language Model from scratch.
"""

import logging
import numpy as np
from layers import Embedding, Linear, Attention, relu, relu_backward


class SimpleLLM:
    """
    A very simple language model with:
    - Token embeddings
    - Position embeddings
    - Self-attention
    - Feed-forward network
    - Output projection
    """
    
    def __init__(self, vocab_size, embed_dim=64, max_seq_len=20, num_layers=1):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        
        # Token embeddings
        self.token_embedding = Embedding(vocab_size, embed_dim)
        
        # Position embeddings (learnable)
        self.pos_embedding = np.random.randn(max_seq_len, embed_dim) * 0.01
        self.grad_pos_embedding = None
        
        # Transformer layers
        self.attention_layers = [Attention(embed_dim) for _ in range(num_layers)]
        self.ff_layers = []
        for _ in range(num_layers):
            ff = {
                'linear1': Linear(embed_dim, embed_dim * 4),
                'linear2': Linear(embed_dim * 4, embed_dim)
            }
            self.ff_layers.append(ff)
        
        # Output projection to vocabulary
        self.output_proj = Linear(embed_dim, vocab_size)
        
        # Layer norm (simplified - just track running mean/std)
        self.layer_norm_eps = 1e-5
    
    def _layer_norm(self, x):
        """Simple layer normalization."""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + self.layer_norm_eps)
    
    def _create_causal_mask(self, seq_len):
        """Create causal mask to prevent looking at future tokens."""
        mask = np.triu(np.ones((seq_len, seq_len)), k=1).astype(bool)
        return mask
    
    def forward(self, token_ids):
        """
        Forward pass through the model.
        token_ids: (batch_size, seq_len)
        Returns: (batch_size, seq_len, vocab_size) logits
        """
        batch_size, seq_len = token_ids.shape
        
        # Token embeddings
        x = self.token_embedding.forward(token_ids)  # (batch_size, seq_len, embed_dim)
        
        # Add position embeddings
        pos_emb = self.pos_embedding[:seq_len]  # (seq_len, embed_dim)
        x = x + pos_emb[np.newaxis, :, :]  # Broadcast to batch
        
        # Create causal mask
        mask = self._create_causal_mask(seq_len)
        
        # Store activations for backward pass
        self.activations = []
        self.ff_hidden_states = []  # Store ReLU outputs for backward pass
        self.activations.append(x.copy())  # After embedding
        
        # Pass through transformer layers
        for i in range(self.num_layers):
            # Self-attention with residual
            attn_out = self.attention_layers[i].forward(x, mask=mask)
            x = x + attn_out  # Residual connection
            x = self._layer_norm(x)
            self.activations.append(x.copy())
            
            # Feed-forward with residual
            ff_hidden = self.ff_layers[i]['linear1'].forward(x)
            self.ff_hidden_states.append(ff_hidden)  # Store input before ReLU for backward
            ff_hidden_relu = relu(ff_hidden)
            ff_out = self.ff_layers[i]['linear2'].forward(ff_hidden_relu)
            x = x + ff_out  # Residual connection
            x = self._layer_norm(x)
            self.activations.append(x.copy())
        
        # Output projection
        logits = self.output_proj.forward(x)
        return logits
    
    def backward(self, grad_output):
        """Backward pass through the model."""
        # Backward through output projection
        grad = self.output_proj.backward(grad_output)
        
        # Layer norm backward (simplified - approximate as identity for educational purposes)
        # In practice, layer norm has a more complex backward pass
        
        # Backward through transformer layers (in reverse)
        ff_idx = len(self.ff_hidden_states) - 1
        for i in range(self.num_layers - 1, -1, -1):
            # Backward through feed-forward
            # Layer norm backward (simplified)
            grad_ff = grad.copy()
            
            # Split residual: one path goes through FF, one is identity
            grad_residual = grad_ff.copy()
            
            # Backward through FF layer 2
            grad_ff = self.ff_layers[i]['linear2'].backward(grad_ff)
            
            # Backward through ReLU
            if ff_idx >= 0:
                grad_ff = relu_backward(self.ff_hidden_states[ff_idx], grad_ff)
                ff_idx -= 1
            
            # Backward through FF layer 1
            grad_ff = self.ff_layers[i]['linear1'].backward(grad_ff)
            
            # Add residual gradient
            grad = grad_residual + grad_ff
            
            # Backward through attention (with residual)
            grad_residual_attn = grad.copy()
            grad_attn = self.attention_layers[i].backward(grad)
            
            # Add residual gradient
            grad = grad_residual_attn + grad_attn
        
        # Backward through position embeddings (accumulate gradient)
        if self.grad_pos_embedding is None:
            self.grad_pos_embedding = np.zeros_like(self.pos_embedding)
        
        seq_len = grad.shape[1]
        self.grad_pos_embedding[:seq_len] += np.sum(grad, axis=0)
        
        # Backward through token embeddings
        self.token_embedding.backward(grad)
    
    def update(self, learning_rate):
        """Update all parameters."""
        # Update token embeddings
        self.token_embedding.update(learning_rate)
        
        # Update position embeddings
        if self.grad_pos_embedding is not None:
            self.pos_embedding -= learning_rate * self.grad_pos_embedding
            self.grad_pos_embedding = None
        
        # Update attention layers
        for attn in self.attention_layers:
            attn.update(learning_rate)
        
        # Update feed-forward layers
        for ff in self.ff_layers:
            ff['linear1'].update(learning_rate)
            ff['linear2'].update(learning_rate)
        
        # Update output projection
        self.output_proj.update(learning_rate)
    
    def generate(self, tokenizer, prompt, max_length=10, temperature=1.0):
        """
        Generate text given a prompt.
        Uses greedy decoding (can be modified for sampling).
        """
        # Encode prompt
        token_ids = tokenizer.encode(prompt)
        if len(token_ids) == 0:
            token_ids = [tokenizer.word_to_idx['<BOS>']]
        
        # Add BOS if not present
        if token_ids[0] != tokenizer.word_to_idx['<BOS>']:
            token_ids = [tokenizer.word_to_idx['<BOS>']] + token_ids
        
        generated = token_ids.copy()
        
        for step in range(max_length):
            # Prepare input (batch_size=1, seq_len=current_length)
            input_ids = np.array([generated])
            
            # Forward pass
            logits = self.forward(input_ids)
            
            # Get logits for next token (last position)
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply softmax to get probabilities
            from layers import softmax
            probs = softmax(next_token_logits)
            
            # Log probabilities for top tokens (DEBUG level)
            top_k = min(5, len(probs))  # Show top 5 tokens
            top_indices = np.argsort(probs)[-top_k:][::-1]  # Get top k indices in descending order
            
            current_text = tokenizer.decode(generated)
            logging.debug(f"Step {step + 1}, Current: '{current_text}'")
            logging.debug("Top token probabilities:")
            for idx in top_indices:
                token = tokenizer.idx_to_word[idx]
                prob = probs[idx]
                logging.debug(f"  '{token}': {prob:.4f} ({prob*100:.2f}%)")
            
            # Sample next token (greedy: take argmax, or sample from distribution)
            next_token = np.argmax(probs)
            next_token_word = tokenizer.idx_to_word[next_token]
            logging.debug(f"Selected token: '{next_token_word}' (prob: {probs[next_token]:.4f})\n")
            
            # Stop if EOS token
            if next_token == tokenizer.word_to_idx['<EOS>']:
                logging.debug("EOS token generated, stopping.")
                break
            
            generated.append(next_token)
        
        # Decode and return
        return tokenizer.decode(generated)

