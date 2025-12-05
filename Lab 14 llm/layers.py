"""
Neural network layers implemented from scratch using NumPy.
"""

import numpy as np


class Embedding:
    """
    The Embedding class transforms discrete token IDs into continuous dense vector representations,
    enabling the neural network to process categorical data in a learnable way.
    This layer is commonly used as the first layer in natural language processing models to map words or tokens to embedding vectors.
    """
    
    def __init__(self, vocab_size, embed_dim):
        # Initialize embeddings with small random values
        self.embed_dim = embed_dim
        self.embeddings = np.random.randn(vocab_size, embed_dim) * 0.01
        self.grad_embeddings = None
    
    def forward(self, token_ids):
        """Forward pass: lookup embeddings for token IDs."""
        self.token_ids = token_ids
        return self.embeddings[token_ids]
    
    def backward(self, grad_output):
        """Backward pass: accumulate gradients."""
        if self.grad_embeddings is None:
            self.grad_embeddings = np.zeros_like(self.embeddings)
        
        # Accumulate gradients for the embeddings that were used
        np.add.at(self.grad_embeddings, self.token_ids, grad_output)
    
    def update(self, learning_rate):
        """Update embeddings using accumulated gradients."""
        if self.grad_embeddings is not None:
            self.embeddings -= learning_rate * self.grad_embeddings
            self.grad_embeddings = None


class Linear:
    """
    The Linear (fully connected) layer performs a linear transformation on the input,
    mapping it from one space to another. It is a fundamental building block for feed-forward networks.
    This layer is used to learn non-linear relationships between input features and output predictions.
    """
    
    def __init__(self, in_features, out_features):
        # Xavier initialization
        self.weight = np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features)
        self.bias = np.zeros(out_features)
        self.grad_weight = None
        self.grad_bias = None
        self.input = None
    
    def forward(self, x):
        """Forward pass: y = xW + b."""
        self.input = x
        return np.dot(x, self.weight) + self.bias
    
    def backward(self, grad_output):
        """Backward pass: compute gradients."""
        # Handle both 2D (batch, features) and 3D (batch, seq_len, features) inputs
        if len(self.input.shape) == 2:
            # 2D case: (batch, in_features)
            batch_size = grad_output.shape[0]
            
            # Gradient w.r.t. weight: input^T @ grad_output
            self.grad_weight = np.dot(self.input.T, grad_output)
            
            # Gradient w.r.t. bias: sum over batch dimension
            self.grad_bias = np.sum(grad_output, axis=0)
            
            # Gradient w.r.t. input: grad_output @ weight^T
            grad_input = np.dot(grad_output, self.weight.T)
        else:
            # 3D case: (batch, seq_len, in_features)
            batch_size, seq_len, _ = self.input.shape
            
            # Reshape for easier computation: flatten batch and seq dimensions
            input_flat = self.input.reshape(-1, self.input.shape[-1])  # (batch*seq, in_features)
            grad_output_flat = grad_output.reshape(-1, grad_output.shape[-1])  # (batch*seq, out_features)
            
            # Gradient w.r.t. weight: sum over all positions
            self.grad_weight = np.dot(input_flat.T, grad_output_flat)
            
            # Gradient w.r.t. bias: sum over batch and sequence dimensions
            self.grad_bias = np.sum(grad_output_flat, axis=0)
            
            # Gradient w.r.t. input: grad_output @ weight^T, then reshape
            grad_input_flat = np.dot(grad_output_flat, self.weight.T)
            grad_input = grad_input_flat.reshape(self.input.shape)
        
        return grad_input
    
    def update(self, learning_rate):
        """Update weights and bias."""
        if self.grad_weight is not None:
            self.weight -= learning_rate * self.grad_weight
            self.bias -= learning_rate * self.grad_bias
            self.grad_weight = None
            self.grad_bias = None


class Attention:
    """
    The Attention class implements the self-attention mechanism,
    allowing the model to focus on relevant parts of the input sequence.
    This is a key component in Transformer-based models for capturing long-range dependencies.
    """
    
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim
        self.query = Linear(embed_dim, embed_dim)
        self.key = Linear(embed_dim, embed_dim)
        self.value = Linear(embed_dim, embed_dim)
        self.scale = np.sqrt(embed_dim)
        self.attention_weights = None
        self.input = None
    
    def forward(self, x, mask=None):
        """
        Forward pass: compute self-attention.
        x: (batch_size, seq_len, embed_dim)
        """
        self.input = x
        batch_size, seq_len, embed_dim = x.shape
        
        # Compute Q, K, V
        Q = self.query.forward(x)  # (batch_size, seq_len, embed_dim)
        K = self.key.forward(x)
        V = self.value.forward(x)
        
        # Compute attention scores: QK^T / sqrt(d)
        scores = np.einsum('bik,bjk->bij', Q, K) / self.scale
        
        # Apply mask if provided (for causal attention)
        # mask[i,j] = True means j > i (future position), should be masked out
        if mask is not None:
            scores = np.where(mask, -1e9, scores)  # Mask out future positions
        
        # Softmax
        exp_scores = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        
        # Apply attention to values
        output = np.einsum('bij,bjk->bik', self.attention_weights, V)
        return output
    
    def backward(self, grad_output):
        """Backward pass through attention."""
        batch_size, seq_len, embed_dim = self.input.shape
        
        # Recompute Q, K, V for backward pass
        Q = self.query.forward(self.input)
        K = self.key.forward(self.input)
        V = self.value.forward(self.input)
        
        # Gradient w.r.t. V
        grad_V = np.einsum('bij,bik->bjk', self.attention_weights, grad_output)
        
        # Gradient w.r.t. attention weights
        grad_attn_weights = np.einsum('bik,bjk->bij', grad_output, V)
        
        # Gradient through softmax
        # d(softmax(x))/dx = softmax(x) * (grad - sum(grad * softmax(x)))
        grad_scores = self.attention_weights * (grad_attn_weights - np.sum(
            self.attention_weights * grad_attn_weights, axis=-1, keepdims=True))
        
        # Gradient through scaling
        grad_scores = grad_scores / self.scale
        
        # Gradient w.r.t. Q and K
        grad_Q = np.einsum('bij,bjk->bik', grad_scores, K)
        grad_K = np.einsum('bij,bik->bjk', grad_scores, Q)
        
        # Backward through linear layers
        grad_input_Q = self.query.backward(grad_Q)
        grad_input_K = self.key.backward(grad_K)
        grad_input_V = self.value.backward(grad_V)
        
        return grad_input_Q + grad_input_K + grad_input_V
    
    def update(self, learning_rate):
        """Update all linear layers."""
        self.query.update(learning_rate)
        self.key.update(learning_rate)
        self.value.update(learning_rate)


def relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)


def relu_backward(x, grad_output):
    """Backward pass for ReLU."""
    return grad_output * (x > 0)


def softmax(x, axis=-1):
    """
    The softmax function converts a list of numbers (logits) into probabilities. 
    It does this by exponentiating each number and dividing by the sum of all exponentials.
    The output values are between 0 and 1 and add up to 1.
    """
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def cross_entropy_loss(predictions, targets):
    """
    Compute cross-entropy loss.
    predictions: (batch_size, seq_len, vocab_size) - logits
    targets: (batch_size, seq_len) - token IDs
    """
    batch_size, seq_len, vocab_size = predictions.shape
    
    # Flatten for easier computation
    pred_flat = predictions.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    # Compute softmax probabilities
    probs = softmax(pred_flat, axis=1)
    
    # Compute loss: -log(p(target))
    loss = -np.mean(np.log(probs[np.arange(len(targets_flat)), targets_flat] + 1e-10))
    
    # Compute gradients
    grad_flat = probs.copy()
    grad_flat[np.arange(len(targets_flat)), targets_flat] -= 1
    grad_flat /= batch_size * seq_len
    
    grad = grad_flat.reshape(batch_size, seq_len, vocab_size)
    
    return loss, grad

