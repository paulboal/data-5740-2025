"""
Training script for the simple LLM.
"""

import argparse
import logging
import numpy as np
from tokenizer import Tokenizer
from model import SimpleLLM
from layers import cross_entropy_loss


def prepare_data(corpus, tokenizer, max_seq_len=20):
    """
    Prepare training data from corpus.
    Creates input-target pairs for next token prediction.
    For each position in a sequence, predict the next token.
    """
    data = []
    
    logging.debug("Preparing training data: creating input-target pairs for next token prediction")
    
    for sentence_idx, sentence in enumerate(corpus):
        # Encode sentence
        tokens = tokenizer.encode(sentence)
        
        # Add BOS and EOS tokens
        tokens = [tokenizer.word_to_idx['<BOS>']] + tokens + [tokenizer.word_to_idx['<EOS>']]
        
        logging.debug(f"Sentence {sentence_idx + 1}: '{sentence}' -> tokens: {[tokenizer.idx_to_word[t] for t in tokens]}")
        
        # Create training examples: for each position, predict next token
        for i in range(len(tokens) - 1):
            # Input: tokens from start up to position i+1 (including the token we'll predict)
            # But we only use up to position i for input, and predict i+1
            input_seq = tokens[:i+1]
            target = tokens[i+1]
            
            # Pad or truncate input sequence to max_seq_len
            if len(input_seq) < max_seq_len:
                input_seq = input_seq + [tokenizer.word_to_idx['<PAD>']] * (max_seq_len - len(input_seq))
            else:
                input_seq = input_seq[-max_seq_len:]  # Take last max_seq_len tokens
            
            input_text = tokenizer.decode([t for t in input_seq if t != tokenizer.word_to_idx['<PAD>']])
            target_text = tokenizer.idx_to_word[target]
            logging.debug(f"  Training example: input='{input_text}' -> target='{target_text}'")
            
            data.append((input_seq, target, len([t for t in input_seq if t != tokenizer.word_to_idx['<PAD>']])))
    
    logging.debug(f"Total training examples created: {len(data)}")
    return data


def train_epoch(model, data, tokenizer, learning_rate=0.01, batch_size=4):
    """
    Train for one epoch.
    """
    total_loss = 0
    num_batches = 0
    
    # Shuffle data
    np.random.shuffle(data)
    logging.debug(f"\nStarting epoch: {len(data)} training examples, batch size: {batch_size}")
    
    # Process in batches
    for batch_idx, i in enumerate(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        batch_num = batch_idx + 1
        
        # Prepare batch
        batch_size_actual = len(batch)
        max_seq_len = len(batch[0][0])
        
        inputs = np.array([item[0] for item in batch])
        targets = np.array([item[1] for item in batch])
        seq_lengths = np.array([item[2] for item in batch])  # Actual sequence lengths
        
        logging.debug(f"\n--- Batch {batch_num} ({batch_size_actual} examples) ---")
        
        # Show what we're training on
        for b in range(batch_size_actual):
            input_seq_clean = [t for t in inputs[b] if t != tokenizer.word_to_idx['<PAD>']]
            input_text = tokenizer.decode(input_seq_clean)
            target_text = tokenizer.idx_to_word[targets[b]]
            logging.debug(f"Example {b+1}: input='{input_text}' -> target='{target_text}'")
        
        # Forward pass
        logging.debug(f"\nForward pass: running model on batch...")
        logits = model.forward(inputs)
        logging.debug(f"Model output shape: {logits.shape} (batch_size={logits.shape[0]}, seq_len={logits.shape[1]}, vocab_size={logits.shape[2]})")
        
        # Get logits at the last non-pad position for each sequence
        # The target is the token that comes after the input sequence
        batch_logits = []
        for b in range(batch_size_actual):
            # Position is seq_length - 1 (last token in input, predict next)
            pos = seq_lengths[b] - 1
            if pos < 0:
                pos = 0
            batch_logits.append(logits[b, pos, :])
        
        batch_logits = np.array(batch_logits)  # (batch_size, vocab_size)
        batch_logits = batch_logits[:, np.newaxis, :]  # (batch_size, 1, vocab_size)
        targets_expanded = targets[:, np.newaxis]  # (batch_size, 1)
        
        # Show predictions vs targets
        logging.debug(f"\nPredictions at last position:")
        for b in range(batch_size_actual):
            pred_token_idx = np.argmax(batch_logits[b, 0, :])
            pred_token = tokenizer.idx_to_word[pred_token_idx]
            target_token = tokenizer.idx_to_word[targets[b]]
            correct = "✓" if pred_token_idx == targets[b] else "✗"
            logging.debug(f"  Example {b+1}: predicted='{pred_token}' (idx={pred_token_idx}), target='{target_token}' (idx={targets[b]}) {correct}")
        
        # Compute loss
        logging.debug(f"\nComputing loss (cross-entropy)...")
        loss, grad = cross_entropy_loss(batch_logits, targets_expanded)
        logging.debug(f"Loss: {loss:.4f}")
        logging.debug(f"Gradient shape: {grad.shape}")
        total_loss += loss
        num_batches += 1
        
        # Expand gradient to match logits shape (only at the prediction position)
        grad_expanded = np.zeros_like(logits)
        for b in range(batch_size_actual):
            pos = seq_lengths[b] - 1
            if pos < 0:
                pos = 0
            grad_expanded[b, pos, :] = grad[b, 0, :]
        
        # Backward pass
        logging.debug(f"Backward pass: computing gradients for all parameters...")
        model.backward(grad_expanded)
        logging.debug(f"Gradients computed, ready to update parameters")
        
        # Update parameters
        logging.debug(f"Updating parameters with learning rate: {learning_rate}")
        model.update(learning_rate)
        logging.debug(f"Parameters updated")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    logging.debug(f"\nEpoch complete: average loss = {avg_loss:.4f} over {num_batches} batches")
    return avg_loss


def train(model, corpus, tokenizer, num_epochs=50, learning_rate=0.01, batch_size=4):
    """
    Main training loop.
    """
    logging.info("Preparing training data...")
    data = prepare_data(corpus, tokenizer, max_seq_len=20)
    logging.info(f"Created {len(data)} training examples")
    
    logging.info("\nStarting training...")
    logging.info("=" * 50)
    logging.debug(f"Training configuration:")
    logging.debug(f"  - Epochs: {num_epochs}")
    logging.debug(f"  - Learning rate: {learning_rate}")
    logging.debug(f"  - Batch size: {batch_size}")
    logging.debug(f"  - Training examples: {len(data)}")
    
    for epoch in range(num_epochs):
        logging.debug(f"\n{'='*50}")
        logging.debug(f"EPOCH {epoch + 1}/{num_epochs}")
        logging.debug(f"{'='*50}")
        loss = train_epoch(model, data, tokenizer, learning_rate, batch_size)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logging.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {loss:.4f}")
    
    logging.info("=" * 50)
    logging.info("Training complete!")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a simple LLM from scratch')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                        help='Set the logging level (default: INFO)')
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(levelname)s: %(message)s'
    )
    
    # Define corpus
    corpus = [
        "the cat sat on the mat",
        "the dog sat on the rug",
        "the cat chased the mouse",
        "the dog chased the ball",
        "the cat slept on the mat",
        "the dog barked at the mailman",
    ]
    
    # Create tokenizer
    logging.info("Building vocabulary...")
    tokenizer = Tokenizer(corpus)
    logging.info(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    logging.info(f"Vocabulary: {tokenizer.vocab}")
    
    # Create model
    logging.info("\nInitializing model...")
    model = SimpleLLM(
        vocab_size=tokenizer.get_vocab_size(),
        embed_dim=32,  # Small for demonstration
        max_seq_len=20,
        num_layers=1
    )
    logging.info("Model created!")
    
    # Train
    train(model, corpus, tokenizer, num_epochs=100, learning_rate=0.01, batch_size=4)
    
    # Test generation
    logging.info("\n" + "=" * 50)
    logging.info("Testing generation:")
    logging.info("=" * 50)
    
    test_prompts = ["the cat", "the dog", "the", "the cow"]
    for prompt in test_prompts:
        generated = model.generate(tokenizer, prompt, max_length=8, temperature=1.0)
        logging.info(f"Prompt: '{prompt}'")
        logging.info(f"Generated: '{generated}'")
        logging.info("")

