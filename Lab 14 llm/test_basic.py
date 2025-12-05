"""
Basic test to verify the code structure works.
Run this after installing numpy: pip install numpy
"""

try:
    import numpy as np
    print("✓ NumPy imported successfully")
    
    from tokenizer import Tokenizer
    print("✓ Tokenizer imported successfully")
    
    from layers import Embedding, Linear, Attention
    print("✓ Layers imported successfully")
    
    from model import SimpleLLM
    print("✓ Model imported successfully")
    
    # Test tokenizer
    corpus = ["the cat sat on the mat", "the dog sat on the rug"]
    tokenizer = Tokenizer(corpus)
    print(f"✓ Tokenizer created with vocab size: {tokenizer.get_vocab_size()}")
    
    # Test model creation
    model = SimpleLLM(vocab_size=tokenizer.get_vocab_size(), embed_dim=32, max_seq_len=10, num_layers=1)
    print("✓ Model created successfully")
    
    # Test forward pass
    test_input = np.array([[tokenizer.word_to_idx['<BOS>'], tokenizer.encode("the cat")[0]]])
    logits = model.forward(test_input)
    print(f"✓ Forward pass successful, output shape: {logits.shape}")
    
    print("\n✅ All basic tests passed! The code structure is correct.")
    print("You can now run 'python3 train.py' to train the model.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install numpy: pip install numpy")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

