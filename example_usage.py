vocab_size = 10000
d_model = 512
num_layers = 6
num_heads = 8
d_ff = 2048

model = TransformerEncoder(vocab_size, d_model, num_layers, num_heads, d_ff)

# Example input (batch_size, sequence_length)
input_sequence = torch.randint(0, vocab_size, (32, 50))

# Forward pass
output = model(input_sequence)
