from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Set the device to GPU if available, otherwise CPU
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the GPT-2 tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id).to(torch_device)

# Prompt for the user input
user_prompt = input("Enter text to generate prompt: ")

# Set fixed values for max words and max sequences
max_words = 1000  # You can adjust this value
max_sequences = 4  # You can adjust this value

# Tokenize the user prompt
model_inputs = tokenizer(user_prompt, return_tensors='pt').to(torch_device)

from transformers import set_seed

# Set a seed for reproducibility
set_seed(42)

# Generate text
sample_outputs = model.generate(
    **model_inputs,
    max_new_tokens=max_words,
    do_sample=True,
    top_k=50,
    top_p=0.95,
    no_repeat_ngram_size=2,
    temperature=0.4,
    num_return_sequences=max_sequences,
)

print("Generated Expansions Based On Context:\n" + 100 * '-')

# Decode the generated sequences
samples = [
    tokenizer.decode(sample_output, skip_special_tokens=True)
    for sample_output in sample_outputs
]

# Select the longest sequence
result = max(samples, key=len)

# Trim the text to end at the last full stop
last_full_stop_index = result.rfind('.')
if last_full_stop_index != -1:
    result = result[:last_full_stop_index + 1]

# Print the final result
print(result)
# Save the model and tokenizer
model.save_pretrained('./results')
tokenizer.save_pretrained('./results')

print()

# Add this line to keep the console window open
input()
