"""
June 2nd, 2026
I haven't tested the code yet (Project Gutenberg servers aren't the strongest right now), but what I have should work. 
As always, a line full of underscores denotes the presence of a separate cell
"""
# 1. Install pip upgrade
!pip install --upgrade pip
__________________________________
# Hyperspecific installs and fixes for my situation
# 1. FIXED INSTALLATION
# Thoroughly uninstall unsloth and all related packages to ensure a clean slate.
!pip uninstall -y unsloth unsloth_zoo transformers datasets trl peft accelerate bitsandbytes xformers torch torchvision torchaudio

# Install the stable release of unsloth and unsloth_zoo from PyPI
!pip install "unsloth[colab-new]" torchvision torchaudio

# Check for any remaining dependency conflicts after installation
!pip check

# Standard imports
import unsloth
from unsloth import FastLanguageModel
import torch
import requests
import re
from trl import SFTTrainer
from transformers import TrainingArguments, TrainerCallback
from datasets import Dataset
______________________________________________________________
# ==========================================
# 2. DATA ACQUISITION & DEEP CLEANING
# ==========================================
import requests
import re

urls = [
    "https://www.gutenberg.org/files/1497/1497-0.txt",   # Plato: The Republic
    "https://www.gutenberg.org/files/1600/1600-0.txt",   # Plato: Symposium 
    "https://www.gutenberg.org/files/1658/1658-0.txt",   # Plato: Phaedo
    "https://www.gutenberg.org/files/6762/6762-0.txt",   # Aristotle: Politics
    "https://www.gutenberg.org/files/2680/2680-0.txt",   # Marcus Aurelius: Meditations
    "https://www.gutenberg.org/files/45109/45109-0.txt", # Epictetus: The Enchiridion
    "https://www.gutenberg.org/files/23639/23639-0.txt", # Plutarch's Morals
    "https://www.gutenberg.org/files/59/59-0.txt",       # Descartes: Discourse on Method...
    "https://www.gutenberg.org/files/25830/25830-0.txt", # Descartes: A Discourse of a Method for the Well Guiding of Reason
    "https://www.gutenberg.org/files/70091/70091-0.txt", # Descartes: Six metaphysical meditations
    "https://www.gutenberg.org/files/3800/3800-0.txt",   # Spinoza: Ethics
    "https://www.gutenberg.org/files/4320/4320.txt",     # Hume: An Enquiry Concerning the Principles of Morals
    "https://www.gutenberg.org/files/34901/34901-0.txt", # Mill: On Liberty
    "https://www.gutenberg.org/files/38158/38158-0.txt", # Mill: Utilitarianism
    "https://www.gutenberg.org/files/1232/1232-0.txt",   # Machiavelli: The Prince
    "https://www.gutenberg.org/files/3207/3207-0.txt",   # Hobbes: Leviathan
    "https://www.gutenberg.org/files/7370/7370-0.txt",   # Locke: Second Treatise of Government
    "https://www.gutenberg.org/files/46333/46333-0.txt", # Rousseau: The Social Contract
    "https://www.gutenberg.org/files/4280/4280-0.txt",   # The Critique of Pure Reason - Kant
    "https://www.gutenberg.org/files/50922/50922-0.txt", # Perpetual Peace: A Philosophical Essay - Kant
    "https://www.gutenberg.org/files/5682/5682-0.txt",   # Fundamental Principles of the Metaphysic of Morals - Kant
    "https://www.gutenberg.org/files/5683/5683-0.txt",   # The Critique of Practical Reason - Kant
    "https://www.gutenberg.org/files/48433/48433-0.txt", # Kant's Critique of Judgement
    "https://www.gutenberg.org/files/52821/52821-0.txt", # Kant's Prolegomena to Any Future Metaphysics
    "https://www.gutenberg.org/files/56811/56811-0.txt", # Hegel: The Phenomenology of Mind 
    "https://www.gutenberg.org/files/41551/41551-0.txt", # Hegel: The Philosophy of History
    "https://www.gutenberg.org/files/51468/51468-0.txt", # Hegel: The Logic 
    "https://www.gutenberg.org/files/58013/58013-0.txt", # Schopenhauer: The Wisdom of Life 
    "https://www.gutenberg.org/files/1998/1998-0.txt",   # Thus Spake Zarathustra - Nietzsche 
    "https://www.gutenberg.org/files/4363/4363-0.txt",   # Beyond Good and Evil - Nietzsche 
    "https://www.gutenberg.org/files/52190/52190-0.txt", # Ecce Homo - Nietzsche 
    "https://www.gutenberg.org/files/51356/51356-0.txt", # The Birth of Tragedy - Nietzsche 
    "https://www.gutenberg.org/files/52319/52319-0.txt", # The Geneaology of Morals - Nietzsche 
    "https://www.gutenberg.org/files/52263/52263-0.txt", # The Twilight of the Idols - Nietzsche 
    "https://www.gutenberg.org/files/19322/19322-0.txt", # The Antichrist - Nietzsche
    "https://www.gutenberg.org/files/38145/38145-0.txt", # Human, All Too Human - Nietzsche 
    "https://www.gutenberg.org/files/37841/37841-0.txt", # Human, All Too Human Book II - Nietzsche
    "https://www.gutenberg.org/files/39955/39955-0.txt", # The Dawn of Day - Nietzsche
    "https://www.gutenberg.org/files/205/205-0.txt",     # Thoreau: Walden, And On The Duty Of Civil Disobedience
    "https://www.gutenberg.org/files/26716/26716-0.txt", # Ruskin: The Crown of Wild Olive
]

def clean_classic_text(text):
    start_markers = ["*** START OF THIS PROJECT", "*** START OF THE PROJECT"]
    end_markers = ["*** END OF THIS PROJECT", "*** END OF THE PROJECT"]
    start_idx = 0
    for m in start_markers:
        found = text.find(m)
        if found != -1:
            start_idx = text.find("\n", found) + 1
            break
    end_idx = len(text)
    for m in end_markers:
        found = text.find(m)
        if found != -1:
            end_idx = found
            break
    clean = text[start_idx:end_idx]
    clean = re.sub(r'CHAPTER\s+[IVXLCDM]+\.?', '', clean, flags=re.IGNORECASE)
    clean = re.sub(r'Chapter\s+\d+\.?', '', clean)
    clean = re.sub(r'\n\s*\n', '\n\n', clean)
    return clean.strip()

print("Downloading the Full Library...")
full_library_text = ""
for i, url in enumerate(urls):
    print(f"Downloading book {i+1}/{len(urls)}...")
    r = requests.get(url)
    r.encoding = 'utf-8-sig'
    full_library_text += clean_classic_text(r.text) + "\n\n"
print("Download Complete!")
____________________________________________________________________________
from datasets import Dataset
from transformers import TrainerCallback

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit", #Llama 3.1 works just as well for this in my experience
    max_seq_length = 2048,
    load_in_4bit = True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 32,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 32,
    lora_dropout = 0,
    bias = "none",
)

def format_prompt(text_chunk):
    # Optimized for a "Philosopher Continuation" persona
    return {
        "text": f"### Instruction:\nContinue the philosophical discourse, adhering to the reasoning, precise vocabulary, and profound insights characteristic of philosophical texts. Expand upon the presented arguments with logical coherence and intellectual rigor.\n\n### Input:\n{text_chunk[:200]}\n\n### Response:\n{text_chunk[200:]}"
    }

chunks = [full_library_text[i:i + 1100] for i in range(0, len(full_library_text), 1100)]
dataset = Dataset.from_dict({"raw_text": chunks})
dataset = dataset.map(lambda x: format_prompt(x["raw_text"]), remove_columns=["raw_text"])

# ==========================================
# 4. MONITORING CALLBACK (The Progress Mirror)
# ==========================================
class ProgressMonitor(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0 and state.global_step > 0:
            print(f"\n--- [PHILOSOPHICAL CHECK-IN: STEP {state.global_step}] ---")
            FastLanguageModel.for_inference(model)
            test_prompt = "The human mind, in its ceaseless quest for understanding, often encounters the paradox that"
            inputs = tokenizer([f"### Instruction:\nContinue the philosophical discourse, adhering to the reasoning, precise vocabulary, and profound insights characteristic of philosophical texts. Expand upon the presented arguments with logical coherence and intellectual rigor.\n\n### Input:\n{test_prompt}\n\n### Response:\n"], return_tensors="pt").to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=60, temperature=0.8)
            print(test_prompt,tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:\n")[-1])
            print("-" * 50)
            model.train() # Resume learning
_________________________________________________________________________________________
from datasets import Dataset

# Rebuild the dataset purely from lists
print("Rebuilding a clean dataset to bypass serialization errors...")
chunks = [full_library_text[i:i + 1100] for i in range(0, len(full_library_text), 1100)]
formatted_texts = [
    f"### Instruction:\nContinue the philosophical discourse, adhering to the reasoning, precise vocabulary, and profound insights characteristic of philosophical texts. Expand upon the presented arguments with logical coherence and intellectual rigor.\n\n### Input:\n{chunk[:200]}\n\n### Response:\n{chunk[200:]}"
    for chunk in chunks
]
clean_dataset = Dataset.from_dict({"text": formatted_texts})

# Pre-tokenize the dataset manually on a SINGLE core to prevent Unsloth's 16-core crash
print("Pre-tokenizing dataset...")
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=2048)

tokenized_dataset = clean_dataset.map(tokenize_function, batched=True, num_proc=1)
print("Dataset pre-tokenized successfully!")

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = tokenized_dataset,
    max_seq_length = 2048,
    args = TrainingArguments(
        num_train_epochs = 4,
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 8,
        warmup_steps = 50,
        learning_rate = 1e-4, 
        lr_scheduler_type = "cosine",
        weight_decay = 0.01,
        fp16 = not torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        output_dir = "philosophy_model_final",
    ),
)

# Add the callback after initialization
trainer.add_callback(ProgressMonitor())

# Begin the training!
trainer.train()
_______________________________________________________________________________
#GENERATION CELL
FastLanguageModel.for_inference(model)
test_prompt = input("")
inputs = tokenizer([f"### Instruction:\nContinue the philosophical discourse, adhering to the reasoning, precise vocabulary, and profound insights characteristic of philosophical texts. Expand upon the presented arguments with logical coherence and intellectual rigor, and maintain a logical sequence of ideas and arguments.\n\n### Input:\n{test_prompt}\n\n### Response:\n"], return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7) # I like to alter the temperature on occasion to see how a model spirals or holds its ground
print(test_prompt)
print(tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Response:\n")[-1])
print("-" * 50)
