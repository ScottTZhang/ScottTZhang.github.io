```
import os
import gc
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence

# -------------------------------
# Step 1: Split your original dataset
# -------------------------------
# This block assumes you have a TSV file named 'your_dataset.tsv'
# with at least two columns: 'source' and 'target'.
if not os.path.exists('train.tsv') or not os.path.exists('val.tsv'):
    # Read the TSV file (ensure your file is named with .tsv if it’s tab-delimited)
    df = pd.read_csv('sample.tsv', sep='\t', encoding='utf-8-sig')
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_df.to_csv('train.tsv', sep='\t', index=False)
    val_df.to_csv('val.tsv', sep='\t', index=False)
    print("Dataset split complete: 'train.tsv' and 'val.tsv' created.")
else:
    print("Train and validation TSV files already exist.")

#-------------------------------
# Step 2: Load the pre-trained model and tokenizer
# -------------------------------
# If you have the model locally, set model_name to the local path (e.g., './mt5-large')
# Otherwise, you can use the identifier 'google/mt5-large' to download from Hugging Face Hub.
model_name = './mt5-large'
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = MT5Tokenizer.from_pretrained(model_name)

# -------------------------------
# Step 3: Load and preprocess the dataset
# -------------------------------
# The CSV files (train.csv and val.csv) must have at least 'source' and 'target' columns.
dataset = load_dataset('csv', data_files={'train': 'train.tsv', 'validation': 'val.tsv'}, delimiter='\t')

def preprocess_function(examples):
    # If your dataset does not include the language token in the source text,
    # you can uncomment the following line to prepend it (e.g., for Czech translation):
    examples['source'] = [f"<2cs> {text}" for text in examples['source']]

    inputs = examples['source']  # For example: "<2cs> Hello, how are you?"
    targets = examples['target']
    
    # print(inputs)
    # print(targets)
    # Tokenize source text with padding and truncation
    model_inputs = tokenizer(inputs, max_length=128, truncation=True, padding='max_length')
    # Tokenize target text (using target tokenizer context)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=128, truncation=True, padding='max_length')
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# Tokenize the datasets and remove the raw text columns to save memory
tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['source', 'target'])

# Print sample and exam data fromat
sample_input_ids = tokenized_datasets['train'][0]['input_ids']
decoded_input = tokenizer.decode(sample_input_ids, skip_special_tokens=True)
print("Decoded input:", decoded_input)

sample_labels = tokenized_datasets['train'][0]['labels']
decoded_labels = tokenizer.decode(sample_labels, skip_special_tokens=True)
print("Decoded labels:", decoded_labels)

def custom_collate_fn(batch):
    if not batch:
        return {'input_ids': torch.tensor([]), 'labels': torch.tensor([])}
    inputs = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100) # -100 is the default ignore index in PyTorch's CrossEntropyLoss
    return {'input_ids': inputs_padded, 'labels': labels_padded}

# Create DataLoaders with multiple workers and pinned memory for efficient data loading
train_dataloader = DataLoader(
    tokenized_datasets['train'],
    batch_size=8,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
    collate_fn=custom_collate_fn  # Use the custom collate function
)

eval_dataloader = DataLoader(
    tokenized_datasets['validation'],
    batch_size=8,
    num_workers=0,
    pin_memory=True,
    collate_fn=custom_collate_fn  # Use the custom collate function
)

# -------------------------------
# Step 4: Configure and apply LoRA
# -------------------------------
# LoRA configuration with reduced rank and scaling factor for lower memory usage
lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=4,               # Reduced rank to save memory
    lora_alpha=16,     # Reduced scaling factor
    lora_dropout=0.1,
    modules_to_save=["q_proj", "v_proj"]  # Specify modules to fully fine-tune
)
model = get_peft_model(model, lora_config)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# -------------------------------
# Step 5: Set up optimizer, scheduler, and mixed precision
# -------------------------------
# 学习率过高可能导致梯度爆炸。尝试降低学习率，例如从 5e-5 降低到 1e-5
# Learning Rate: Even though you lowered it to 1e-5, you might try lowering it further (e.g., 5e-6) to see if the loss stabilizes.
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-6, weight_decay=0.01)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
scaler = torch.amp.GradScaler('cuda')

# -------------------------------
# Step 6: Training Loop
# -------------------------------
num_epochs = 3
gradient_accumulation_steps = 4
logging_steps = 200  # Log every 200 global steps
eval_steps = 500     # Evaluate every 500 global steps

global_step = 0
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    for step, batch in enumerate(train_dataloader):
        # Detect if batch is empty
        if len(batch) == 0:
            print("Empty batch detected!")
            continue
        
        # Move batch tensors to GPU
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        
        #print("Input IDs:", input_ids)
        #print("Labels:", labels)
        
        # Exam inputs and labels
        if torch.isnan(input_ids).any() or torch.isinf(input_ids).any():
            print("Invalid input IDs detected!")
            continue
        if torch.isnan(labels).any() or torch.isinf(labels).any():
            print("Invalid labels detected!")
            continue
        
        with torch.amp.autocast('cuda'):
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / gradient_accumulation_steps  # Normalize loss
            # Temporarily disable AMP
            loss.backward()
        
        # Detect loss value
        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print("NaN or Inf loss detected!")
            continue
            
        scaler.scale(loss).backward()
        
        if (step + 1) % gradient_accumulation_steps == 0:
            # Start of Add gradient clipping
            scaler.unscale_(optimizer)  # 取消缩放以进行梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            # End of Add gradient clipping
        
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1

            # Logging training loss
            if global_step % logging_steps == 0:
                print(f"Epoch {epoch+1}, Global Step {global_step}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")

            # Evaluate every eval_steps
            if global_step % eval_steps == 0:
                model.eval()
                total_eval_loss = 0.0
                num_eval_batches = 0
                with torch.no_grad():
                    for eval_batch in eval_dataloader:
                        input_ids = eval_batch['input_ids'].to(device)
                        labels = eval_batch['labels'].to(device)
                        with torch.amp.autocast('cuda'):
                            eval_outputs = model(input_ids=input_ids, labels=labels)
                            total_eval_loss += eval_outputs.loss.item()
                        num_eval_batches += 1
                        torch.cuda.empty_cache()
                        gc.collect()
                avg_eval_loss = total_eval_loss / num_eval_batches
                print(f"Epoch {epoch+1}, Global Step {global_step}, Validation Loss: {avg_eval_loss:.4f}")
                model.train()

        # Clear unused GPU memory
        torch.cuda.empty_cache()
        gc.collect()
    scheduler.step()


    # Save checkpoint at the end of each epoch
    checkpoint_path = f'./en2cz_model_epoch_{epoch+1}'
    model.save_pretrained(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

# -------------------------------
# Step 7: Save the Final Model
# -------------------------------
final_save_path = './en2cz_model_20250216'
model.save_pretrained(final_save_path)
tokenizer.save_pretrained(final_save_path)
print("Training complete. Model saved to", final_save_path)
```
Dataset has 26000 lines.

The result prints: 

NaN or Inf loss detected!

NaN or Inf loss detected!

NaN or Inf loss detected!
