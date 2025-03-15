
Data format (.tsv):
| source	| target |
| ----------- | ----------- |
|Prefix	|Prefix |
|Please set password for this {0}.	|Nastavte heslo pro tento {0}.|

New version of code:
```
import os
import gc
import torch
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from torch.optim.lr_scheduler import StepLR
from torch.nn.utils.rnn import pad_sequence
from functools import partial

# Set up logging for detailed information.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def prepare_dataset():
    # Split your dataset into train and validation sets if needed.
    if not os.path.exists('train.tsv') or not os.path.exists('val.tsv'):
        df = pd.read_csv('cleaned_file.tsv', sep='\t', encoding='utf-8-sig', on_bad_lines='skip')
        train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
        train_df.to_csv('train.tsv', sep='\t', index=False)
        val_df.to_csv('val.tsv', sep='\t', index=False)
        logging.info("Dataset split complete: 'train.tsv' and 'val.tsv' created.")
    else:
        logging.info("Train and validation TSV files already exist.")

def load_and_tokenize(tokenizer):
    dataset = load_dataset('csv', data_files={'train': 'train.tsv', 'validation': 'val.tsv'}, delimiter='\t')

    def preprocess_function(examples):
        inputs = [f"<2cs> {str(text) if text is not None else ''}" for text in examples['source']]
        targets = [str(text) if text is not None else '' for text in examples['target']]
        filtered_inputs, filtered_targets = [], []
        for inp, tgt in zip(inputs, targets):
            if inp.strip() and tgt.strip():
                filtered_inputs.append(inp)
                filtered_targets.append(tgt)
        if not filtered_inputs or not filtered_targets:
            return {'input_ids': [], 'labels': []}
        model_inputs = tokenizer(filtered_inputs, max_length=128, truncation=True, padding='max_length')
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(filtered_targets, max_length=128, truncation=True, padding='max_length')
        model_inputs['labels'] = labels['input_ids']
        return model_inputs

    tokenized_datasets = dataset.map(preprocess_function, batched=True, remove_columns=['source', 'target'])
    return tokenized_datasets

def custom_collate_fn(batch, tokenizer):
    if not batch:
        return {'input_ids': torch.tensor([]), 'labels': torch.tensor([])}
    inputs = [torch.tensor(item['input_ids']) for item in batch]
    labels = [torch.tensor(item['labels']) for item in batch]
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)  # -100 is the ignore index
    return {'input_ids': inputs_padded, 'labels': labels_padded}

def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            num_batches += 1
    avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
    logging.info(f"Validation Loss: {avg_loss:.4f}")
    model.train()
    return avg_loss

def main():
    prepare_dataset()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # If GPU memory is a concern, consider using a smaller model (e.g., 'google/mt5-small')
    model_name = './mt5-large'
    model = MT5ForConditionalGeneration.from_pretrained(model_name)
    tokenizer = MT5Tokenizer.from_pretrained(model_name)
    
    # Enable gradient checkpointing to reduce memory usage.
    model.gradient_checkpointing_enable()
    model.to(device)
    
    tokenized_datasets = load_and_tokenize(tokenizer)
    
    # Use a lower batch size to further reduce memory footprint.
    batch_size = 2  # Reduced from 4
    
    # Use functools.partial to pass the tokenizer to the collate function.
    collate_fn = partial(custom_collate_fn, tokenizer=tokenizer)
    
    train_dataloader = DataLoader(
        tokenized_datasets['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Using 0 workers reduces memory overhead on Windows.
        pin_memory=True,
        collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        tokenized_datasets['validation'],
        batch_size=batch_size,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Configure LoRA if required (currently commented out)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=4,
        lora_alpha=16,
        lora_dropout=0.1,
        modules_to_save=["q_proj", "v_proj"]
    )
    # model = get_peft_model(model, lora_config)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.01)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.9)
    scaler = torch.cuda.amp.GradScaler()
    
    num_epochs = 3
    gradient_accumulation_steps = 4
    logging_steps = 200
    eval_steps = 500
    global_step = 0
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_dataloader):
            # Skip empty or invalid batches.
            if not batch or 'input_ids' not in batch or batch['input_ids'].numel() == 0:
                logging.warning("Empty or invalid batch detected; skipping.")
                continue
            
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            if not torch.isfinite(input_ids).all():
                logging.warning("Non-finite input_ids detected; skipping batch.")
                continue
            if not torch.isfinite(labels).all():
                logging.warning("Non-finite labels detected; skipping batch.")
                continue
            
            with torch.cuda.amp.autocast():
                outputs = model(input_ids=input_ids, labels=labels)
                if not torch.isfinite(outputs.logits).all():
                    logging.warning("Non-finite logits detected; recomputing without autocast.")
                    with torch.cuda.amp.autocast(enabled=False):
                        outputs = model(input_ids=input_ids, labels=labels)
                    if not torch.isfinite(outputs.logits).all():
                        logging.error("Non-finite logits persist even in FP32; skipping batch.")
                        continue
                loss = outputs.loss / gradient_accumulation_steps
            
            if not torch.isfinite(loss):
                logging.error(f"Non-finite loss at epoch {epoch+1}, step {step+1}; skipping batch.")
                continue
            
            scaler.scale(loss).backward()
            
            if (step + 1) % gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                
                if global_step % logging_steps == 0:
                    logging.info(f"Epoch {epoch+1}, Global Step {global_step}, Loss: {loss.item() * gradient_accumulation_steps:.4f}")
                
                if global_step % eval_steps == 0:
                    evaluate(model, eval_dataloader, device)
            
            # Clear unused GPU memory after processing each batch.
            torch.cuda.empty_cache()
            gc.collect()
        scheduler.step()
        
        checkpoint_path = f'./en2cz_model_epoch_{epoch+1}'
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        logging.info(f"Checkpoint saved at {checkpoint_path}")
    
    final_save_path = './en2cz_model_final'
    model.save_pretrained(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    logging.info(f"Training complete. Model saved to {final_save_path}")

if __name__ == '__main__':
    main()
```
---
## Dataset Preparation
Function: `prepare_dataset()`

- **Purpose:**

  Checks whether the training and validation files (train.tsv and val.tsv) exist. If not, it reads a raw dataset file (cleaned_file.tsv), splits it into training and validation sets using an 80/20 split, and saves them as TSV files.

- **Key Techniques:**

  - File Checking: Prevents re-splitting if the files already exist.

  - Train-Test Split: Uses a fixed random seed (random_state=42) for reproducibility.

---
## Tokenization and Preprocessing
Function: `load_and_tokenize(tokenizer)`

- **Purpose:**
  Loads the dataset from the TSV files using Hugging Face’s load_dataset function. Then, it applies a nested preprocess_function to tokenize and preprocess the data.

Details in preprocess_function:

- **Input and Target Construction:**

  The source texts are prefixed with a special token (here, "<2cs>") and converted to strings. The targets are similarly processed.

  - Filtering:

  Only non-empty pairs are kept to avoid empty examples.

  - Tokenization:

  Both source (inputs) and target (labels) texts are tokenized with a fixed maximum sequence length (128 tokens) and are padded to a uniform length.

  - Label Preparation:

  The tokenized targets are stored under the key 'labels' so that the model can use them during training.

- **Mapping:**

  The dataset.map method applies the preprocessing in a batched manner, and the raw text columns are removed to reduce memory overhead.

---
## Custom Collate Function
Function: `custom_collate_fn(batch, tokenizer)`

- **Purpose:**

  Handles the collation of individual examples into a single batch. This function is critical for:

- **Padding:**

  It pads sequences to ensure that all examples in a batch have the same length.

- **Tensor Conversion:**

  It converts lists of token IDs into PyTorch tensors.

- **Padding Values:**

  Uses the tokenizer’s pad_token_id for input sequences and -100 for labels (a value typically used to ignore padding in the loss calculation).

- **Partial Application:**

  The function is wrapped using functools.partial to “bake in” the tokenizer argument so that it is picklable (important when using DataLoader workers).

---
## Evaluation Function
Function: `evaluate(model, dataloader, device)`

- **Purpose:**

  Runs the model in evaluation mode on a given validation DataLoader.

- **Key Points:**

  - No Gradient Calculation:

    Uses torch.no_grad() to avoid unnecessary gradient computations during evaluation.

  - Automatic Mixed Precision (AMP):

    Runs the forward pass under torch.cuda.amp.autocast() to leverage lower-precision computations for speed and reduced memory usage.

  - Loss Aggregation:

    Computes the average loss over all validation batches and logs the result.

---
## Main Function and Training Loop
### Function: `main()`
- **Dataset and Model Initialization:**
  
  Calls `prepare_dataset()` to ensure the data is ready.

  Loads the pretrained MT5 model and its corresponding tokenizer.

- **Device Management:**

  Sets up the device (GPU if available, otherwise CPU).

- **Gradient Checkpointing:**

  Calls `model.gradient_checkpointing_enable()` to reduce memory usage by recomputing intermediate activations during the backward pass. This is especially useful for very large models.

- **Tokenization:**

  Prepares the dataset using load_and_tokenize(tokenizer).

- **DataLoader Setup:**

  Sets a small batch size (2) to keep memory usage low.

  Uses a custom collate function (with the tokenizer already provided) to prepare batches.

- Optional LoRA Configuration:

  A configuration for LoRA (Low-Rank Adaptation) is prepared (though the actual application is commented out).

- **LoRA Optimization:**

  When enabled, LoRA updates only low-rank matrices within certain modules (e.g., q_proj and v_proj), greatly reducing the number of trainable parameters and thus memory usage during fine-tuning.

- **Optimizer, Scheduler, and Mixed Precision Setup:**

  - Optimizer: Uses AdamW with weight decay and a very low learning rate.

  - Scheduler: Uses StepLR to reduce the learning rate over epochs.

  - Mixed Precision:

    Uses `torch.cuda.amp.GradScaler()` to manage the scaling of gradients when training with mixed precision. This improves performance and reduces memory consumption.
    
- **Training Loop**:

  - Gradient Accumulation:

    Loss is divided by gradient_accumulation_steps to simulate a larger batch size without requiring the memory of a single large batch.

  - Mixed Precision and Loss Scaling:

    The forward pass is wrapped with torch.cuda.amp.autocast(), and the scaled loss is backpropagated using the GradScaler.

  - Gradient Clipping:

    Gradients are clipped to a maximum norm (1.0) to stabilize training.

  - Logging and Evaluation:

    Periodic logging of loss and evaluation on the validation set are performed.

  - Memory Management:

    After each batch, explicit calls to torch.cuda.empty_cache() and gc.collect() help to free unused GPU memory.

- **Checkpointing**:

  At the end of each epoch, the model and tokenizer are saved. A final checkpoint is saved after training completes.

---
## Training results and issue:

- Results:

  ```
  Line  343: 2025-03-06 04:07:43,424 - INFO - Epoch 2, Global Step 3600, Loss: 4.7958
  Line 1115: 2025-03-06 04:21:34,620 - INFO - Epoch 2, Global Step 3800, Loss: 3.8345
  Line 1877: 2025-03-06 04:34:48,126 - INFO - Epoch 2, Global Step 4000, Loss: 2.6528
  Line 1878: 2025-03-06 04:38:02,593 - INFO - Validation Loss: nan
  Line 2647: 2025-03-06 04:51:36,512 - INFO - Epoch 2, Global Step 4200, Loss: 1.8896
  Line 3416: 2025-03-06 05:05:06,259 - INFO - Epoch 2, Global Step 4400, Loss: 1.1527
  Line 3805: 2025-03-06 05:15:13,463 - INFO - Validation Loss: nan
  Line 4200: 2025-03-06 05:21:54,331 - INFO - Epoch 2, Global Step 4600, Loss: 1.8585
  Line 4988: 2025-03-06 05:35:50,887 - INFO - Epoch 2, Global Step 4800, Loss: 0.7073
  Line 5782: 2025-03-06 05:49:16,600 - INFO - Epoch 2, Global Step 5000, Loss: 0.9007
  Line 5783: 2025-03-06 05:52:31,622 - INFO - Validation Loss: nan
  Line 6577: 2025-03-06 06:06:45,737 - INFO - Epoch 2, Global Step 5200, Loss: 0.5821
  Line 7381: 2025-03-06 06:20:47,805 - INFO - Epoch 3, Global Step 5400, Loss: 0.8262
  Line 7783: 2025-03-06 06:31:00,309 - INFO - Validation Loss: nan
  Line 8183: 2025-03-06 06:37:53,569 - INFO - Epoch 3, Global Step 5600, Loss: 0.7085
  ```
- Issues:

  `model = get_peft_model(model, lora_config)` is commented out. Still encounter runtime error:

	Even the loss is low, the test sentences for translation still bad like trash, lol

  ```
  RuntimeError: CUDA error: out of memory
  CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
  For debugging consider passing CUDA_LAUNCH_BLOCKING=1
  Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
  ```

### After enabling `model = get_peft_model(model, lora_config)`
- Results: 
  ```
	Line  363: 2025-03-06 14:52:16,427 - INFO - Epoch 3, Global Step 5800, Loss: 33.1783
	Line 1164: 2025-03-06 14:58:36,783 - INFO - Epoch 3, Global Step 6000, Loss: 33.2051
	Line 1958: 2025-03-06 15:08:22,505 - INFO - Epoch 3, Global Step 6200, Loss: 35.6984
	Line 2755: 2025-03-06 15:14:38,682 - INFO - Epoch 3, Global Step 6400, Loss: 36.6631
	Line 3552: 2025-03-06 15:24:28,047 - INFO - Epoch 3, Global Step 6600, Loss: 34.2763
	Line 4352: 2025-03-06 15:30:50,879 - INFO - Epoch 3, Global Step 6800, Loss: 30.9876
	Line 5146: 2025-03-06 15:37:10,508 - INFO - Epoch 3, Global Step 7000, Loss: 34.8415
	Line 5945: 2025-03-06 15:46:55,176 - INFO - Epoch 3, Global Step 7200, Loss: 32.9306
	Line 6745: 2025-03-06 15:53:16,560 - INFO - Epoch 3, Global Step 7400, Loss: 31.7479
	Line 7544: 2025-03-06 16:03:04,357 - INFO - Epoch 3, Global Step 7600, Loss: 33.7565
	Line 8342: 2025-03-06 16:09:24,471 - INFO - Epoch 3, Global Step 7800, Loss: 31.3005
    ```
---
### 优化技术总结:
- **显存优化**

  梯度检查点（Gradient Checkpointing）：牺牲计算时间换取显存，通过重计算中间激活而非存储。

  小批次训练（Batch Size=2）：降低单步显存需求，适合低显存设备。

  混合精度训练（FP16）：使用autocast和GradScaler减少显存占用并加速计算。

  参数高效微调（LoRA）：可选方案，通过低秩适配器更新部分参数，大幅减少显存需求。

- **训练效率优化**

  梯度累积（Gradient Accumulation=4）：累积多个小批次的梯度再更新，模拟大批次训练。

  动态填充（Dynamic Padding）：仅填充到批次内最大长度，减少无效计算。

  数据加载优化：pin_memory=True加速CPU到GPU数据传输，num_workers=0避免Windows多进程问题。

- **稳定性与收敛性**

  梯度裁剪（Gradient Clipping）：限制梯度最大范数，防止梯度爆炸。

  学习率调度（StepLR）：逐步衰减学习率，平衡收敛速度与稳定性。

  非数值过滤：跳过含NaN/Inf的批次，避免训练崩溃。

- **模型保存与恢复**

  定期检查点：每epoch保存模型，防止训练中断丢失进度。

  最终模型保存：训练完成后保存完整模型和分词器。
