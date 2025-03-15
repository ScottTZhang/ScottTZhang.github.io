
### This is a test artical about how to use github.io, also.
## My PC configuration:
1. CPU: AMD Ryzen 9 7900X 12-Core Processor 4.70 GHz,
2. Computer with 32.0 GB RAM (31.6 GB available),
3. System type: 64-bit operating system, x64-based processor,
4. Gigabyte NVIDIA GeForce RTX 4070 Ti Super Eagle.

## 1) Everything must be run using x64 Command prompt:
Press **Win + S**, search for **x64 Native Tools Command Prompt for VS 2022**, and open it.

## 2) Create a Virtual Environment
Create a virtual environment on a non-C drive, for example: 
in Command prompt: `python -m venv D:\myenv`

## 3) Activate the virtual environment in Command prompt:
`D:\myenv\Scripts\activate`
After activation, your prompt will change to show the virtual environment name, e.g.:
**(myenv) C:\Users\YourUsername> **
> [Note] Use a Virtual Environment:
> It’s a good practice to always use a virtual environment for Python projects to avoid dependency conflicts and ensure reproducibility.

> Document Your Environment:
>If you’re not using a virtual environment, at least document the installed packages using: 
>`pip freeze > requirements.txt` in Command prompt. This file can be used to recreate the environment later.

## 4) Install Required Libraries
Install the necessary libraries in the virtual environment: 
in Command prompt: `pip install transformers torch sentencepiece`

**[Issue]**
*Getting requirements to build wheel ... error*
*error: subprocess-exited-with-error*

**[solution]** Install Build Tools
- On Windows, install the Build Tools for Visual Studio:
- Download and install the Build Tools for Visual Studio.
- During installation, select the C++ build tools workload.
- Make sure following are chekced:
  1. MSVC v143 - VS 2022 C++ x64/x86 build tools
  2. Windows 10 SDK（Windows 10 SDK) -- I installed 11 SDK
  3. C++ CMake tools
- Restart your computer after installation.

**[Issue]** Encountered following when pip install:

*build\lib.win32-cpython-313\sentencepiece_sentencepiece.cp313-win_amd64.pyd : fatal error LNK1120: 117 个无法解析的外部命令*
*error: command 'D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\HostX86\x86\link.exe' failed with exit code 1120*

**[Solution]**
- Open a 64-bit command prompt: Press Win + S, search for **x64 Native Tools Command Prompt for VS 2022**, and open it.
- In the command prompt, activate your Python environment (if using a virtual environment): in Command Prompt: `D:\myenv\Scripts\activate`
- Try installing the package again: in Command Prompt: `pip install sentencepiece`

## 5) Change model cache path
By default, Hugging Face is downloaded to `C:\Users.cache\huggingface`. You can set system env var TRANSFORMERS_CACHE to change cache path.

In Command Prompt: **(myenv) D:\my_models>**`set TRANSFORMERS_CACHE=D:\huggingface_models`

To set TRANSFORMERS_CACHE permanently (this writes to your user environment variables),

In Command Prompt: **(myenv) D:\my_models>**`setx TRANSFORMERS_CACHE "D:\huggingface_models"`

## 6) Download the Model

6.1) Run the following code to download the mT5-large model:

python:

```
from transformers import MT5ForConditionalGeneration, T5Tokenizer
model_name = "google/mt5-large"
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
```
The model files will be downloaded to the D:\huggingface_models directory.
6.2) Save the Model to a Non-C Drive
Save the model and tokenizer to a specified directory, e.g., D:\my_models\mt5-large: 
python:
```
model.save_pretrained("D:/my_models/mt5-large")
tokenizer.save_pretrained("D:/my_models/mt5-large")
```
6.3) Load the Model from a Non-C Drive
You can load the model directly from the non-C drive without re-downloading:
python:
```
from transformers import MT5ForConditionalGeneration, T5Tokenizer
model = MT5ForConditionalGeneration.from_pretrained("D:/my_models/mt5-large")
tokenizer = T5Tokenizer.from_pretrained("D:/my_models/mt5-large")
```
## 7) Run the Model
5.1 Example Code
Below is a complete example for English-to-Czech translation:
```
from transformers import MT5ForConditionalGeneration, T5Tokenizer
import torch

# Load the model and tokenizer
model = MT5ForConditionalGeneration.from_pretrained("D:/my_models/mt5-large").to("cuda")
tokenizer = T5Tokenizer.from_pretrained("D:/my_models/mt5-large")

# Input text
input_text = "Translate English to Czech: Hello, how are you?"

# Tokenize and encode
inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to("cuda")

# Generate translation
with torch.no_grad():
    outputs = model.generate(**inputs)

# Decode the translation
translated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f"Translated: {translated_text}")
```
When I put the above in a text.py and run it, I encountered the **[issue]**:

*File "D:\myenv\Lib\site-packages\torch\cuda_init_.py", line 310, in _lazy_init*
*raise AssertionError("Torch not compiled with CUDA enabled")*
*AssertionError: Torch not compiled with CUDA enabled*

**[Solution]**
- `pip uninstall torch torchvision torchaudio`
- `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

The output will be **Translated: <extra_id_0>: Hello, how are you? Hello, how are you? Hello, how are you?**

That means the model is not fine-tuned. Now the model is ready to perform fine-tune!!!
