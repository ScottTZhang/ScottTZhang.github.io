### This is a test artical about how to use github.io, also.
My PC configuration:

1. Everything must be run using x64 Command prompt:
Press Win + S, search for x64 Native Tools Command Prompt for VS 2022, and open it.

2. Activate the virtual environment in Command prompt:
D:\myenv\Scripts\activate

1.2 Create a Virtual Environment
Create a virtual environment on a non-C drive, for example: in bash
python -m venv D:\myenv 
Activate the virtual environment:
Windows: D:\myenv\Scripts\activate

After activation, your prompt will change to show the virtual environment name, e.g.:
(myenv) C:\Users\YourUsername> 
Use a Virtual Environment:
It’s a good practice to always use a virtual environment for Python projects to avoid dependency conflicts and ensure reproducibility.
Example: bash

python -m venv D:\myenv
D:\myenv\Scripts\activate
pip install transformers torch sentencepiece
Document Your Environment:
If you’re not using a virtual environment, at least document the installed packages using: in bash
pip freeze > requirements.txt This file can be used to recreate the environment later.

1.3 Install Required Libraries
Install the necessary libraries in the virtual environment: in bash
pip install transformers torch sentencepiece
Getting requirements to build wheel ... error
error: subprocess-exited-with-error
[solution 2.1] Install Build Tools
On Windows, install the Build Tools for Visual Studio:
Download and install the Build Tools for Visual Studio.
During installation, select the C++ build tools workload.
Make sure following are chekced:
MSVC v143 - VS 2022 C++ x64/x86 生成工具（MSVC v143 - VS 2022 C++ x64/x86 build tools）
Windows 10 SDK（Windows 10 SDK) -- I installed 11 SDK
C++ CMake 工具（C++ CMake tools）
Restart your computer after installation.

Encountered following when pip install:
build\lib.win32-cpython-313\sentencepiece_sentencepiece.cp313-win_amd64.pyd : fatal error LNK1120: 117 个无法解析的外部命令
error: command 'D:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.43.34808\bin\HostX86\x86\link.exe' failed with exit code 1120
[Solution]
Open a 64-bit command prompt:
Press Win + S, search for x64 Native Tools Command Prompt for VS 2022, and open it.

In the command prompt, activate your Python environment (if using a virtual environment): in bash
D:\myenv\Scripts\activate
Try installing the package again: in bash
pip install sentencepiece

2.1 Change model cache path
By default, Hugging Face is downloaded to C:\Users.cache\huggingface。You can set system env var TRANSFORMERS_CACHE to change cache path。

Add system env variable from bash:
set TRANSFORMERS_CACHE=D:\huggingface_models

2.2 Download the Model
Run the following code to download the mT5-large model:

python

from transformers import MT5ForConditionalGeneration, T5Tokenizer

model_name = "google/mt5-large"
model = MT5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)
The model files will be downloaded to the D:\huggingface_models directory.

3. Save the Model to a Non-C Drive
Save the model and tokenizer to a specified directory, e.g., D:\my_models\mt5-large: in python
model.save_pretrained("D:/my_models/mt5-large")
tokenizer.save_pretrained("D:/my_models/mt5-large")

4. Load the Model from a Non-C Drive
You can load the model directly from the non-C drive without re-downloading:
in python
from transformers import MT5ForConditionalGeneration, T5Tokenizer
model = MT5ForConditionalGeneration.from_pretrained("D:/my_models/mt5-large")
tokenizer = T5Tokenizer.from_pretrained("D:/my_models/mt5-large")

Run the Model
5.1 Example Code
Below is a complete example for English-to-Czech translation:
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

When I put the above in a text.py and run it, I encountered the issue:
File "D:\myenv\Lib\site-packages\torch\cuda_init_.py", line 310, in _lazy_init
raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled
[Solution]

pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

Now the model is ready to perform fine-tune!!!
