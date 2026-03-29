# Neraium + AUX-TIME test - Google Colab

Copy each block below into a separate Colab cell and run in order.

---

## Cell 1: Clone repo and go to project

```python
# Clone (replace with your repo URL if different)
!git clone https://github.com/YOUR_USER/neraium-core-1.git
%cd neraium-core-1
```

If you uploaded the project to Colab or Drive instead:

```python
# %cd /content/neraium-core-1
# or: %cd /content/drive/MyDrive/neraium-core-1
```

---

## Cell 2: Set AUX-TIME API credentials (paste your key here)

```python
import os

# Paste your AUX-TIME API key and URL. Do not share this notebook with the key visible.
os.environ["AUX_TIME_API_KEY"] = "YOUR_AUX_TIME_API_KEY_HERE"
os.environ["AUX_TIME_TIME_URL"] = "https://api-v2.aux-time.com/time"
```

---

## Cell 3: Install dependencies

```python
!pip install -q numpy
!pip install -q -e .
```

---

## Cell 4: Run the upgraded multinode test (Neraium + AUX-TIME)

```python
!python run_upgraded_multinode_test.py
```

---

## Cell 5: Load and show JSON results (optional)

```python
import json
from pathlib import Path

path = Path("upgraded_multinode_test_results.json")
if path.exists():
    data = json.loads(path.read_text(encoding="utf-8"))
    print("AUX-TIME configured:", data.get("aux_time_api_configured"))
    print("AUX-TIME URL:", data.get("aux_time_time_url"))
    print("AUX-TIME used for disturbed_time:", data.get("aux_time_used_for_disturbed_time"))
    print("\nNodes:", data.get("nodes"))
    print("Conditions:", data.get("conditions"))
else:
    print("Results file not found. Run Cell 4 first.")
```

---

## Cell 6: Download results to your machine (optional)

```python
from google.colab import files

path = "upgraded_multinode_test_results.json"
if Path(path).exists():
    files.download(path)
else:
    print("Run the test (Cell 4) first.")
```

---

## Alternative: Use Colab Secrets for the API key (recommended)

Instead of pasting the key in Cell 2, use the key icon in the left sidebar to add a secret named `AUX_TIME_API_KEY`, then run:

```python
import os
from google.colab import userdata

os.environ["AUX_TIME_API_KEY"] = userdata.get("AUX_TIME_API_KEY")
os.environ["AUX_TIME_TIME_URL"] = os.environ.get("AUX_TIME_TIME_URL", "https://api-v2.aux-time.com/time")
```

---

## All-in-one cell (clone + install + set key + run)

If you prefer one cell (replace `YOUR_AUX_TIME_API_KEY_HERE` with your key or use Secrets):

```python
import os
import subprocess
import sys

# 1) Clone (change URL if needed)
subprocess.run(["git", "clone", "https://github.com/YOUR_USER/neraium-core-1.git"], check=True)
os.chdir("neraium-core-1")

# 2) AUX-TIME (set key; or use userdata.get("AUX_TIME_API_KEY") if using Colab Secrets)
os.environ["AUX_TIME_API_KEY"] = os.environ.get("AUX_TIME_API_KEY", "YOUR_AUX_TIME_API_KEY_HERE")
os.environ["AUX_TIME_TIME_URL"] = os.environ.get("AUX_TIME_TIME_URL", "https://api-v2.aux-time.com/time")

# 3) Install
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", "."], check=True)

# 4) Run test
subprocess.run([sys.executable, "run_upgraded_multinode_test.py"], check=True)
```
