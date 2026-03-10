Training workspace

- Install virtualenv and create one, then install requirements:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

- To run training for RNN/LSTM/Attention with automatic CUDA detection:

```powershell
python run_all_training.py
```
