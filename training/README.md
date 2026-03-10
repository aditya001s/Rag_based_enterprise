# Training pipeline

Place your dataset as a CSV with two columns: `text` and `label` (integer labels).

Install dependencies:

```bash
pip install -r requirements.txt
```

Run training (example):

```bash
python train.py --data-path path/to/data.csv --model lstm --epochs 5 --batch-size 32 --save-dir ./models
```

Models supported: `rnn`, `lstm`, `attn` (self-attention using PyTorch TransformerEncoder).

If you have an unlabeled PDF (like `C:\project\NIST.SP.800-53r5.pdf`) and want embeddings / retrieval:

1. Extract chunks from the PDF:

```bash
python extract_pdf.py --pdf "C:\\project\\NIST.SP.800-53r5.pdf" --out extracted_chunks.csv
```

2. Compute embeddings (default model `all-MiniLM-L6-v2`):

```bash
python embed.py --csv extracted_chunks.csv --out-emb embeddings.npy --out-meta metas.json
```

3. Use `embeddings.npy` and `metas.json` for retrieval or downstream tasks. For supervised training, create a CSV with `text,label` and run `train.py`.

