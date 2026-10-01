# Laya Studio for Data

[![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/echoghimire/agentic-notebooks/blob/main/notebooks/laya-studio/laya-studio.ipynb)

By Er. Gunjan Ghimire. A web studio on port 7860 for labelling bills, receipts and fintech text, fine-tuning [Laya](https://github.com/NandhaKishorM/laya) on Kaggle GPUs, and testing the result.

- Secrets: `LAYA_TUNNEL_TOKEN` (required), `LAYA_UI_PASSWORD`, `HF_TOKEN` (optional)
- Accelerator: GPU T4 x2
- Laya answers typed questions (choice / yes-no / score) about text. It does not read images: run OCR first.
- Regenerate: `python src/build_notebook.py`
