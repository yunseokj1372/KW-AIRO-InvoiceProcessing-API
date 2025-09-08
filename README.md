# KW-AIRO-InvoiceProcessing-API

```bash
printf 'y\y\n' conda create -n invoice python=3.11.7 pip
conda activate invoice
```

## Requirements

```bash
pip install -r requirements.txt

printf "y" conda install -c conda-forge pymupdf==1.24.2
```

### Production Mode (Dev Mode localhost)
```bash
tmux new -s invoice

uvicorn app.main:app --host 0.0.0.0 --port 8001
```