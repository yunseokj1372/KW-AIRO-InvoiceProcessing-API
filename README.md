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

### Development Mode
```bash
tmux new -s invoice

uvicorn app.main:app --host localhost --port 8001
```