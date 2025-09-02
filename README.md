# KW-AIRO-InvoiceProcessing-API

```bash
conda create -n invoice python=3.11.7 pip
conda activate invoice
```

## Requirements

```bash
pip install -r requirements.txt
```

### Development Mode
```bash
uvicorn app.main:app --host localhost --port 6000
```