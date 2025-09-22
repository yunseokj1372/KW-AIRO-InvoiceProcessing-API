# KW-AIRO-InvoiceProcessing-API

```bash
printf 'y\y\n' conda create -n invoice python=3.11.7 pip
conda activate invoice
```

## Requirements

```bash
pip install -r requirements.txt

pip install --upgrade pymupdf==1.26.4
```

### Production Mode (Dev Mode localhost)
```bash
tmux new -s invoice

tmux attach -t invoice

uvicorn app.main:app --host 0.0.0.0 --port 8001
```

## API Authentication

The API is secured with API key authentication. Add your API key to the `.env` file:

```env
API_KEY=your-secure-api-key-here
```

All requests must include the API key in the header:
```
x-api-key: your-secure-api-key-here
```

## API Endpoints

### Health Check
```bash
GET http://192.168.9.172:8001/
Headers:
  x-api-key: your-api-key
```

### Process Invoices
```bash
POST http://192.168.9.172:8001/output/single
Headers:
  Content-Type: application/json
  x-api-key: your-api-key
Body:
{
    "filekey": "your-file-key",
    "filetype": "ASIS"
}
```

### Supported File Types
- `ASIS`
- `DR` 
- `LGBGOOD`
- `LGPARTS`
- `SGBGOOD`
- `SGPARTS`
- `SRA`