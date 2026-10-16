# Receipt MLOps — Ekstraksi Nota Keuangan ke JSON

Prototipe backend FastAPI yang menerima gambar nota, memprosesnya dengan Qwen2.5-VL-7B (via LM Studio), dan mengembalikan data keuangan terstruktur dalam JSON.

## Arsitektur
Client (upload JPG/PNG/WebP) -> FastAPI -> Qwen2.5-VL via LM Studio (OpenAI-compatible API) -> JSON

## Menjalankan Lokal (WSL)
```bash
cp .env.example .env
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```
Swagger UI: http://localhost:8000/docs

Catatan WSL: aktifkan `networkingMode=mirrored` di `%USERPROFILE%\.wslconfig` agar `127.0.0.1:1234` dapat menjangkau LM Studio di Windows.

## Docker
```bash
cp .env.example .env.docker
sed -i 's#^MODEL_BASE_URL=.*#MODEL_BASE_URL=http://host.docker.internal:1234/v1#' .env.docker
docker build -t receipt-mlops .
docker run --rm -p 8000:8000 --env-file .env.docker receipt-mlops
```

## Endpoint
| Method | Path | Keterangan |
|---|---|---|
| GET | /health | Cek status server |
| POST | /api/receipts/extract | Upload gambar nota (field `file`) |

## Pengembangan dari Modul
- Prompt diperjelas: nominal tanpa pemisah ribuan, currency `IDR`, TOTAL bukan uang tunai.
- Normalisasi currency `Rp` -> `IDR`.
- Validasi backend `validate_receipt()`: kewajaran nominal IDR, qty x harga = subtotal item, jumlah item = subtotal, subtotal - diskon + pajak = total.
- Metadata `needs_review` dan `warnings` untuk hasil yang perlu dicek manual.
- Logging output mentah model dan error.

## Hasil Uji (samples/nota1_jelas.png)
| Field | Sebelum perbaikan prompt | Sesudah |
|---|---|---|
| currency | Rp | IDR |
| total | 56.000 (terbaca 56.0) | 56000 |
| processing_status | success (data salah) | success (data benar) |

Waktu inferensi: sekitar 10,7 detik per nota (Qwen2.5-VL-7B Q4_K_M).
