import os
import json
import base64
import io
import logging
from datetime import date
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("receipt-api")

app = FastAPI(title="Receipt Finance API", version="1.0.0")

client = AsyncOpenAI(
    base_url=os.getenv("MODEL_BASE_URL", "http://127.0.0.1:1234/v1"),
    api_key=os.getenv("MODEL_API_KEY", "lm-studio"),
    timeout=120.0,
)
MODEL_NAME = os.getenv("MODEL_NAME", "")
MAX_BYTES = int(os.getenv("MAX_FILE_MB", "8")) * 1024 * 1024


class ReceiptItem(BaseModel):
    name: str
    quantity: Optional[float] = None
    unit_price: Optional[float] = None
    subtotal: Optional[float] = None


class ReceiptData(BaseModel):
    merchant: Optional[str] = None
    transaction_date: Optional[date] = None
    currency: str = "IDR"
    items: list[ReceiptItem] = Field(default_factory=list)
    subtotal: Optional[float] = None
    tax: Optional[float] = None
    discount: Optional[float] = None
    total: Optional[float] = None
    payment_method: Optional[str] = None


class ReceiptResponse(BaseModel):
    success: bool
    data: ReceiptData
    metadata: dict


# [TAMBAHAN] validasi aritmetika & kewajaran oleh backend
def validate_receipt(d: ReceiptData) -> list[str]:
    warnings = []
    tol = 1.0
    money = [("subtotal", d.subtotal), ("tax", d.tax), ("discount", d.discount), ("total", d.total)]
    money += [(f"items[{i}].unit_price", it.unit_price) for i, it in enumerate(d.items)]
    money += [(f"items[{i}].subtotal", it.subtotal) for i, it in enumerate(d.items)]
    if d.currency == "IDR":
        for name, v in money:
            if v is not None and v != 0 and (v != int(v) or abs(v) < 100):
                warnings.append(f"{name}={v} tidak wajar untuk IDR (kemungkinan salah baca pemisah ribuan)")
    for i, it in enumerate(d.items):
        if None not in (it.quantity, it.unit_price, it.subtotal):
            if abs(it.quantity * it.unit_price - it.subtotal) > tol:
                warnings.append(f"items[{i}]: quantity x unit_price != subtotal")
    item_subs = [it.subtotal for it in d.items]
    if d.subtotal is not None and item_subs and None not in item_subs:
        if abs(sum(item_subs) - d.subtotal) > tol:
            warnings.append(f"jumlah subtotal item ({sum(item_subs)}) != subtotal ({d.subtotal})")
    if d.subtotal is not None and d.total is not None:
        expected = d.subtotal - (d.discount or 0) + (d.tax or 0)
        if abs(expected - d.total) > tol:
            warnings.append(f"subtotal - discount + tax ({expected}) != total tercetak ({d.total})")
    if d.total is None:
        warnings.append("total tidak terbaca")
    return warnings


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/api/receipts/extract", response_model=ReceiptResponse)
async def extract_receipt(file: UploadFile = File(...)):
    if not MODEL_NAME:
        raise HTTPException(500, "MODEL_NAME belum dikonfigurasi")

    if file.content_type not in {"image/jpeg", "image/png", "image/webp"}:
        raise HTTPException(415, "Format gambar tidak didukung")

    raw = await file.read(MAX_BYTES + 1)
    if not raw or len(raw) > MAX_BYTES:
        raise HTTPException(413, "Gambar kosong atau melebihi batas ukuran")

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw))
        if image.format not in {"JPEG", "PNG", "WEBP"}:
            raise HTTPException(415, "Format gambar tidak didukung")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "Berkas bukan gambar yang valid")

    mime = {"JPEG": "image/jpeg", "PNG": "image/png", "WEBP": "image/webp"}[image.format]
    encoded = base64.b64encode(raw).decode("ascii")
    image_url = f"data:{mime};base64,{encoded}"

    prompt = """
Ekstrak data nota keuangan dari gambar.
Kembalikan satu objek JSON dengan kunci:
merchant, transaction_date, currency, items, subtotal,
tax, discount, total, payment_method.
Setiap item memiliki name, quantity, unit_price, subtotal.
Gunakan angka untuk nominal, tanggal YYYY-MM-DD,
dan null untuk data yang tidak diketahui.
Jangan mengarang informasi. Pertahankan total yang tercetak.
Aturan tambahan:
- Nominal ditulis sebagai bilangan bulat TANPA titik/koma pemisah ribuan.
  Contoh: "Rp 56.000" ditulis 56000, bukan 56.000.
- currency gunakan kode ISO, misalnya "IDR" untuk Rupiah.
- total adalah angka pada baris TOTAL, bukan uang tunai yang dibayar atau kembalian.
"""

    try:
        result = await client.chat.completions.create(
            model=MODEL_NAME,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
        )
        content = result.choices[0].message.content
        logger.info("Output mentah model: %s", content)
        if not content:
            raise ValueError("Output model kosong")

        start = content.find("{")
        end = content.rfind("}")
        if start < 0 or end < start:
            raise ValueError("JSON tidak ditemukan")

        payload = json.loads(content[start:end + 1])
        if str(payload.get("currency", "")).strip().lower() in {"rp", "rp.", "rupiah"}:
            payload["currency"] = "IDR"  # [TAMBAHAN] normalisasi mata uang
        data = ReceiptData.model_validate(payload)
        warnings = validate_receipt(data)  # [TAMBAHAN]
        if warnings:
            logger.warning("Perlu cek manual: %s", warnings)

        return ReceiptResponse(
            success=True,
            data=data,
            metadata={
                "model": MODEL_NAME,
                "processing_status": "needs_review" if warnings else "success",  # [TAMBAHAN]
                "needs_review": bool(warnings),  # [TAMBAHAN]
                "warnings": warnings,  # [TAMBAHAN]
            },
        )
    except HTTPException:
        raise
    except Exception:
        logger.exception("Ekstraksi gagal")
        raise HTTPException(502, "Inferensi gagal atau output model tidak valid")
    finally:
        await file.close()