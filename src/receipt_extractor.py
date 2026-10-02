import json
import os
import re
from pathlib import Path

import lmstudio as lms
from pydantic import BaseModel, ValidationError

ROOT = Path(__file__).resolve().parents[1]
IMAGE_PATH = ROOT / "data" / "raw" / "nota-sample.png"
OUTPUT_PATH = ROOT / "reports" / "receipt.json"
MODEL_ID = os.environ["LM_STUDIO_MODEL"]

PROMPT = """Baca nota pada gambar. Keluarkan HANYA JSON valid dengan format:
{
  "merchant": "nama toko",
  "tanggal": "DD/MM/YYYY",
  "items": [{"nama": "nama item", "qty": 1, "harga_satuan": 0}],
  "subtotal": 0,
  "pajak": 0,
  "total": 0
}
Aturan:
- Semua angka berupa integer tanpa titik/koma pemisah ribuan (contoh: 95000).
- "harga_satuan" adalah harga untuk 1 unit item (bukan qty x harga).
- Jika pajak tidak terlihat, isi 0.
- Jangan mengarang data yang tidak ada di nota.
"""


class Item(BaseModel):
    nama: str
    qty: int
    harga_satuan: int


class Receipt(BaseModel):
    merchant: str
    tanggal: str
    items: list[Item]
    subtotal: int
    pajak: int
    total: int


def clean_json(text: str) -> str:
    """Buang pembungkus ```json ... ``` jika model menambahkannya."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"Output model tidak berisi JSON:\n{text}")
    return match.group(0)


def check_numbers(receipt: Receipt) -> list[str]:
    """Cek konsistensi angka pada nota."""
    warnings = []
    items_sum = sum(item.qty * item.harga_satuan for item in receipt.items)
    if items_sum != receipt.subtotal:
        warnings.append(f"Jumlah item ({items_sum}) != subtotal ({receipt.subtotal})")
    if receipt.subtotal + receipt.pajak != receipt.total:
        warnings.append(
            f"Subtotal + pajak ({receipt.subtotal + receipt.pajak}) != total ({receipt.total})"
        )
    return warnings


def main():
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"Gambar tidak ditemukan: {IMAGE_PATH}")

    image = lms.prepare_image(str(IMAGE_PATH))
    model = lms.llm(MODEL_ID)

    chat = lms.Chat()
    chat.add_user_message(PROMPT, images=[image])
    prediction = model.respond(chat)

    try:
        data = json.loads(clean_json(prediction.content))
        receipt = Receipt(**data)
    except (ValueError, ValidationError) as error:
        print("Output mentah model:\n", prediction.content)
        raise SystemExit(f"Gagal validasi JSON: {error}")

    warnings = check_numbers(receipt)
    result = {
        "model": MODEL_ID,
        "source_image": str(IMAGE_PATH.relative_to(ROOT)),
        "receipt": receipt.model_dump(),
        "validation": {"passed": not warnings, "warnings": warnings},
    }

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print(f"Hasil tersimpan di: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
