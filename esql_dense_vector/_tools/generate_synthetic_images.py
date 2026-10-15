#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pillow==12.2.0",
#   "pyarrow==25.0.1",
# ]
# ///
"""
Generates the synthetic image corpus that the image DENSE_VECTOR benchmarks read, as NDJSON and as
Parquet.

Each document is `{"id": "img_<i>", "img": "data:image/png;base64,..."}`: a 64x64 PNG with a
gradient background and one shape, both derived from `i`, so the corpus is the same on every run.
The pattern repeats every 768 images, so a 1000-image corpus holds 232 repeats. The images carry no
meaning; they are for timing image embedding, not for judging its quality.

These are the images of the EXP-02 round 8 and round 12 benchmarks: with pillow 12.2.0 the NDJSON
is byte for byte the file those rounds used. pillow and pyarrow are pinned so that the PNG and
Parquet bytes, and so their checksums, are reproducible. Both files are read back and compared before the checksums are printed. Uploading is
left to you.

Usage:
    uv run esql_dense_vector/_tools/generate_synthetic_images.py --docs 1000 --out ~/image-corpus

    aws s3 cp ~/image-corpus/images_1k.ndjson s3://<bucket>/images/1k/ndjson/images_1k.ndjson
    aws s3 cp ~/image-corpus/images_1k.parquet s3://<bucket>/images/1k/parquet/images_1k.parquet
"""

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageDraw

SIZE = 64


def size_label(docs):
    return f"{docs // 1000}k" if docs % 1000 == 0 else str(docs)


def make_image(i):
    img = Image.new("RGB", (SIZE, SIZE))
    px = img.load()
    r0, g0, b0 = (i * 37) % 256, (i * 53) % 256, (i * 97) % 256
    for x in range(SIZE):
        for y in range(SIZE):
            px[x, y] = ((r0 + x * 2) % 256, (g0 + y * 2) % 256, (b0 + (x + y)) % 256)
    draw = ImageDraw.Draw(img)
    cx, cy = (i * 7) % SIZE, (i * 11) % SIZE
    color = ((i * 13) % 256, (i * 17) % 256, (i * 19) % 256)
    if i % 3 == 0:
        draw.ellipse([cx - 12, cy - 12, cx + 12, cy + 12], fill=color)
    elif i % 3 == 1:
        draw.rectangle([cx - 10, cy - 10, cx + 10, cy + 10], fill=color)
    else:
        draw.polygon([(cx, cy - 12), (cx - 12, cy + 10), (cx + 12, cy + 10)], fill=color)
    return img


def data_uri(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def write_ndjson(path, ids, images):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for doc_id, uri in zip(ids, images):
            f.write(json.dumps({"id": doc_id, "img": uri}) + "\n")


def write_parquet(path, ids, images):
    table = pa.table({"id": pa.array(ids, pa.string()), "img": pa.array(images, pa.string())})
    pq.write_table(table, path, compression="zstd")


def verify(ndjson_path, parquet_path, ids, images):
    with open(ndjson_path, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    if [d["id"] for d in docs] != ids or [d["img"] for d in docs] != images:
        raise SystemExit(f"{ndjson_path} does not read back as written")
    table = pq.read_table(parquet_path)
    if table.column("id").to_pylist() != ids or table.column("img").to_pylist() != images:
        raise SystemExit(f"{parquet_path} does not read back as written")
    if len(set(ids)) != len(ids):
        raise SystemExit("the corpus contains duplicate ids")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=int, default=1000, help="number of images (default: 1000)")
    parser.add_argument("--out", type=Path, default=Path("."), help="directory for the two files (default: current)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"images_{size_label(args.docs)}"
    ndjson_path = args.out / f"{name}.ndjson"
    parquet_path = args.out / f"{name}.parquet"

    ids = [f"img_{i}" for i in range(args.docs)]
    images = [data_uri(make_image(i)) for i in range(args.docs)]
    write_ndjson(ndjson_path, ids, images)
    write_parquet(parquet_path, ids, images)
    verify(ndjson_path, parquet_path, ids, images)

    for path in (ndjson_path, parquet_path):
        print(f"{sha256(path)}  {path.stat().st_size:>12}  {path}")


if __name__ == "__main__":
    main()
