#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "pyarrow==25.0.1",
#   "zstandard",
# ]
# ///
"""
Cuts the msmarco-v2 slice that the federated DENSE_VECTOR benchmarks read from S3, as NDJSON and
as Parquet.

The slice is the first N passages of the first shard of the `msmarco-v2-vector` corpus, keeping
only `id` (the corpus `docid`) and `text`; the Cohere embeddings are dropped. A passage with no
`text` falls back to its `title`, and one with neither, or with no id, is skipped. The ids are the
original MS MARCO passage ids, so the slice joins with other msmarco-v2 corpora such as
`msmarco-jina-10m-vector`.

The source is streamed and the download stops after N passages, so a 10k slice reads a few MB of
the shard rather than all of it.

pyarrow is pinned so that the Parquet bytes, and so their checksums, are reproducible. Both files
are read back and compared before the checksums are printed. Uploading is left to you.

Usage:
    uv run esql_dense_vector/_tools/prepare_msmarco_slice.py --docs 10000 --out ~/msmarco-slices

    aws s3 cp ~/msmarco-slices/msmarco_10k.ndjson s3://<bucket>/msmarco/10k/ndjson/msmarco_10k.ndjson
    aws s3 cp ~/msmarco-slices/msmarco_10k.parquet s3://<bucket>/msmarco/10k/parquet/msmarco_10k.parquet
"""

import argparse
import hashlib
import io
import json
import urllib.request
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import zstandard

DEFAULT_SOURCE = "https://rally-tracks.elastic.co/cohere-msmarco-v2-embed-english-v3/cohere-documents-01.json.zst"


def size_label(docs):
    return f"{docs // 1000}k" if docs % 1000 == 0 else str(docs)


def read_passages(source, docs):
    ids, texts = [], []
    with urllib.request.urlopen(source) as response:
        with zstandard.ZstdDecompressor().stream_reader(response) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    doc = json.loads(line)
                except json.JSONDecodeError:
                    continue
                doc_id = doc.get("docid") or doc.get("id") or doc.get("_id")
                text = doc.get("text") or doc.get("title")
                if doc_id is None or not text:
                    continue
                ids.append(doc_id)
                texts.append(text)
                if len(ids) >= docs:
                    break
    if len(ids) < docs:
        raise SystemExit(f"source ended after {len(ids)} passages, {docs} requested")
    return ids, texts


def write_ndjson(path, ids, texts):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for doc_id, text in zip(ids, texts):
            f.write(json.dumps({"id": doc_id, "text": text}) + "\n")


def write_parquet(path, ids, texts):
    table = pa.table({"id": pa.array(ids, pa.string()), "text": pa.array(texts, pa.string())})
    pq.write_table(table, path, compression="zstd")


def verify(ndjson_path, parquet_path, ids, texts):
    with open(ndjson_path, encoding="utf-8") as f:
        docs = [json.loads(line) for line in f]
    if [d["id"] for d in docs] != ids or [d["text"] for d in docs] != texts:
        raise SystemExit(f"{ndjson_path} does not read back as written")
    table = pq.read_table(parquet_path)
    if table.column("id").to_pylist() != ids or table.column("text").to_pylist() != texts:
        raise SystemExit(f"{parquet_path} does not read back as written")
    if len(set(ids)) != len(ids):
        raise SystemExit("the slice contains duplicate ids")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=int, default=10000, help="number of passages in the slice (default: 10000)")
    parser.add_argument("--out", type=Path, default=Path("."), help="directory for the two files (default: current)")
    parser.add_argument("--source", default=DEFAULT_SOURCE, help="zstd-compressed NDJSON corpus shard to cut from")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    name = f"msmarco_{size_label(args.docs)}"
    ndjson_path = args.out / f"{name}.ndjson"
    parquet_path = args.out / f"{name}.parquet"

    ids, texts = read_passages(args.source, args.docs)
    write_ndjson(ndjson_path, ids, texts)
    write_parquet(parquet_path, ids, texts)
    verify(ndjson_path, parquet_path, ids, texts)

    for path in (ndjson_path, parquet_path):
        print(f"{sha256(path)}  {path.stat().st_size:>12}  {path}")


if __name__ == "__main__":
    main()
