## esql_dense_vector

Measures the ES|QL `DENSE_VECTOR` command embedding msmarco-v2 passages, or synthetic images with `data: images`, read
either from an index or from an S3 dataset:

```
FROM <source> | LIMIT n | DENSE_VECTOR emb = text WITH {"inference_id": "<inference_id>"} | KEEP id, emb
```

Each `n` in `sizes` is its own task, run by one client: `warmup_iterations` runs, then `iterations` measured runs. The
`esql` operation records the server-side `took` of each run (`detailed-results`).

### Data

10,000 msmarco-v2 passages (`id`, `text`): the first passages of the first shard of the `msmarco-v2-vector` corpus,
cut by `_tools/prepare_msmarco_slice.py`. The same rows sit in `s3://lz-esql-dense-vector-eu-central-1/msmarco/10k/` as
NDJSON (`ndjson/msmarco_10k.ndjson`) and Parquet (`parquet/msmarco_10k.parquet`).

* `source: index` bulk-loads the NDJSON into `index_name`. Rally downloads it from S3, which needs `boto3` and read
  access to the bucket on the load driver.
* `source: s3` registers an S3 data source and a dataset on `dataset_resource`, read by the Elasticsearch nodes with
  their instance role (`auth: managed_identity`). The track turns on `esql.external.managed_identity.enabled` first.

### Challenge `embed` (default)

1. Raise `esql.command.dense_vector.limit` to the largest size.
2. Load the index, or register the data source and dataset.
3. Create the `inference_id` endpoint, unless `create_e5_endpoint` is `false`. It uses the `elasticsearch`
   service with e5, the same settings as the default `.multilingual-e5-small-elasticsearch` endpoint:
   - one thread;
   - adaptive allocations from 0 to 32;
   - sentence chunking.

   The model is `.multilingual-e5-small_linux-x86_64`, so the ML nodes must be Linux x86_64. Machine
   learning must be enabled; with Rally's default car, add the `x-pack-ml` mixin.
4. `wait-for-inference-endpoint`: call the endpoint through `_inference` until it returns a
   non-empty embedding. Every error, and every answer without an embedding, is retried, so a model
   that is still deploying or scaling up is waited for, not measured.
5. `null-check-gate`, unless `null_check` is `false`: once the endpoint is online, one null check of
   `null_check_gate_rows` rows, before any warm-up or measured run.
   - It proves that the source, `DENSE_VECTOR` and the endpoint return real vectors for every row.
   - With `--on-error=abort` a failure stops the race here.
6. `embed-<n>` for each size, then `null-check-<n>` unless `null_check` is `false`.
   - `null-check-<n>` runs the same `LIMIT n | DENSE_VECTOR` once, ending in
     `EVAL is_null = CASE(emb IS NULL, 1, 0) | STATS row_count = COUNT(*), null_count = SUM(is_null)`.
   - It fails the sample, and with `--on-error=abort` the race, when any embedding is null or fewer
     than `n` rows came back.
   - The counts are logged to `rally.log` (`DENSE_VECTOR null check: rows=[…] nulls=[…]`) and stored
     in the sample's metadata (`rows`, `nulls`).

### Challenge `embed-profiled`

The same as `embed`, with an `embed-<n>-profile` task after each `embed-<n>` (and before its `null-check-<n>`). It runs the same query with the ES|QL
profile on (`esql-profile`), so each sample records the time of every driver and operator. Profiled runs are kept
apart from `embed-<n>` so the profile's own cost does not reach the measured runs.

### Challenges `cosine-profiled` and `knn-profiled`

These have the same setup, null checks and profiled runs as `embed-profiled`. The measured query
compares each row's embedding with a reference vector: the embedding of document `reference_doc_id`
(default `00_0`).

```
cosine: FROM <source> | LIMIT n | DENSE_VECTOR dv = text WITH {…} | EVAL sim = v_cosine(<reference>, dv) | KEEP id, sim
knn:    FROM <source> | LIMIT n | DENSE_VECTOR vec = text WITH {…} | WHERE knn(vec, <reference>, {"similarity": <knn_similarity>}) | KEEP id
```

After `null-check-gate` and before any warm-up:

1. `compute-reference-vector` embeds the reference document with DENSE_VECTOR itself and stores the
   vector on the load driver. The queries then use exactly what DENSE_VECTOR produces for that
   document with this endpoint.
2. `correctness-gate` runs the query on the reference document alone:
   - cosine: its `sim` must be at least `reference_min_similarity`;
   - knn: the query must return exactly that document.

   With `--on-error=abort` a failure stops the race here.

Each measured and profiled task puts the stored vector into its query (`__REFERENCE_VECTOR__`) at
run time.

The knn queries run `knn` on a vector computed at query time. That needs the
`"pragma": {"knn_runtime_field": true}` request option, which only snapshot builds of Elasticsearch
accept. After each size, `knn-rows-<n>` records how many rows the filter kept (`returned_rows`).
Its query puts `LIMIT n` right after `knn`: `knn` needs a LIMIT after it to set its k, and the
LIMIT that ES|QL adds at the end of a query does not reach `knn` past a `STATS`.

### Challenge `image-profiled` (with `data: images`)

Embeds images through a multimodal endpoint, by default the EIS endpoint `.jina-clip-v2`:

```
FROM <source> | LIMIT n | DENSE_VECTOR v = img WITH {"inference_id": ".jina-clip-v2", "type": "image"} | EVAL m = v_magnitude(v) | KEEP id, m
```

The data is 1,000 synthetic 64x64 PNG images (`id`, `img` as a base64 data URI), made by
`_tools/generate_synthetic_images.py`, in `s3://lz-esql-dense-vector-eu-central-1/images/1k/` as NDJSON
(`ndjson/images_1k.ndjson`) and Parquet (`parquet/images_1k.parquet`). The index is `img_local` (`index_images.json`)
and the dataset `img_s3`.

The elasticsearch service on ML nodes embeds text only, so on a self-managed cluster the images go to EIS through
Cloud Connected Mode:

1. `configure-ccm` reads an Elastic Cloud API key from `ccm_api_key_file` on the load driver and sends it to
   `PUT _inference/_ccm`. The key is a file, not a track parameter, because Rally stores track parameters with the race
   and in its log. The file can be written after the race starts; the task waits for it for `ccm_api_key_file_wait`
   seconds.
2. `wait-for-inference-endpoint` runs a one-row DENSE_VECTOR query until it returns a vector, as the EIS endpoints
   appear once EIS has authorized the cluster.

Then the same checks as `cosine-profiled`: `null-check-gate`, `compute-reference-vector` and `correctness-gate`
(document `img_0` against its own vector). After that, for each size: the measured runs, the profiled runs and
`null-check-<n>`. `size_overrides` sets other run counts for some sizes, for example fewer runs for 1,000 images.

### Parameters

* `source` (default: `index`): `index` or `s3`
* `inference_id` (default: `esql-dv-benc-e5`): the `text_embedding` inference endpoint the queries use
* `create_e5_endpoint` (default: `true`): create `inference_id` as an e5 endpoint; set `false` to use an existing one
* `e5_model_id` (default: `.multilingual-e5-small_linux-x86_64`), `e5_num_threads` (default: 1),
  `e5_min_allocations` (default: 0), `e5_max_allocations` (default: 32)
* `create_endpoint_timeout` (default: `5m`), `create_endpoint_request_timeout` (default: 360): how
  long one attempt at creating the endpoint may wait for the model, on the server and in the client
* `create_endpoint_retries` (default: 10), `create_endpoint_retry_wait` (default: 30): retries of
  `create-inference-endpoint`, and the seconds between them. Before each retry the endpoint and the
  model are deleted, so a failed model download starts over
* `data` (default: `msmarco`): `images` reads the image corpus and changes the defaults of `sizes` (`[1, 10, 100, 1000]`),
  `inference_id` (`.jina-clip-v2`), `index_name` (`img_local`), `dataset` (`img_s3`), `data_source` (`img_src`),
  `dataset_resource` (the image Parquet file), `reference_doc_id` (`img_0`), `create_e5_endpoint` (`false`) and
  `wait_for_endpoint_retries` (40)
* `ccm_api_key_file` (default: `~/.esql-dense-vector-ccm-api-key`), `ccm_api_key_file_wait` (default: 1800): the file
  holding the Elastic Cloud API key for `configure-ccm`, and how many seconds to wait for it (`data: images`)
* `size_overrides` (default: `{}`): per-size run counts, keyed by size, for example
  `{"1000": {"warmup_iterations": 2, "iterations": 10, "profile_warmup_iterations": 2, "profile_iterations": 3}}`
* `null_check` (default: `true`): run `null-check-gate` before the first size and `null-check-<n>` after each size
* `null_check_gate_rows` (default: 100): rows checked by `null-check-gate`
* `reference_doc_id` (default: `00_0`): the document whose embedding is the reference vector (`cosine-profiled`, `knn-profiled`)
* `reference_min_similarity` (default: 0.9999): the least `sim` the cosine correctness gate accepts
* `knn_similarity` (default: 0.6): the `similarity` threshold of the knn queries
* `sizes` (default: `[1, 10, 100, 1000, 10000]`): row counts, one task each
* `warmup_iterations` (default: 2), `iterations` (default: 10)
* `profile_warmup_iterations` (default: `warmup_iterations`), `profile_iterations` (default: 10): for `embed-<n>-profile`
* `request_timeout` (default: 600): seconds per `embed-<n>` request
* `index_name` (default: `msmarco_local`), `bulk_size` (default: 1000), `corpus_base_url` (default: the bucket's `ndjson` prefix)
* `dataset` (default: `msmarco_s3`), `data_source` (default: `msmarco_src`), `s3_region` (default: `eu-central-1`)
* `dataset_resource` (default: the Parquet file): the `s3://` URI the dataset reads
* `wait_for_endpoint_retries` (default: 10), `wait_for_endpoint_retry_wait` (default: 30): retries of
  `wait-for-inference-endpoint`, and the seconds between them
