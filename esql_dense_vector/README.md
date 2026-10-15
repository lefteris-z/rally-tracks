## esql_dense_vector

Measures the ES|QL `DENSE_VECTOR` command embedding msmarco-v2 passages, read either from an index or from an S3 dataset:

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
     `EVAL is_null = CASE(emb IS NULL, 1, 0) | STATS rows = COUNT(*), nulls = SUM(is_null)`.
   - It fails the sample, and with `--on-error=abort` the race, when any embedding is null or fewer
     than `n` rows came back.
   - The counts are logged to `rally.log` (`DENSE_VECTOR null check: rows=[…] nulls=[…]`) and stored
     in the sample's metadata (`rows`, `nulls`).

### Challenge `embed-profiled`

The same as `embed`, with an `embed-<n>-profile` task after each `embed-<n>` (and before its `null-check-<n>`). It runs the same query with the ES|QL
profile on (`esql-profile`), so each sample records the time of every driver and operator. Profiled runs are kept
apart from `embed-<n>` so the profile's own cost does not reach the measured runs.

### Parameters

* `source` (default: `index`): `index` or `s3`
* `inference_id` (default: `esql-dv-benc-e5`): the `text_embedding` inference endpoint the queries use
* `create_e5_endpoint` (default: `true`): create `inference_id` as an e5 endpoint; set `false` to use an existing one
* `e5_model_id` (default: `.multilingual-e5-small_linux-x86_64`), `e5_num_threads` (default: 1),
  `e5_min_allocations` (default: 0), `e5_max_allocations` (default: 32)
* `create_endpoint_timeout` (default: `10m`): how long creating the endpoint may wait for the model
* `null_check` (default: `true`): run `null-check-gate` before the first size and `null-check-<n>` after each size
* `null_check_gate_rows` (default: 100): rows checked by `null-check-gate`
* `sizes` (default: `[1, 10, 100, 1000, 10000]`): row counts, one task each
* `warmup_iterations` (default: 2), `iterations` (default: 10)
* `profile_warmup_iterations` (default: `warmup_iterations`), `profile_iterations` (default: 10): for `embed-<n>-profile`
* `request_timeout` (default: 600): seconds per `embed-<n>` request
* `index_name` (default: `msmarco_local`), `bulk_size` (default: 1000), `corpus_base_url` (default: the bucket's `ndjson` prefix)
* `dataset` (default: `msmarco_s3`), `data_source` (default: `msmarco_src`), `s3_region` (default: `eu-central-1`)
* `dataset_resource` (default: the Parquet file): the `s3://` URI the dataset reads
* `wait_for_endpoint_retries` (default: 10), `wait_for_endpoint_retry_wait` (default: 30): retries of
  `wait-for-inference-endpoint`, and the seconds between them
