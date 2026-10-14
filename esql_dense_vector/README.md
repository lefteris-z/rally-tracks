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

### Challenge `embed`

1. Raise `esql.command.dense_vector.limit` to the largest size.
2. Load the index, or register the data source and dataset.
3. Call the inference endpoint until it answers, so a model deployment is not measured.
4. `embed-<n>` for each size.

### Parameters

* `source` (default: `index`): `index` or `s3`
* `inference_id` (default: `.multilingual-e5-small-elasticsearch`): a `text_embedding` inference endpoint
* `sizes` (default: `[1, 10, 100, 1000, 10000]`): row counts, one task each
* `warmup_iterations` (default: 2), `iterations` (default: 10)
* `request_timeout` (default: 600): seconds per `embed-<n>` request
* `index_name` (default: `msmarco_local`), `bulk_size` (default: 1000), `corpus_base_url` (default: the bucket's `ndjson` prefix)
* `dataset` (default: `msmarco_s3`), `data_source` (default: `msmarco_src`), `s3_region` (default: `eu-central-1`)
* `dataset_resource` (default: the Parquet file): the `s3://` URI the dataset reads
* `warm_up_retries` (default: 60): attempts, 10 seconds apart, for the endpoint warm-up
