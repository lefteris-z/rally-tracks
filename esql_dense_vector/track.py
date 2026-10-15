import asyncio
import json
import os
import tempfile

from esrally.driver import runner

REFERENCE_VECTOR_PLACEHOLDER = "__REFERENCE_VECTOR__"


def _reference_vector_path(reference_name):
    return os.path.join(tempfile.gettempdir(), f"esql-dense-vector-reference-{reference_name}.json")


_reference_vector_literals = {}


def _reference_vector_literal(reference_name):
    """Returns the stored reference vector as an ES|QL list literal, read once per file version."""
    path = _reference_vector_path(reference_name)
    key = (path, os.path.getmtime(path))
    if key not in _reference_vector_literals:
        with open(path, encoding="utf-8") as f:
            _reference_vector_literals[key] = json.dumps(json.load(f))
    return _reference_vector_literals[key]


def _with_reference_vector(params):
    """A copy of params whose query has the stored reference vector in place of the placeholder."""
    params = dict(params)
    params["query"] = runner.mandatory(params, "query", "reference vector query").replace(
        REFERENCE_VECTOR_PLACEHOLDER, _reference_vector_literal(runner.mandatory(params, "reference-name", "reference vector query"))
    )
    if "body" in params:
        params["body"] = json.loads(json.dumps(params["body"]))
    return params


async def _run_esql(es, params, runner_instance):
    params, request_params, transport_params, headers = runner_instance._transport_request_params(params)
    es = es.options(**transport_params)
    body = dict(params.get("body", {}))
    body["query"] = runner.mandatory(params, "query", runner_instance)
    return await es.perform_request(method="POST", path="/_query", headers=headers or None, body=body, params=request_params)


class WaitForInferenceEndpoint(runner.Runner):
    """
    Calls a text_embedding inference endpoint until it returns a non-empty embedding, so that
    deploying or scaling up its model is not part of any measured task, and measuring starts only
    on an endpoint that works.

    Every error, and every answer without an embedding, is retried, up to ``retries`` times and
    ``retry-wait-period`` seconds apart: while a model deploys, an endpoint can answer with errors
    that Rally's own retry does not cover, as that one only retries connection errors and HTTP 408.
    """

    async def __call__(self, es, params):
        params, request_params, transport_params, headers = self._transport_request_params(params)
        es = es.options(**transport_params)
        inference_id = runner.mandatory(params, "inference-id", self)
        retries = params.get("retries", 10)
        wait_seconds = params.get("retry-wait-period", 30)
        for attempt in range(retries + 1):
            try:
                response = await es.perform_request(
                    method="POST",
                    path=f"/_inference/text_embedding/{inference_id}",
                    headers=headers or None,
                    body={"input": ["is this endpoint ready"]},
                    params=request_params,
                )
                embedding = response["text_embedding"][0]["embedding"]
                if not embedding:
                    raise ValueError("the endpoint answered with an empty embedding")
                return {"weight": 1, "unit": "ops", "success": True, "attempts": attempt + 1, "dimensions": len(embedding)}
            except Exception as e:
                if attempt == retries:
                    raise
                self.logger.info(
                    "Inference endpoint [%s] is not ready (attempt %d of %d): %s. Retrying in [%s] seconds.",
                    inference_id,
                    attempt + 1,
                    retries + 1,
                    e,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)

    def __repr__(self, *args, **kwargs):
        return "wait-for-inference-endpoint"


class CreateInferenceEndpoint(runner.Runner):
    """
    Creates an inference endpoint, and tries again when the creation fails, up to ``retries`` times
    and ``retry-wait-period`` seconds apart.

    For the ``elasticsearch`` service, creating the endpoint stores it and then downloads and deploys
    the model, so a failed model download leaves the endpoint and a partly imported model behind.
    Both are deleted before each new attempt. The client's own retry is off for the creation: it
    resends the request after a 502, and the second request then fails because the first one already
    stored the endpoint.
    """

    async def __call__(self, es, params):
        params, request_params, transport_params, headers = self._transport_request_params(params)
        task_type = runner.mandatory(params, "task-type", self)
        inference_id = runner.mandatory(params, "inference-id", self)
        body = runner.mandatory(params, "body", self)
        model_id = body.get("service_settings", {}).get("model_id")
        retries = params.get("retries", 10)
        wait_seconds = params.get("retry-wait-period", 30)
        endpoint_path = f"/_inference/{task_type}/{inference_id}"
        for attempt in range(retries + 1):
            try:
                await es.options(**{**transport_params, "max_retries": 0}).perform_request(
                    method="PUT", path=endpoint_path, headers=headers or None, body=body, params=request_params
                )
                return {"weight": 1, "unit": "ops", "success": True, "attempts": attempt + 1}
            except Exception as e:
                if attempt == retries:
                    raise
                self.logger.warning(
                    "Creating inference endpoint [%s] failed (attempt %d of %d): %s. Deleting it and model [%s], retrying in [%s] seconds.",
                    inference_id,
                    attempt + 1,
                    retries + 1,
                    e,
                    model_id,
                    wait_seconds,
                )
            cleanup_paths = [endpoint_path] + ([f"/_ml/trained_models/{model_id}"] if model_id else [])
            for path in cleanup_paths:
                try:
                    await es.options(**{**transport_params, "ignore_status": 404}).perform_request(
                        method="DELETE", path=path, headers=headers or None, params={"force": "true"}
                    )
                except Exception as e:
                    self.logger.warning("Deleting [%s] before the next attempt failed: %s", path, e)
            await asyncio.sleep(wait_seconds)

    def __repr__(self, *args, **kwargs):
        return "create-inference-endpoint"


class ConfigureCCM(runner.Runner):
    """
    Connects the cluster to the Elastic Inference Service (Cloud Connected Mode), so that EIS endpoints
    such as ``.jina-clip-v2`` exist on a self-managed cluster.

    The Elastic Cloud API key is read from ``api-key-file`` on the load driver, not passed as a track
    parameter: Rally stores track parameters with the race and writes them to its log. The file may be
    written after the race has started; it is awaited for up to ``file-wait-seconds``. The key is
    never logged or returned.
    """

    async def __call__(self, es, params):
        params, request_params, transport_params, headers = self._transport_request_params(params)
        path = os.path.expanduser(runner.mandatory(params, "api-key-file", self))
        wait_seconds = params.get("file-wait-seconds", 1800)
        waited = 0
        while not os.path.exists(path):
            if waited >= wait_seconds:
                raise FileNotFoundError(f"no CCM API key file [{path}] after {waited} seconds")
            if waited % 60 == 0:
                self.logger.info("Waiting for the CCM API key file [%s] (%d of %d seconds)", path, waited, wait_seconds)
            await asyncio.sleep(10)
            waited += 10
        with open(path, encoding="utf-8") as f:
            api_key = f.read().strip()
        if not api_key:
            raise ValueError(f"the CCM API key file [{path}] is empty")
        await es.options(**transport_params).perform_request(
            method="PUT", path="/_inference/_ccm", headers=headers or None, body={"api_key": api_key}, params=request_params
        )
        self.logger.info("Cloud Connected Mode configured with the key from [%s]", path)
        return {"weight": 1, "unit": "ops", "success": True, "waited_seconds": waited}

    def __repr__(self, *args, **kwargs):
        return "configure-ccm"


class WaitForDenseVector(runner.Runner):
    """
    Runs an ES|QL query until its first value is a non-empty vector, up to ``retries`` times and
    ``retry-wait-period`` seconds apart, so that measuring starts only once the whole DENSE_VECTOR path
    works. Used for endpoints that the cluster does not create itself, such as EIS endpoints that
    appear once the service has authorized the cluster.

    Every error, and every answer without a vector, is retried.
    """

    async def __call__(self, es, params):
        retries = params.get("retries", 10)
        wait_seconds = params.get("retry-wait-period", 30)
        for attempt in range(retries + 1):
            try:
                response = await _run_esql(es, dict(params), self)
                values = response["values"]
                if not values or not values[0] or not values[0][0]:
                    raise ValueError("the query answered without a vector")
                return {"weight": 1, "unit": "ops", "success": True, "attempts": attempt + 1, "dimensions": len(values[0][0])}
            except Exception as e:
                if attempt == retries:
                    raise
                self.logger.info(
                    "DENSE_VECTOR is not ready (attempt %d of %d): %s. Retrying in [%s] seconds.", attempt + 1, retries + 1, e, wait_seconds
                )
                await asyncio.sleep(wait_seconds)

    def __repr__(self, *args, **kwargs):
        return "wait-for-dense-vector"


class DenseVectorNullCheck(runner.Runner):
    """
    Runs one ES|QL query that embeds the same rows as a measured task and counts the null embeddings,
    so that a run where the endpoint returns nulls cannot pass for a fast one.

    The query must end in ``STATS row_count = COUNT(*), null_count = SUM(...)``; ``nulls`` cannot be
    used as a column name, as it is an ES|QL keyword (``NULLS FIRST``). The sample fails when any
    embedding is null, or when ``expected-rows`` is set and a different number of rows came back.
    """

    async def __call__(self, es, params):
        params, request_params, transport_params, headers = self._transport_request_params(params)
        es = es.options(**transport_params)
        query = runner.mandatory(params, "query", self)
        expected_rows = params.get("expected-rows")
        response = await es.perform_request(
            method="POST",
            path="/_query",
            headers=headers or None,
            body={"query": query},
            params=request_params,
        )
        columns = [column["name"] for column in response["columns"]]
        row = dict(zip(columns, response["values"][0])) if response["values"] else {}
        rows = row.get("row_count") or 0
        nulls = row.get("null_count") or 0
        passed = nulls == 0 and (expected_rows is None or rows == expected_rows)
        log = self.logger.info if passed else self.logger.warning
        log("DENSE_VECTOR null check: rows=[%s] nulls=[%s] expected rows=[%s] query=[%s]", rows, nulls, expected_rows, query)
        result = {"weight": 1, "unit": "ops", "success": passed, "rows": rows, "nulls": nulls, "expected_rows": expected_rows}
        if not passed:
            result["error-type"] = "null-check"
            result["error-description"] = f"{nulls} null embeddings in {rows} rows (expected {expected_rows} rows)"
        return result

    def __repr__(self, *args, **kwargs):
        return "dense-vector-null-check"


class ComputeReferenceVector(runner.Runner):
    """
    Embeds the reference document with DENSE_VECTOR and stores the vector on the load driver, so that
    the queries that compare against it use exactly the vector DENSE_VECTOR produces for that
    document with this endpoint.

    The query must return one row whose first column is the embedding.
    """

    async def __call__(self, es, params):
        reference_name = runner.mandatory(params, "reference-name", self)
        response = await _run_esql(es, params, self)
        values = response["values"]
        if len(values) != 1 or not values[0] or not values[0][0]:
            raise ValueError(f"expected one row with an embedding for reference [{reference_name}], got {len(values)} rows")
        vector = values[0][0]
        with open(_reference_vector_path(reference_name), "w", encoding="utf-8") as f:
            json.dump(vector, f)
        self.logger.info("Reference vector [%s] stored: %d dimensions", reference_name, len(vector))
        return {"weight": 1, "unit": "ops", "success": True, "dimensions": len(vector)}

    def __repr__(self, *args, **kwargs):
        return "compute-reference-vector"


class ReferenceVectorCorrectness(runner.Runner):
    """
    Checks, before anything is measured, that a query comparing the reference document with its own
    vector gives the expected answer.

    With ``check: cosine`` the query must return one row whose ``sim`` is at least ``min-similarity``.
    With ``check: present`` it must return exactly one row, whose ``id`` is ``expected-id``.
    """

    async def __call__(self, es, params):
        check = runner.mandatory(params, "check", self)
        if check not in ("cosine", "present"):
            raise ValueError(f"unknown check [{check}]")
        params = _with_reference_vector(params)
        response = await _run_esql(es, params, self)
        columns = [column["name"] for column in response["columns"]]
        rows = [dict(zip(columns, values)) for values in response["values"]]
        if check == "cosine":
            min_similarity = params.get("min-similarity", 0.9999)
            similarity = rows[0].get("sim") if len(rows) == 1 else None
            passed = similarity is not None and similarity >= min_similarity
            detail = f"sim={similarity} (minimum {min_similarity}) over {len(rows)} rows"
        else:
            expected_id = runner.mandatory(params, "expected-id", self)
            ids = [row.get("id") for row in rows]
            passed = ids == [expected_id]
            detail = f"ids={ids} (expected [{expected_id}])"
        log = self.logger.info if passed else self.logger.warning
        log("Reference vector correctness [%s]: %s", check, detail)
        result = {"weight": 1, "unit": "ops", "success": passed, "check": check, "detail": detail}
        if not passed:
            result["error-type"] = "correctness"
            result["error-description"] = f"{check} check failed: {detail}"
        return result

    def __repr__(self, *args, **kwargs):
        return "reference-vector-correctness"


class EsqlWithReferenceVector(runner.Esql):
    """Rally's ``esql`` operation, with the stored reference vector put into the query first."""

    async def __call__(self, es, params):
        return await super().__call__(es, _with_reference_vector(params))

    def __repr__(self, *args, **kwargs):
        return "esql-with-reference-vector"


class EsqlProfileWithReferenceVector(runner.EsqlProfile):
    """Rally's ``esql-profile`` operation, with the stored reference vector put into the query first."""

    async def __call__(self, es, params):
        return await super().__call__(es, _with_reference_vector(params))

    def __repr__(self, *args, **kwargs):
        return "esql-profile-with-reference-vector"


class EsqlStatsWithReferenceVector(runner.Runner):
    """
    Runs a query that ends in ``STATS`` once, with the stored reference vector put in, and records
    every column of its single row in the sample's metadata. It records; it does not judge.
    """

    async def __call__(self, es, params):
        response = await _run_esql(es, _with_reference_vector(params), self)
        columns = [column["name"] for column in response["columns"]]
        row = dict(zip(columns, response["values"][0])) if response["values"] else {}
        self.logger.info("ES|QL stats: %s", row)
        return {"weight": 1, "unit": "ops", "success": True, **row}

    def __repr__(self, *args, **kwargs):
        return "esql-stats-with-reference-vector"


def register(registry):
    registry.register_runner("create-inference-endpoint", CreateInferenceEndpoint(), async_runner=True)
    registry.register_runner("configure-ccm", ConfigureCCM(), async_runner=True)
    registry.register_runner("wait-for-inference-endpoint", WaitForInferenceEndpoint(), async_runner=True)
    registry.register_runner("wait-for-dense-vector", WaitForDenseVector(), async_runner=True)
    registry.register_runner("dense-vector-null-check", DenseVectorNullCheck(), async_runner=True)
    registry.register_runner("compute-reference-vector", ComputeReferenceVector(), async_runner=True)
    registry.register_runner("reference-vector-correctness", ReferenceVectorCorrectness(), async_runner=True)
    registry.register_runner("esql-with-reference-vector", EsqlWithReferenceVector(), async_runner=True)
    registry.register_runner("esql-profile-with-reference-vector", EsqlProfileWithReferenceVector(), async_runner=True)
    registry.register_runner("esql-stats-with-reference-vector", EsqlStatsWithReferenceVector(), async_runner=True)
