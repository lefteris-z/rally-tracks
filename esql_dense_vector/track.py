import asyncio

from esrally.driver import runner


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


def register(registry):
    registry.register_runner("wait-for-inference-endpoint", WaitForInferenceEndpoint(), async_runner=True)
    registry.register_runner("dense-vector-null-check", DenseVectorNullCheck(), async_runner=True)
