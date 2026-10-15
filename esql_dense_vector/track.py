import asyncio
import json
import os
import tempfile
import urllib.error
import urllib.request

from esrally.driver import runner

REFERENCE_VECTOR_PLACEHOLDER = "__REFERENCE_VECTOR__"
CLOUD_API_URL = "https://cloud.elastic.co/api/v1"


def _cloud_api_request(method, url, api_key, body=None, timeout=60):
    """Calls the Elastic Cloud API and returns (status, parsed body). Errors are returned, not raised."""
    request = urllib.request.Request(url, method=method, data=json.dumps(body).encode() if body is not None else None)
    request.add_header("Authorization", f"apiKey {api_key}")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.status
    except urllib.error.HTTPError as e:
        raw = e.read()
        status = e.code
    try:
        return status, json.loads(raw) if raw else {}
    except ValueError:
        return status, {"raw": raw[:500].decode(errors="replace")}


async def _cloud_api(method, url, api_key, body=None):
    status, response = await asyncio.to_thread(_cloud_api_request, method, url, api_key, body)
    if status >= 300:
        errors = response.get("errors") if isinstance(response, dict) else None
        detail = [{k: e.get(k) for k in ("code", "message")} for e in errors] if errors else response
        raise RuntimeError(f"Elastic Cloud API {method} {url.split('/api/v1', 1)[-1]} returned {status}: {detail}")
    return response


def _ccm_registration_path(cluster_uuid):
    return os.path.join(tempfile.gettempdir(), f"esql-dense-vector-ccm-{cluster_uuid}.json")


async def _read_secret_file(path, wait_seconds, logger, what):
    waited = 0
    while not os.path.exists(path):
        if waited >= wait_seconds:
            raise FileNotFoundError(f"no {what} file [{path}] after {waited} seconds")
        if waited % 60 == 0:
            logger.info("Waiting for the %s file [%s] (%d of %d seconds)", what, path, waited, wait_seconds)
        await asyncio.sleep(10)
        waited += 10
    with open(path, encoding="utf-8") as f:
        secret = f.read().strip()
    if not secret:
        raise ValueError(f"the {what} file [{path}] is empty")
    return secret, waited


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
    Connects the cluster to the Elastic Inference Service through Cloud Connect, so that EIS endpoints
    such as ``.jina-clip-v2`` exist on a self-managed cluster. These are the steps Kibana's Cloud
    Connect page takes:

    1. register the cluster (its UUID, name, version and license) with the Elastic Cloud API;
    2. enable the ``eis`` service for that registration, which returns an EIS key;
    3. give the EIS key to Elasticsearch with ``PUT _inference/_ccm``.

    The registration ID is written next to the reference vectors, so that ``disconnect-ccm`` can
    delete the registration at the end of the race.

    The Cloud Connect API key (role ``cloud-connected-admin`` for all connected clusters) is read from
    ``api-key-file`` on the load driver, not passed as a track parameter: Rally stores track parameters
    with the race and writes them to its log. The file may be written after the race has started; it
    is awaited for up to ``file-wait-seconds``. Neither key is logged or returned.
    """

    async def __call__(self, es, params):
        params, request_params, transport_params, headers = self._transport_request_params(params)
        api_key, waited = await _read_secret_file(
            os.path.expanduser(runner.mandatory(params, "api-key-file", self)),
            params.get("file-wait-seconds", 1800),
            self.logger,
            "Cloud Connect API key",
        )
        cloud_api_url = params.get("cloud-api-url", CLOUD_API_URL)
        es = es.options(**transport_params)
        info = await es.perform_request(method="GET", path="/", headers=headers or None)
        license_info = (await es.perform_request(method="GET", path="/_license", headers=headers or None))["license"]
        cluster_uuid = info["cluster_uuid"]
        registration = await _cloud_api(
            "POST",
            f"{cloud_api_url}/cloud-connected/clusters",
            api_key,
            {
                "self_managed_cluster": {"id": cluster_uuid, "name": info["cluster_name"], "version": info["version"]["number"]},
                "license": {"type": license_info["type"], "uid": license_info["uid"]},
                "name": f"esql-dense-vector-{cluster_uuid}",
            },
        )
        registration_id = registration["id"]
        with open(_ccm_registration_path(cluster_uuid), "w", encoding="utf-8") as f:
            json.dump({"id": registration_id, "cloud-api-url": cloud_api_url}, f)
        self.logger.info("Cluster [%s] registered with Cloud Connect as [%s]", cluster_uuid, registration_id)
        enabled = await _cloud_api(
            "PATCH", f"{cloud_api_url}/cloud-connected/clusters/{registration_id}", api_key, {"services": {"eis": {"enabled": True}}}
        )
        eis_key = (enabled.get("keys") or {}).get("eis")
        if not eis_key:
            raise ValueError(f"enabling EIS for Cloud Connect registration [{registration_id}] returned no EIS key")
        await es.perform_request(
            method="PUT", path="/_inference/_ccm", headers=headers or None, body={"api_key": eis_key}, params=request_params
        )
        self.logger.info("EIS enabled for Cloud Connect registration [%s] and set in Elasticsearch", registration_id)
        return {
            "weight": 1,
            "unit": "ops",
            "success": True,
            "waited_seconds": waited,
            "cluster_uuid": cluster_uuid,
            "cloud_connect_id": registration_id,
        }

    def __repr__(self, *args, **kwargs):
        return "configure-ccm"


class DisconnectCCM(runner.Runner):
    """
    Deletes the Cloud Connect registration that ``configure-ccm`` made for this cluster, as every
    experiment installs a new cluster and registers it again. It records and does not fail: a
    registration it cannot delete stays in the Elastic Cloud organization and is logged with its ID.
    """

    async def __call__(self, es, params):
        api_key, _ = await _read_secret_file(
            os.path.expanduser(runner.mandatory(params, "api-key-file", self)), 0, self.logger, "Cloud Connect API key"
        )
        cluster_uuid = (await es.perform_request(method="GET", path="/"))["cluster_uuid"]
        path = _ccm_registration_path(cluster_uuid)
        if not os.path.exists(path):
            self.logger.warning("No Cloud Connect registration recorded for cluster [%s]", cluster_uuid)
            return {"weight": 1, "unit": "ops", "success": True, "deleted": False}
        with open(path, encoding="utf-8") as f:
            registration = json.load(f)
        try:
            await _cloud_api("DELETE", f"{registration['cloud-api-url']}/cloud-connected/clusters/{registration['id']}", api_key)
        except Exception as e:
            self.logger.warning("Cloud Connect registration [%s] was not deleted: %s", registration["id"], e)
            return {"weight": 1, "unit": "ops", "success": True, "deleted": False, "cloud_connect_id": registration["id"]}
        os.remove(path)
        self.logger.info("Cloud Connect registration [%s] deleted", registration["id"])
        return {"weight": 1, "unit": "ops", "success": True, "deleted": True, "cloud_connect_id": registration["id"]}

    def __repr__(self, *args, **kwargs):
        return "disconnect-ccm"


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
    registry.register_runner("disconnect-ccm", DisconnectCCM(), async_runner=True)
    registry.register_runner("wait-for-inference-endpoint", WaitForInferenceEndpoint(), async_runner=True)
    registry.register_runner("wait-for-dense-vector", WaitForDenseVector(), async_runner=True)
    registry.register_runner("dense-vector-null-check", DenseVectorNullCheck(), async_runner=True)
    registry.register_runner("compute-reference-vector", ComputeReferenceVector(), async_runner=True)
    registry.register_runner("reference-vector-correctness", ReferenceVectorCorrectness(), async_runner=True)
    registry.register_runner("esql-with-reference-vector", EsqlWithReferenceVector(), async_runner=True)
    registry.register_runner("esql-profile-with-reference-vector", EsqlProfileWithReferenceVector(), async_runner=True)
    registry.register_runner("esql-stats-with-reference-vector", EsqlStatsWithReferenceVector(), async_runner=True)
