import asyncio

from esrally.driver import runner


class WarmUpInferenceEndpoint(runner.Runner):
    """
    Calls a text_embedding inference endpoint until it answers, so that deploying or scaling up its
    model is not part of any measured task.

    Every error is retried, up to ``retries`` times and ``retry-wait-period`` seconds apart: while a
    model deploys, an endpoint can answer with errors that Rally's own retry does not cover, as that
    one only retries connection errors and HTTP 408.
    """

    async def __call__(self, es, params):
        params, request_params, transport_params, headers = self._transport_request_params(params)
        es = es.options(**transport_params)
        inference_id = runner.mandatory(params, "inference-id", self)
        retries = params.get("retries", 10)
        wait_seconds = params.get("retry-wait-period", 30)
        for attempt in range(retries + 1):
            try:
                await es.perform_request(
                    method="POST",
                    path=f"/_inference/text_embedding/{inference_id}",
                    headers=headers or None,
                    body={"input": ["warm up"]},
                    params=request_params,
                )
                return {"weight": 1, "unit": "ops", "success": True, "attempts": attempt + 1}
            except Exception as e:
                if attempt == retries:
                    raise
                self.logger.info(
                    "Inference endpoint [%s] did not answer (attempt %d of %d): %s. Retrying in [%s] seconds.",
                    inference_id,
                    attempt + 1,
                    retries + 1,
                    e,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)

    def __repr__(self, *args, **kwargs):
        return "warm-up-inference-endpoint"


def register(registry):
    registry.register_runner("warm-up-inference-endpoint", WarmUpInferenceEndpoint(), async_runner=True)
