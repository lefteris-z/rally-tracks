from esrally.driver import runner


def register(registry):
    # The first request to an inference endpoint can fail while its model deploys, so the warm-up
    # retries Rally's own raw-request until the endpoint answers.
    registry.register_runner("retryable-raw-request", runner.Retry(runner.RawRequest()), async_runner=True)
