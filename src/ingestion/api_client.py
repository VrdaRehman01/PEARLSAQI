import time
import requests


class APIClient:

    def __init__(
        self,
        timeout=60,
        max_retries=5,
        retry_delay=5
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    def get(self, url, params=None):

        last_error = None

        for attempt in range(1, self.max_retries + 1):

            try:

                response = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout
                )

                # Rate limited
                if response.status_code == 429:

                    retry_after = response.headers.get(
                        "Retry-After"
                    )

                    if retry_after:
                        wait_time = int(retry_after)
                    else:
                        wait_time = self.retry_delay * attempt

                    print(
                        f"Rate limit reached. "
                        f"Waiting {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

                response.raise_for_status()

                return response.json()

            except requests.RequestException as error:

                last_error = error

                print(
                    f"API request failed "
                    f"(attempt {attempt}/{self.max_retries}): "
                    f"{error}"
                )

                if attempt < self.max_retries:

                    wait_time = (
                        self.retry_delay * attempt
                    )

                    print(
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

        raise RuntimeError(
            f"API request failed after "
            f"{self.max_retries} attempts: "
            f"{last_error}"
        )