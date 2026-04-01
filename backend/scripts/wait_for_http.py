import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


DEFAULT_TIMEOUT_SECONDS = 60
DEFAULT_SLEEP_SECONDS = 1


def wait_for_url(url: str, timeout_seconds: int) -> int:
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=5) as response:
                status_code = getattr(response, "status", response.getcode())
                if 200 <= status_code < 300:
                    print(f"HTTP dependency is ready: {url}")
                    return 0
        except (HTTPError, URLError, TimeoutError):
            pass

        print(f"Waiting for HTTP dependency: {url}")
        time.sleep(DEFAULT_SLEEP_SECONDS)

    print(
        f"Timed out waiting for HTTP dependency after {timeout_seconds}s: {url}",
        file=sys.stderr,
    )
    return 1


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python scripts/wait_for_http.py <url> [timeout_seconds]", file=sys.stderr)
        return 2

    url = sys.argv[1]
    timeout_seconds = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_TIMEOUT_SECONDS
    return wait_for_url(url, timeout_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
