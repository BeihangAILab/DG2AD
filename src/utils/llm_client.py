import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable
from urllib.parse import urlparse


_LOOPBACK_HOSTS = {"localhost", "127.0.0.1", "::1"}


def validate_endpoint_configuration(api_key, base_url, model_name):
    """Validate one OpenAI-compatible endpoint without exposing credentials."""
    normalized_base_url = str(base_url or "").strip()
    normalized_model_name = str(model_name or "").strip()
    normalized_api_key = str(api_key or "").strip()
    if not normalized_base_url:
        raise RuntimeError("LLM_BASE_URL is required for training")
    if not normalized_model_name:
        raise RuntimeError("LLM_MODEL_NAME is required for training")
    parsed = urlparse(normalized_base_url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise RuntimeError("LLM_BASE_URL must be an absolute http:// or https:// URL")
    if parsed.username is not None or parsed.password is not None:
        raise RuntimeError("LLM_BASE_URL must not contain embedded credentials")
    if not normalized_api_key and parsed.hostname.lower() not in _LOOPBACK_HOSTS:
        raise RuntimeError("LLM_API_KEY is required for a non-loopback OpenAI-compatible endpoint")
    return normalized_api_key, normalized_base_url, normalized_model_name


def _request_chat_completion(
    api_key,
    base_url,
    model_name,
    messages,
    temperature,
    timeout_seconds,
    retries,
    on_warning=None,
):
    """Call an OpenAI-compatible endpoint with OS/environment proxy support."""
    import urllib.error
    import urllib.request

    api_key, base_url, model_name = validate_endpoint_configuration(api_key, base_url, model_name)
    api_url = base_url.rstrip("/") + "/chat/completions"
    attempts = max(1, int(retries) + 1)
    for attempt in range(attempts):
        try:
            payload = json.dumps(
                {
                    "model": model_name,
                    "messages": messages,
                    "temperature": temperature,
                }
            ).encode("utf-8")
            headers = {
                "User-Agent": "DGA2D/1.0",
                "Content-Type": "application/json",
            }
            if api_key:
                headers["Authorization"] = "Bearer " + api_key
            request = urllib.request.Request(
                api_url,
                data=payload,
                method="POST",
                headers=headers,
            )
            try:
                with urllib.request.urlopen(request, timeout=float(timeout_seconds)) as response:
                    status = int(response.status)
                    body = response.read()
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                body = exc.read()
            if status >= 400:
                body_hash = hashlib.sha256(body).hexdigest()[:16]
                raise RuntimeError(f"LLM API returned HTTP {status} (response_sha256={body_hash})")
            data = json.loads(body)
            return (
                data["choices"][0]["message"]["content"],
                data.get("usage", {}).get("total_tokens", 0),
            )
        except Exception as exc:
            if attempt == attempts - 1:
                raise
            wait_seconds = 2 * (attempt + 1)
            message = (
                f"LLM API call failed (attempt {attempt + 1}/{attempts}, "
                f"error={type(exc).__name__}). "
                f"Retrying in {wait_seconds}s."
            )
            if on_warning is not None:
                on_warning(message)
            time.sleep(wait_seconds)


@dataclass
class LLMClient:
    api_key: str
    base_url: str
    model_name: str
    temperature: float = 0.7
    max_calls: int = 64
    request_timeout_seconds: float = 60.0
    retries: int = 2
    calls_used: int = 0
    warning_callback: Callable[[str], None] | None = None
    audit_records: list[dict] = field(default_factory=list)

    def chat(self, messages, temperature=None):
        if self.calls_used >= self.max_calls:
            raise RuntimeError(f"LLM call budget exhausted ({self.max_calls})")
        self.calls_used += 1
        selected_temperature = self.temperature if temperature is None else float(temperature)
        request_bytes = json.dumps(
            messages,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        record = {
            "call_index": self.calls_used,
            "model": self.model_name,
            "temperature": selected_temperature,
            "started_at_utc": datetime.now(timezone.utc).isoformat(),
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
        }
        started = time.perf_counter()
        try:
            response, tokens = _request_chat_completion(
                self.api_key,
                self.base_url,
                self.model_name,
                messages,
                selected_temperature,
                self.request_timeout_seconds,
                self.retries,
                self.warning_callback,
            )
            record.update(
                {
                    "status": "ok",
                    "tokens": int(tokens or 0),
                    "response_sha256": hashlib.sha256(str(response).encode("utf-8")).hexdigest(),
                }
            )
            return response, tokens
        except Exception as exc:
            record.update(
                {
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_sha256": hashlib.sha256(str(exc).encode("utf-8")).hexdigest(),
                }
            )
            raise
        finally:
            record["elapsed_seconds"] = round(time.perf_counter() - started, 6)
            self.audit_records.append(record)

    def state_dict(self):
        return {
            "max_calls": self.max_calls,
            "calls_used": self.calls_used,
            "audit_records": list(self.audit_records),
        }

    def load_state_dict(self, state):
        self.max_calls = int(state.get("max_calls", self.max_calls))
        self.calls_used = int(state.get("calls_used", 0))
        self.audit_records = list(state.get("audit_records", []))

    def public_config(self):
        """Return non-secret endpoint metadata suitable for a run manifest."""

        parsed = urlparse(self.base_url)
        return {
            "model": self.model_name,
            "temperature": self.temperature,
            "endpoint_scheme": parsed.scheme,
            "endpoint_host": parsed.hostname,
            "max_calls": self.max_calls,
        }


def create_client(
    api_key,
    base_url,
    model_name,
    temperature=0.7,
    max_calls=64,
    request_timeout_seconds=60.0,
    retries=2,
    warning_callback=None,
):
    """Build one run-scoped client without global provider state."""
    api_key, base_url, model_name = validate_endpoint_configuration(api_key, base_url, model_name)
    return LLMClient(
        api_key=api_key,
        base_url=base_url,
        model_name=model_name,
        temperature=float(temperature),
        max_calls=int(max_calls),
        request_timeout_seconds=float(request_timeout_seconds),
        retries=int(retries),
        warning_callback=warning_callback,
    )
