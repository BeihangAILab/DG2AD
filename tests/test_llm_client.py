import json
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from src.core.trainer import _json_safe_config
from src.utils.llm_client import _request_chat_completion, create_client


class _Response:
    status = 200

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def read(self):
        return json.dumps(
            {
                "choices": [{"message": {"content": "OK"}}],
                "usage": {"total_tokens": 3},
            }
        ).encode("utf-8")


class LLMClientTests(unittest.TestCase):
    @patch("urllib.request.urlopen", return_value=_Response())
    def test_openai_request_uses_proxy_capable_urllib(self, mocked_urlopen):
        content, tokens = _request_chat_completion(
            api_key="test-key",
            base_url="https://example.test/v1/",
            model_name="test-model",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0.2,
            timeout_seconds=12.0,
            retries=0,
        )

        self.assertEqual((content, tokens), ("OK", 3))
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "https://example.test/v1/chat/completions")
        self.assertEqual(request.method, "POST")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(mocked_urlopen.call_args.kwargs["timeout"], 12.0)

    @patch("urllib.request.urlopen", return_value=_Response())
    def test_loopback_endpoint_may_omit_authorization(self, mocked_urlopen):
        client = create_client(
            api_key="",
            base_url="http://127.0.0.1:8000/v1",
            model_name="local-proposer",
            retries=0,
        )

        self.assertEqual(client.chat([{"role": "user", "content": "hello"}]), ("OK", 3))
        request = mocked_urlopen.call_args.args[0]
        self.assertIsNone(request.get_header("Authorization"))
        self.assertEqual(len(client.audit_records), 1)
        audit = client.audit_records[0]
        self.assertEqual(audit["status"], "ok")
        self.assertEqual(audit["tokens"], 3)
        self.assertEqual(audit["model"], "local-proposer")
        self.assertNotIn("hello", json.dumps(audit))
        self.assertNotIn("api_key", json.dumps(audit).lower())

        restored = create_client(
            api_key="",
            base_url="http://127.0.0.1:8000/v1",
            model_name="local-proposer",
            retries=0,
        )
        restored.load_state_dict(client.state_dict())
        self.assertEqual(restored.audit_records, client.audit_records)

    def test_remote_endpoint_requires_api_key(self):
        with self.assertRaisesRegex(RuntimeError, "LLM_API_KEY"):
            create_client(
                api_key="",
                base_url="https://example.test/v1",
                model_name="remote-model",
            )

    def test_endpoint_and_model_are_required(self):
        with self.assertRaisesRegex(RuntimeError, "LLM_BASE_URL"):
            create_client(api_key="secret", base_url="", model_name="model")
        with self.assertRaisesRegex(RuntimeError, "LLM_MODEL_NAME"):
            create_client(
                api_key="secret",
                base_url="https://example.test/v1",
                model_name="",
            )

    def test_endpoint_rejects_embedded_credentials(self):
        with self.assertRaisesRegex(RuntimeError, "embedded credentials"):
            create_client(
                api_key="secret",
                base_url="https://user:" + "password" + "@example.test/v1",
                model_name="model",
            )

    def test_manifest_config_redacts_nested_secret_names(self):
        cfg = OmegaConf.create(
            {
                "llm": {"api_key": "do-not-write", "model_name": "model"},
                "nested": {
                    "authorization": "Bearer private",
                    "items": [{"refresh_token": "private"}],
                },
            }
        )
        serialized = _json_safe_config(cfg)
        text = json.dumps(serialized)
        self.assertNotIn("do-not-write", text)
        self.assertNotIn("Bearer private", text)
        self.assertEqual(serialized["llm"]["api_key"], "<redacted>")
        self.assertEqual(serialized["nested"]["items"][0]["refresh_token"], "<redacted>")


if __name__ == "__main__":
    unittest.main()
