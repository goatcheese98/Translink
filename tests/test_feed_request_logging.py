"""Offline CLI regression: never use credentials, network, or repository data."""

import contextlib
import io
import os
from pathlib import Path
import runpy
import tempfile
import traceback
import unittest
from unittest.mock import Mock, patch

import requests
from google.transit import gtfs_realtime_pb2


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/test_gtfs_rt_trip_updates.py"
CANARY = "NONSECRET_LOGGING_CANARY_917"
URL = f"https://gtfsapi.translink.ca/v3/gtfsrealtime?apikey={CANARY}"


class FeedRequestLoggingTests(unittest.TestCase):
    def run_script(self, *, error=None, status=200, content=None, key=CANARY):
        feed = gtfs_realtime_pb2.FeedMessage()
        feed.header.gtfs_realtime_version = "2.0"
        feed.header.timestamp = 123
        entity = feed.entity.add()
        entity.id = "offline-entity"
        payload = feed.SerializeToString() if content is None else content
        response = requests.Response()
        response.status_code = status
        response.url = URL
        response.reason = f"untrusted reason {URL}"
        response._content = payload
        get = Mock(return_value=response, side_effect=error)
        stdout, stderr = io.StringIO(), io.StringIO()
        exit_code = 0
        old_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with (
                    patch.dict(os.environ, {"TRANSLINK_API_KEY": key}, clear=True),
                    patch("dotenv.load_dotenv") as dotenv,
                    patch("requests.get", get),
                    patch("socket.socket", side_effect=AssertionError("Network blocked")),
                    patch("time.time", return_value=456),
                    contextlib.redirect_stdout(stdout),
                    contextlib.redirect_stderr(stderr),
                ):
                    try:
                        runpy.run_path(str(SCRIPT), run_name="__main__")
                    except SystemExit as exc:
                        exit_code = exc.code
                    except Exception:
                        exit_code = 1
                        # Model the stderr produced by an uncaught CLI exception.
                        traceback.print_exc()
                dotenv.assert_called_once_with()
                files = list(Path(directory).rglob("*.pb"))
                saved = [path.read_bytes() for path in files]
            finally:
                os.chdir(old_cwd)
        if key:
            get.assert_called_once_with(URL, timeout=30)
        else:
            get.assert_not_called()
        for output in (stdout.getvalue(), stderr.getvalue()):
            self.assertNotIn(CANARY, output)
            self.assertNotIn(URL, output)
            self.assertNotIn("apikey=", output)
            self.assertNotIn("untrusted reason", output)
        return exit_code, stdout.getvalue(), stderr.getvalue(), saved, payload

    def test_success_preserves_download_and_diagnostics(self):
        code, out, err, saved, payload = self.run_script()
        self.assertEqual(code, 0)
        self.assertEqual(err, "")
        self.assertEqual(saved, [payload])
        for diagnostic in ("HTTP status: 200", "Feed timestamp: 123",
                           "Number of entities: 1", "offline-entity",
                           "SCRIPT FINISHED SUCCESSFULLY"):
            self.assertIn(diagnostic, out)

    def test_transport_errors(self):
        for error_type in (requests.Timeout, requests.ConnectionError,
                           requests.TooManyRedirects, requests.RequestException):
            with self.subTest(error_type=error_type.__name__):
                code, out, err, saved, _ = self.run_script(
                    error=error_type(f"request failed {URL}"))
                self.assertNotEqual(code, 0)
                self.assertIn("Request failed", err)
                self.assertEqual(saved, [])
                self.assertNotIn("SUCCESSFULLY", out)

    def test_http_errors(self):
        for status in (401, 403, 429, 500):
            with self.subTest(status=status):
                code, out, err, saved, _ = self.run_script(status=status)
                self.assertNotEqual(code, 0)
                self.assertIn(f"HTTP status: {status}", out)
                self.assertIn("Request failed", err)
                self.assertEqual(saved, [])

    def test_invalid_feed_keeps_raw_download(self):
        code, _, _, saved, payload = self.run_script(content=b"\xff")
        self.assertNotEqual(code, 0)
        self.assertEqual(saved, [payload])

    def test_missing_key_does_not_request(self):
        code, _, err, saved, _ = self.run_script(key="")
        self.assertNotEqual(code, 0)
        self.assertIn("TRANSLINK_API_KEY not found", err)
        self.assertEqual(saved, [])


if __name__ == "__main__":
    unittest.main()
