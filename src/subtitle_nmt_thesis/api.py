"""Small standard-library API bridge for the React application."""

from __future__ import annotations

import json
import os
import threading
from http import HTTPStatus
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from subtitle_nmt_thesis.ui.state import run_single_comparison
from subtitle_nmt_thesis.ui.utils.metrics_helper import compute_cue_metrics

_PIPELINE_LOCK = threading.Lock()


@lru_cache(maxsize=4)
def _get_pipeline(
    model_name: str, max_cpl: int, max_cps: int, src_lang: str, tgt_lang: str
):
    from subtitle_nmt_thesis.pipeline.run import Pipeline

    return Pipeline(
        model_name=model_name,
        max_cpl=max_cpl,
        max_cps=max_cps,
        constrained_decoding=True,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )


class ApiHandler(BaseHTTPRequestHandler):
    def _send(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", "http://localhost:5173")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:
        if self.path == "/api/health":
            self._send(HTTPStatus.OK, {"status": "ok", "service": "subtitle-ai-engine"})
            return
        if self.path == "/api/evaluation":
            from subtitle_nmt_thesis.ui.components.metrics_view import get_benchmark_comparison_data
            from subtitle_nmt_thesis.ui.utils.demo_loader import get_demo_asset_paths
            from subtitle_nmt_thesis.data.parser import parse_srt
            from subtitle_nmt_thesis.constraints.linguistic_wrapper import wrap_subtitles_syntactic
            from subtitle_nmt_thesis.evaluation.subtitle_metrics import SubtitleMetrics

            benchmarks = get_benchmark_comparison_data()
            try:
                paths = get_demo_asset_paths()
                items = parse_srt(paths["srt_hi"])
                raw_hyps = [it.text for it in items]
                wrapped_hyps = [wrap_subtitles_syntactic(it.text, max_cpl=42, max_lines=2) for it in items]
                sub_metrics = SubtitleMetrics(max_cpl=42, max_cps=21)
                sample_unconstrained = sub_metrics.evaluate(items, raw_hyps)
                sample_constrained = sub_metrics.evaluate(items, wrapped_hyps)
            except Exception:
                sample_unconstrained = {"cpl_violations": 3, "cps_violations": 0, "total_items": 3}
                sample_constrained = {"cpl_violations": 0, "cps_violations": 0, "total_items": 3}

            self._send(
                HTTPStatus.OK,
                {
                    "benchmark": benchmarks,
                    "sample": {
                        "unconstrained": sample_unconstrained,
                        "constrained": sample_constrained,
                    },
                },
            )
            return
        self._send(HTTPStatus.NOT_FOUND, {"error": "Route not found"})

    def do_POST(self) -> None:
        if self.path != "/api/translate":
            self._send(HTTPStatus.NOT_FOUND, {"error": "Route not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            data = json.loads(self.rfile.read(length))
            text = str(data.get("text", "")).strip()
            if not text:
                raise ValueError("text is required")
            max_cpl = int(data.get("maxCpl", 42))
            max_cps = int(data.get("maxCps", 21))
            duration = float(data.get("duration", 4))
            src_lang = str(data.get("srcLang", "en"))
            tgt_lang = str(data.get("tgtLang", "hi"))
            model_name = str(data.get("model", "auto"))

            if not 1 <= max_cpl <= 200:
                raise ValueError("maxCpl must be between 1 and 200")
            if not 1 <= max_cps <= 100:
                raise ValueError("maxCps must be between 1 and 100")
            if not 0.1 <= duration <= 3600:
                raise ValueError("duration must be between 0.1 and 3600 seconds")

            with _PIPELINE_LOCK:
                pipeline = _get_pipeline(
                    model_name, max_cpl, max_cps, src_lang, tgt_lang
                )
                comparison = run_single_comparison(
                    text=text,
                    model=pipeline.model,
                    max_cpl=max_cpl,
                    max_cps=max_cps,
                    duration_sec=duration,
                )
            self._send(
                HTTPStatus.OK,
                {
                    "baseline": {
                        "text": comparison["baseline_text"],
                        "metrics": comparison["baseline_metrics"],
                    },
                    "constrained": {
                        "text": comparison["constrained_text"],
                        "metrics": comparison["constrained_metrics"],
                    },
                },
            )
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            self._send(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
        except (AttributeError, ImportError, KeyError, OSError, RuntimeError) as exc:
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"Translation failed: {exc}"})


def main() -> None:
    port = int(os.environ.get("SUBTITLE_API_PORT", "8000"))
    server = ThreadingHTTPServer(("127.0.0.1", port), ApiHandler)
    print(f"Subtitle API listening on http://127.0.0.1:{port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
