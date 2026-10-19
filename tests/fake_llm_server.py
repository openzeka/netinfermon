#!/usr/bin/env python3
"""Stand-in OpenAI-compatible server for end-to-end tests of netinfermon (no GPU, no model).

Streams one token per chunk: TTFT = 40 ms + 3 µs per prompt token, then 2 ms per token.
Honours max_tokens and ignore_eos/min_tokens; reports usage (prompt_tokens = chars / 4).

    fake_llm_server.py [port]      (default 30099)
"""

import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "fake/model-for-tests"


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": MODEL, "object": "model",
                                                           "max_model_len": 65536}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        text = "".join(m["content"] for m in req["messages"])
        p_tok = max(1, len(text) // 4)
        n = int(req.get("max_tokens", 16))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.040 + p_tok * 3e-6)

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        for i in range(n):
            send({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": MODEL,
                  "choices": [{"index": 0, "delta": {"content": "tok "}, "finish_reason": None}]})
            time.sleep(0.002)
        send({"id": "x", "object": "chat.completion.chunk", "created": 0, "model": MODEL, "choices": [],
              "usage": {"prompt_tokens": p_tok, "completion_tokens": n, "total_tokens": p_tok + n}})
        self.wfile.write(b"data: [DONE]\n\n")


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 30099), H).serve_forever()
