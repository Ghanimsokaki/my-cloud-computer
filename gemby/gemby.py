#!/usr/bin/env python3
"""Gemby CLI.

Loads the Hugging Face model `Hwiiiiiiii/Gemby-2-agent`, lets you chat with it,
serve it as an OpenAI-compatible API, and manage API keys.

Quick start
-----------
    # 1. Create an API key (printed once, stored hashed in ~/.gemby/keys.json)
    gemby key new

    # 2. Log in to Hugging Face so the gated model can be downloaded
    huggingface-cli login            # or: export HF_TOKEN=hf_xxx

    # 3. Download / cache the model
    gemby pull

    # 4. Talk to it
    gemby chat                       # interactive
    gemby serve --port 8000          # OpenAI-compatible server

Environment
-----------
    GEMBY_MODEL        model repo id         (default: Hwiiiiiiii/Gemby-2-agent)
    GEMBY_BASE_MODEL   optional architecture donor when config.json is custom
    HF_TOKEN           Hugging Face token for the gated model
    GEMBY_HOME         state dir             (default: ~/.gemby)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import secrets
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_MODEL = "Hwiiiiiiii/Gemby-2-agent"
GEMBY_HOME = Path(os.environ.get("GEMBY_HOME", Path.home() / ".gemby"))
KEYS_FILE = GEMBY_HOME / "keys.json"
KEY_PREFIX = "gemby-"


# --------------------------------------------------------------------------- #
# API key management
# --------------------------------------------------------------------------- #
def _load_keys() -> dict:
    if not KEYS_FILE.exists():
        return {"keys": []}
    try:
        return json.loads(KEYS_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return {"keys": []}


def _save_keys(data: dict) -> None:
    GEMBY_HOME.mkdir(parents=True, exist_ok=True)
    KEYS_FILE.write_text(json.dumps(data, indent=2))
    try:
        os.chmod(KEYS_FILE, 0o600)
    except OSError:
        pass


def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def key_new(name: str | None = None) -> str:
    """Generate, store and return a new API key."""
    raw = KEY_PREFIX + secrets.token_urlsafe(32)
    data = _load_keys()
    data["keys"].append(
        {
            "name": name or "default",
            "prefix": raw[: len(KEY_PREFIX) + 6],
            "sha256": _hash(raw),
            "created": datetime.now(timezone.utc).isoformat(),
            "last_used": None,
        }
    )
    _save_keys(data)
    return raw


def key_list() -> list[dict]:
    return _load_keys()["keys"]


def key_revoke(key_or_prefix: str) -> int:
    data = _load_keys()
    before = len(data["keys"])
    data["keys"] = [
        k
        for k in data["keys"]
        if k["sha256"] != _hash(key_or_prefix) and k["prefix"] != key_or_prefix
    ]
    _save_keys(data)
    return before - len(data["keys"])


def key_verify(key: str) -> bool:
    """Return True if `key` matches a stored, non-revoked key."""
    if not key:
        return False
    digest = _hash(key)
    for entry in key_list():
        if entry.get("sha256") == digest:
            entry["last_used"] = datetime.now(timezone.utc).isoformat()
            _save_keys({"keys": key_list()})
            return True
    return False


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _hf_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        or None
    )


class GembyRuntime:
    """Holds a loaded model + tokenizer and exposes generate()."""

    def __init__(self, model, tokenizer, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    # -- prompt building ---------------------------------------------------- #
    def _build_inputs(self, messages: list[dict]):
        import torch

        tok = self.tokenizer
        text = None
        if getattr(tok, "chat_template", None):
            try:
                text = tok.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                text = None
        if text is None:  # repo has no tokenizer_config -> no chat template
            lines = [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in messages]
            lines.append("assistant:")
            text = "\n".join(lines)
        return tok(text, return_tensors="pt").to(self.device)

    # -- generation --------------------------------------------------------- #
    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> str:
        import torch

        inputs = self._build_inputs(messages)
        do_sample = temperature is not None and temperature > 0
        gen_kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        with torch.no_grad():
            out = self.model.generate(**inputs, **gen_kwargs)
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ):
        """Yield generated text chunks (requires transformers TextIteratorStreamer)."""
        import torch
        from transformers import TextIteratorStreamer

        inputs = self._build_inputs(messages)
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        do_sample = temperature is not None and temperature > 0
        gen_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            streamer=streamer,
        )
        if do_sample:
            gen_kwargs.update(temperature=temperature, top_p=top_p)
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()


def _resolve_dtype(dtype: str, device: str):
    import torch

    if dtype == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype]


def load_runtime(
    model_id: str,
    base_model: str | None = None,
    dtype: str = "auto",
    device: str | None = None,
    trust_remote_code: bool = True,
) -> GembyRuntime:
    """Download (if needed) and load the model. Handles the custom `gemby`
    model_type that plain transformers cannot resolve on its own."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            f"[gemby] missing dependency '{exc.name}'. Install the runtime with:\n"
            "        pip install -r /workspace/gemby/requirements.txt"
        ) from exc

    token = _hf_token()
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = _resolve_dtype(dtype, device)

    print(f"[gemby] loading tokenizer: {model_id}", file=sys.stderr)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, token=token, trust_remote_code=trust_remote_code
        )
    except Exception as exc:  # tokenizer may lack tokenizer_config.json
        print(f"[gemby] tokenizer load failed ({exc}); retrying manually", file=sys.stderr)
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast(
            tokenizer_file=str(Path(_hub_cache_dir(model_id)) / "tokenizer.json")
        )

    print(f"[gemby] loading model: {model_id} ({dtype} on {device})", file=sys.stderr)
    load_kwargs = dict(
        token=token,
        trust_remote_code=trust_remote_code,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        device_map=device if device == "cpu" else "auto",
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    except ValueError as exc:
        msg = str(exc)
        if "model type" not in msg and "unrecognized" not in msg.lower():
            raise
        if not base_model:
            raise SystemExit(
                f"[gemby] '{model_id}' declares a custom model type that transformers\n"
                f"cannot build: {msg}\n"
                "Re-run with --base-model <hf-repo-id> (a decoder-only architecture\n"
                "with matching weights, e.g. a Qwen2/Llama variant) so Gemby can\n"
                "instantiate the config and load the state dict. See README notes."
            ) from exc
        print(f"[gemby] custom model type -> using --base-model {base_model}", file=sys.stderr)
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(base_model, token=token)
        model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
        state = _load_state_dict(model_id)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing or unexpected:
            print(
                f"[gemby] warning: {len(missing)} missing / {len(unexpected)} "
                "unexpected weight keys",
                file=sys.stderr,
            )

    model.eval()
    return GembyRuntime(model, tokenizer, device)


def _hub_cache_dir(model_id: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_id, token=_hf_token()))


def _load_state_dict(model_id: str):
    from safetensors.torch import load_file
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(model_id, token=_hf_token()))
    weights = path / "model.safetensors"
    if not weights.exists():
        weights = path / "downloaded_existing_model" / "model.safetensors"
    return load_file(str(weights))


def pull_model(model_id: str, base_model: str | None = None, dtype: str = "auto") -> None:
    if not _hf_token():
        raise SystemExit(
            "[gemby] HF_TOKEN is not set. This model is gated: accept its license at\n"
            f"        https://huggingface.co/{model_id}\n"
            "        then run `huggingface-cli login` or `export HF_TOKEN=hf_...`."
        )
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            f"[gemby] missing dependency '{exc.name}'. Install the runtime with:\n"
            "        pip install -r /workspace/gemby/requirements.txt"
        ) from exc
    print(f"[gemby] downloading {model_id} ...", file=sys.stderr)
    path = snapshot_download(
        model_id,
        token=_hf_token(),
        ignore_patterns=["*.jsonl"],  # skip the multi-GB training datasets
    )
    print(f"[gemby] cached at: {path}")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_pull(args) -> int:
    pull_model(args.model, args.base_model, args.dtype)
    return 0


def cmd_chat(args) -> int:
    runtime = load_runtime(args.model, args.base_model, args.dtype, args.device)
    print("Gemby ready. Type 'exit' or Ctrl-D to quit.")
    history: list[dict] = []
    if args.system:
        history.append({"role": "system", "content": args.system})
    while True:
        try:
            user = input("you> ").strip()
        except EOFError:
            print()
            break
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        history.append({"role": "user", "content": user})
        reply = runtime.generate(history, args.max_tokens, args.temperature, args.top_p)
        print(f"gemby> {reply}")
        history.append({"role": "assistant", "content": reply})
    return 0


def cmd_serve(args) -> int:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    if not key_list():
        print(
            "[gemby] no API key exists yet. Run `gemby key new` first.",
            file=sys.stderr,
        )
        return 1

    runtime = load_runtime(args.model, args.base_model, args.dtype, args.device)

    class Handler(BaseHTTPRequestHandler):
        server_version = "Gemby/1.0"

        def _json(self, code: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            header = self.headers.get("Authorization", "")
            if not header.startswith("Bearer "):
                return False
            return key_verify(header[len("Bearer "):].strip())

        def log_message(self, fmt, *a):  # quieter logs
            sys.stderr.write("[gemby] " + fmt % a + "\n")

        def do_GET(self):
            if self.path.startswith("/v1/models"):
                if not self._authorized():
                    return self._json(401, {"error": {"message": "invalid API key"}})
                return self._json(200, {"object": "list", "data": [
                    {"id": args.model, "object": "model", "owned_by": "gemby"}]})
            self._json(404, {"error": {"message": "not found"}})

        def do_POST(self):
            if not self.path.startswith("/v1/chat/completions"):
                return self._json(404, {"error": {"message": "not found"}})
            if not self._authorized():
                return self._json(401, {"error": {"message": "invalid API key"}})
            try:
                length = int(self.headers.get("Content-Length", 0))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                return self._json(400, {"error": {"message": "invalid JSON body"}})

            messages = payload.get("messages", [])
            max_tokens = int(payload.get("max_tokens", 512))
            temperature = float(payload.get("temperature", 0.7))
            top_p = float(payload.get("top_p", 0.95))
            created = int(time.time())

            if payload.get("stream"):
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                for chunk in runtime.stream(messages, max_tokens, temperature, top_p):
                    data = {
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": args.model,
                        "choices": [{"index": 0, "delta": {"content": chunk}}],
                    }
                    self.wfile.write(f"data: {json.dumps(data)}\n\n".encode())
                    self.wfile.flush()
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return

            text = runtime.generate(messages, max_tokens, temperature, top_p)
            self._json(200, {
                "id": f"chatcmpl-{secrets.token_hex(8)}",
                "object": "chat.completion",
                "created": created,
                "model": args.model,
                "choices": [{
                    "index": 0,
                    "message": {"role": "assistant", "content": text},
                    "finish_reason": "stop",
                }],
                "usage": {},
            })

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[gemby] serving {args.model} at http://{args.host}:{args.port}/v1 "
          f"(Bearer auth required)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[gemby] stopped")
    return 0


def cmd_key(args) -> int:
    if args.key_command == "new":
        raw = key_new(args.name)
        print(raw)
        print("\nStore it now - it is only shown once.", file=sys.stderr)
    elif args.key_command == "list":
        entries = key_list()
        if not entries:
            print("(no keys)")
            return 0
        print(f"{'PREFIX':<16}{'NAME':<16}{'CREATED':<26}LAST USED")
        for e in entries:
            print(f"{e['prefix']:<16}{e['name']:<16}{e['created']:<26}{e.get('last_used') or '-'}")
    elif args.key_command == "revoke":
        n = key_revoke(args.key)
        print(f"revoked {n} key(s)")
    return 0


def cmd_info(args) -> int:
    token_state = "set" if _hf_token() else "NOT set (gated model cannot be downloaded)"
    print(f"model id      : {args.model}")
    print(f"base model    : {args.base_model or '-'}")
    print(f"HF token      : {token_state}")
    print(f"state dir     : {GEMBY_HOME}")
    print(f"keys stored   : {len(key_list())} ({KEYS_FILE})")
    try:
        import torch

        print(f"torch         : {torch.__version__} (cuda={torch.cuda.is_available()})")
    except ImportError:
        print("torch         : not installed")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gemby", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=os.environ.get("GEMBY_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-model", default=os.environ.get("GEMBY_BASE_MODEL"))
    parser.add_argument("--dtype", default="auto",
                        choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--device", default=None, choices=["cpu", "cuda"])

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("pull", help="download/cache the model").set_defaults(func=cmd_pull)

    chat = sub.add_parser("chat", help="interactive chat")
    chat.add_argument("--system", default=None)
    chat.add_argument("--max-tokens", type=int, default=512)
    chat.add_argument("--temperature", type=float, default=0.7)
    chat.add_argument("--top-p", type=float, default=0.95)
    chat.set_defaults(func=cmd_chat)

    serve = sub.add_parser("serve", help="OpenAI-compatible API server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.set_defaults(func=cmd_serve)

    key = sub.add_parser("key", help="manage API keys")
    key_sub = key.add_subparsers(dest="key_command", required=True)
    k_new = key_sub.add_parser("new", help="create a new API key")
    k_new.add_argument("--name", default=None)
    key_sub.add_parser("list", help="list API keys")
    k_rev = key_sub.add_parser("revoke", help="revoke a key (full key or prefix)")
    k_rev.add_argument("key")
    key.set_defaults(func=cmd_key)

    sub.add_parser("info", help="show resolved config").set_defaults(func=cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "device", None) is None:
        args.device = None
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
