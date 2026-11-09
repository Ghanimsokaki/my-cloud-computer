#!/usr/bin/env python3
"""Gemby CLI.

Loads the Hugging Face model `Hwiiiiiiii/Gemby-2-agent`, lets you chat with it,
serve it as an OpenAI-compatible API, and manage API keys.

Quick start
-----------
    # 1. Provide a Hugging Face token (the model is gated). Either:
    #      - add a Codespaces secret named HF_TOKEN, or
    #      - put HF_TOKEN=hf_xxx in .env (git-ignored), or
    #      - export HF_TOKEN=hf_xxx
    # 2. Create an API key (printed once, stored hashed in ~/.gemby/keys.json)
    gemby key new
    # 3. Download / cache the model
    gemby pull
    # 4. Talk to it
    gemby chat                       # interactive
    gemby serve --port 8000          # OpenAI-compatible server

About the model
---------------
`Hwiiiiiiii/Gemby-2-agent` is a Llama-style decoder built from scratch with a
custom `"model_type": "gemby"` and non-standard weight names, so upstream
`AutoModelForCausalLM` cannot load it. This CLI remaps the custom weights onto a
standard Llama architecture (`embed_tokens`, `layers.*.attention.*`, `norm1/2`)
and builds the chat prompt from the model's own special tokens
(`<|system|>`, `<|user|>`, `<|assistant|>`).

Environment
-----------
    GEMBY_MODEL   model repo id (default: Hwiiiiiiii/Gemby-2-agent)
    HF_TOKEN      Hugging Face token for the gated model
    GEMBY_HOME    state dir (default: ~/.gemby)
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
KEY_PREFIX = "gemby-"


# --------------------------------------------------------------------------- #
# .env loading (no external dependency)
# --------------------------------------------------------------------------- #
def _load_dotenv() -> None:
    for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parent.parent / ".env"):
        if not candidate.is_file():
            continue
        for line in candidate.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_dotenv()

GEMBY_HOME = Path(os.environ.get("GEMBY_HOME", Path.home() / ".gemby"))
KEYS_FILE = GEMBY_HOME / "keys.json"

# Assets we actually need (skips the multi-GB *.jsonl training dumps).
MODEL_FILES = ["config.json", "tokenizer.json", "model.safetensors"]

# Special tokens used by the Gemby tokenizer.
ROLE_TOKENS = {"system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>"}


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
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def _need_deps(exc: ImportError):
    raise SystemExit(
        f"[gemby] missing dependency '{exc.name}'. Install the runtime with:\n"
        "        pip install -r requirements.txt"
    ) from exc


def _download_assets(model_id: str) -> Path:
    """Download config/tokenizer/weights into the HF cache; return the dir."""
    if not _hf_token():
        raise SystemExit(
            "[gemby] HF_TOKEN is not set. This model is gated: accept its license at\n"
            f"        https://huggingface.co/{model_id}\n"
            "        then add a Codespaces secret named HF_TOKEN, or put it in .env."
        )
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        _need_deps(exc)
    for filename in MODEL_FILES:
        hf_hub_download(model_id, filename, token=_hf_token())
    return Path(hf_hub_download(model_id, "config.json", token=_hf_token())).parent


def _remap_state_dict(state: dict) -> dict:
    """Map Gemby's custom tensor names onto standard Llama names."""
    out = {}
    for name, tensor in state.items():
        if name == "embed_tokens.weight":
            new = "model.embed_tokens.weight"
        elif name == "norm.weight":
            new = "model.norm.weight"
        elif name.startswith("layers."):
            new = ("model." + name
                   .replace(".attention.", ".self_attn.")
                   .replace(".norm1.", ".input_layernorm.")
                   .replace(".norm2.", ".post_attention_layernorm."))
        else:
            new = name
        out[new] = tensor
    return out


def _build_llama_config(cfg: dict):
    from transformers import LlamaConfig

    return LlamaConfig(
        vocab_size=cfg["vocab_size"],
        hidden_size=cfg["hidden_size"],
        intermediate_size=cfg["intermediate_size"],
        num_hidden_layers=cfg["num_hidden_layers"],
        num_attention_heads=cfg["num_attention_heads"],
        num_key_value_heads=cfg.get("num_key_value_heads", cfg["num_attention_heads"]),
        max_position_embeddings=cfg.get("max_position_embeddings", 4096),
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-5),
        tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        bos_token_id=cfg.get("bos_token_id"),
        eos_token_id=cfg.get("eos_token_id"),
        pad_token_id=cfg.get("pad_token_id"),
        attention_bias=False,
        mlp_bias=False,
    )


def _load_gemby(model_id: str, dtype_name: str, device: str):
    """Build a Llama model and load Gemby's custom-named weights into it."""
    import torch
    from safetensors.torch import load_file
    from transformers import LlamaForCausalLM

    try:
        from transformers import AutoConfig, PreTrainedTokenizerFast
    except ImportError as exc:
        _need_deps(exc)

    local_dir = _download_assets(model_id)
    cfg = json.loads((local_dir / "config.json").read_text())
    torch_dtype = _resolve_dtype(dtype_name, device)

    print(f"[gemby] building Llama architecture ({cfg['num_hidden_layers']} layers, "
          f"{cfg['hidden_size']} hidden) from custom config", file=sys.stderr)
    llama_cfg = _build_llama_config(cfg)
    try:
        model = LlamaForCausalLM._from_config(llama_cfg, torch_dtype=torch_dtype)
    except TypeError:
        try:
            model = LlamaForCausalLM._from_config(llama_cfg, dtype=torch_dtype)
        except TypeError:
            model = LlamaForCausalLM(llama_cfg)
    if next(model.parameters()).dtype != torch_dtype:
        model = model.to(torch_dtype)

    state = _remap_state_dict(load_file(str(local_dir / "model.safetensors")))
    if llama_cfg.tie_word_embeddings and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if unexpected:
        raise SystemExit(f"[gemby] unexpected weight keys: {unexpected[:5]}")
    real_missing = [m for m in missing if "lm_head" not in m and "rotary" not in m]
    if real_missing:
        print(f"[gemby] warning: {len(real_missing)} missing weights "
              f"(e.g. {real_missing[:3]})", file=sys.stderr)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(local_dir / "tokenizer.json"),
        bos_token="<|bos|>",
        eos_token="<|eos|>",
        unk_token="<|unk|>",
        pad_token="<|pad|>",
    )
    model.eval()
    if device == "cuda":
        model = model.to("cuda")
    return GembyRuntime(model, tokenizer, device, prompt_fn=_gemby_prompt)


def _load_generic(model_id: str, dtype_name: str, device: str, trust_remote_code: bool):
    """Fallback: standard HF load for any non-custom model type."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    token = _hf_token()
    torch_dtype = _resolve_dtype(dtype_name, device)
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=token, trust_remote_code=trust_remote_code
    )
    kwargs = dict(token=token, trust_remote_code=trust_remote_code,
                  low_cpu_mem_usage=True,
                  device_map=device if device == "cpu" else "auto")
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype, **kwargs)
    except TypeError:  # newer transformers renamed torch_dtype -> dtype
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype, **kwargs)
    model.eval()
    return GembyRuntime(model, tokenizer, device, prompt_fn=None)


def _gemby_prompt(messages: list[dict]) -> str:
    parts = []
    for m in messages:
        tag = ROLE_TOKENS.get(m.get("role", "user"), ROLE_TOKENS["user"])
        parts.append(f"{tag}{m.get('content', '')}")
    parts.append(ROLE_TOKENS["assistant"])
    return "".join(parts)


def _resolve_dtype(dtype: str, device: str):
    import torch

    if dtype == "auto":
        if device.startswith("cuda"):
            return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.float32
    return {"float32": torch.float32, "float16": torch.float16,
            "bfloat16": torch.bfloat16}[dtype]


class GembyRuntime:
    """Loaded model + tokenizer with a generate()/stream() interface."""

    def __init__(self, model, tokenizer, device: str, prompt_fn=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.prompt_fn = prompt_fn

    def _build_inputs(self, messages: list[dict]):
        tok = self.tokenizer
        text = None
        if self.prompt_fn is not None:
            text = self.prompt_fn(messages)
        elif getattr(tok, "chat_template", None):
            try:
                text = tok.apply_chat_template(messages, tokenize=False,
                                               add_generation_prompt=True)
            except Exception:
                text = None
        if text is None:
            text = "\n".join(f"{m.get('role','user')}: {m.get('content','')}"
                             for m in messages) + "\nassistant:"
        return tok(text, return_tensors="pt", add_special_tokens=False).to(self.device)

    def _gen_kwargs(self, max_new_tokens, temperature, top_p):
        do_sample = temperature is not None and temperature > 0
        kwargs = dict(
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
        )
        if do_sample:
            kwargs.update(temperature=temperature, top_p=top_p)
        return kwargs

    def generate(self, messages, max_new_tokens=512, temperature=0.7, top_p=0.95) -> str:
        import torch

        inputs = self._build_inputs(messages)
        with torch.no_grad():
            out = self.model.generate(**inputs,
                                      **self._gen_kwargs(max_new_tokens, temperature, top_p))
        new_tokens = out[0][inputs["input_ids"].shape[-1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    def stream(self, messages, max_new_tokens=512, temperature=0.7, top_p=0.95):
        import torch
        from transformers import TextIteratorStreamer

        inputs = self._build_inputs(messages)
        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)
        kwargs = dict(**inputs, **self._gen_kwargs(max_new_tokens, temperature, top_p),
                      streamer=streamer)
        thread = threading.Thread(target=self.model.generate, kwargs=kwargs)
        thread.start()
        for chunk in streamer:
            if chunk:
                yield chunk
        thread.join()


def load_runtime(model_id: str, dtype: str = "auto", device: str | None = None,
                 trust_remote_code: bool = True) -> GembyRuntime:
    try:
        import torch
        from transformers import AutoConfig
    except ImportError as exc:
        _need_deps(exc)

    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    token = _hf_token()
    if not token:
        raise SystemExit(
            "[gemby] HF_TOKEN is not set. This model is gated: accept its license at\n"
            f"        https://huggingface.co/{model_id}\n"
            "        then add a Codespaces secret named HF_TOKEN, or put it in .env."
        )
    try:
        hf_cfg = AutoConfig.from_pretrained(model_id, token=token,
                                            trust_remote_code=trust_remote_code)
        model_type = getattr(hf_cfg, "model_type", "")
    except Exception:
        model_type = ""

    print(f"[gemby] model_type={model_type or '?'} -> device={device} dtype={dtype}",
          file=sys.stderr)
    if model_type == "gemby":
        return _load_gemby(model_id, dtype, device)
    return _load_generic(model_id, dtype, device, trust_remote_code)


def pull_model(model_id: str) -> None:
    local_dir = _download_assets(model_id)
    total = sum((local_dir / f).stat().st_size for f in MODEL_FILES if (local_dir / f).exists())
    print(f"[gemby] cached at: {local_dir}  ({total / 1e9:.1f} GB)")


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_pull(args) -> int:
    pull_model(args.model)
    return 0


def cmd_chat(args) -> int:
    runtime = load_runtime(args.model, args.dtype, args.device)
    print("Gemby ready. Type 'exit' or Ctrl-D to quit.")
    history = []
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
        print("[gemby] no API key exists yet. Run `gemby key new` first.", file=sys.stderr)
        return 1

    runtime = load_runtime(args.model, args.dtype, args.device)

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
            return header.startswith("Bearer ") and key_verify(header[len("Bearer "):].strip())

        def log_message(self, fmt, *a):
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
                    data = {"object": "chat.completion.chunk", "created": created,
                            "model": args.model,
                            "choices": [{"index": 0, "delta": {"content": chunk}}]}
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
                "choices": [{"index": 0,
                             "message": {"role": "assistant", "content": text},
                             "finish_reason": "stop"}],
                "usage": {},
            })

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[gemby] serving {args.model} at http://{args.host}:{args.port}/v1 "
          "(Bearer auth required)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[gemby] stopped")
    return 0


def cmd_key(args) -> int:
    if args.key_command == "new":
        print(key_new(args.name))
        print("\nStore it now - it is only shown once.", file=sys.stderr)
    elif args.key_command == "list":
        entries = key_list()
        if not entries:
            print("(no keys)")
            return 0
        print(f"{'PREFIX':<16}{'NAME':<16}{'CREATED':<28}LAST USED")
        for e in entries:
            print(f"{e['prefix']:<16}{e['name']:<16}{e['created']:<28}{e.get('last_used') or '-'}")
    elif args.key_command == "revoke":
        print(f"revoked {key_revoke(args.key)} key(s)")
    return 0


def cmd_info(args) -> int:
    token_state = "set" if _hf_token() else "NOT set (gated model cannot be downloaded)"
    print(f"model id      : {args.model}")
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
    _load_dotenv()
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
