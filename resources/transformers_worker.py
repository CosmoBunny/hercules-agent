#!/usr/bin/env python3
"""Hercules Transformers worker: isolated SafeTensors/PyTorch inference.

Protocol: JSON Lines on stdout (machine-readable ONLY — all logging goes
to stderr). One JSON object per line, with a versioned handshake.

Modes:
    worker.py --model-path DIR --device auto|cpu|cuda|mps
        Serve mode: load model, print ready, serve generate requests.
    worker.py --check-deps
        Print dependency/device JSON and exit (Rust availability probe).

Cancellation: Rust sends {"type":"cancel","request_id":...}; the
generation thread observes a StoppingCriteria event per token. If the
worker ignores cancellation, Rust kills the process group (hard path).
"""

import argparse
import json
import os
import sys
import threading

PROTOCOL_VERSION = 1


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def check_deps():
    info = {"type": "deps", "protocol_version": PROTOCOL_VERSION}
    try:
        import transformers

        info["transformers"] = getattr(transformers, "__version__", "unknown")
    except Exception as e:
        info["transformers"] = None
        info["transformers_error"] = str(e)
    try:
        import torch

        info["torch"] = getattr(torch, "__version__", "unknown")
        info["cuda_available"] = bool(torch.cuda.is_available())
        try:
            info["cuda_device_count"] = int(torch.cuda.device_count())
        except Exception:
            info["cuda_device_count"] = 0
        info["mps_available"] = bool(
            getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        )
    except Exception as e:
        info["torch"] = None
        info["torch_error"] = str(e)
        info["cuda_available"] = False
        info["cuda_device_count"] = 0
        info["mps_available"] = False
    emit(info)


def resolve_device(requested):
    import torch

    req = (requested or "auto").lower()
    if req == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        return "cuda"
    if req == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError("MPS requested but unavailable")
        return "mps"
    if req == "cpu":
        return "cpu"
    # auto
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default=None)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--check-deps", action="store_true")
    args = ap.parse_args()

    if args.check_deps:
        check_deps()
        return 0

    if not args.model_path or not os.path.isdir(args.model_path):
        emit({"type": "error", "code": "model_not_found",
              "message": "model directory missing: %r" % (args.model_path,)})
        return 2

    # Handshake FIRST (protocol version gates everything), then the slow
    # load. Rust's timeout therefore covers version mismatch fast and
    # model loading explicitly, not opaquely.
    emit({"type": "hello", "protocol_version": PROTOCOL_VERSION})

    try:
        import torch
        from transformers import (AutoConfig, AutoModelForCausalLM,
                                  AutoTokenizer, TextIteratorStreamer)
        from transformers.generation.stopping_criteria import StoppingCriteria
    except Exception as e:
        emit({"type": "error", "code": "dependency_missing",
              "message": "transformers/torch import failed: %s" % e})
        return 3

    try:
        device = resolve_device(args.device)
    except Exception as e:
        emit({"type": "error", "code": "device_unavailable", "message": str(e)})
        return 4

    try:
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=False)
        architecture = (getattr(config, "architectures", None) or [None])[0]
        model_type = getattr(config, "model_type", None)
        if not architecture:
            architecture = model_type or "unknown"
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
        dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path, torch_dtype=dtype, trust_remote_code=False)
        model.to(device)
        model.eval()
    except Exception as e:
        emit({"type": "error", "code": "load_failed",
              "message": "model load failed: %s" % e})
        return 5

    try:
        dev_name = torch.cuda.get_device_name(0) if device == "cuda" else device
    except Exception:
        dev_name = device

    emit({"type": "ready", "protocol_version": PROTOCOL_VERSION,
          "architecture": architecture, "model_type": model_type,
          "device": device, "device_name": str(dev_name)})

    cancel_events = {}
    cancel_lock = threading.Lock()

    class CancelCriteria(StoppingCriteria):
        def __init__(self, event):
            self.event = event

        def __call__(self, input_ids, scores, **kwargs):
            return self.event.is_set()

    def stream_tokens_from_generate(req):
        # Run blocking generate() on a thread while the main thread pumps
        # streamer text AND watches stdin for cancel messages.
        import queue
        import select

        rid = req.get("request_id", "")
        token_q = queue.Queue()
        with cancel_lock:
            ev = cancel_events.setdefault(rid, threading.Event())
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                        skip_special_tokens=True)

        def run_gen():
            try:
                prompt = req.get("prompt", "")
                max_new = int(req.get("max_new_tokens", 128) or 128)
                inputs = tokenizer(prompt, return_tensors="pt").to(device)
                gen_kwargs = dict(inputs, streamer=streamer,
                                  max_new_tokens=max_new,
                                  stopping_criteria=[CancelCriteria(ev)],
                                  pad_token_id=tokenizer.eos_token_id)
                # Temperature is opt-in: absent means greedy (do_sample off).
                try:
                    temp = req.get("temperature", None)
                    if temp is not None:
                        gen_kwargs["temperature"] = float(temp)
                        gen_kwargs["do_sample"] = True
                except Exception:
                    pass
                model.generate(**gen_kwargs)
                token_q.put(("gen_done", None))
            except Exception as e:
                token_q.put(("gen_error", str(e)))

        def pump():
            try:
                for text in streamer:
                    token_q.put(("token", text))
            except Exception as e:
                token_q.put(("pump_error", str(e)))
                return
            token_q.put(("pump_done", None))

        threading.Thread(target=run_gen, daemon=True).start()
        threading.Thread(target=pump, daemon=True).start()
        while True:
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if rlist:
                line = sys.stdin.readline()
                if not line:
                    break
                try:
                    msg = json.loads(line)
                except Exception:
                    continue
                if (msg.get("type") == "cancel"
                        and msg.get("request_id", "") in (rid, "")):
                    ev.set()
            try:
                kind, payload = token_q.get(timeout=0.05)
            except Exception:
                continue
            if kind == "token":
                emit({"type": "token", "request_id": rid, "text": payload})
            elif kind == "gen_error":
                msg = str(payload)
                code = "out_of_memory" if "out of memory" in msg.lower() \
                    else "generation_failed"
                emit({"type": "error", "request_id": rid, "code": code,
                      "message": msg})
                return
            elif kind == "pump_error":
                emit({"type": "error", "request_id": rid,
                      "code": "generation_failed",
                      "message": "streamer failed: %s" % payload})
                return
            elif kind == "pump_done":
                # Streamer exhausted: generation fully delivered.
                emit({"type": "cancelled" if ev.is_set() else "done",
                      "request_id": rid})
                return
            # "gen_done" alone doesn't terminate: trailing streamer text
            # still drains through pump_done.

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except Exception:
            emit({"type": "error", "code": "protocol_error",
                  "message": "malformed JSON from host"})
            continue
        mtype = msg.get("type")
        if mtype == "generate":
            stream_tokens_from_generate(msg)
            with cancel_lock:
                cancel_events.pop(msg.get("request_id", ""), None)
        elif mtype == "cancel":
            rid = msg.get("request_id", "")
            with cancel_lock:
                if rid in cancel_events:
                    cancel_events[rid].set()
                else:
                    # Unknown/stale request: acknowledge so Rust never hangs.
                    emit({"type": "cancelled", "request_id": rid})
        elif mtype == "shutdown":
            emit({"type": "bye", "protocol_version": PROTOCOL_VERSION})
            return 0
        else:
            emit({"type": "error", "code": "protocol_error",
                  "message": "unknown message type: %r" % (mtype,)})
    return 0


if __name__ == "__main__":
    sys.exit(main())
