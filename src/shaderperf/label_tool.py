"""Local, blinded web UI for labeling perceptual-calibration pairs."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PAIR_ID_RE = re.compile(r"^[0-9a-f]{10}$")
VALID_LABELS = {"acceptable", "unacceptable"}


def load_labels(path: Path) -> dict:
    """Read a label mapping, treating a missing file as an empty store."""

    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("labels.json must contain an object")
    return data


def write_labels_atomic(path: Path, labels: dict) -> None:
    """Atomically replace a label store using a temp file beside its target."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_name = handle.name
            json.dump(labels, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        if temp_name is not None:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
        raise


def update_label_atomic(path: Path, pair_id: str, label: str, utc: str) -> dict:
    """Merge one label into the current store and persist it atomically."""

    if not PAIR_ID_RE.fullmatch(pair_id):
        raise ValueError("invalid pair id")
    if label not in VALID_LABELS:
        raise ValueError("invalid label")
    labels = load_labels(path)
    labels[pair_id] = {"label": label, "utc": utc}
    write_labels_atomic(path, labels)
    return labels


def load_pair_order(directory: Path) -> list[str]:
    """Load only opaque IDs from manifest order; ignore all pair metadata."""

    manifest = json.loads((directory / "manifest.json").read_text())
    pairs = manifest.get("pairs")
    if not isinstance(pairs, list):
        raise ValueError("manifest.json must contain a pairs list")
    pair_ids = [str(pair["pair_id"]) for pair in pairs]
    if not pair_ids or any(not PAIR_ID_RE.fullmatch(value) for value in pair_ids):
        raise ValueError("manifest contains no pairs or an invalid pair id")
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("manifest contains duplicate pair ids")
    return pair_ids


def left_is_reference(pair_id: str) -> bool:
    """Choose sides deterministically from the opaque id."""

    digest = hashlib.sha256(pair_id.encode("ascii")).digest()
    return digest[0] % 2 == 0


def _next_unlabeled(pair_ids: list[str], labels: dict, start: int):
    for index in range(max(0, start), len(pair_ids)):
        if pair_ids[index] not in labels:
            return index, pair_ids[index]
    return None


def _page(pair_id: str, index: int, total: int, labeled: int) -> bytes:
    safe_pair_id = html.escape(pair_id)
    left_url = f"/image/{safe_pair_id}/left"
    right_url = f"/image/{safe_pair_id}/right"
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Perceptual calibration</title>
<style>
  :root {{ color-scheme: dark; font-family: system-ui, sans-serif; }}
  body {{ margin: 0 auto; max-width: 1500px; padding: 20px; background: #111; color: #eee; }}
  header, .actions {{ display: flex; gap: 12px; align-items: center; justify-content: center; flex-wrap: wrap; }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 18px 0; }}
  img {{ display: block; width: 100%; height: auto; image-rendering: auto; }}
  .overlay {{ position: relative; width: min(70vw, 800px); margin: 12px auto; aspect-ratio: 1; }}
  .overlay img {{ position: absolute; inset: 0; }}
  #overlay-right {{ opacity: 0; }}
  button {{ border: 1px solid #777; border-radius: 7px; padding: 11px 18px; font-size: 16px; cursor: pointer; }}
  .acceptable {{ background: #1f7a42; color: white; }}
  .unacceptable {{ background: #963b3b; color: white; }}
  .skip {{ background: #444; color: white; }}
  @media (max-width: 800px) {{ .pair {{ grid-template-columns: 1fr; }} }}
</style>
</head>
<body>
<header>
  <span>Pair {index + 1} of {total}</span>
  <span>Labeled {labeled}; remaining {total - labeled}</span>
  <span>Shortcuts: A / U / S</span>
</header>
<div class="pair">
  <img src="{left_url}" alt="Image A">
  <img src="{right_url}" alt="Image B">
</div>
<div class="actions"><button id="flicker" type="button">Start flicker</button></div>
<div class="overlay">
  <img src="{left_url}" alt="Overlay image A">
  <img id="overlay-right" src="{right_url}" alt="Overlay image B">
</div>
<div class="actions">
  <button class="acceptable" data-action="acceptable">Acceptable (A)</button>
  <button class="unacceptable" data-action="unacceptable">Unacceptable (U)</button>
  <button class="skip" data-action="skip">Skip (S)</button>
</div>
<script>
(() => {{
  const pairId = {json.dumps(pair_id)};
  const pairIndex = {index};
  const overlay = document.getElementById('overlay-right');
  const flickerButton = document.getElementById('flicker');
  let timer = null;
  let showingRight = false;
  flickerButton.addEventListener('click', () => {{
    if (timer !== null) {{
      clearInterval(timer); timer = null; overlay.style.opacity = '0';
      flickerButton.textContent = 'Start flicker'; return;
    }}
    timer = setInterval(() => {{
      showingRight = !showingRight;
      overlay.style.opacity = showingRight ? '1' : '0';
    }}, 350);
    flickerButton.textContent = 'Stop flicker';
  }});
  async function submit(action) {{
    const body = new URLSearchParams({{pair_id: pairId, index: String(pairIndex), action}});
    const response = await fetch('/label', {{method: 'POST', body}});
    if (!response.ok) {{ alert(await response.text()); return; }}
    const result = await response.json();
    location.href = result.next;
  }}
  document.querySelectorAll('[data-action]').forEach(button =>
    button.addEventListener('click', () => submit(button.dataset.action)));
  document.addEventListener('keydown', event => {{
    if (event.repeat) return;
    const key = event.key.toLowerCase();
    if (key === 'a') submit('acceptable');
    if (key === 'u') submit('unacceptable');
    if (key === 's') submit('skip');
  }});
}})();
</script>
</body>
</html>
"""
    return document.encode("utf-8")


def _finished_page(total: int, labeled: int) -> bytes:
    return (
        "<!doctype html><meta charset='utf-8'><title>Calibration complete</title>"
        "<body style='font-family:system-ui;text-align:center;padding:4rem'>"
        f"<h1>Pass complete</h1><p>Labeled {labeled} of {total} pairs.</p>"
        "<p>Reload from the beginning to revisit skipped pairs.</p></body>"
    ).encode("utf-8")


def make_handler(directory: Path, pair_ids: list[str]):
    labels_path = directory / "labels.json"

    class LabelHandler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, content_type: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                query = parse_qs(parsed.query)
                try:
                    start = int(query.get("index", ["0"])[0])
                    labels = load_labels(labels_path)
                except (ValueError, OSError, json.JSONDecodeError) as exc:
                    self._send(HTTPStatus.BAD_REQUEST, str(exc).encode(), "text/plain")
                    return
                item = _next_unlabeled(pair_ids, labels, start)
                labeled_count = sum(pair_id in labels for pair_id in pair_ids)
                if item is None:
                    body = _finished_page(len(pair_ids), labeled_count)
                else:
                    index, pair_id = item
                    body = _page(pair_id, index, len(pair_ids), labeled_count)
                self._send(HTTPStatus.OK, body, "text/html; charset=utf-8")
                return

            match = re.fullmatch(r"/image/([0-9a-f]{10})/(left|right)", parsed.path)
            if match is not None and match.group(1) in pair_ids:
                pair_id, side = match.groups()
                left_ref = left_is_reference(pair_id)
                wants_ref = left_ref if side == "left" else not left_ref
                suffix = "ref" if wants_ref else "var"
                image_path = directory / f"{pair_id}_{suffix}.png"
                try:
                    body = image_path.read_bytes()
                except OSError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send(HTTPStatus.OK, body, "image/png")
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/label":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                fields = parse_qs(self.rfile.read(length).decode("utf-8"))
                pair_id = fields["pair_id"][0]
                index = int(fields["index"][0])
                action = fields["action"][0]
                if index < 0 or index >= len(pair_ids) or pair_ids[index] != pair_id:
                    raise ValueError("pair and index do not match manifest order")
                if action in VALID_LABELS:
                    utc = dt.datetime.now(dt.timezone.utc).isoformat().replace(
                        "+00:00", "Z"
                    )
                    update_label_atomic(labels_path, pair_id, action, utc)
                elif action != "skip":
                    raise ValueError("invalid action")
            except (
                KeyError,
                IndexError,
                UnicodeError,
                ValueError,
                OSError,
                json.JSONDecodeError,
            ) as exc:
                self._send(HTTPStatus.BAD_REQUEST, str(exc).encode(), "text/plain")
                return
            body = json.dumps({"next": f"/?index={index + 1}"}).encode("utf-8")
            self._send(HTTPStatus.OK, body, "application/json")

        def log_message(self, format: str, *args) -> None:
            print(f"{self.address_string()} - {format % args}")

    return LabelHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("results/calibration"),
        help="directory containing manifest and pair PNGs",
    )
    parser.add_argument("--port", type=int, default=8765)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("error: --port must be between 1 and 65535")
    try:
        pair_ids = load_pair_order(args.dir)
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(f"error: {exc}") from exc
    server = HTTPServer(("127.0.0.1", args.port), make_handler(args.dir, pair_ids))
    print(f"labeling {len(pair_ids)} pairs at http://127.0.0.1:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
