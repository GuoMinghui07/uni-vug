#!/usr/bin/env python3
"""Interactive multi-turn multimodal chat app for Interleaved UMM."""

from __future__ import annotations

import argparse
import copy
import mimetypes
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import gradio as gr
import numpy as np
from PIL import Image

from umm import ALLOWED_RESOLUTIONS, DEFAULT_CHECKPOINT, build_default_engine

IMAGE_EXTS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interleaved UMM interactive multimodal chat app.")
    parser.add_argument("--config", type=str, default="~/uni-vug/t2i_training_single_ds/config/dit_training.yaml")
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--latent-stats", type=str, default=None)
    parser.add_argument("--disable-latent-norm", action="store_true")
    parser.add_argument("--host", default="0.0.0.0", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share link.")
    parser.add_argument("--max-queue", type=int, default=8, help="Max queued requests in Gradio.")
    return parser.parse_args()


def _extract_paths(files: Any) -> List[str]:
    if files is None:
        return []
    if not isinstance(files, (list, tuple)):
        files = [files]

    paths: List[str] = []
    for item in files:
        if item is None:
            continue
        if isinstance(item, str):
            paths.append(item)
            continue
        if isinstance(item, dict):
            candidate = item.get("path") or item.get("name") or item.get("orig_name")
            if candidate:
                paths.append(str(candidate))
                continue
        candidate = getattr(item, "path", None)
        if candidate:
            paths.append(str(candidate))
            continue
        name = getattr(item, "name", None)
        if name:
            paths.append(str(name))
    return paths


def _infer_media_type(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if isinstance(mime, str) and mime.startswith("image/"):
        return "image"
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTS:
        return "image"
    return "unknown"


def _load_images(paths: Sequence[str]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for path in paths:
        media_type = _infer_media_type(path)
        if media_type != "image":
            raise ValueError(f"Unsupported file type (image only): {Path(path).name}")
        images.append(Image.open(path).convert("RGB"))
    return images


def _format_user_preview(user_text: str, paths: Sequence[str]) -> str:
    lines: List[str] = []
    text = (user_text or "").strip()
    if text:
        lines.append(text)
    for path in paths:
        lines.append(f"[image] {Path(path).name}")
    return "\n".join(lines).strip()


def _default_state() -> Dict[str, Any]:
    return {"history": [], "pending": None}


def _normalize_state(state: Any) -> Dict[str, Any]:
    if isinstance(state, dict):
        history = state.get("history")
        pending = state.get("pending")
        return {
            "history": list(history) if isinstance(history, list) else [],
            "pending": pending if isinstance(pending, dict) or pending is None else None,
        }
    return _default_state()


def _detect_chatbot_messages_mode(chatbot: Any) -> bool:
    """Detect whether this gr.Chatbot instance expects messages-format payloads."""
    try:
        chatbot.postprocess([{"role": "user", "content": "ping"}])
        return True
    except Exception:
        pass
    try:
        chatbot.postprocess([["ping", "pong"]])
        return False
    except Exception:
        pass
    desc = str(getattr(chatbot, "_value_description", "")).lower()
    if "tuple" in desc and "message" not in desc:
        return False
    return True


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        ctype = str(content.get("type", "")).strip().lower()
        if ctype == "text":
            return str(content.get("text", "") or "")
        if "text" in content:
            return str(content.get("text", "") or "")
        if "path" in content:
            return f"[file] {Path(str(content['path'])).name}"
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            part = _content_to_text(item).strip()
            if part:
                parts.append(part)
        return "\n".join(parts).strip()
    return str(content)


def _chatbot_to_pairs(chat_history: Any, messages_mode: bool) -> List[List[str]]:
    if not isinstance(chat_history, list):
        return []
    if not messages_mode:
        pairs: List[List[str]] = []
        for item in chat_history:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                pairs.append([_content_to_text(item[0]), _content_to_text(item[1])])
        return pairs

    pairs: List[List[str]] = []
    pending_user: Optional[str] = None
    for msg in chat_history:
        if not isinstance(msg, dict):
            continue
        role = str(msg.get("role", "")).strip().lower()
        content = _content_to_text(msg.get("content"))
        if role == "user":
            if pending_user is not None:
                pairs.append([pending_user, ""])
            pending_user = content
        elif role == "assistant":
            if pending_user is None:
                pairs.append(["", content])
            else:
                pairs.append([pending_user, content])
                pending_user = None
    if pending_user is not None:
        pairs.append([pending_user, ""])
    return pairs


def _pairs_to_chatbot(pairs: List[List[str]], messages_mode: bool) -> Any:
    if not messages_mode:
        return pairs
    messages: List[Dict[str, str]] = []
    for item in pairs:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        user_text_raw = _content_to_text(item[0])
        assistant_text_raw = _content_to_text(item[1])
        if user_text_raw.strip():
            messages.append({"role": "user", "content": user_text_raw})
        if assistant_text_raw.strip():
            messages.append({"role": "assistant", "content": assistant_text_raw})
    return messages


def _image_to_markdown(image: np.ndarray) -> str:
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    pil = Image.fromarray(image)
    import io
    import base64

    buffer = io.BytesIO()
    pil.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return (
        '<img src="data:image/png;base64,'
        f"{encoded}"
        '" style="max-width:100%; border-radius:12px; border:1px solid #d9dde7;" />'
    )


def _render_assistant_blocks(blocks: List[Dict[str, Any]]) -> str:
    rendered: List[str] = []
    for block in blocks:
        btype = str(block.get("type", "text"))
        if btype == "image_pending":
            txt = str(block.get("text", "")).strip()
            if txt:
                rendered.append(f"> {txt}")
            continue
        if btype == "image":
            md = str(block.get("markdown", "")).strip()
            if not md:
                img = block.get("image")
                if isinstance(img, np.ndarray):
                    md = _image_to_markdown(img)
            if md:
                rendered.append(md)
            continue
        text = str(block.get("text", ""))
        if text:
            rendered.append(text)
    return "\n\n".join(rendered).strip() or "..."


def add_user_turn(
    user_text: str,
    files: Any,
    chat_history: List[List[str]] | None,
    app_state: Dict[str, Any] | None,
):
    chat = copy.deepcopy(chat_history or [])
    state = _normalize_state(app_state)
    paths = _extract_paths(files)

    text = (user_text or "").strip()
    if not text and not paths:
        raise ValueError("Please enter text or upload at least one image.")

    preview = _format_user_preview(text, paths)
    chat.append([preview or "[empty]", "..."])
    state["pending"] = {"text": text, "paths": paths}

    status = f"Queued | {len(paths)} image(s)"
    return chat, state, "", None, status, ""


def generate_assistant_turn(
    chat_history: List[List[str]] | None,
    app_state: Dict[str, Any] | None,
    resolution: int,
    steps: int,
    sample_guidance_scale: float,
    seed: float,
    max_images: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    engine,
):
    chat = copy.deepcopy(chat_history or [])
    state = _normalize_state(app_state)
    pending = state.get("pending")

    if not isinstance(pending, dict):
        yield chat, state, "No queued user turn.", ""
        return
    if not chat:
        chat.append(["[empty]", "..."])
    if not isinstance(chat[-1], list) or len(chat[-1]) != 2:
        chat.append([str(chat[-1]), "..."])
    else:
        chat[-1][1] = "..."

    user_text = str(pending.get("text", "") or "")
    paths = list(pending.get("paths", []) or [])
    planner_raw = ""

    try:
        images = _load_images(paths)
    except Exception as exc:
        err = f"Error: {type(exc).__name__}: {exc}"
        chat[-1][1] = err
        state["pending"] = None
        yield chat, state, err, planner_raw
        return

    blocks: List[Dict[str, Any]] = []
    active_text_seg: Optional[int] = None
    active_text_idx: Optional[int] = None
    active_image_idx: Optional[int] = None

    start_t = time.time()
    status = "Generating..."
    yield chat, state, status, planner_raw

    try:
        for event in engine.respond_stream(
            user_text=user_text,
            user_images=images,
            history_state=state.get("history", []),
            resolution=int(resolution),
            num_steps=int(steps),
            sample_guidance_scale=float(sample_guidance_scale),
            seed=int(seed),
            max_images=int(max_images),
            max_new_tokens=int(max_new_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        ):
            evt = str(event.get("event", ""))

            if evt == "raw_delta":
                planner_raw = str(event.get("raw", planner_raw))
                continue

            elif evt == "plan_ready":
                planner_raw = str(event.get("planner_raw", ""))
                planned = event.get("planned_segments", []) or []
                n_text = sum(1 for seg in planned if isinstance(seg, dict) and seg.get("type") == "text")
                n_image = sum(1 for seg in planned if isinstance(seg, dict) and seg.get("type") == "image")
                status = f"Plan ready | text={n_text} image={n_image}"

            elif evt == "text_delta":
                seg_idx = int(event.get("segment_index", -1))
                partial = str(event.get("text", ""))
                is_final = bool(event.get("is_final", False))

                if active_text_seg != seg_idx:
                    active_text_seg = seg_idx
                    blocks.append({"type": "text", "text": partial})
                    active_text_idx = len(blocks) - 1
                elif active_text_idx is not None and 0 <= active_text_idx < len(blocks):
                    blocks[active_text_idx]["text"] = partial

                status = "Streaming text..."
                if is_final:
                    active_text_seg = None
                    active_text_idx = None

            elif evt == "image_start":
                prompt = str(event.get("prompt", ""))
                if len(prompt) > 90:
                    prompt = prompt[:90] + "..."
                blocks.append({"type": "image_pending", "text": f"🎨 Generating image: {prompt}"})
                active_image_idx = len(blocks) - 1
                status = "Rendering image..."

            elif evt == "image_done":
                segment = event.get("segment", {}) or {}
                img = segment.get("image")
                if isinstance(img, np.ndarray):
                    replacement = {"type": "image", "markdown": _image_to_markdown(img)}
                else:
                    replacement = {"type": "text", "text": "[image result missing]"}
                if active_image_idx is not None and 0 <= active_image_idx < len(blocks):
                    blocks[active_image_idx] = replacement
                else:
                    blocks.append(replacement)
                active_image_idx = None
                status = "Image generated"

            elif evt == "image_error":
                err = str(event.get("error", "[image generation error]"))
                replacement = {"type": "text", "text": err}
                if active_image_idx is not None and 0 <= active_image_idx < len(blocks):
                    blocks[active_image_idx] = replacement
                else:
                    blocks.append(replacement)
                active_image_idx = None
                status = "Image generation failed"

            elif evt == "final":
                state["history"] = list(event.get("history", []))
                state["pending"] = None
                planner_raw = str(event.get("planner_raw", planner_raw))
                elapsed = time.time() - start_t
                status = f"Done | {elapsed:.1f}s"

            chat[-1][1] = _render_assistant_blocks(blocks)
            yield chat, state, status, planner_raw

    except Exception as exc:
        err = f"Error: {type(exc).__name__}: {exc}"
        chat[-1][1] = err
        state["pending"] = None
        yield chat, state, err, planner_raw


def clear_conversation():
    return [], _default_state(), "", None, "Conversation cleared.", ""


def build_ui(engine) -> gr.Blocks:
    css = """
    .app-wrap {max-width: 1280px; margin: 0 auto;}
    .chat-wrap {border: 1px solid #d9dde7; border-radius: 16px; overflow: hidden;}
    #status-box {min-height: 40px;}
    """

    with gr.Blocks() as demo:
        with gr.Column(elem_classes=["app-wrap"]):
            gr.Markdown("## Interleaved UMM Chat")
            gr.Markdown("Uses qwen3_app-style UI and streaming pipeline. Supports text + multi-image input.")

            with gr.Row():
                with gr.Column(scale=4):
                    chatbot = gr.Chatbot(
                        label="Conversation",
                        height=620,
                        elem_classes=["chat-wrap"],
                    )
                    chatbot_uses_messages = _detect_chatbot_messages_mode(chatbot)
                    print(f"[umm_app] Chatbot format detected: {'messages' if chatbot_uses_messages else 'legacy-tuples'}")
                    user_text = gr.Textbox(
                        label="Message",
                        placeholder="Type a message (Enter to send). You can upload multiple images below.",
                        lines=1,
                        max_lines=8,
                    )
                    uploads = gr.Files(
                        label="Attachments (multi-image)",
                        file_count="multiple",
                        file_types=["image"],
                    )
                    with gr.Row():
                        send_btn = gr.Button("Send", variant="primary")
                        clear_btn = gr.Button("Clear")

                with gr.Column(scale=2):
                    resolution = gr.Dropdown(
                        choices=list(ALLOWED_RESOLUTIONS),
                        value=448,
                        label="Resolution",
                        allow_custom_value=False,
                    )
                    steps = gr.Slider(label="FM Steps", minimum=1, maximum=50, step=1, value=10)
                    sample_guidance_scale = gr.Slider(
                        label="Sample Guidance Scale",
                        minimum=0.0,
                        maximum=10.0,
                        step=0.1,
                        value=7.0,
                    )
                    seed = gr.Number(label="Seed (-1=random)", value=-1, precision=0)
                    max_images = gr.Slider(label="Max Images / Turn", minimum=0, maximum=10, step=1, value=2)
                    max_new_tokens = gr.Slider(label="Planner Max New Tokens", minimum=64, maximum=1024, step=32, value=512)
                    temperature = gr.Slider(label="Planner Temperature", minimum=0.0, maximum=1.5, step=0.05, value=0.25)
                    top_p = gr.Slider(label="Planner Top-p", minimum=0.05, maximum=1.0, step=0.05, value=0.9)
                    status = gr.Markdown("Ready.", elem_id="status-box")
                    planner_debug = gr.Textbox(label="Raw Model Stream", lines=10, interactive=False)

            app_state = gr.State(_default_state())

            def add_user_turn_ui(
                user_text: str,
                files: Any,
                chatbot_history: Any,
                app_state_in: Dict[str, Any] | None,
            ):
                pairs = _chatbot_to_pairs(chatbot_history, chatbot_uses_messages)
                out_chat, out_state, out_text, out_files, out_status, out_planner = add_user_turn(
                    user_text=user_text,
                    files=files,
                    chat_history=pairs,
                    app_state=app_state_in,
                )
                return _pairs_to_chatbot(out_chat, chatbot_uses_messages), out_state, out_text, out_files, out_status, out_planner

            def generate_assistant_turn_ui(
                chatbot_history: Any,
                app_state_in: Dict[str, Any] | None,
                resolution: int,
                steps: int,
                sample_guidance_scale: float,
                seed: float,
                max_images: int,
                max_new_tokens: int,
                temperature: float,
                top_p: float,
            ) -> Iterator[Tuple[Any, Dict[str, Any], str, str]]:
                pairs = _chatbot_to_pairs(chatbot_history, chatbot_uses_messages)
                for out_chat, out_state, out_status, out_planner in generate_assistant_turn(
                    chat_history=pairs,
                    app_state=app_state_in,
                    resolution=resolution,
                    steps=steps,
                    sample_guidance_scale=sample_guidance_scale,
                    seed=seed,
                    max_images=max_images,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    engine=engine,
                ):
                    yield _pairs_to_chatbot(out_chat, chatbot_uses_messages), out_state, out_status, out_planner

            submit_event = send_btn.click(
                fn=add_user_turn_ui,
                inputs=[user_text, uploads, chatbot, app_state],
                outputs=[chatbot, app_state, user_text, uploads, status, planner_debug],
                queue=False,
            )
            submit_event.then(
                fn=generate_assistant_turn_ui,
                inputs=[
                    chatbot,
                    app_state,
                    resolution,
                    steps,
                    sample_guidance_scale,
                    seed,
                    max_images,
                    max_new_tokens,
                    temperature,
                    top_p,
                ],
                outputs=[chatbot, app_state, status, planner_debug],
            )

            enter_event = user_text.submit(
                fn=add_user_turn_ui,
                inputs=[user_text, uploads, chatbot, app_state],
                outputs=[chatbot, app_state, user_text, uploads, status, planner_debug],
                queue=False,
            )
            enter_event.then(
                fn=generate_assistant_turn_ui,
                inputs=[
                    chatbot,
                    app_state,
                    resolution,
                    steps,
                    sample_guidance_scale,
                    seed,
                    max_images,
                    max_new_tokens,
                    temperature,
                    top_p,
                ],
                outputs=[chatbot, app_state, status, planner_debug],
            )

            clear_btn.click(
                fn=clear_conversation,
                inputs=None,
                outputs=[chatbot, app_state, user_text, uploads, status, planner_debug],
                queue=False,
            )

    setattr(demo, "_app_css", css)
    return demo


def resolve_config_path(config_arg: str) -> Path:
    raw = Path(config_arg).expanduser()
    candidates = []

    # 1) Exact path as given (with "~" resolved).
    candidates.append(raw)

    # 2) If relative, try from the current working directory explicitly.
    if not raw.is_absolute():
        candidates.append(Path.cwd() / raw)

    # 3) Common repo config locations by basename.
    repo_root = Path(__file__).resolve().parent.parent
    candidates.append(repo_root / "t2i_training_single_ds" / "config" / raw.name)
    candidates.append(repo_root / "training" / "config" / raw.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = "', '".join(str(p) for p in candidates)
    raise FileNotFoundError(f"Config not found: '{config_arg}'. Tried: '{tried}'.")


def main() -> None:
    args = parse_args()
    config_path = resolve_config_path(args.config)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: '{checkpoint}'.")

    engine = build_default_engine(
        config_path=str(config_path),
        checkpoint_path=str(checkpoint),
        device=args.device,
        precision=args.precision,
        sample_steps=args.sample_steps,
        latent_stats=args.latent_stats,
        disable_latent_norm=args.disable_latent_norm,
    )

    demo = build_ui(engine)
    launch_kwargs = {
        "server_name": args.host,
        "server_port": int(args.port),
        "share": bool(args.share),
        "show_error": True,
        "theme": gr.themes.Soft(),
        "css": getattr(demo, "_app_css", None),
    }
    demo.queue(max_size=int(args.max_queue), default_concurrency_limit=1).launch(**launch_kwargs)


if __name__ == "__main__":
    main()
