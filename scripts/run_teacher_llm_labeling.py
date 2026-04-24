#!/usr/bin/env python
"""
Run teacher LLM/VLM labeling on nuScenes CoC teacher requests.

Reads teacher_requests.jsonl, calls a real LLM/VLM API (OpenAI-compatible),
and writes teacher_responses.jsonl for downstream assembly.

Supports:
  - 阿里百炼 Qwen-VL-Max（推荐，支持图像，对齐官方 VLM 标注）
  - OpenAI API (gpt-4o, gpt-4-turbo, etc.)
  - DeepSeek / 智谱 / Kimi（文本模式，无图像）
  - Local Ollama  (--base-url http://localhost:11434/v1 --model qwen2.5:7b)

Usage examples:
  # ★ 推荐：阿里百炼 qwen-vl-max（VLM，传图像）
  python scripts/run_teacher_llm_labeling.py \
    --requests-input out/nuscenes_coc_teacher_requests_full.jsonl \
    --responses-output out/nuscenes_coc_teacher_responses.jsonl \
    --base-url https://dashscope.aliyuncs.com/compatible-mode/v1 \
    --api-key sk-你的key \
    --model qwen-vl-max \
    --with-images \
    --cameras CAM_FRONT CAM_FRONT_LEFT CAM_FRONT_RIGHT \
    --image-frames -2 -1 0

  # 纯文本模式（DeepSeek，无图像，便宜但精度低）
  python scripts/run_teacher_llm_labeling.py \
    --requests-input out/nuscenes_coc_teacher_requests_full.jsonl \
    --responses-output out/nuscenes_coc_teacher_responses.jsonl \
    --base-url https://api.deepseek.com \
    --api-key sk-你的key \
    --model deepseek-chat

  # OpenAI gpt-4o（VLM，传图像）
  python scripts/run_teacher_llm_labeling.py \
    --requests-input out/nuscenes_coc_teacher_requests_full.jsonl \
    --responses-output out/nuscenes_coc_teacher_responses.jsonl \
    --api-key sk-... \
    --model gpt-4o \
    --with-images

  # After running, assemble final dataset:
  python scripts/build_teacher_labeling_assets.py \
    --input out/nuscenes_coc_teacher_input_full.json \
    --requests-output out/nuscenes_coc_teacher_requests_full.jsonl \
    --teacher-responses out/nuscenes_coc_teacher_responses.jsonl \
    --final-output out/nuscenes_coc_final_teacher.json
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nuscenes_coc.exporter import read_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run teacher LLM labeling via OpenAI-compatible API.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--requests-input", required=True,
        help="Path to teacher_requests.jsonl",
    )
    parser.add_argument(
        "--responses-output", required=True,
        help="Path to write teacher_responses.jsonl",
    )
    parser.add_argument(
        "--api-key", default=None,
        help="API key (or set OPENAI_API_KEY env var). Use 'ollama' for local Ollama.",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="Custom API base URL. Default: OpenAI official. "
             "For Ollama: http://localhost:11434/v1",
    )
    parser.add_argument(
        "--model", default="gpt-4o",
        help="Model name (default: gpt-4o). For Ollama e.g. qwen2.5:7b",
    )
    parser.add_argument(
        "--max-samples", type=int, default=-1,
        help="Process at most N samples (-1 = all)",
    )
    parser.add_argument(
        "--retry", type=int, default=3,
        help="Number of retries on API error (default: 3)",
    )
    parser.add_argument(
        "--retry-delay", type=float, default=5.0,
        help="Seconds to wait between retries (default: 5.0)",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Skip already processed request_ids in responses-output",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print first request prompt and exit without calling API",
    )
    # ── VLM 图像参数 ──────────────────────────────────────────────────────
    parser.add_argument(
        "--with-images", action="store_true",
        help="传入摄像头图像（VLM 模式，推荐 qwen-vl-max / gpt-4o）",
    )
    parser.add_argument(
        "--cameras",
        nargs="+",
        default=["CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT"],
        metavar="CAM",
        help="要传入的摄像头列表（默认：前3路）。可选：CAM_BACK CAM_BACK_LEFT CAM_BACK_RIGHT",
    )
    parser.add_argument(
        "--image-frames",
        nargs="+",
        type=int,
        default=[-2, -1, 0],
        metavar="IDX",
        help="要传入的帧 relative_index（默认：-2 -1 0 即最近3帧）",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _encode_image_base64(image_path: str) -> Optional[str]:
    """将本地图像文件编码为 base64 字符串，失败返回 None。"""
    try:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    except Exception as e:
        print(f"  [WARN] Cannot read image {image_path}: {e}")
        return None


def _select_image_assets(request: Dict, cameras: List[str], frame_indices: List[int]) -> List[Dict]:
    """从 image_assets 中筛选指定帧+摄像头。"""
    assets = request.get("image_assets", [])
    selected = []
    for asset in assets:
        if asset["relative_index"] in frame_indices and asset["camera_name"] in cameras:
            selected.append(asset)
    # 按 (relative_index, camera_name) 排序
    selected.sort(key=lambda a: (a["relative_index"], cameras.index(a["camera_name"])
                                  if a["camera_name"] in cameras else 99))
    return selected


# ---------------------------------------------------------------------------
# Prompt building (text-only, no images)
# ---------------------------------------------------------------------------

def _build_text_messages(request: Dict) -> List[Dict]:
    """
    Build OpenAI messages list from a teacher request.
    Uses only structured_context (text), ignores image_assets.
    """
    messages: List[Dict] = []

    # System message
    for msg in request["messages"]:
        if msg["role"] == "system":
            messages.append({"role": "system", "content": msg["content"]})
            break

    # User message: combine text + structured_context JSON
    user_content_parts: List[str] = []
    for msg in request["messages"]:
        if msg["role"] == "user":
            for part in msg["content"]:
                if part["type"] == "text":
                    user_content_parts.append(part["text"])
                elif part["type"] == "structured_context":
                    ctx_json = json.dumps(part["data"], ensure_ascii=False, indent=2)
                    user_content_parts.append(
                        f"\n以下是结构化场景上下文（JSON）：\n```json\n{ctx_json}\n```"
                    )
            break

    # Append weak label hint to help the model
    hint = request.get("weak_label_hint")
    if hint:
        hint_json = json.dumps(hint, ensure_ascii=False, indent=2)
        user_content_parts.append(
            f"\n以下是规则生成的弱标签（仅供参考，可以修正）：\n```json\n{hint_json}\n```"
        )

    messages.append({"role": "user", "content": "\n".join(user_content_parts)})
    return messages


def _build_vlm_messages(
    request: Dict,
    cameras: List[str],
    frame_indices: List[int],
) -> List[Dict]:
    """
    构建 VLM（视觉语言模型）消息列表，图像以 base64 内嵌方式传入。
    兼容 qwen-vl-max / gpt-4o 的 image_url 格式。
    """
    messages: List[Dict] = []

    # System message
    for msg in request["messages"]:
        if msg["role"] == "system":
            messages.append({"role": "system", "content": msg["content"]})
            break

    # 收集图像资产
    selected_assets = _select_image_assets(request, cameras, frame_indices)

    user_parts = []

    # 1. 文字指令
    for msg in request["messages"]:
        if msg["role"] == "user":
            for part in msg["content"]:
                if part["type"] == "text":
                    user_parts.append({"type": "text", "text": part["text"]})
            break

    # 2. 图像块（按帧分组，加帧标签）
    current_frame = None
    for asset in selected_assets:
        if asset["relative_index"] != current_frame:
            current_frame = asset["relative_index"]
            label = "当前帧" if current_frame == 0 else f"历史帧 t{current_frame:+d}"
            user_parts.append({"type": "text", "text": f"\n【{label}】"})

        b64 = _encode_image_base64(asset["path"])
        if b64:
            cam_label = asset["camera_name"].replace("CAM_", "").replace("_", " ").title()
            user_parts.append({"type": "text", "text": f"[{cam_label}]"})
            user_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })

    # 3. 结构化上下文
    for msg in request["messages"]:
        if msg["role"] == "user":
            for part in msg["content"]:
                if part["type"] == "structured_context":
                    ctx_json = json.dumps(part["data"], ensure_ascii=False, indent=2)
                    user_parts.append({
                        "type": "text",
                        "text": f"\n以下是结构化场景上下文（JSON）：\n```json\n{ctx_json}\n```",
                    })
            break

    # 4. 弱标签提示
    hint = request.get("weak_label_hint")
    if hint:
        hint_json = json.dumps(hint, ensure_ascii=False, indent=2)
        user_parts.append({
            "type": "text",
            "text": f"\n以下是规则生成的弱标签（仅供参考，可以修正）：\n```json\n{hint_json}\n```",
        })

    if not selected_assets:
        print("  [WARN] No images found/encoded, falling back to text-only mode")
        return _build_text_messages(request)

    messages.append({"role": "user", "content": user_parts})
    return messages


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def _call_llm(
    client,
    model: str,
    messages: List[Dict],
    temperature: float,
    response_schema: Optional[Dict],
) -> str:
    """Call OpenAI-compatible API and return raw response text."""
    kwargs: Dict = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    # Use json_object mode if schema provided (supported by most APIs)
    if response_schema is not None:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


def _parse_response(raw: str, request_id: str) -> Optional[Dict]:
    """Try to parse LLM output as JSON. Strip markdown fences if present."""
    text = raw.strip()
    # Strip ```json ... ``` or ``` ... ```
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON parse failed for {request_id}: {e}")
        return None


# Known VLM aliases → canonical CoC categories
# These are terms qwen-vl-max uses naturally that must be remapped.
_LONGITUDINAL_ALIASES: Dict[str, str] = {
    "gentle_accelerate": "set_speed_tracking",
    "gentle_decelerate": "lead_obstacle_following",
    "maintain_speed": "set_speed_tracking",
    "cruise": "set_speed_tracking",
    "accelerate": "set_speed_tracking",
    "decelerate": "lead_obstacle_following",
    "brake": "stop_static_constraint",
}
_LATERAL_ALIASES: Dict[str, str] = {
    "go_straight": "lane_keeping_centering",
    "keep_lane": "lane_keeping_centering",
    "steer_left": "in_lane_nudge_left",
    "steer_right": "in_lane_nudge_right",
    "sharp_steer_left": "turn_left",
    "sharp_steer_right": "turn_right",
}


def _remap_aliases(label: Dict) -> Dict:
    """Remap known VLM alias terms to canonical CoC category names in-place."""
    dd = label.get("driving_decision", {})
    lon = dd.get("longitudinal", "")
    lat = dd.get("lateral", "")
    remapped = []
    if lon in _LONGITUDINAL_ALIASES:
        new_lon = _LONGITUDINAL_ALIASES[lon]
        remapped.append(f"longitudinal '{lon}' → '{new_lon}'")
        dd["longitudinal"] = new_lon
    if lat in _LATERAL_ALIASES:
        new_lat = _LATERAL_ALIASES[lat]
        remapped.append(f"lateral '{lat}' → '{new_lat}'")
        dd["lateral"] = new_lat
    if remapped:
        print(f"  [REMAP] " + "; ".join(remapped))
    return label


def _validate_label(label: Dict) -> bool:
    """Basic structural validation of teacher label."""
    import sys
    from pathlib import Path
    _root = Path(__file__).resolve().parent.parent
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))
    from nuscenes_coc.constants import LONGITUDINAL_DECISIONS, LATERAL_DECISIONS

    if not isinstance(label, dict):
        return False
    if "driving_decision" not in label or "coc_reasoning" not in label:
        return False
    dd = label["driving_decision"]
    if "longitudinal" not in dd or "lateral" not in dd:
        return False
    lon = dd["longitudinal"]
    lat = dd["lateral"]
    if lon not in LONGITUDINAL_DECISIONS:
        print(f"  [WARN] Invalid longitudinal category from VLM: '{lon}' — will fallback to rule label")
        return False
    if lat not in LATERAL_DECISIONS:
        print(f"  [WARN] Invalid lateral category from VLM: '{lat}' — will fallback to rule label")
        return False
    reasoning = label.get("coc_reasoning", "")
    if not reasoning or len(reasoning.strip()) < 10:
        return False
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Import openai lazily
    try:
        import openai
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai")
        sys.exit(1)

    # Resolve API key
    import os
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: No API key provided. Use --api-key or set OPENAI_API_KEY env var.")
        sys.exit(1)

    # Build client
    client_kwargs: Dict = {"api_key": api_key}
    if args.base_url:
        client_kwargs["base_url"] = args.base_url
    client = openai.OpenAI(**client_kwargs)

    # Load requests
    requests_path = Path(args.requests_input)
    if not requests_path.exists():
        print(f"ERROR: requests file not found: {requests_path}")
        sys.exit(1)
    requests = read_jsonl(requests_path)
    print(f"Loaded {len(requests)} teacher requests from {requests_path}")

    if args.max_samples >= 0:
        requests = requests[: args.max_samples]
        print(f"Limited to first {args.max_samples} samples")

    # Resume: load already-processed IDs
    responses_path = Path(args.responses_output)
    existing_ids: set = set()
    existing_responses: List[Dict] = []
    if args.resume and responses_path.exists():
        existing_responses = read_jsonl(responses_path)
        existing_ids = {r["request_id"] for r in existing_responses}
        print(f"Resume mode: {len(existing_ids)} already processed")

    # Dry run
    if args.dry_run:
        sample_req = requests[0]
        if args.with_images:
            msgs = _build_vlm_messages(sample_req, args.cameras, args.image_frames)
            img_count = sum(1 for m in msgs for p in (m["content"] if isinstance(m["content"], list) else [])
                            if isinstance(p, dict) and p.get("type") == "image_url")
            print(f"\n=== DRY RUN (VLM 模式): {img_count} 张图像 ===")
        else:
            msgs = _build_text_messages(sample_req)
            print("\n=== DRY RUN (文本模式) ===")
        for m in msgs:
            role = m["role"]
            content = m["content"]
            if isinstance(content, list):
                text_parts = [p["text"] for p in content if isinstance(p, dict) and p.get("type") == "text"]
                preview = "\n".join(text_parts)[:600]
            else:
                preview = content[:600]
            if len(preview) == 600:
                preview += "..."
            print(f"\n[{role.upper()}]\n{preview}")
        print("\n=== End dry run. Exiting. ===")
        return

    # Process
    new_responses: List[Dict] = []
    success_count = 0
    fail_count = 0

    for i, request in enumerate(requests):
        request_id = request["request_id"]
        sample_id = request["sample_id"]

        if request_id in existing_ids:
            print(f"[{i+1}/{len(requests)}] SKIP (already done): {sample_id}")
            continue

        print(f"[{i+1}/{len(requests)}] Processing: {sample_id}")

        if args.with_images:
            messages = _build_vlm_messages(request, args.cameras, args.image_frames)
        else:
            messages = _build_text_messages(request)
        response_schema = request.get("response_format", {}).get("json_schema")

        raw_output: Optional[str] = None
        label: Optional[Dict] = None
        error_msg: Optional[str] = None

        for attempt in range(1, args.retry + 1):
            try:
                raw_output = _call_llm(
                    client,
                    args.model,
                    messages,
                    args.temperature,
                    response_schema,
                )
                label = _parse_response(raw_output, request_id)
                if label is not None:
                    label = _remap_aliases(label)
                if label is not None and _validate_label(label):
                    break
                else:
                    error_msg = "Invalid label structure or reasoning template"
                    print(f"  [WARN] Attempt {attempt}: {error_msg}")
            except Exception as e:
                error_msg = str(e)
                print(f"  [WARN] Attempt {attempt} API error: {error_msg}")
                if attempt < args.retry:
                    time.sleep(args.retry_delay)

        if label is not None and _validate_label(label):
            response = {
                "request_id": request_id,
                "sample_id": sample_id,
                "teacher_source": f"llm:{args.model}",
                "teacher_label": {
                    "driving_decision": label["driving_decision"],
                    "critical_components": label.get("critical_components", []),
                    "coc_reasoning": label["coc_reasoning"],
                },
                "confidence": "high",
                "evidence": [f"llm_model:{args.model}"],
                "raw_output": raw_output,
            }
            new_responses.append(response)
            success_count += 1
            print(f"  [OK] {label['driving_decision']['longitudinal']} / "
                  f"{label['driving_decision']['lateral']}")
        else:
            # Fallback to weak_label_hint
            hint = request.get("weak_label_hint", {})
            response = {
                "request_id": request_id,
                "sample_id": sample_id,
                "teacher_source": "llm_fallback_to_rule",
                "teacher_label": {
                    "driving_decision": hint.get("driving_decision", {"longitudinal": "none", "lateral": "none"}),
                    "critical_components": hint.get("critical_components", []),
                    "coc_reasoning": hint.get("coc_reasoning", "Maintain current behavior based on rule-based annotation."),
                },
                "confidence": "low",
                "evidence": [f"llm_failed:{error_msg or 'unknown'}"],
                "raw_output": raw_output,
            }
            new_responses.append(response)
            fail_count += 1
            print(f"  [FAIL] Fell back to rule hint. Error: {error_msg}")

    # Write output (append to existing if resume)
    all_responses = existing_responses + new_responses
    responses_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(responses_path, all_responses)

    print(f"\nDone. Success: {success_count}, Failed/fallback: {fail_count}")
    print(f"Total responses written: {len(all_responses)} -> {responses_path}")
    print(
        "\nNext step: assemble final dataset with:\n"
        f"  python scripts/build_teacher_labeling_assets.py \\\n"
        f"    --input out/nuscenes_coc_teacher_input_full.json \\\n"
        f"    --requests-output out/nuscenes_coc_teacher_requests_full.jsonl \\\n"
        f"    --teacher-responses {responses_path} \\\n"
        f"    --final-output out/nuscenes_coc_final_teacher.json"
    )


if __name__ == "__main__":
    main()
