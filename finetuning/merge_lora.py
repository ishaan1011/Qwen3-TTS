# coding=utf-8
"""
Merge a LoRA adapter (produced by sft_12hz_lora.py) into the base Qwen3-TTS
model and save in CustomVoice format. The merged checkpoint is drop-in
compatible with audition.py and the production vllm-omni inference path —
i.e., from the consumer's perspective it is indistinguishable from a full-SFT
checkpoint.

Usage:
  python merge_lora.py \
      --adapter_dir /home/ubuntu/models/ishaan-lora/checkpoint-epoch-3/adapter \
      --output_dir  /home/ubuntu/models/ishaan-lora-merged/checkpoint-epoch-3

The script reads lora_meta.json (sibling of adapter_dir) for speaker_name and
init_model_path, and reads speaker_embedding.pt to write into codec_embedding
row 3000. Override with --init_model_path / --speaker_name / --speaker_embedding
if needed (e.g., merging an adapter without the meta sidecar).
"""
import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file


def _resolve_to_local_dir(model_path: str) -> str:
    """If model_path is a local directory, return it as-is. Otherwise treat
    it as an HF Hub repo ID and resolve to the cached snapshot directory
    (downloading if needed). shutil.copytree needs a real filesystem path.
    """
    if Path(model_path).is_dir():
        return str(model_path)
    from huggingface_hub import snapshot_download
    return snapshot_download(repo_id=model_path)


def save_custom_voice_checkpoint(
    merged_model,
    output_dir: str,
    init_model_path: str,
    speaker_name: str,
    speaker_embedding: torch.Tensor,
):
    """Mirror the save logic at the end of sft_12hz.py.

    Copies the base model directory as a template, rewrites config to declare
    custom_voice / spk_id mapping, drops speaker_encoder weights from the
    state_dict, writes the speaker embedding into codec_embedding row 3000,
    and saves model.safetensors.
    """
    local_init = _resolve_to_local_dir(init_model_path)

    os.makedirs(output_dir, exist_ok=True)
    shutil.copytree(local_init, output_dir, dirs_exist_ok=True)

    input_config_file = os.path.join(local_init, "config.json")
    output_config_file = os.path.join(output_dir, "config.json")
    with open(input_config_file, "r", encoding="utf-8") as f:
        config_dict = json.load(f)
    config_dict["tts_model_type"] = "custom_voice"
    talker_config = config_dict.get("talker_config", {})
    talker_config["spk_id"] = {speaker_name: 3000}
    talker_config["spk_is_dialect"] = {speaker_name: False}
    config_dict["talker_config"] = talker_config
    with open(output_config_file, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    state_dict = {
        k: v.detach().to("cpu") for k, v in merged_model.state_dict().items()
    }

    drop_prefix = "speaker_encoder"
    keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
    for k in keys_to_drop:
        del state_dict[k]

    weight = state_dict["talker.model.codec_embedding.weight"]
    state_dict["talker.model.codec_embedding.weight"][3000] = (
        speaker_embedding.detach().to(weight.device).to(weight.dtype)
    )

    save_file(state_dict, os.path.join(output_dir, "model.safetensors"))


def resolve_meta(adapter_dir: Path,
                 init_model_path: str = None,
                 speaker_name: str = None,
                 speaker_embedding_path: Path = None):
    """Read lora_meta.json + speaker_embedding.pt sibling to adapter_dir,
    with optional CLI overrides. Returns (init_model_path, speaker_name,
    speaker_embedding_tensor).
    """
    epoch_dir = adapter_dir.parent
    meta_path = epoch_dir / "lora_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    init_model_path = init_model_path or meta.get("init_model_path")
    speaker_name = speaker_name or meta.get("speaker_name")
    if not init_model_path:
        raise SystemExit("init_model_path missing — pass --init_model_path or include lora_meta.json")
    if not speaker_name:
        raise SystemExit("speaker_name missing — pass --speaker_name or include lora_meta.json")

    se_path = speaker_embedding_path or (epoch_dir / "speaker_embedding.pt")
    if not se_path.exists():
        raise SystemExit(f"speaker_embedding.pt not found at {se_path}")
    speaker_embedding = torch.load(str(se_path), map_location="cpu")
    return init_model_path, speaker_name, speaker_embedding


def merge_adapter(adapter_dir, output_dir, *,
                 init_model_path=None, speaker_name=None,
                 speaker_embedding_path=None, device="cuda:0",
                 verbose=True):
    """Programmatic API: load base -> attach adapter -> merge -> save
    CustomVoice checkpoint. Returns the output_dir path on success.

    Heavy: loads the full base model (~3.4 GB bf16) into VRAM. Caller is
    responsible for sequencing if running over multiple adapters.
    """
    adapter_dir = Path(adapter_dir).expanduser().resolve()
    output_dir = str(output_dir)

    init_model_path, speaker_name, speaker_embedding = resolve_meta(
        adapter_dir,
        init_model_path=init_model_path,
        speaker_name=speaker_name,
        speaker_embedding_path=Path(speaker_embedding_path) if speaker_embedding_path else None,
    )

    if verbose:
        print(f"[merge] loading base model from {init_model_path}")
    qwen3tts = Qwen3TTSModel.from_pretrained(
        init_model_path,
        device_map=device,
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )

    if verbose:
        print(f"[merge] attaching LoRA adapter from {adapter_dir}")
    peft_model = PeftModel.from_pretrained(qwen3tts.model, str(adapter_dir))

    if verbose:
        print(f"[merge] merging LoRA weights into base")
    merged = peft_model.merge_and_unload(progressbar=verbose)

    if verbose:
        print(f"[merge] writing CustomVoice checkpoint -> {output_dir}")
    save_custom_voice_checkpoint(
        merged_model=merged,
        output_dir=output_dir,
        init_model_path=init_model_path,
        speaker_name=speaker_name,
        speaker_embedding=speaker_embedding,
    )
    return output_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True,
                    help="Directory containing PEFT adapter (adapter_model.safetensors etc.).")
    ap.add_argument("--output_dir", required=True,
                    help="Destination directory for the merged CustomVoice checkpoint.")
    ap.add_argument("--init_model_path", default=None,
                    help="Base model path. Defaults to value in lora_meta.json.")
    ap.add_argument("--speaker_name", default=None,
                    help="Speaker name to register at codec slot 3000. Defaults to lora_meta.json.")
    ap.add_argument("--speaker_embedding", default=None,
                    help="Path to speaker_embedding.pt. Defaults to <adapter_dir>/../speaker_embedding.pt.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    merge_adapter(
        adapter_dir=args.adapter_dir,
        output_dir=args.output_dir,
        init_model_path=args.init_model_path,
        speaker_name=args.speaker_name,
        speaker_embedding_path=args.speaker_embedding,
        device=args.device,
    )

    print(f"[done] merged checkpoint saved. Audition with:")
    print(f"  python finetuning/audition.py \\")
    print(f"      --checkpoint_dir {args.output_dir} \\")
    print(f"      --speaker <name from lora_meta.json> \\")
    print(f"      --output_dir <out>")


if __name__ == "__main__":
    main()
