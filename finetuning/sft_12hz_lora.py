# coding=utf-8
"""
LoRA fine-tuning variant of sft_12hz.py.

Mirrors the full-SFT loop one-for-one (data, loss, optimizer, gradient
clipping, accelerate config) so the only variable in the validation A/B is
LoRA-vs-full-SFT — not anything else.

Trainable parameters:
  - Low-rank adapters on every {q,k,v,o}_proj and {gate,up,down}_proj across
    talker.model.layers AND talker.code_predictor.model.layers (suffix match
    via PEFT target_modules).
  - Full-rank training of codec_embedding, codec_head, lm_head, text_projection
    via PEFT modules_to_save. These are small / output-side modules where
    full-rank training matters disproportionately for prosody quality.

Frozen:
  - speaker_encoder (the full-SFT script also bypasses gradients on it via
    .detach() and drops it from the saved state_dict; LoRA matches that).
  - text_embedding (kept at base — phoneme/text representations transfer).

Saved per epoch:
  - <output>/checkpoint-epoch-N/adapter/         PEFT adapter (~50–150 MB)
  - <output>/checkpoint-epoch-N/speaker_embedding.pt
  - <output>/checkpoint-epoch-N/lora_meta.json   {speaker_name, init_model_path}

Merging into a CustomVoice checkpoint compatible with audition.py / vllm-omni
is a separate step — see finetuning/merge_lora.py. Merge is post-training so
the same training run can audition multiple epochs without leaving merged
state behind in the PEFT model (merge_and_unload is destructive).

Validation usage (matches the recommended A/B path):
  python sft_12hz_lora.py \
      --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
      --output_model_path /home/ubuntu/models/ishaan-lora \
      --train_jsonl train_with_codes.jsonl \
      --batch_size 2 --lr 1e-4 --num_epochs 5 \
      --speaker_name ishaan --lora_rank 32 --lora_alpha 64
"""
import argparse
import json
import os

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from peft import LoraConfig, get_peft_model
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

DEFAULT_LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Full-rank training of small / output-side modules where rank-restriction
# would cap prosody quality. Names use full paths (suffix-matched by PEFT) to
# discriminate between the talker's codec_embedding (a single nn.Embedding)
# and the code_predictor's codec_embedding (an nn.ModuleList of 15 Embeddings).
#
# The 15 sub-codebook lm_head Linears and the 15 sub-codebook codec_embedding
# Embeddings are wrapped INDIVIDUALLY (rather than the ModuleList parent),
# because PEFT's ModulesToSaveWrapper only proxies forward() — wrapping a
# ModuleList would break the indexed access (lm_head[i](x)) used in
# modeling_qwen3_tts.py.
NUM_SUB_CODE_GROUPS = 15

DEFAULT_MODULES_TO_SAVE = (
    [
        "talker.model.codec_embedding",  # speaker slot lives here at row 3000
        "codec_head",                    # talker's main codec output (nn.Linear)
        "text_projection",               # talker's text-to-talker resize MLP
        "small_to_mtp_projection",       # code_predictor input projection
    ]
    + [f"lm_head.{i}" for i in range(NUM_SUB_CODE_GROUPS)]
    + [f"code_predictor.model.codec_embedding.{i}" for i in range(NUM_SUB_CODE_GROUPS)]
)


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str,
                        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output_lora")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="LoRA wants ~10x higher LR than full SFT (full was 1e-5).")
    parser.add_argument("--num_epochs", type=int, default=5)
    parser.add_argument("--speaker_name", type=str, default="speaker_lora")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64,
                        help="Conventionally 2x lora_rank.")
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    args = parser.parse_args()

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with="tensorboard",
    )

    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(args.init_model_path)

    train_data = [json.loads(line) for line in open(args.train_jsonl).readlines()]
    dataset = TTSDataset(train_data, qwen3tts.processor, config)
    train_dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=dataset.collate_fn,
    )

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=DEFAULT_LORA_TARGETS,
        modules_to_save=DEFAULT_MODULES_TO_SAVE,
        lora_dropout=args.lora_dropout,
        bias="none",
    )

    model = get_peft_model(qwen3tts.model, lora_config)
    if accelerator.is_main_process:
        model.print_trainable_parameters()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader,
    )

    target_speaker_embedding = None
    model.train()

    for epoch in range(args.num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                input_ids = batch["input_ids"]
                codec_ids = batch["codec_ids"]
                ref_mels = batch["ref_mels"]
                text_embedding_mask = batch["text_embedding_mask"]
                codec_embedding_mask = batch["codec_embedding_mask"]
                attention_mask = batch["attention_mask"]
                codec_0_labels = batch["codec_0_labels"]
                codec_mask = batch["codec_mask"]

                speaker_embedding = model.speaker_encoder(
                    ref_mels.to(model.device).to(model.dtype)
                ).detach()
                if target_speaker_embedding is None:
                    target_speaker_embedding = speaker_embedding

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = (
                    model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                )
                input_codec_embedding = (
                    model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                )
                input_codec_embedding[:, 6, :] = speaker_embedding

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](
                        codec_ids[:, :, i]
                    )
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True,
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, :-1]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states,
                )

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(trainable_params, 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 10 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            epoch_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            adapter_dir = os.path.join(epoch_dir, "adapter")
            os.makedirs(adapter_dir, exist_ok=True)

            unwrapped = accelerator.unwrap_model(model)
            # save_pretrained writes both LoRA weights AND modules_to_save.
            unwrapped.save_pretrained(adapter_dir)

            # Stash the speaker embedding so merge_lora.py can write it into
            # codec_embedding row 3000 without re-running the speaker_encoder.
            torch.save(
                target_speaker_embedding[0].detach().to("cpu"),
                os.path.join(epoch_dir, "speaker_embedding.pt"),
            )

            with open(os.path.join(epoch_dir, "lora_meta.json"), "w") as f:
                json.dump(
                    {
                        "speaker_name": args.speaker_name,
                        "init_model_path": args.init_model_path,
                        "lora_rank": args.lora_rank,
                        "lora_alpha": args.lora_alpha,
                        "lora_dropout": args.lora_dropout,
                        "epoch": epoch,
                    },
                    f,
                    indent=2,
                )

            accelerator.print(f"[epoch {epoch}] saved adapter -> {adapter_dir}")


if __name__ == "__main__":
    train()
