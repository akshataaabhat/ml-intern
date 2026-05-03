# ML-Intern Task: Train-Eval Research Loop for 12 OCR Models

## Objective
Run a full data-generation → LoRA fine-tuning → evaluation → targeted-retrain research loop
across all 12 OCR models listed below. The pipeline is:

1. **Generate synthetic training data** with DocuMint (scan-augmented document images + OCR ground truth)
2. **Convert** to `dataset.jsonl` with `make_jsonl.py`
3. **Train** each model with the provided LoRA script
4. **Eval** each fine-tuned model on IntentBench (F1, CER, WER)
5. **Second-pass**: for any model with F1 < 0.75, generate a harder targeted dataset and retrain once

---

## Models to train (in order)

| # | Model ID (HuggingFace hub) |
|---|---|
| 1 | opendatalab/MinerU2.5-Pro-2604 |
| 2 | THUDM/GLM-OCR |
| 3 | PaddlePaddle/PaddleOCR-VL-1.5 |
| 4 | Tencent/Youtu-Parsing |
| 5 | AIDC-AI/Ovis2.6-30B-A3B |
| 6 | Logics-AI/Logics-Parsing-v2 |
| 7 | ERNIE-Bot/Qianfan-OCR |
| 8 | fireredteam/FireRed-OCR-2B |
| 9 | deepseek-ai/DeepSeek-OCR-2 |
| 10 | Qwen/Qwen3-30B-A3B |
| 11 | stepfun-ai/dots.mocr |
| 12 | openbmb/MiniCPM-o-4.5 |

> If any of these model IDs are not found on the hub, search for the closest match
> (e.g. search HuggingFace for "MinerU2.5-Pro" or "GLM-OCR") and use the correct slug.
> Log any substitutions made.

---

## Phase 0 — Synthetic data generation with DocuMint

Before any training begins, generate the full synthetic dataset using DocuMint.
The repo lives at `VLMSyn/` (clone it if not present: `git clone <DocuMint repo>`).

### Step 0a — Install dependencies

```bash
cd VLMSyn/
pip install -r requirements.txt
pip install augraphy>=8.2.6 opencv-python>=4.5.0
```

### Step 0b — Generate base dataset (all effects enabled)

```bash
python generate.py \
  --input amasked_pdfs/ \
  --images images/ \
  --ocr ocr_flat/ \
  --scan-count 1300 \
  --clean-count 700 \
  --dpi 150 \
  --effects office,aged,dark,phone,washed,crumpled \
  --wrinkle-texture wrinkles.jpg \
  --prob-fold 0.35 \
  --prob-on-paper 0.20 \
  --prob-augraphy 0.45 \
  --prob-perspective 0.25 \
  --seed 42
```

> If `wrinkles.jpg` does not exist, skip `--prob-fold` and `--wrinkle-texture` for now
> and note that fold/wrinkle augmentation is disabled for the base run.

### Step 0c — Convert to JSONL

```bash
python make_jsonl.py
# produces dataset.jsonl — this is the file passed to --dataset in training
```

Log: total sample count in `dataset.jsonl`, split between scan vs clean images.

### Step 0d — Harder targeted dataset (used in second-pass retraining)

Generate a separate harder variant with more aggressive degradation for struggling models.
Write it to `dataset_hard.jsonl`:

```bash
python generate.py \
  --input amasked_pdfs/ \
  --images images_hard/ \
  --ocr ocr_flat_hard/ \
  --scan-count 800 \
  --clean-count 200 \
  --dpi 120 \
  --effects aged,dark,crumpled \
  --wrinkle-texture wrinkles.jpg \
  --prob-fold 0.55 \
  --prob-on-paper 0.30 \
  --prob-augraphy 0.70 \
  --prob-perspective 0.40 \
  --seed 99

# Convert to JSONL
python make_jsonl.py \
  --images images_hard/ \
  --ocr ocr_flat_hard/ \
  --output dataset_hard.jsonl
```

> Check if `make_jsonl.py` accepts `--images`, `--ocr`, `--output` flags. If not, temporarily
> symlink or copy the hard images/ocr folders and rename the output manually.

---

## Cluster setup

- **Hardware**: 8× NVIDIA A100 40GB GPUs on a single node (or equivalent cluster)
- **Step 1 — GPU check**: Run `nvidia-smi` and identify which GPU indices are currently free
  (memory used < 2GB). Log the free GPU list.
- **Allocation**: Use 2 GPUs per training job (`--nproc_per_node=2`). Assign the two lowest-index
  free GPUs to each job. If fewer than 2 GPUs are free, wait and poll every 60s.
- **Ports**: Assign a unique `--master_port` per job starting from 29509, incrementing by 1.
  For vLLM serving after training, assign ports starting from 5534, incrementing by 1.

---

## Training script

Run the following for each model, substituting `{MODEL_ID}`, `{OUTPUT_DIR}`, `{CUDA_DEVICES}`,
and `{MASTER_PORT}` appropriately:

```bash
CUDA_VISIBLE_DEVICES={CUDA_DEVICES} \
PYTORCH_ALLOC_CONF=expandable_segments:True \
USE_HF=1 \
torchrun \
    --nproc_per_node=2 \
    --master_port={MASTER_PORT} \
    -m swift.cli.sft \
    --model {MODEL_ID} \
    --dataset 'dataset.jsonl' \
    --train_type lora \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 5e-5 \
    --attn_impl flash_attn \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing true \
    --save_total_limit 5 \
    --load_args false \
    --logging_steps 1 \
    --max_length 20000 \
    --warmup_ratio 0.1 \
    --weight_decay 0.01 \
    --dataloader_num_workers 4 \
    --dataset_num_proc 4 \
    --output_dir 'output/{MODEL_SHORT_NAME}/lora' \
    --logging_dir 'output/{MODEL_SHORT_NAME}/lora' \
    --eval_steps 5 \
    --save_steps 100 \
    --split_dataset_ratio 0.1 \
    --metric_for_best_model loss \
    --greater_is_better false \
    --load_best_model_at_end true \
    --early_stop_interval 3 \
    --use_logits_to_keep true
```

Where `{MODEL_SHORT_NAME}` is a short slug like `mineru`, `glm-ocr`, `paddleocr-vl`, etc.

**If flash_attn is not available** for a given model architecture, fall back to `--attn_impl sdpa`.

**If a model is too large** for 2× A100-40GB (e.g. 30B+ models like Ovis2.6-30B or Qwen3-30B-A3B):
- Increase to 4 GPUs (`--nproc_per_node=4`, `CUDA_VISIBLE_DEVICES=0,1,2,3`)
- Add `--deepspeed zero3` if needed
- Reduce `--max_length 8000` if OOM persists
- Log the adjustments made

---

## Post-training: merge adapter and serve

After each training completes, merge the LoRA adapter into the base model and serve it:

```bash
# Merge
swift export \
    --model {MODEL_ID} \
    --adapters output/{MODEL_SHORT_NAME}/lora/best \
    --merge_lora true \
    --output_dir output/{MODEL_SHORT_NAME}/merged

# Serve with vLLM
python -m vllm.entrypoints.openai.api_server \
    --model output/{MODEL_SHORT_NAME}/merged \
    --port {VLLM_PORT} \
    --trust-remote-code \
    --max-model-len 8192 \
    &
```

Wait for the server to be healthy (`curl http://0.0.0.0:{VLLM_PORT}/health`) before running eval.

---

## Evaluation

Use the following eval script (already written — do not modify the eval logic):

```python
# eval.py — runs as-is, pointed at intentbench_gt.json
# MODELS dict must be updated per model being evaluated
```

For each model evaluation run, update the `MODELS` dict in `eval.py` to point to the correct
served model and port:

```python
MODELS = {
    "{MODEL_SHORT_NAME}": {
        "model_name": "{MERGED_MODEL_NAME}",   # as reported by vLLM
        "port": {VLLM_PORT},
        "query": QUERY_NO,
    },
}
```

Then run:

```bash
python eval.py intentbench_gt.json results/{MODEL_SHORT_NAME}_eval.json
```

After eval, shut down the vLLM server for that model before moving to the next one.

---

## Results tracking

After all 12 models are evaluated, aggregate results into a single comparison file:

```bash
python - <<'EOF'
import json, glob, os

rows = []
for path in sorted(glob.glob("results/*_eval.json")):
    name = os.path.basename(path).replace("_eval.json", "")
    with open(path) as f:
        data = json.load(f)
    summary = list(data["summary"].values())[0]
    rows.append({"model": name, **summary})

rows.sort(key=lambda r: -r["f1"])

print(f"{'Model':<30} {'F1':>8} {'CER':>8} {'WER':>8}")
print("-" * 58)
for r in rows:
    print(f"{r['model']:<30} {r['f1']:>8.4f} {r['cer']:>8.4f} {r['wer']:>8.4f}")

with open("results/summary_all_models.json", "w") as f:
    json.dump(rows, f, indent=2)
print("\nSaved to results/summary_all_models.json")
EOF
```

---

## Success criteria

- [ ] All 12 models trained (or documented why a model was skipped/substituted)
- [ ] All 12 merged checkpoints exist at `output/{MODEL_SHORT_NAME}/merged`
- [ ] All 12 eval result JSONs exist at `results/{MODEL_SHORT_NAME}_eval.json`
- [ ] Final summary table printed and saved to `results/summary_all_models.json`
- [ ] Best model by F1 identified and highlighted

---

## Error handling

- If training fails (OOM, missing dependency, unsupported architecture): log the error,
  try the fallback adjustments above, and if still failing mark the model as FAILED in the
  summary with a note. Do not block other models.
- If swift SFT doesn't support a model's architecture: try `--train_type full` with a lower
  learning rate (`1e-6`) as a fallback.
- If vLLM can't serve the merged model: try `--dtype bfloat16` or `--quantization awq` as fallback.
- Always prefer a working but degraded run over a clean skip — a model with a slightly wrong
  config still gives us signal.