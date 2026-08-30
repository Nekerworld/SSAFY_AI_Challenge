# ============================================================
# Qwen2.5-VL 3B + QLoRA
# VQA Multiple Choice - Final Local Training / Dev / Test
#
# 실제 데이터 구조:
# train.csv: id,path,question,a,b,c,d,answer
# dev.csv  : id,path,question,a,b,c,d,answer1,...,answer5
# test.csv : id,path,question,a,b,c,d
#
# 이미지:
# ./train, ./dev, ./test
# ============================================================

import os
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any

import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from transformers import (
    Qwen3VLForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)

from tqdm.auto import tqdm


# ============================================================
# 1. Config
# ============================================================

# .py로 실행하면 스크립트가 있는 폴더를 데이터 루트로 사용.
# 노트북에서 실행하면 현재 작업 폴더를 사용.
try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd().resolve()

MODEL_ID = "Qwen/Qwen3VLForConditionalGeneration"

SEED = 42

# 200개 제한이 과제 조건이면 유지.
# 전체 train 5,073개를 쓰고 싶으면 None으로 변경.
TRAIN_LIMIT = None

# Qwen2.5-VL pixel budget.
# VRAM 부족 시 MAX_PIXELS를 384 * 384 정도로 낮출 수 있음.
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 512 * 28 * 28

NUM_EPOCHS = 20
TRAIN_LIMIT = 200
EVAL_EVERY = 10

TRAIN_BATCH_SIZE = 8
GRAD_ACCUM = 4

LR = 1e-4
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.03

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

# 매 epoch에는 dev 전체 4,413개 대신 고정 subset만 평가.
# 학습 종료 후 best model로 dev 전체를 한 번 평가.

SAVE_DIR = PROJECT_DIR / "model" / "qwen2_5_vl_3b_lora"
SUBMISSION_PATH = PROJECT_DIR / "submission" / "submission.csv"
VALIDATION_PREDICTION_PATH = (PROJECT_DIR / "submission" / "validation_predictions.csv")
LOGIT_PATH = PROJECT_DIR / "submission" / "prediction_logits.csv"

Image.MAX_IMAGE_PIXELS = None


# ============================================================
# 2. Reproducibility / CUDA
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 필요합니다. 현재 torch.cuda.is_available() == False 입니다."
    )

torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0")
BF16_SUPPORTED = torch.cuda.is_bf16_supported()
COMPUTE_DTYPE = torch.bfloat16 if BF16_SUPPORTED else torch.float16
USE_SCALER = COMPUTE_DTYPE == torch.float16

print("Project dir :", PROJECT_DIR)
print("Device      :", DEVICE)
print("GPU         :", torch.cuda.get_device_name(0))
print("Compute dtype:", COMPUTE_DTYPE)
print("GradScaler  :", USE_SCALER)


# ============================================================
# 3. Helpers - paths / validation
# ============================================================

def resolve_image_path(path_value) -> Path:
    """CSV의 상대 이미지 경로를 프로젝트 폴더 기준으로 절대경로화."""
    path = Path(str(path_value))

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path


def validate_image_paths(df, name, sample_only=False):
    """이미지 경로 오류를 학습 중이 아니라 시작 시점에 발견."""
    if sample_only:
        check_df = df.head(min(20, len(df)))
    else:
        check_df = df

    missing = []

    for path_value in check_df["path"]:
        path = resolve_image_path(path_value)
        if not path.exists():
            missing.append(str(path))

            if len(missing) >= 10:
                break

    if missing:
        raise FileNotFoundError(
            f"{name} 이미지 경로를 찾을 수 없습니다. 예시:\n"
            + "\n".join(missing)
        )


def load_rgb_image(path_value):
    path = resolve_image_path(path_value)

    with Image.open(path) as image:
        return image.convert("RGB")


# ============================================================
# 4. Load actual CSVs
# ============================================================

TRAIN_CSV = PROJECT_DIR / "train.csv"
DEV_CSV = PROJECT_DIR / "dev.csv"
TEST_CSV = PROJECT_DIR / "test.csv"

train_df = pd.read_csv(TRAIN_CSV)
dev_df = pd.read_csv(DEV_CSV)
test_df = pd.read_csv(TEST_CSV)

TRAIN_REQUIRED = {
    "id", "path", "question", "a", "b", "c", "d", "answer"
}

DEV_REQUIRED = {
    "id", "path", "question", "a", "b", "c", "d",
    "answer1", "answer2", "answer3", "answer4", "answer5",
}

TEST_REQUIRED = {
    "id", "path", "question", "a", "b", "c", "d"
}


def require_columns(df, required, name):
    missing = set(required) - set(df.columns)

    if missing:
        raise ValueError(
            f"{name} missing columns: {sorted(missing)}"
        )


require_columns(train_df, TRAIN_REQUIRED, "train.csv")
require_columns(dev_df, DEV_REQUIRED, "dev.csv")
require_columns(test_df, TEST_REQUIRED, "test.csv")

# 실제 answer 값 검사
valid_choices = {"a", "b", "c", "d"}

bad_train_answers = (
    train_df["answer"]
    .dropna()
    .astype(str)
    .str.strip()
    .str.lower()
)

if not set(bad_train_answers.unique()).issubset(valid_choices):
    raise ValueError(
        "train.csv의 answer에 a/b/c/d 외 값이 있습니다."
    )

# 이미지 폴더 구조도 초기에 검증.
validate_image_paths(train_df, "train.csv")
validate_image_paths(dev_df, "dev.csv")
validate_image_paths(test_df, "test.csv")


# ============================================================
# 5. Train subset / fixed dev subset
# ============================================================

def stratified_train_sample(df, n, seed):
    df = df.reset_index(drop=True)

    if n is None or n >= len(df):
        return df

    if n <= 0:
        raise ValueError(
            "TRAIN_LIMIT은 양수 또는 None이어야 합니다."
        )

    rng = random.Random(seed)
    choices = ["a", "b", "c", "d"]

    base = n // len(choices)
    remainder = n % len(choices)

    selected_parts = []

    extra_choices = choices.copy()
    rng.shuffle(extra_choices)
    extra_choices = set(extra_choices[:remainder])

    for choice in choices:
        group = df[
            df["answer"].astype(str).str.strip().str.lower()
            == choice
        ]

        take = base + (
            1 if choice in extra_choices else 0
        )

        if len(group) < take:
            raise ValueError(
                f"answer={choice} 샘플 수가 "
                f"{take}개보다 적습니다."
            )

        selected_parts.append(
            group.sample(
                n=take,
                random_state=seed + ord(choice),
            )
        )

    sampled = pd.concat(
        selected_parts,
        ignore_index=True,
    )

    return sampled.sample(
        frac=1.0,
        random_state=seed,
    ).reset_index(drop=True)

train_pool, valid_subset = train_test_split(
    train_df,
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["answer"],
)

train_pool = train_pool.reset_index(drop=True)
valid_subset = valid_subset.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_subset = stratified_train_sample(
    train_pool,
    TRAIN_LIMIT,
    SEED,
)

print("\n===== DATA =====")
print("Original labeled train:", len(train_df))
print("Train pool            :", len(train_pool))
print("Train actually used   :", len(train_subset))
print("Validation            :", len(valid_subset))
print("Test                  :", len(test_df))
print(
    "Train labels          :",
    train_subset["answer"]
    .astype(str)
    .str.strip()
    .str.lower()
    .value_counts()
    .sort_index()
    .to_dict()
)

# ============================================================
# 6. Processor
# ============================================================

processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    min_pixels=MIN_PIXELS,
    max_pixels=MAX_PIXELS,
    trust_remote_code=True,
)

if processor.tokenizer.pad_token_id is None:
    processor.tokenizer.pad_token = processor.tokenizer.eos_token

print("Padding side  :", processor.tokenizer.padding_side)


# ============================================================
# 7. 4-bit QLoRA base model
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)

base_model = Qwen3VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
)

base_model.config.use_cache = False

base_model = prepare_model_for_kbit_training(
    base_model,
    use_gradient_checkpointing=True,
)


# ============================================================
# 8. LoRA - language model only
# ============================================================

LLM_TARGET_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

target_modules = []

for name, module in base_model.named_modules():
    # vision / visual 이름이 포함된 module은 무조건 제외.
    lower_name = name.lower()

    if "visual" in lower_name or "vision" in lower_name:
        continue

    if any(
        name.endswith(suffix)
        for suffix in LLM_TARGET_SUFFIXES
    ):
        target_modules.append(name)

target_modules = sorted(set(target_modules))

if not target_modules:
    raise RuntimeError(
        "Language-model LoRA target을 찾지 못했습니다. "
        "설치된 transformers 버전에서 모델 module 구조를 확인하세요."
    )

if any(
    "visual" in name.lower() or "vision" in name.lower()
    for name in target_modules
):
    raise RuntimeError(
        "Vision module이 LoRA target에 포함되었습니다."
    )

print("\n===== LORA TARGET =====")
print("Target count:", len(target_modules))
for name in target_modules[:20]:
    print(" ", name)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    target_modules=target_modules,
    task_type="CAUSAL_LM",
)

model = get_peft_model(
    base_model,
    lora_config,
)

model.print_trainable_parameters()


# ============================================================
# 9. Prompt
# ============================================================

SYSTEM_INSTRUCT = (
    "You are a visual multiple-choice question answering assistant. "
    "Inspect the image and question carefully. "
    "Answer using exactly one lowercase letter: a, b, c, or d. "
    "Do not provide any explanation."
)


def build_mc_prompt(question, a, b, c, d):
    return (
        f"{question}\n\n"
        f"(a) {a}\n"
        f"(b) {b}\n"
        f"(c) {c}\n"
        f"(d) {d}\n\n"
        "정답을 a, b, c, d 중 하나의 소문자 한 글자로만 출력하세요."
    )


def build_prompt_messages(row, image):
    user_text = build_mc_prompt(
        str(row["question"]),
        str(row["a"]),
        str(row["b"]),
        str(row["c"]),
        str(row["d"]),
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SYSTEM_INSTRUCT,
                }
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": user_text,
                },
            ],
        },
    ]


# ============================================================
# 10. Train Dataset
# ============================================================

class VQAMCDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_rgb_image(row["path"])

        gold = str(row["answer"]).strip().lower()

        if gold not in valid_choices:
            raise ValueError(
                f"Invalid answer idx={idx}: {gold!r}"
            )

        prompt_messages = build_prompt_messages(
            row,
            image,
        )

        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": gold,
                    }
                ],
            }
        ]

        return {
            "prompt_messages": prompt_messages,
            "full_messages": full_messages,
            "image": image,
            "gold": gold,
        }


# ============================================================
# 11. Train collator
#     Loss only on assistant answer + assistant closing tokens
# ============================================================

@dataclass
class TrainDataCollator:
    processor: Any

    def __call__(self, batch):
        images = [
            sample["image"]
            for sample in batch
        ]

        full_texts = []
        prompt_texts = []

        for sample in batch:
            full_text = self.processor.apply_chat_template(
                sample["full_messages"],
                tokenize=False,
                add_generation_prompt=False,
            )

            prompt_text = self.processor.apply_chat_template(
                sample["prompt_messages"],
                tokenize=False,
                add_generation_prompt=True,
            )

            full_texts.append(full_text)
            prompt_texts.append(prompt_text)

        full_enc = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        prompt_enc = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        labels = torch.full_like(
            full_enc["input_ids"],
            fill_value=-100,
        )

        for i in range(len(batch)):
            full_positions = (
                full_enc["attention_mask"][i]
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )

            prompt_positions = (
                prompt_enc["attention_mask"][i]
                .nonzero(as_tuple=False)
                .squeeze(-1)
            )

            full_ids = full_enc["input_ids"][
                i,
                full_positions,
            ]

            prompt_ids = prompt_enc["input_ids"][
                i,
                prompt_positions,
            ]

            prompt_len = int(prompt_ids.numel())
            full_len = int(full_ids.numel())

            if prompt_len >= full_len:
                raise RuntimeError(
                    "Label masking failed: "
                    f"prompt_len={prompt_len}, "
                    f"full_len={full_len}"
                )

            # full token sequence가 prompt token sequence를
            # 실제 prefix로 포함하는지 검사.
            if not torch.equal(
                full_ids[:prompt_len].cpu(),
                prompt_ids.cpu(),
            ):
                raise RuntimeError(
                    "Prompt/full token prefix mismatch. "
                    "현재 tokenizer/chat template에서 "
                    "assistant-only label masking을 안전하게 만들 수 없습니다."
                )

            answer_positions = full_positions[
                prompt_len:
            ]

            # 이 과제의 supervised target은 a/b/c/d 한 글자뿐이다.
            # <|im_end|> 같은 assistant 종료 토큰에는 loss를 걸지 않는다.
            answer_token_position = answer_positions[0]

            labels[
                i,
                answer_token_position,
            ] = full_enc["input_ids"][
                i,
                answer_token_position,
            ]

        full_enc["labels"] = labels
        return full_enc


# ============================================================
# 12. Train Loader
# ============================================================

train_ds = VQAMCDataset(
    train_subset
)

train_loader = DataLoader(
    train_ds,
    batch_size=TRAIN_BATCH_SIZE,
    shuffle=True,
    collate_fn=TrainDataCollator(processor),
    num_workers=0,   # Windows + PIL + GPU 학습에서 안정성 우선
    pin_memory=False,
)

print("Train batches:", len(train_loader))


# ============================================================
# 13. Label sanity check
# ============================================================

debug_batch = next(iter(train_loader))

debug_labels = debug_batch["labels"][0]
debug_target_ids = debug_labels[
    debug_labels != -100
]

debug_target_text = processor.tokenizer.decode(
    debug_target_ids,
    skip_special_tokens=False,
)

print("\n===== LABEL CHECK =====")
print(repr(debug_target_text))
print("=======================\n")

if not any(
    letter in debug_target_text.lower()
    for letter in valid_choices
):
    raise RuntimeError(
        "Label sanity check failed: "
        "assistant target에 a/b/c/d가 보이지 않습니다."
    )


# ============================================================
# 14. Autocast
# ============================================================

def autocast_context():
    return torch.autocast(
        device_type="cuda",
        dtype=COMPUTE_DTYPE,
    )


# ============================================================
# 15. Determine exact answer token IDs from Qwen chat template
#
# tokenizer.encode("a")를 무조건 믿지 않고,
# 실제 assistant chat template에서 답 글자가 시작되는 토큰을 찾음.
# ============================================================

CHOICES = ("a", "b", "c", "d")


def discover_choice_token_ids():
    # 이미지 내용 자체는 tokenize=False chat template 문자열 생성에는
    # 필요하지 않으므로 임의 PIL 이미지를 사용.
    dummy_image = Image.new("RGB", (28, 28))

    dummy_row = {
        "question": "Choose the correct option.",
        "a": "A",
        "b": "B",
        "c": "C",
        "d": "D",
    }

    prompt_messages = build_prompt_messages(
        dummy_row,
        dummy_image,
    )

    prompt_text = processor.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    prompt_ids = processor.tokenizer(
        prompt_text,
        add_special_tokens=False,
    )["input_ids"]

    result = {}

    for choice in CHOICES:
        full_messages = prompt_messages + [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": choice,
                    }
                ],
            }
        ]

        full_text = processor.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )

        full_ids = processor.tokenizer(
            full_text,
            add_special_tokens=False,
        )["input_ids"]

        if full_ids[:len(prompt_ids)] != prompt_ids:
            raise RuntimeError(
                f"Chat-template token prefix mismatch for choice={choice!r}."
            )

        suffix = full_ids[len(prompt_ids):]

        if not suffix:
            raise RuntimeError(
                f"choice={choice!r}의 assistant token suffix가 비어 있습니다."
            )

        result[choice] = suffix[0]

        decoded_first = processor.tokenizer.decode(
            [suffix[0]],
            skip_special_tokens=False,
        )

        print(
            f"choice={choice!r} "
            f"first_token={suffix[0]} "
            f"decoded={decoded_first!r} "
            f"full_suffix={suffix}"
        )

    # 네 선택지가 같은 첫 token을 가지면 1-step logits 비교 불가능.
    if len(set(result.values())) != len(CHOICES):
        raise RuntimeError(
            "a/b/c/d가 서로 다른 first token으로 표현되지 않습니다. "
            "1-step MC logits 평가를 사용할 수 없습니다."
        )

    return result


choice_token_ids = discover_choice_token_ids()

print("Choice token IDs:", choice_token_ids)


# ============================================================
# 16. Fast MC prediction using next-token logits
# ============================================================

def predict_row(row, return_scores=False):
    image = load_rgb_image(
        row["path"]
    )

    messages = build_prompt_messages(
        row,
        image,
    )

    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[text],
        images=[image],
        padding=False,
        return_tensors="pt",
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode(), autocast_context():
        outputs = model(
            **inputs,
            use_cache=False,
        )

        # batch_size=1, padding=False:
        # 마지막 prompt token 위치의 logits가 다음 answer token 분포.
        next_token_logits = outputs.logits[
            0,
            -1,
        ].float()

    scores = {
        choice: next_token_logits[
            token_id
        ].item()
        for choice, token_id
        in choice_token_ids.items()
    }

    pred = max(
        scores,
        key=scores.get,
    )

    if return_scores:
        return pred, scores

    return pred

def evaluate_accuracy(df, desc="Validation"):
    model.eval()

    correct = 0
    predictions = []

    bar = tqdm(
        range(len(df)),
        desc=desc,
        unit="sample",
    )

    for idx in bar:
        row = df.iloc[idx]

        pred = predict_row(row)
        gold = str(row["answer"]).strip().lower()

        predictions.append(pred)

        if pred == gold:
            correct += 1

        accuracy = correct / (idx + 1)

        bar.set_postfix(
            acc=f"{accuracy:.4f}"
        )

    final_accuracy = (
        correct / len(df)
        if len(df)
        else 0.0
    )

    return {
        "accuracy": final_accuracy,
        "correct": correct,
        "total": len(df),
        "predictions": predictions,
    }

# ============================================================
# 17. Dev multi-annotator metrics
# ============================================================


# ============================================================
# 18. Optimizer / Scheduler
# ============================================================

trainable_params = [
    parameter
    for parameter in model.parameters()
    if parameter.requires_grad
]

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)

steps_per_epoch = math.ceil(
    len(train_loader)
    / GRAD_ACCUM
)

num_training_steps = (
    NUM_EPOCHS
    * steps_per_epoch
)

num_warmup_steps = int(
    num_training_steps
    * WARMUP_RATIO
)

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_SCALER,
)

print("\n===== OPTIMIZER =====")
print("Optimizer steps / epoch:", steps_per_epoch)
print("Total optimizer steps  :", num_training_steps)
print("Warmup steps           :", num_warmup_steps)


# ============================================================
# 19. Training state
# ============================================================

best_val_accuracy = -1.0
best_epoch = -1
best_state = None

history = []

optimizer.zero_grad(
    set_to_none=True
)

# ============================================================
# 20. Training loop
# ============================================================

for epoch in range(NUM_EPOCHS):
    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------
    model.train()
    model.config.use_cache = False

    train_loss_sum = 0.0
    train_batch_count = 0

    num_batches = len(
        train_loader
    )

    progress_bar = tqdm(
        train_loader,
        desc=(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            "[train]"
        ),
        unit="batch",
    )

    for step, batch in enumerate(
        progress_bar,
        start=1,
    ):
        batch = {
            key: value.to(DEVICE)
            for key, value in batch.items()
        }

        # 마지막 accumulation group이 GRAD_ACCUM보다 작으면
        # 실제 group 크기로 나누어 gradient scale을 보정.
        group_start = (
            ((step - 1) // GRAD_ACCUM)
            * GRAD_ACCUM
            + 1
        )

        accum_divisor = min(
            GRAD_ACCUM,
            num_batches - group_start + 1,
        )

        with autocast_context():
            outputs = model(
                **batch,
                use_cache=False,
            )

            raw_loss = outputs.loss
            loss = raw_loss / accum_divisor

        if not torch.isfinite(
            raw_loss
        ).item():
            raise RuntimeError(
                "Non-finite loss detected: "
                f"epoch={epoch + 1}, "
                f"step={step}, "
                f"loss={raw_loss.item()}"
            )

        if USE_SCALER:
            scaler.scale(
                loss
            ).backward()
        else:
            loss.backward()

        train_loss_sum += (
            raw_loss.detach()
            .float()
            .item()
        )

        train_batch_count += 1

        should_step = (
            step % GRAD_ACCUM == 0
            or step == num_batches
        )

        if should_step:
            if USE_SCALER:
                scaler.unscale_(
                    optimizer
                )

            torch.nn.utils.clip_grad_norm_(
                trainable_params,
                max_norm=MAX_GRAD_NORM,
            )

            if USE_SCALER:
                scaler.step(
                    optimizer
                )
                scaler.update()
            else:
                optimizer.step()

            scheduler.step()

            optimizer.zero_grad(
                set_to_none=True
            )

        avg_train_loss = (
            train_loss_sum
            / train_batch_count
        )

        progress_bar.set_postfix(
            loss=f"{avg_train_loss:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
        )

    avg_train_loss = (
        train_loss_sum
        / max(train_batch_count, 1)
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    should_evaluate = (
        (epoch + 1) % EVAL_EVERY == 0
        or (epoch + 1) == NUM_EPOCHS
    )

    current_lr = scheduler.get_last_lr()[0]

    if not should_evaluate:
        print(
            f"\n[Epoch {epoch + 1}/{NUM_EPOCHS}]"
        )
        print(
            f"train_loss={avg_train_loss:.4f}"
        )
        print("validation skipped")

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "valid_accuracy": None,
                "lr": current_lr,
            }
        )

        continue

    val_result = evaluate_accuracy(
        valid_subset,
        desc=(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} "
            "[validation]"
        ),
    )

    val_accuracy = val_result["accuracy"]

    print(
        f"\n[Epoch {epoch + 1}/{NUM_EPOCHS}]"
    )
    print(
        f"train_loss={avg_train_loss:.4f}"
    )
    print(
        f"valid_acc={val_accuracy:.4f} "
        f"({val_result['correct']}/"
        f"{val_result['total']})"
    )

    history.append(
        {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "valid_accuracy": val_accuracy,
            "lr": current_lr,
        }
    )

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_epoch = epoch + 1

        adapter_state = get_peft_model_state_dict(
            model
        )

        best_state = {
            key: value.detach().cpu().clone()
            for key, value in adapter_state.items()
        }

        print(
            "Best checkpoint updated | "
            f"epoch={best_epoch}, "
            f"accuracy={best_val_accuracy:.4f}"
        )

# ============================================================
# 21. Restore / save best model
# ============================================================

if best_state is None:
    raise RuntimeError(
        "Best adapter state가 생성되지 않았습니다."
    )

set_peft_model_state_dict(
    model,
    best_state,
)

model.eval()

history_df = pd.DataFrame(
    history
)

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

model.save_pretrained(
    SAVE_DIR
)

processor.save_pretrained(
    SAVE_DIR
)

history_df.to_csv(
    SAVE_DIR / "training_history.csv",
    index=False,
)

print("\n===== BEST MODEL =====")
print("Best epoch             :", best_epoch)
print("Best validation accuracy:", best_val_accuracy)
print("Saved                  :", SAVE_DIR)


# ============================================================
# 22. Final validation
# ============================================================

final_val = evaluate_accuracy(
    valid_subset,
    desc="Final validation",
)

final_val_predictions = final_val["predictions"]

print("\n===== FINAL VALIDATION =====")
print("Best epoch:", best_epoch)
print(
    "Accuracy:",
    f"{final_val['accuracy']:.4f}",
    f"({final_val['correct']}/"
    f"{final_val['total']})",
)

validation_prediction_rows = []

for idx, pred in enumerate(final_val_predictions):
    row = valid_subset.iloc[idx]
    gold = str(row["answer"]).strip().lower()

    validation_prediction_rows.append(
        {
            "id": row["id"],
            "gold": gold,
            "pred": pred,
            "correct": pred == gold,
        }
    )

pd.DataFrame(
    validation_prediction_rows
).to_csv(
    VALIDATION_PREDICTION_PATH,
    index=False,
)

print(
    "Saved validation predictions:",
    VALIDATION_PREDICTION_PATH,
)


# ============================================================
# 23. Test inference
# ============================================================

preds = []
score_rows = []

for idx in tqdm(
    range(len(test_df)),
    desc="Test inference",
    unit="sample",
):
    row = test_df.iloc[idx]

    pred, scores = predict_row(
        row,
        return_scores=True,
    )

    preds.append(pred)

    score_rows.append(
        {
            "id": row["id"],
            "pred": pred,
            **{
                f"logit_{choice}": scores[choice]
                for choice in CHOICES
            },
        }
    )

    if idx < 10:
        print(
            f"\n{idx}: "
            f"pred={pred}, "
            f"scores={scores}"
        )


# ============================================================
# 24. Prediction diagnostics / submission
# ============================================================

pred_series = pd.Series(
    preds
)

print(
    "\n===== TEST PREDICTION DISTRIBUTION ====="
)

print(
    pred_series
    .value_counts()
    .sort_index()
)

print(
    "\n===== TEST PREDICTION RATIO ====="
)

print(
    pred_series
    .value_counts(normalize=True)
    .sort_index()
)

submission = pd.DataFrame(
    {
        "id": test_df["id"],
        "answer": preds,
    }
)

if len(submission) != len(test_df):
    raise RuntimeError(
        "Submission row count mismatch."
    )

if not set(
    submission["answer"].unique()
).issubset(valid_choices):
    raise RuntimeError(
        "Submission answer에 a/b/c/d 외 값이 있습니다."
    )

submission.to_csv(
    SUBMISSION_PATH,
    index=False,
)

pd.DataFrame(
    score_rows
).to_csv(
    LOGIT_PATH,
    index=False,
)

print("\n===== DONE =====")
print("Submission :", SUBMISSION_PATH)
print("Test logits:", LOGIT_PATH)
print("Rows       :", len(submission))
print(submission.head(10))
