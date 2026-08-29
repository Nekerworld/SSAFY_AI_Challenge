# ============================================================
# SSAFY-8B Simple Baseline
#
# Purpose
# ------------------------------------------------------------
# 복잡한 SSAFY-T / SSAFY-T+ 기능을 모두 제거하고,
# Qwen3-VL-8B 자체의 순수 backbone 성능을 확인하기 위한 baseline.
#
# Architecture
# ------------------------------------------------------------
# - Qwen3-VL-8B-Instruct
# - 4-bit NF4 QLoRA
# - LoRA r=16 / alpha=32
# - Language LoRA only
# - Gold train only
# - 90% train / 10% fixed validation
# - 1 epoch
# - LR = 1e-4
# - assistant answer token only CE loss
# - direct a/b/c/d next-token logit inference
#
# NOT USED
# ------------------------------------------------------------
# - Surprise Replay
# - Margin Loss
# - Vision LoRA
# - DEV pseudo labels
# - Augmentation
# - Option permutation
# - Semantic scorer
# - Second pass
# - Ensemble
# ============================================================


# ============================================================
# 0. Imports
# ============================================================

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
)

from tqdm.auto import tqdm


# ============================================================
# 1. Project / Config
# ============================================================

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd().resolve()


MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

SEED = 42


# ------------------------------------------------------------
# Image resolution
#
# 3090 Ti 24GB 기준으로 보수적인 설정.
# 성능 테스트 후 MAX_PIXELS를 512 * 28 * 28까지
# 올려볼 수 있음.
# ------------------------------------------------------------

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 448 * 28 * 28


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

NUM_EPOCHS = 1

LR = 1e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.03

MAX_GRAD_NORM = 1.0


# ------------------------------------------------------------
# RTX 3090 Ti 24GB
#
# 먼저 batch=2 권장.
#
# OOM 발생:
# TRAIN_BATCH_SIZE = 1
# GRAD_ACCUM = 16
#
# 로 변경.
# ------------------------------------------------------------

TRAIN_BATCH_SIZE = 2
GRAD_ACCUM = 8


# ------------------------------------------------------------
# Validation / Test
# ------------------------------------------------------------

EVAL_BATCH_SIZE = 4


# ------------------------------------------------------------
# LoRA
#
# Discussion 조건
# ------------------------------------------------------------

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ============================================================
# 2. Paths
# ============================================================

TRAIN_CSV = PROJECT_DIR / "train.csv"
DEV_CSV = PROJECT_DIR / "dev.csv"
TEST_CSV = PROJECT_DIR / "test.csv"

SAVE_DIR = (
    PROJECT_DIR
    / "model"
    / "qwen3_vl_8b_simple"
)

SUBMISSION_DIR = (
    PROJECT_DIR
    / "submission"
)

SUBMISSION_PATH = (
    SUBMISSION_DIR
    / "submission_8b_simple.csv"
)

VALIDATION_PATH = (
    SUBMISSION_DIR
    / "validation_8b_simple.csv"
)

LOGIT_PATH = (
    SUBMISSION_DIR
    / "logits_8b_simple.csv"
)


Image.MAX_IMAGE_PIXELS = None


# ============================================================
# 3. Reproducibility / CUDA
# ============================================================

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 필요합니다."
    )

torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda:0")

BF16_SUPPORTED = (
    torch.cuda.is_bf16_supported()
)

COMPUTE_DTYPE = (
    torch.bfloat16
    if BF16_SUPPORTED
    else torch.float16
)

USE_SCALER = (
    COMPUTE_DTYPE == torch.float16
)


print("Project       :", PROJECT_DIR)
print("GPU           :", torch.cuda.get_device_name(0))
print("Compute dtype :", COMPUTE_DTYPE)
print("GradScaler    :", USE_SCALER)


# ============================================================
# 4. Helpers
# ============================================================

CHOICES = (
    "a",
    "b",
    "c",
    "d",
)

VALID_CHOICES = set(
    CHOICES
)


def resolve_image_path(path_value):

    path = Path(
        str(path_value)
    )

    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path


def load_rgb_image(path_value):

    path = resolve_image_path(
        path_value
    )

    with Image.open(path) as image:
        return image.convert("RGB")


def validate_image_paths(
    df,
    name,
):

    missing = []

    for value in df["path"]:

        path = resolve_image_path(
            value
        )

        if not path.exists():

            missing.append(
                str(path)
            )

            if len(missing) >= 10:
                break

    if missing:

        raise FileNotFoundError(
            f"{name} 이미지 누락:\n"
            + "\n".join(missing)
        )


def autocast_context():

    return torch.autocast(
        device_type="cuda",
        dtype=COMPUTE_DTYPE,
    )


def sanitize_inputs(inputs):

    # 일부 processor 환경에서 생성될 수 있음.
    inputs.pop(
        "token_type_ids",
        None,
    )

    return inputs


# ============================================================
# 5. Load Data
# ============================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

test_df = pd.read_csv(
    TEST_CSV
)


TRAIN_REQUIRED = {
    "id",
    "path",
    "question",
    "a",
    "b",
    "c",
    "d",
    "answer",
}

TEST_REQUIRED = {
    "id",
    "path",
    "question",
    "a",
    "b",
    "c",
    "d",
}


def require_columns(
    df,
    required,
    name,
):

    missing = (
        required
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"{name} missing columns: "
            f"{sorted(missing)}"
        )


require_columns(
    train_df,
    TRAIN_REQUIRED,
    "train.csv",
)

require_columns(
    test_df,
    TEST_REQUIRED,
    "test.csv",
)


# ------------------------------------------------------------
# Answer validation
# ------------------------------------------------------------

answers = (
    train_df["answer"]
    .astype(str)
    .str.strip()
    .str.lower()
)

if not set(
    answers.unique()
).issubset(
    VALID_CHOICES
):

    raise ValueError(
        "train answer에 "
        "a/b/c/d 이외 값이 존재합니다."
    )


validate_image_paths(
    train_df,
    "train"
)

validate_image_paths(
    test_df,
    "test"
)


# ============================================================
# 6. Fixed 90 / 10 Split
#
# 5,073개라면 대략
#
# train      ≈ 4,565
# validation ≈   508
# ============================================================

train_subset, valid_subset = (
    train_test_split(
        train_df,

        test_size=0.10,

        random_state=SEED,

        stratify=(
            train_df["answer"]
        ),
    )
)


train_subset = (
    train_subset
    .reset_index(drop=True)
)

valid_subset = (
    valid_subset
    .reset_index(drop=True)
)

test_df = (
    test_df
    .reset_index(drop=True)
)


print("\n===== DATA =====")

print(
    "Total train :",
    len(train_df)
)

print(
    "Train       :",
    len(train_subset)
)

print(
    "Validation  :",
    len(valid_subset)
)

print(
    "Test        :",
    len(test_df)
)


print(
    "\nTrain label distribution:"
)

print(
    train_subset[
        "answer"
    ]
    .value_counts()
    .sort_index()
)


# ============================================================
# 7. Processor
# ============================================================

processor = (
    AutoProcessor.from_pretrained(
        MODEL_ID,

        min_pixels=MIN_PIXELS,

        max_pixels=MAX_PIXELS,

        trust_remote_code=True,
    )
)


# inference batching을 위해 left padding

processor.tokenizer.padding_side = (
    "left"
)


if (
    processor.tokenizer.pad_token_id
    is None
):

    processor.tokenizer.pad_token = (
        processor.tokenizer.eos_token
    )


print(
    "\nPadding side:",
    processor.tokenizer.padding_side
)


# ============================================================
# 8. 4-bit NF4 Model
# ============================================================

bnb_config = (
    BitsAndBytesConfig(

        load_in_4bit=True,

        bnb_4bit_use_double_quant=True,

        bnb_4bit_quant_type="nf4",

        bnb_4bit_compute_dtype=(
            COMPUTE_DTYPE
        ),
    )
)


base_model = (
    Qwen3VLForConditionalGeneration
    .from_pretrained(

        MODEL_ID,

        quantization_config=(
            bnb_config
        ),

        device_map={
            "": 0
        },

        trust_remote_code=True,

        # Windows에서 flash-attn 설치가 번거로우므로
        # PyTorch SDPA 사용.
        attn_implementation="sdpa",
    )
)


base_model.config.use_cache = False


# ============================================================
# 9. Prepare QLoRA
# ============================================================

base_model = (
    prepare_model_for_kbit_training(

        base_model,

        use_gradient_checkpointing=True,
    )
)


# ============================================================
# 10. Language-only LoRA
#
# Discussion baseline을 단순하게 재현하기 위해
# Vision LoRA는 사용하지 않음.
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


for name, module in (
    base_model.named_modules()
):

    lower_name = (
        name.lower()
    )

    # Vision tower 제외
    if (
        "vision" in lower_name
        or
        "visual" in lower_name
    ):
        continue

    if any(
        name.endswith(suffix)
        for suffix
        in LLM_TARGET_SUFFIXES
    ):

        target_modules.append(
            name
        )


target_modules = sorted(
    set(target_modules)
)


if not target_modules:

    raise RuntimeError(
        "LoRA target module을 "
        "찾지 못했습니다."
    )


print(
    "\n===== LoRA ====="
)

print(
    "Target modules:",
    len(target_modules)
)

print(
    "r:",
    LORA_R
)

print(
    "alpha:",
    LORA_ALPHA
)


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
# 11. Prompt
# ============================================================

SYSTEM_INSTRUCT = (
    "You are a visual multiple-choice "
    "question answering assistant. "
    "Inspect the image and question carefully. "
    "Answer using exactly one lowercase letter: "
    "a, b, c, or d. "
    "Do not provide any explanation."
)


def build_mc_prompt(
    question,
    a,
    b,
    c,
    d,
):

    return (
        f"{question}\n\n"
        f"(a) {a}\n"
        f"(b) {b}\n"
        f"(c) {c}\n"
        f"(d) {d}\n\n"
        "정답을 a, b, c, d 중 하나의 "
        "소문자 한 글자로만 출력하세요."
    )


def build_prompt_messages(
    row,
    image,
):

    user_text = build_mc_prompt(

        str(
            row["question"]
        ),

        str(
            row["a"]
        ),

        str(
            row["b"]
        ),

        str(
            row["c"]
        ),

        str(
            row["d"]
        ),
    )


    return [

        {
            "role":
                "system",

            "content": [
                {
                    "type":
                        "text",

                    "text":
                        SYSTEM_INSTRUCT,
                }
            ],
        },

        {
            "role":
                "user",

            "content": [

                {
                    "type":
                        "image",

                    "image":
                        image,
                },

                {
                    "type":
                        "text",

                    "text":
                        user_text,
                },
            ],
        },
    ]


# ============================================================
# 12. Dataset
# ============================================================

class VQADataset(Dataset):

    def __init__(
        self,
        df,
    ):

        self.df = (
            df.reset_index(
                drop=True
            )
        )


    def __len__(
        self,
    ):

        return len(
            self.df
        )


    def __getitem__(
        self,
        idx,
    ):

        row = (
            self.df.iloc[
                idx
            ]
        )


        image = load_rgb_image(
            row["path"]
        )


        gold = (
            str(
                row["answer"]
            )
            .strip()
            .lower()
        )


        if gold not in VALID_CHOICES:

            raise ValueError(
                f"Invalid answer: {gold}"
            )


        prompt_messages = (
            build_prompt_messages(
                row,
                image,
            )
        )


        full_messages = (
            prompt_messages
            + [
                {
                    "role":
                        "assistant",

                    "content": [
                        {
                            "type":
                                "text",

                            "text":
                                gold,
                        }
                    ],
                }
            ]
        )


        return {

            "image":
                image,

            "prompt_messages":
                prompt_messages,

            "full_messages":
                full_messages,
        }


# ============================================================
# 13. Assistant-only Loss Collator
#
# loss는 정답 a/b/c/d 한 글자에만 적용.
#
# prompt/system/question/options에는 loss 없음.
# ============================================================

@dataclass
class TrainCollator:

    processor: Any


    def __call__(
        self,
        batch,
    ):

        images = [
            sample[
                "image"
            ]
            for sample
            in batch
        ]


        prompt_texts = []

        full_texts = []


        for sample in batch:

            prompt_text = (
                self.processor
                .apply_chat_template(

                    sample[
                        "prompt_messages"
                    ],

                    tokenize=False,

                    add_generation_prompt=True,
                )
            )


            full_text = (
                self.processor
                .apply_chat_template(

                    sample[
                        "full_messages"
                    ],

                    tokenize=False,

                    add_generation_prompt=False,
                )
            )


            prompt_texts.append(
                prompt_text
            )

            full_texts.append(
                full_text
            )


        # ----------------------------------------------------
        # Full sequence
        # ----------------------------------------------------

        full_enc = self.processor(

            text=full_texts,

            images=images,

            padding=True,

            return_tensors="pt",
        )


        # ----------------------------------------------------
        # Prompt-only sequence
        # ----------------------------------------------------

        prompt_enc = self.processor(

            text=prompt_texts,

            images=images,

            padding=True,

            return_tensors="pt",
        )


        full_enc = sanitize_inputs(
            full_enc
        )

        prompt_enc = sanitize_inputs(
            prompt_enc
        )


        # ----------------------------------------------------
        # labels = -100 everywhere
        # ----------------------------------------------------

        labels = torch.full_like(

            full_enc[
                "input_ids"
            ],

            fill_value=-100,
        )


        # ----------------------------------------------------
        # 정답 첫 token만 supervision
        # ----------------------------------------------------

        for i in range(
            len(batch)
        ):

            full_positions = (

                full_enc[
                    "attention_mask"
                ][i]

                .nonzero(
                    as_tuple=False
                )

                .squeeze(-1)
            )


            prompt_positions = (

                prompt_enc[
                    "attention_mask"
                ][i]

                .nonzero(
                    as_tuple=False
                )

                .squeeze(-1)
            )


            full_ids = (

                full_enc[
                    "input_ids"
                ][
                    i,
                    full_positions
                ]
            )


            prompt_ids = (

                prompt_enc[
                    "input_ids"
                ][
                    i,
                    prompt_positions
                ]
            )


            prompt_len = int(
                prompt_ids.numel()
            )


            full_len = int(
                full_ids.numel()
            )


            if prompt_len >= full_len:

                raise RuntimeError(
                    "Prompt/full length error."
                )


            # 실제 prefix인지 확인

            if not torch.equal(

                full_ids[
                    :prompt_len
                ].cpu(),

                prompt_ids.cpu(),
            ):

                raise RuntimeError(
                    "Prompt/full token prefix mismatch."
                )


            answer_positions = (

                full_positions[
                    prompt_len:
                ]
            )


            answer_position = (
                answer_positions[0]
            )


            labels[
                i,
                answer_position
            ] = (

                full_enc[
                    "input_ids"
                ][
                    i,
                    answer_position
                ]
            )


        full_enc[
            "labels"
        ] = labels


        return full_enc


# ============================================================
# 14. DataLoader
# ============================================================

train_ds = VQADataset(
    train_subset
)


train_loader = DataLoader(

    train_ds,

    batch_size=(
        TRAIN_BATCH_SIZE
    ),

    shuffle=True,

    collate_fn=(
        TrainCollator(
            processor
        )
    ),

    num_workers=0,

    pin_memory=False,
)


print(
    "\nTrain batches:",
    len(train_loader)
)


# ============================================================
# 15. Label sanity check
# ============================================================

debug_batch = next(
    iter(train_loader)
)


debug_labels = (
    debug_batch[
        "labels"
    ][0]
)


target_ids = (
    debug_labels[
        debug_labels != -100
    ]
)


target_text = (
    processor.tokenizer.decode(

        target_ids,

        skip_special_tokens=False,
    )
)


print(
    "\n===== LABEL CHECK ====="
)

print(
    repr(
        target_text
    )
)

print(
    "=======================\n"
)


if not any(
    choice
    in target_text.lower()

    for choice
    in CHOICES
):

    raise RuntimeError(
        "Answer-token masking failed."
    )


# ============================================================
# 16. Discover actual a/b/c/d token IDs
# ============================================================

def discover_choice_token_ids():

    dummy_image = Image.new(
        "RGB",
        (28, 28),
    )


    dummy_row = {

        "question":
            "Choose the correct option.",

        "a":
            "A",

        "b":
            "B",

        "c":
            "C",

        "d":
            "D",
    }


    prompt_messages = (
        build_prompt_messages(

            dummy_row,

            dummy_image,
        )
    )


    prompt_text = (
        processor
        .apply_chat_template(

            prompt_messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )


    prompt_ids = (
        processor.tokenizer(

            prompt_text,

            add_special_tokens=False,
        )[
            "input_ids"
        ]
    )


    result = {}


    for choice in CHOICES:

        full_messages = (

            prompt_messages
            + [
                {
                    "role":
                        "assistant",

                    "content": [
                        {
                            "type":
                                "text",

                            "text":
                                choice,
                        }
                    ],
                }
            ]
        )


        full_text = (
            processor
            .apply_chat_template(

                full_messages,

                tokenize=False,

                add_generation_prompt=False,
            )
        )


        full_ids = (
            processor.tokenizer(

                full_text,

                add_special_tokens=False,
            )[
                "input_ids"
            ]
        )


        if (
            full_ids[
                :len(prompt_ids)
            ]
            != prompt_ids
        ):

            raise RuntimeError(
                f"Choice token prefix mismatch: {choice}"
            )


        suffix = (
            full_ids[
                len(prompt_ids):
            ]
        )


        if not suffix:

            raise RuntimeError(
                f"No token for choice={choice}"
            )


        result[
            choice
        ] = suffix[0]


        print(
            f"{choice}: "
            f"id={suffix[0]}, "
            f"decoded="
            f"{processor.tokenizer.decode([suffix[0]])!r}"
        )


    if (
        len(
            set(
                result.values()
            )
        )
        != 4
    ):

        raise RuntimeError(
            "a/b/c/d first token IDs are not unique."
        )


    return result


choice_token_ids = (
    discover_choice_token_ids()
)


CHOICE_TOKEN_TENSOR = (
    torch.tensor(

        [
            choice_token_ids[
                choice
            ]
            for choice
            in CHOICES
        ],

        dtype=torch.long,

        device=DEVICE,
    )
)


# ============================================================
# 17. Optimizer / Scheduler
# ============================================================

trainable_params = [

    parameter

    for parameter
    in model.parameters()

    if parameter.requires_grad
]


optimizer = (
    torch.optim.AdamW(

        trainable_params,

        lr=LR,

        weight_decay=(
            WEIGHT_DECAY
        ),
    )
)


steps_per_epoch = (
    math.ceil(

        len(train_loader)
        /
        GRAD_ACCUM
    )
)


total_steps = (

    NUM_EPOCHS
    *
    steps_per_epoch
)


warmup_steps = int(

    total_steps
    *
    WARMUP_RATIO
)


scheduler = (
    get_linear_schedule_with_warmup(

        optimizer,

        num_warmup_steps=(
            warmup_steps
        ),

        num_training_steps=(
            total_steps
        ),
    )
)


scaler = (
    torch.amp.GradScaler(

        "cuda",

        enabled=USE_SCALER,
    )
)


print(
    "\n===== OPTIMIZER ====="
)

print(
    "Effective batch:",
    TRAIN_BATCH_SIZE
    * GRAD_ACCUM
)

print(
    "Optimizer steps:",
    total_steps
)

print(
    "Warmup steps:",
    warmup_steps
)


# ============================================================
# 18. Training
# ============================================================

optimizer.zero_grad(
    set_to_none=True
)


for epoch in range(
    NUM_EPOCHS
):

    model.train()

    model.config.use_cache = False


    running_loss = 0.0

    batch_count = 0


    num_batches = len(
        train_loader
    )


    bar = tqdm(

        train_loader,

        desc=(
            f"Epoch "
            f"{epoch + 1}/"
            f"{NUM_EPOCHS}"
        ),

        unit="batch",
    )


    for step, batch in enumerate(
        bar,
        start=1,
    ):

        batch = sanitize_inputs(
            batch
        )


        batch = {

            key:
                value.to(
                    DEVICE
                )

            for key, value
            in batch.items()
        }


        # ----------------------------------------------------
        # Last incomplete accumulation group correction
        # ----------------------------------------------------

        group_start = (

            (
                (step - 1)
                // GRAD_ACCUM
            )

            * GRAD_ACCUM

            + 1
        )


        accum_divisor = min(

            GRAD_ACCUM,

            num_batches
            - group_start
            + 1,
        )


        with autocast_context():

            outputs = model(

                **batch,

                use_cache=False,
            )


            raw_loss = (
                outputs.loss
            )


            loss = (
                raw_loss
                /
                accum_divisor
            )


        if not torch.isfinite(
            raw_loss
        ).item():

            raise RuntimeError(
                f"Non-finite loss: "
                f"{raw_loss.item()}"
            )


        if USE_SCALER:

            scaler.scale(
                loss
            ).backward()

        else:

            loss.backward()


        running_loss += (

            raw_loss
            .detach()
            .float()
            .item()
        )


        batch_count += 1


        should_step = (

            step
            % GRAD_ACCUM
            == 0

            or

            step
            == num_batches
        )


        if should_step:

            if USE_SCALER:

                scaler.unscale_(
                    optimizer
                )


            torch.nn.utils.clip_grad_norm_(

                trainable_params,

                MAX_GRAD_NORM,
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


        avg_loss = (

            running_loss
            /
            batch_count
        )


        bar.set_postfix(

            loss=(
                f"{avg_loss:.4f}"
            ),

            lr=(
                f"{scheduler.get_last_lr()[0]:.2e}"
            ),
        )


train_loss = (

    running_loss
    /
    max(
        batch_count,
        1
    )
)


print(
    "\n===== TRAIN COMPLETE ====="
)

print(
    "Train loss:",
    train_loss
)


# ============================================================
# 19. Batched direct MC inference
# ============================================================

def predict_logits(
    df,
    desc,
):

    model.eval()


    all_logits = []


    for start in tqdm(

        range(
            0,
            len(df),
            EVAL_BATCH_SIZE,
        ),

        desc=desc,

        unit="batch",
    ):

        end = min(

            start
            + EVAL_BATCH_SIZE,

            len(df),
        )


        part = df.iloc[
            start:end
        ]


        texts = []

        images = []


        for _, row in (
            part.iterrows()
        ):

            image = load_rgb_image(
                row["path"]
            )


            messages = (
                build_prompt_messages(
                    row,
                    image,
                )
            )


            text = (
                processor
                .apply_chat_template(

                    messages,

                    tokenize=False,

                    add_generation_prompt=True,
                )
            )


            texts.append(
                text
            )


            images.append(
                image
            )


        inputs = (
            processor(

                text=texts,

                images=images,

                padding=True,

                return_tensors="pt",
            )
        )


        inputs = sanitize_inputs(
            inputs
        )


        inputs = {

            key:
                value.to(
                    DEVICE
                )

            for key, value
            in inputs.items()
        }


        with (
            torch.inference_mode(),
            autocast_context(),
        ):

            outputs = model(

                **inputs,

                use_cache=False,

                # 마지막 token position만 vocabulary projection
                logits_to_keep=1,
            )


            vocab_logits = (

                outputs.logits[
                    :,
                    -1,
                    :
                ]

                .float()
            )


            choice_logits = (

                vocab_logits
                .index_select(

                    dim=-1,

                    index=(
                        CHOICE_TOKEN_TENSOR
                    ),
                )
            )


        all_logits.append(
            choice_logits.cpu()
        )


    return torch.cat(
        all_logits,
        dim=0,
    )


# ============================================================
# 20. Validation
# ============================================================

validation_logits = (
    predict_logits(

        valid_subset,

        desc="Validation",
    )
)


validation_pred_idx = (

    validation_logits
    .argmax(
        dim=1
    )
)


validation_predictions = [

    CHOICES[
        int(index)
    ]

    for index
    in validation_pred_idx.tolist()
]


validation_gold = [

    str(answer)
    .strip()
    .lower()

    for answer
    in valid_subset[
        "answer"
    ]
]


correct = sum(

    pred == gold

    for pred, gold
    in zip(

        validation_predictions,

        validation_gold,
    )
)


validation_accuracy = (

    correct
    /
    len(valid_subset)
)


print(
    "\n===== VALIDATION ====="
)

print(
    f"Accuracy: "
    f"{validation_accuracy:.5f}"
)

print(
    f"Correct : "
    f"{correct}/"
    f"{len(valid_subset)}"
)


# ============================================================
# 21. Save validation predictions
# ============================================================

SUBMISSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


validation_output = (
    pd.DataFrame(
        {
            "id":
                valid_subset[
                    "id"
                ],

            "gold":
                validation_gold,

            "pred":
                validation_predictions,

            "correct":
                [
                    p == g

                    for p, g
                    in zip(
                        validation_predictions,
                        validation_gold,
                    )
                ],
        }
    )
)


validation_output.to_csv(

    VALIDATION_PATH,

    index=False,
)


# ============================================================
# 22. Save LoRA Adapter
# ============================================================

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


print(
    "\nSaved model:",
    SAVE_DIR
)


# ============================================================
# 23. Test inference
# ============================================================

test_logits = (
    predict_logits(

        test_df,

        desc="Test",
    )
)


test_pred_idx = (

    test_logits
    .argmax(
        dim=1
    )
)


test_predictions = [

    CHOICES[
        int(index)
    ]

    for index
    in test_pred_idx.tolist()
]


# ============================================================
# 24. Prediction diagnostics
# ============================================================

print(
    "\n===== TEST DISTRIBUTION ====="
)


print(
    pd.Series(
        test_predictions
    )
    .value_counts()
    .sort_index()
)


print(
    "\n===== TEST RATIO ====="
)


print(
    pd.Series(
        test_predictions
    )
    .value_counts(
        normalize=True
    )
    .sort_index()
)


# ============================================================
# 25. Submission
# ============================================================

submission = (
    pd.DataFrame(
        {
            "id":
                test_df[
                    "id"
                ],

            "answer":
                test_predictions,
        }
    )
)


if (
    len(submission)
    != len(test_df)
):

    raise RuntimeError(
        "Submission row mismatch."
    )


if not set(
    submission[
        "answer"
    ].unique()
).issubset(
    VALID_CHOICES
):

    raise RuntimeError(
        "Invalid answer detected."
    )


submission.to_csv(

    SUBMISSION_PATH,

    index=False,
)


# ============================================================
# 26. Save logits
# ============================================================

logit_output = (
    pd.DataFrame(
        {
            "id":
                test_df[
                    "id"
                ],

            "pred":
                test_predictions,

            "logit_a":
                test_logits[
                    :,
                    0
                ].numpy(),

            "logit_b":
                test_logits[
                    :,
                    1
                ].numpy(),

            "logit_c":
                test_logits[
                    :,
                    2
                ].numpy(),

            "logit_d":
                test_logits[
                    :,
                    3
                ].numpy(),
        }
    )
)


logit_output.to_csv(

    LOGIT_PATH,

    index=False,
)


# ============================================================
# 27. Final summary
# ============================================================

print(
    "\n"
    + "=" * 60
)

print(
    "SSAFY 8B SIMPLE BASELINE"
)

print(
    "=" * 60
)


print(
    "Model      :",
    MODEL_ID
)

print(
    "LoRA       :",
    f"r={LORA_R}, "
    f"alpha={LORA_ALPHA}"
)

print(
    "Epoch      :",
    NUM_EPOCHS
)

print(
    "LR         :",
    LR
)

print(
    "Train size :",
    len(train_subset)
)

print(
    "Valid size :",
    len(valid_subset)
)

print(
    "Train loss :",
    train_loss
)

print(
    "Valid acc  :",
    validation_accuracy
)

print(
    "Submission :",
    SUBMISSION_PATH
)

print(
    "=" * 60
)