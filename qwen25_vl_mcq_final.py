# ============================================================
# Qwen3-VL-2B + 4bit QLoRA
# SSAFY Recycling VQA Multiple Choice
#
# Features
# ------------------------------------------------------------
# 1. Gold train 90/10 stratified split
# 2. Qwen3-VL-2B-Instruct
# 3. 4bit NF4 QLoRA
# 4. Language-model LoRA only
# 5. Direct 4-choice Cross Entropy
# 6. Hard-negative margin loss
# 7. Titans-inspired surprise replay
# 8. Batched validation / test inference
# 9. logits_to_keep=1 for memory/speed
# 10. Confidence-based high-resolution second pass
# 11. Validation-based second-pass threshold calibration
# 12. Optional dev pseudo-label Stage 2
#     - human agreement >= 4/5
#     - teacher prediction agreement
#     - teacher confidence margin
# 13. Best checkpoint restoration
# 14. Final submission + diagnostic logits
# ============================================================


# ============================================================
# 0. Imports
# ============================================================

import os
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
)

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
# 1. Project / Config
# ============================================================

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd().resolve()


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------

SEED = 42


# ------------------------------------------------------------
# Train data
#
# None = train pool 전체 사용
#
# 성능을 목표로 한다면 None 권장
# ------------------------------------------------------------

TRAIN_LIMIT = None


# ------------------------------------------------------------
# Image resolution
#
# Pass 1:
# 비교적 저렴한 해상도
#
# Pass 2:
# confidence가 낮은 문제만 고해상도
# ------------------------------------------------------------

LOW_MIN_PIXELS = 224 * 224
LOW_MAX_PIXELS = 384 * 384

HIGH_MIN_PIXELS = 256 * 28 * 28
HIGH_MAX_PIXELS = 768 * 28 * 28


# ------------------------------------------------------------
# Stage 1 training
#
# 전체 4,500개 정도를 쓰므로 예전의 200개 × 20epoch보다
# epoch 수를 훨씬 줄이는 것이 적절함.
# ------------------------------------------------------------

NUM_EPOCHS = 8

EVAL_EVERY = 2


# ------------------------------------------------------------
# RTX 3090 Ti 24GB
#
# 먼저 batch=8 시도.
# OOM이면 4로 내릴 것.
# ------------------------------------------------------------

TRAIN_BATCH_SIZE = 8

VALID_BATCH_SIZE = 8

TEST_BATCH_SIZE = 8

GRAD_ACCUM = 2


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------

LR = 1e-4

WEIGHT_DECAY = 0.01

MAX_GRAD_NORM = 1.0

WARMUP_RATIO = 0.03


# ------------------------------------------------------------
# LoRA
# ------------------------------------------------------------

LORA_R = 8

LORA_ALPHA = 16

LORA_DROPOUT = 0.05


# ============================================================
# 2. MC loss
# ============================================================

# 4-choice CE가 primary objective

MC_CE_WEIGHT = 1.0


# ------------------------------------------------------------
# Hard-negative ranking
#
# correct logit이 hardest wrong보다
# 최소 MARGIN 만큼 높게 만들기
# ------------------------------------------------------------

USE_MARGIN_LOSS = True

MARGIN_LOSS_WEIGHT = 0.20

MARGIN = 1.0


# ============================================================
# 3. Surprise Replay
#
# Titans의 surprise-memory 개념을
# hard-example replay로 단순화
# ============================================================

USE_SURPRISE_REPLAY = True

SURPRISE_REPLAY_RATIO = 0.25

SURPRISE_EMA_BETA = 0.80


# ============================================================
# 4. Second-pass inference
# ============================================================

USE_SECOND_PASS = True


# validation에서 탐색할 margin threshold

SECOND_PASS_THRESHOLDS = [
    0.0,
    0.10,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
]


# 높은 해상도 second-pass logit 비중

SECOND_PASS_WEIGHT = 0.70


# ============================================================
# 5. Optional DEV pseudo-label Stage 2
# ============================================================

USE_DEV_PSEUDO_STAGE = True


# 사람 5명 중 최소 4명 일치

PSEUDO_MIN_HUMAN_CONF = 0.80


# teacher top1-top2 margin

PSEUDO_MIN_TEACHER_MARGIN = 1.0


# pseudo label의 loss 영향

PSEUDO_BASE_WEIGHT = 0.50


# Stage 2

PSEUDO_EPOCHS = 2

PSEUDO_LR = 2e-5


# ============================================================
# 6. Paths
# ============================================================

TRAIN_CSV = PROJECT_DIR / "train.csv"

DEV_CSV = PROJECT_DIR / "dev.csv"

TEST_CSV = PROJECT_DIR / "test.csv"


SAVE_DIR = (
    PROJECT_DIR
    / "model"
    / "qwen3_vl_2b_vqa"
)


SUBMISSION_DIR = (
    PROJECT_DIR
    / "submission"
)


SUBMISSION_PATH = (
    SUBMISSION_DIR
    / "submission.csv"
)


VALIDATION_PREDICTION_PATH = (
    SUBMISSION_DIR
    / "validation_predictions.csv"
)


TEST_LOGIT_PATH = (
    SUBMISSION_DIR
    / "prediction_logits.csv"
)


PSEUDO_PATH = (
    SUBMISSION_DIR
    / "dev_pseudo_labels.csv"
)


Image.MAX_IMAGE_PIXELS = None


# ============================================================
# 7. Seeds / CUDA
# ============================================================

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU가 필요합니다."
    )


torch.cuda.manual_seed_all(SEED)


DEVICE = torch.device(
    "cuda:0"
)


BF16_SUPPORTED = (
    torch.cuda.is_bf16_supported()
)


COMPUTE_DTYPE = (
    torch.bfloat16
    if BF16_SUPPORTED
    else torch.float16
)


USE_SCALER = (
    COMPUTE_DTYPE
    == torch.float16
)


print(
    "Project dir  :",
    PROJECT_DIR
)

print(
    "GPU          :",
    torch.cuda.get_device_name(0)
)

print(
    "Compute dtype:",
    COMPUTE_DTYPE
)

print(
    "GradScaler   :",
    USE_SCALER
)


# ============================================================
# 8. General helpers
# ============================================================

CHOICES = (
    "a",
    "b",
    "c",
    "d",
)


CHOICE_TO_INDEX = {
    choice: idx
    for idx, choice
    in enumerate(CHOICES)
}


VALID_CHOICES = set(
    CHOICES
)


DEV_ANSWER_COLS = [
    "answer1",
    "answer2",
    "answer3",
    "answer4",
    "answer5",
]


def resolve_image_path(
    path_value,
):

    path = Path(
        str(path_value)
    )

    if not path.is_absolute():

        path = (
            PROJECT_DIR
            / path
        )

    return path


def load_rgb_image(
    path_value,
):

    path = resolve_image_path(
        path_value
    )

    with Image.open(
        path
    ) as image:

        return image.convert(
            "RGB"
        )


def validate_image_paths(
    df,
    name,
):

    missing = []

    for path_value in df["path"]:

        path = resolve_image_path(
            path_value
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


def sanitize_inputs(
    inputs,
):

    # Qwen3-VL 환경에 따라 processor가
    # token_type_ids를 만들 수 있음.

    inputs.pop(
        "token_type_ids",
        None,
    )

    return inputs


# ============================================================
# 9. Load CSV
# ============================================================

train_df = pd.read_csv(
    TRAIN_CSV
)

dev_df = pd.read_csv(
    DEV_CSV
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


DEV_REQUIRED = {
    "id",
    "path",
    "question",
    "a",
    "b",
    "c",
    "d",
    *DEV_ANSWER_COLS,
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
            f"{name} missing: "
            f"{sorted(missing)}"
        )


require_columns(
    train_df,
    TRAIN_REQUIRED,
    "train.csv",
)

require_columns(
    dev_df,
    DEV_REQUIRED,
    "dev.csv",
)

require_columns(
    test_df,
    TEST_REQUIRED,
    "test.csv",
)


validate_image_paths(
    train_df,
    "train"
)

validate_image_paths(
    dev_df,
    "dev"
)

validate_image_paths(
    test_df,
    "test"
)


# ============================================================
# 10. Train / Validation split
#
# validation은 절대 pseudo training에 넣지 않음
# ============================================================

train_pool, valid_subset = (
    train_test_split(
        train_df,
        test_size=0.10,
        random_state=SEED,
        stratify=train_df["answer"],
    )
)


train_pool = (
    train_pool
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


# ============================================================
# 11. Optional train limit
# ============================================================

if (
    TRAIN_LIMIT is not None
    and TRAIN_LIMIT
    < len(train_pool)
):

    # answer 비율 보존

    train_subset, _ = (
        train_test_split(
            train_pool,
            train_size=TRAIN_LIMIT,
            random_state=SEED,
            stratify=train_pool[
                "answer"
            ],
        )
    )

    train_subset = (
        train_subset
        .reset_index(drop=True)
    )

else:

    train_subset = (
        train_pool.copy()
    )


# gold sample weight

train_subset[
    "sample_weight"
] = 1.0


print(
    "\n===== DATA ====="
)

print(
    "Original train:",
    len(train_df)
)

print(
    "Train pool    :",
    len(train_pool)
)

print(
    "Train used    :",
    len(train_subset)
)

print(
    "Validation    :",
    len(valid_subset)
)

print(
    "Dev           :",
    len(dev_df)
)

print(
    "Test          :",
    len(test_df)
)


# ============================================================
# 12. Processor
#
# 중요:
# padding_side='left'
#
# 이렇게 하면 batch에서도 마지막 token이
# 모든 샘플의 실제 generation position이 된다.
#
# 그 결과 logits_to_keep=1 사용 가능.
# ============================================================

processor = (
    AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=LOW_MIN_PIXELS,
        max_pixels=LOW_MAX_PIXELS,
        trust_remote_code=True,
    )
)


high_processor = (
    AutoProcessor.from_pretrained(
        MODEL_ID,
        min_pixels=HIGH_MIN_PIXELS,
        max_pixels=HIGH_MAX_PIXELS,
        trust_remote_code=True,
    )
)


for proc in [
    processor,
    high_processor,
]:

    proc.tokenizer.padding_side = (
        "left"
    )

    if (
        proc.tokenizer.pad_token_id
        is None
    ):

        proc.tokenizer.pad_token = (
            proc.tokenizer.eos_token
        )


print(
    "Padding side:",
    processor.tokenizer.padding_side
)


# ============================================================
# 13. Model
# ============================================================

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=(
        COMPUTE_DTYPE
    ),
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

        # Windows에서 flash-attn 설치 문제를 피하면서
        # PyTorch SDPA 사용
        attn_implementation="sdpa",
    )
)


base_model.config.use_cache = False


base_model = (
    prepare_model_for_kbit_training(
        base_model,
        use_gradient_checkpointing=True,
    )
)


# ============================================================
# 14. Language-only LoRA
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

    lower = name.lower()

    if (
        "vision" in lower
        or "visual" in lower
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
    "\nLoRA target count:",
    len(target_modules)
)


for name in target_modules[:20]:

    print(
        " ",
        name
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
# 15. Prompt
# ============================================================

SYSTEM_INSTRUCT = (
    "You are a visual multiple-choice "
    "question answering assistant. "
    "Inspect the image carefully. "
    "Answer using exactly one lowercase "
    "letter: a, b, c, or d. "
    "Do not explain."
)


def build_mc_prompt(
    question,
    a,
    b,
    c,
    d,
    focus_choices=None,
):

    base = (
        f"{question}\n\n"
        f"(a) {a}\n"
        f"(b) {b}\n"
        f"(c) {c}\n"
        f"(d) {d}\n"
    )

    if focus_choices is not None:

        x, y = focus_choices

        base += (
            "\nThe first evaluation was uncertain. "
            f"Pay particular attention to options "
            f"{x} and {y}. "
            "Inspect the visual evidence again and "
            "choose the better-supported answer.\n"
        )

    base += (
        "\n정답을 a, b, c, d 중 "
        "하나의 소문자 한 글자로만 출력하세요."
    )

    return base


def build_prompt_messages(
    row,
    image,
    focus_choices=None,
):

    user_text = build_mc_prompt(
        str(row["question"]),
        str(row["a"]),
        str(row["b"]),
        str(row["c"]),
        str(row["d"]),
        focus_choices=focus_choices,
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
# 16. Discover actual answer token IDs
# ============================================================

def discover_choice_token_ids():

    dummy_image = Image.new(
        "RGB",
        (28, 28)
    )

    dummy_row = {
        "question": "Choose one option.",
        "a": "A",
        "b": "B",
        "c": "C",
        "d": "D",
    }

    prompt_messages = (
        build_prompt_messages(
            dummy_row,
            dummy_image,
        )
    )

    prompt_text = (
        processor.apply_chat_template(
            prompt_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    )

    prompt_ids = (
        processor.tokenizer(
            prompt_text,
            add_special_tokens=False,
        )["input_ids"]
    )

    result = {}

    for choice in CHOICES:

        full_messages = (
            prompt_messages
            + [
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
        )

        full_text = (
            processor.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        )

        full_ids = (
            processor.tokenizer(
                full_text,
                add_special_tokens=False,
            )["input_ids"]
        )

        if (
            full_ids[
                :len(prompt_ids)
            ]
            != prompt_ids
        ):

            raise RuntimeError(
                "Chat-template prefix mismatch."
            )

        suffix = (
            full_ids[
                len(prompt_ids):
            ]
        )

        if not suffix:

            raise RuntimeError(
                f"No token for {choice}"
            )

        result[
            choice
        ] = suffix[0]

        print(
            f"{choice}: "
            f"{suffix[0]} -> "
            f"{processor.tokenizer.decode([suffix[0]])!r}"
        )

    if (
        len(
            set(result.values())
        )
        != 4
    ):

        raise RuntimeError(
            "a/b/c/d가 서로 다른 "
            "single first token이 아닙니다."
        )

    return result


choice_token_ids = (
    discover_choice_token_ids()
)


CHOICE_TOKEN_TENSOR = torch.tensor(
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


print(
    "Choice token IDs:",
    choice_token_ids
)


# ============================================================
# 17. Training Dataset
# ============================================================

class VQAMCDataset(Dataset):

    def __init__(
        self,
        df,
    ):

        self.df = (
            df.reset_index(
                drop=True
            )
        )


    def __len__(self):

        return len(
            self.df
        )


    def __getitem__(
        self,
        idx,
    ):

        row = (
            self.df.iloc[idx]
        )

        image = load_rgb_image(
            row["path"]
        )

        gold = (
            str(row["answer"])
            .strip()
            .lower()
        )

        if gold not in VALID_CHOICES:

            raise ValueError(
                f"Invalid answer: "
                f"{gold}"
            )

        messages = (
            build_prompt_messages(
                row,
                image,
            )
        )

        weight = float(
            row.get(
                "sample_weight",
                1.0,
            )
        )

        return {
            "messages": messages,
            "image": image,
            "target": (
                CHOICE_TO_INDEX[
                    gold
                ]
            ),
            "sample_idx": idx,
            "sample_weight": weight,
        }


# ============================================================
# 18. MC Training Collator
#
# 정답 token을 input에 넣지 않는다.
#
# assistant generation prompt 직후의
# next-token logits를 직접 4-way 분류에 사용.
# ============================================================

@dataclass
class MCTrainCollator:

    processor: Any


    def __call__(
        self,
        batch,
    ):

        texts = []

        images = []

        targets = []

        sample_indices = []

        sample_weights = []


        for sample in batch:

            text = (
                self.processor
                .apply_chat_template(
                    sample[
                        "messages"
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

            texts.append(
                text
            )

            images.append(
                sample["image"]
            )

            targets.append(
                sample["target"]
            )

            sample_indices.append(
                sample["sample_idx"]
            )

            sample_weights.append(
                sample[
                    "sample_weight"
                ]
            )


        enc = self.processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt",
        )


        enc = sanitize_inputs(
            enc
        )


        enc["mc_target"] = (
            torch.tensor(
                targets,
                dtype=torch.long,
            )
        )


        enc["sample_idx"] = (
            torch.tensor(
                sample_indices,
                dtype=torch.long,
            )
        )


        enc["sample_weight"] = (
            torch.tensor(
                sample_weights,
                dtype=torch.float32,
            )
        )


        return enc


# ============================================================
# 19. Build Loader with Surprise Replay
# ============================================================

train_ds = VQAMCDataset(
    train_subset
)


surprise_scores = np.ones(
    len(train_ds),
    dtype=np.float32,
)


def build_train_loader(
    dataset,
    surprise=None,
):

    base_indices = list(
        range(
            len(dataset)
        )
    )


    if (
        USE_SURPRISE_REPLAY
        and surprise is not None
        and len(dataset) > 0
    ):

        extra_count = int(
            len(dataset)
            * SURPRISE_REPLAY_RATIO
        )

        weights = np.asarray(
            surprise,
            dtype=np.float64,
        )

        weights = np.maximum(
            weights,
            1e-6,
        )


        replay_indices = (
            random.choices(
                base_indices,
                weights=weights.tolist(),
                k=extra_count,
            )
        )


        epoch_indices = (
            base_indices
            + replay_indices
        )

    else:

        epoch_indices = (
            base_indices
        )


    epoch_dataset = Subset(
        dataset,
        epoch_indices,
    )


    return DataLoader(
        epoch_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True,
        collate_fn=MCTrainCollator(
            processor
        ),
        num_workers=0,
        pin_memory=False,
    )


# ============================================================
# 20. Loss function
# ============================================================

def compute_mc_losses(
    choice_logits,
    target,
    sample_weight,
):

    # --------------------------------------------------------
    # CE per sample
    # --------------------------------------------------------

    ce_per_sample = (
        F.cross_entropy(
            choice_logits,
            target,
            reduction="none",
        )
    )


    # --------------------------------------------------------
    # hardest negative margin
    # --------------------------------------------------------

    correct_logits = (
        choice_logits
        .gather(
            1,
            target.unsqueeze(1),
        )
        .squeeze(1)
    )


    wrong_mask = torch.ones_like(
        choice_logits,
        dtype=torch.bool,
    )


    wrong_mask.scatter_(
        1,
        target.unsqueeze(1),
        False,
    )


    wrong_logits = (
        choice_logits
        .masked_fill(
            ~wrong_mask,
            float("-inf"),
        )
    )


    hardest_wrong = (
        wrong_logits.max(
            dim=1
        ).values
    )


    margin_per_sample = (
        F.relu(
            MARGIN
            - correct_logits
            + hardest_wrong
        )
    )


    weighted_ce = (
        ce_per_sample
        * sample_weight
    )


    weighted_margin = (
        margin_per_sample
        * sample_weight
    )


    mc_loss = (
        weighted_ce.mean()
    )


    margin_loss = (
        weighted_margin.mean()
    )


    if USE_MARGIN_LOSS:

        total = (
            MC_CE_WEIGHT
            * mc_loss
            +
            MARGIN_LOSS_WEIGHT
            * margin_loss
        )

    else:

        total = (
            MC_CE_WEIGHT
            * mc_loss
        )


    return (
        total,
        ce_per_sample,
        mc_loss,
        margin_loss,
        correct_logits,
        hardest_wrong,
    )


# ============================================================
# 21. Batch prediction
# ============================================================

def predict_logits_df(
    df,
    batch_size,
    processor_obj,
    focus_pairs=None,
    desc="Inference",
):

    model.eval()


    all_choice_logits = []


    for start in tqdm(
        range(
            0,
            len(df),
            batch_size,
        ),
        desc=desc,
        unit="batch",
    ):

        end = min(
            start + batch_size,
            len(df),
        )


        batch_df = (
            df.iloc[
                start:end
            ]
        )


        images = []

        texts = []


        for local_idx, (
            _,
            row
        ) in enumerate(
            batch_df.iterrows()
        ):

            global_idx = (
                start
                + local_idx
            )


            image = load_rgb_image(
                row["path"]
            )


            focus = None

            if (
                focus_pairs
                is not None
            ):

                focus = (
                    focus_pairs[
                        global_idx
                    ]
                )


            messages = (
                build_prompt_messages(
                    row,
                    image,
                    focus_choices=focus,
                )
            )


            text = (
                processor_obj
                .apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )


            images.append(
                image
            )

            texts.append(
                text
            )


        inputs = (
            processor_obj(
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
            key: value.to(
                DEVICE
            )
            for key, value
            in inputs.items()
        }


        with (
            torch.inference_mode(),
            autocast_context()
        ):

            outputs = model(
                **inputs,
                use_cache=False,

                # 매우 중요:
                # vocab projection을 마지막 token에만 수행
                logits_to_keep=1,
            )


            last_logits = (
                outputs.logits[
                    :,
                    -1,
                    :
                ]
                .float()
            )


            choice_logits = (
                last_logits.index_select(
                    dim=-1,
                    index=(
                        CHOICE_TOKEN_TENSOR
                    ),
                )
            )


        all_choice_logits.append(
            choice_logits.cpu()
        )


    return torch.cat(
        all_choice_logits,
        dim=0,
    )


# ============================================================
# 22. Accuracy helpers
# ============================================================

def gold_tensor(
    df,
):

    return torch.tensor(
        [
            CHOICE_TO_INDEX[
                str(answer)
                .strip()
                .lower()
            ]
            for answer
            in df["answer"]
        ],
        dtype=torch.long,
    )


def accuracy_from_logits(
    logits,
    gold,
):

    pred = (
        logits.argmax(
            dim=1
        )
    )

    accuracy = (
        pred.eq(gold)
        .float()
        .mean()
        .item()
    )

    return (
        accuracy,
        pred,
    )


def confidence_margin(
    logits,
):

    top2 = torch.topk(
        logits,
        k=2,
        dim=1,
    ).values

    return (
        top2[:, 0]
        - top2[:, 1]
    )


# ============================================================
# 23. Optimizer creation
# ============================================================

trainable_params = [
    param
    for param
    in model.parameters()
    if param.requires_grad
]


def make_optimizer_scheduler(
    lr,
    epochs,
    loader_length,
):

    optimizer = (
        torch.optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=(
                WEIGHT_DECAY
            ),
        )
    )


    steps_per_epoch = (
        math.ceil(
            loader_length
            / GRAD_ACCUM
        )
    )


    total_steps = (
        max(
            1,
            epochs
            * steps_per_epoch
        )
    )


    warmup_steps = int(
        total_steps
        * WARMUP_RATIO
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


    return (
        optimizer,
        scheduler,
    )


# ============================================================
# 24. Initial Loader / Optimizer
# ============================================================

initial_loader = (
    build_train_loader(
        train_ds,
        surprise_scores,
    )
)


optimizer, scheduler = (
    make_optimizer_scheduler(
        LR,
        NUM_EPOCHS,
        len(initial_loader),
    )
)


scaler = torch.amp.GradScaler(
    "cuda",
    enabled=USE_SCALER,
)


# ============================================================
# 25. Training function
# ============================================================

def train_one_epoch(
    epoch,
    dataset,
    loader,
    optimizer,
    scheduler,
    surprise=None,
    desc_prefix="Stage1",
):

    model.train()

    model.config.use_cache = False


    optimizer.zero_grad(
        set_to_none=True
    )


    train_loss_sum = 0.0

    train_ce_sum = 0.0

    train_margin_sum = 0.0


    correct = 0

    seen = 0


    # surprise 업데이트용

    surprise_accum = {}

    surprise_count = {}


    num_batches = len(
        loader
    )


    progress = tqdm(
        loader,
        desc=(
            f"{desc_prefix} "
            f"Epoch {epoch}"
        ),
        unit="batch",
    )


    for step, batch in enumerate(
        progress,
        start=1,
    ):

        mc_target = (
            batch.pop(
                "mc_target"
            )
            .to(DEVICE)
        )


        sample_idx = (
            batch.pop(
                "sample_idx"
            )
        )


        sample_weight = (
            batch.pop(
                "sample_weight"
            )
            .to(DEVICE)
        )


        batch = sanitize_inputs(
            batch
        )


        batch = {
            key: value.to(
                DEVICE
            )
            for key, value
            in batch.items()
        }


        # 마지막 accumulation 그룹 보정

        group_start = (
            ((step - 1)
             // GRAD_ACCUM)
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
                logits_to_keep=1,
            )


            vocab_logits = (
                outputs.logits[
                    :,
                    -1,
                    :
                ]
            )


            choice_logits = (
                vocab_logits.index_select(
                    dim=-1,
                    index=(
                        CHOICE_TOKEN_TENSOR
                    ),
                )
            )


            (
                raw_loss,
                ce_per_sample,
                mc_loss,
                margin_loss,
                _,
                _,
            ) = compute_mc_losses(
                choice_logits,
                mc_target,
                sample_weight,
            )


            loss = (
                raw_loss
                / accum_divisor
            )


        if not torch.isfinite(
            raw_loss
        ).item():

            raise RuntimeError(
                "Non-finite loss."
            )


        if USE_SCALER:

            scaler.scale(
                loss
            ).backward()

        else:

            loss.backward()


        # ----------------------------------------------------
        # train metrics
        # ----------------------------------------------------

        predictions = (
            choice_logits
            .detach()
            .argmax(dim=1)
        )


        correct += (
            predictions
            .eq(mc_target)
            .sum()
            .item()
        )


        batch_size = (
            mc_target.size(0)
        )


        seen += batch_size


        train_loss_sum += (
            raw_loss.detach()
            .float()
            .item()
            * batch_size
        )


        train_ce_sum += (
            mc_loss.detach()
            .float()
            .item()
            * batch_size
        )


        train_margin_sum += (
            margin_loss.detach()
            .float()
            .item()
            * batch_size
        )


        # ----------------------------------------------------
        # Surprise memory
        # ----------------------------------------------------

        if (
            surprise is not None
        ):

            ce_cpu = (
                ce_per_sample
                .detach()
                .float()
                .cpu()
                .numpy()
            )


            idx_cpu = (
                sample_idx
                .cpu()
                .numpy()
            )


            for idx_value, loss_value in zip(
                idx_cpu,
                ce_cpu,
            ):

                idx_value = int(
                    idx_value
                )

                surprise_accum[
                    idx_value
                ] = (
                    surprise_accum.get(
                        idx_value,
                        0.0,
                    )
                    + float(
                        loss_value
                    )
                )


                surprise_count[
                    idx_value
                ] = (
                    surprise_count.get(
                        idx_value,
                        0,
                    )
                    + 1
                )


        # ----------------------------------------------------
        # Optimizer step
        # ----------------------------------------------------

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


        progress.set_postfix(
            loss=(
                f"{train_loss_sum / seen:.4f}"
            ),
            acc=(
                f"{correct / seen:.4f}"
            ),
            lr=(
                f"{scheduler.get_last_lr()[0]:.2e}"
            ),
        )


    # --------------------------------------------------------
    # Update surprise EMA
    # --------------------------------------------------------

    if (
        surprise is not None
    ):

        for idx_value in surprise_accum:

            current = (
                surprise_accum[
                    idx_value
                ]
                /
                surprise_count[
                    idx_value
                ]
            )


            surprise[
                idx_value
            ] = (
                SURPRISE_EMA_BETA
                * surprise[
                    idx_value
                ]
                +
                (
                    1.0
                    - SURPRISE_EMA_BETA
                )
                * current
            )


    return {
        "loss": (
            train_loss_sum / seen
        ),
        "ce": (
            train_ce_sum / seen
        ),
        "margin_loss": (
            train_margin_sum / seen
        ),
        "accuracy": (
            correct / seen
        ),
    }


# ============================================================
# 26. Save adapter helper
# ============================================================

def clone_adapter_state():

    state = (
        get_peft_model_state_dict(
            model
        )
    )

    return {
        key: (
            value.detach()
            .cpu()
            .clone()
        )
        for key, value
        in state.items()
    }


# ============================================================
# 27. Stage 1 training
# ============================================================

best_state = None

best_val_accuracy = -1.0

best_epoch = -1


history = []


for epoch in range(
    1,
    NUM_EPOCHS + 1,
):

    train_loader = (
        build_train_loader(
            train_ds,
            surprise_scores,
        )
    )


    metrics = train_one_epoch(
        epoch=epoch,
        dataset=train_ds,
        loader=train_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        surprise=(
            surprise_scores
            if USE_SURPRISE_REPLAY
            else None
        ),
        desc_prefix="Stage1",
    )


    should_eval = (
        epoch % EVAL_EVERY == 0
        or epoch == NUM_EPOCHS
    )


    val_accuracy = None


    if should_eval:

        val_logits = (
            predict_logits_df(
                valid_subset,
                VALID_BATCH_SIZE,
                processor,
                desc=(
                    f"Validation E{epoch}"
                ),
            )
        )


        val_gold = gold_tensor(
            valid_subset
        )


        (
            val_accuracy,
            _,
        ) = accuracy_from_logits(
            val_logits,
            val_gold,
        )


        print(
            f"\nEpoch {epoch}: "
            f"val_acc="
            f"{val_accuracy:.4f}"
        )


        if (
            val_accuracy
            > best_val_accuracy
        ):

            best_val_accuracy = (
                val_accuracy
            )

            best_epoch = epoch

            best_state = (
                clone_adapter_state()
            )


            print(
                "★ Best Stage1 checkpoint"
            )


    history.append(
        {
            "stage": "stage1",
            "epoch": epoch,
            "train_loss": metrics[
                "loss"
            ],
            "train_accuracy": metrics[
                "accuracy"
            ],
            "valid_accuracy": (
                val_accuracy
            ),
        }
    )


# ============================================================
# 28. Restore Stage1 best
# ============================================================

if best_state is None:

    raise RuntimeError(
        "Stage1 best model 없음."
    )


set_peft_model_state_dict(
    model,
    best_state,
)


print(
    "\nStage1 best epoch:",
    best_epoch
)

print(
    "Stage1 best accuracy:",
    best_val_accuracy
)


# ============================================================
# 29. Optional DEV pseudo-label generation
# ============================================================

def human_majority(
    row,
):

    answers = []

    for col in DEV_ANSWER_COLS:

        value = row[col]

        if pd.isna(value):
            continue

        value = (
            str(value)
            .strip()
            .lower()
        )

        if value in VALID_CHOICES:

            answers.append(
                value
            )


    if not answers:

        return (
            None,
            0.0,
        )


    counts = {
        choice: answers.count(
            choice
        )
        for choice
        in CHOICES
    }


    max_count = max(
        counts.values()
    )


    winners = [
        choice
        for choice, count
        in counts.items()
        if count == max_count
    ]


    # tie 제외

    if len(winners) != 1:

        return (
            None,
            0.0,
        )


    winner = winners[0]


    confidence = (
        max_count
        / len(answers)
    )


    return (
        winner,
        confidence,
    )


stage1_best_state = {
    key: value.clone()
    for key, value
    in best_state.items()
}


if USE_DEV_PSEUDO_STAGE:

    print(
        "\n===== DEV TEACHER INFERENCE ====="
    )


    dev_logits = (
        predict_logits_df(
            dev_df,
            VALID_BATCH_SIZE,
            processor,
            desc="Dev teacher",
        )
    )


    dev_pred_idx = (
        dev_logits.argmax(
            dim=1
        )
    )


    dev_margin = (
        confidence_margin(
            dev_logits
        )
    )


    pseudo_rows = []


    for idx in range(
        len(dev_df)
    ):

        row = dev_df.iloc[
            idx
        ]


        (
            human_label,
            human_conf,
        ) = human_majority(
            row
        )


        if human_label is None:
            continue


        if (
            human_conf
            < PSEUDO_MIN_HUMAN_CONF
        ):
            continue


        teacher_choice = (
            CHOICES[
                int(
                    dev_pred_idx[
                        idx
                    ].item()
                )
            ]
        )


        teacher_margin = float(
            dev_margin[
                idx
            ].item()
        )


        if (
            teacher_choice
            != human_label
        ):
            continue


        if (
            teacher_margin
            < PSEUDO_MIN_TEACHER_MARGIN
        ):
            continue


        record = (
            row.to_dict()
        )


        record["answer"] = (
            human_label
        )


        record[
            "sample_weight"
        ] = (
            PSEUDO_BASE_WEIGHT
            * human_conf
        )


        record[
            "human_confidence"
        ] = human_conf


        record[
            "teacher_margin"
        ] = teacher_margin


        pseudo_rows.append(
            record
        )


    pseudo_df = pd.DataFrame(
        pseudo_rows
    )


    print(
        "Pseudo samples:",
        len(pseudo_df)
    )


    SUBMISSION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    pseudo_df.to_csv(
        PSEUDO_PATH,
        index=False,
    )


# ============================================================
# 30. Stage 2 pseudo fine-tuning
# ============================================================

if (
    USE_DEV_PSEUDO_STAGE
    and len(pseudo_df) > 0
):

    # --------------------------------------------
    # Gold train + filtered pseudo
    # validation은 포함하지 않음
    # --------------------------------------------

    gold_stage2 = (
        train_subset.copy()
    )


    gold_stage2[
        "sample_weight"
    ] = 1.0


    pseudo_train_cols = (
        list(
            gold_stage2.columns
        )
    )


    pseudo_for_train = (
        pseudo_df[
            pseudo_train_cols
        ].copy()
    )


    stage2_df = pd.concat(
        [
            gold_stage2,
            pseudo_for_train,
        ],
        ignore_index=True,
    )


    stage2_ds = VQAMCDataset(
        stage2_df
    )


    stage2_surprise = np.ones(
        len(stage2_ds),
        dtype=np.float32,
    )


    stage2_loader = (
        build_train_loader(
            stage2_ds,
            stage2_surprise,
        )
    )


    optimizer2, scheduler2 = (
        make_optimizer_scheduler(
            PSEUDO_LR,
            PSEUDO_EPOCHS,
            len(stage2_loader),
        )
    )


    stage2_best_acc = (
        best_val_accuracy
    )


    stage2_best_state = (
        clone_adapter_state()
    )


    for pseudo_epoch in range(
        1,
        PSEUDO_EPOCHS + 1,
    ):

        stage2_loader = (
            build_train_loader(
                stage2_ds,
                stage2_surprise,
            )
        )


        metrics = train_one_epoch(
            epoch=pseudo_epoch,
            dataset=stage2_ds,
            loader=stage2_loader,
            optimizer=optimizer2,
            scheduler=scheduler2,
            surprise=(
                stage2_surprise
            ),
            desc_prefix="Stage2",
        )


        val_logits = (
            predict_logits_df(
                valid_subset,
                VALID_BATCH_SIZE,
                processor,
                desc=(
                    "Stage2 validation"
                ),
            )
        )


        val_gold = gold_tensor(
            valid_subset
        )


        (
            stage2_acc,
            _,
        ) = accuracy_from_logits(
            val_logits,
            val_gold,
        )


        print(
            f"Stage2 epoch "
            f"{pseudo_epoch}: "
            f"{stage2_acc:.4f}"
        )


        history.append(
            {
                "stage": "stage2",
                "epoch": pseudo_epoch,
                "train_loss": metrics[
                    "loss"
                ],
                "train_accuracy": metrics[
                    "accuracy"
                ],
                "valid_accuracy": (
                    stage2_acc
                ),
            }
        )


        if (
            stage2_acc
            > stage2_best_acc
        ):

            stage2_best_acc = (
                stage2_acc
            )

            stage2_best_state = (
                clone_adapter_state()
            )


    # --------------------------------------------
    # Stage2가 좋아진 경우만 채택
    # --------------------------------------------

    if (
        stage2_best_acc
        > best_val_accuracy
    ):

        print(
            "★ Stage2 improved model"
        )


        best_val_accuracy = (
            stage2_best_acc
        )


        best_state = (
            stage2_best_state
        )

    else:

        print(
            "Stage2 did not improve. "
            "Restoring Stage1."
        )


        best_state = (
            stage1_best_state
        )


# ============================================================
# 31. Restore final best
# ============================================================

set_peft_model_state_dict(
    model,
    best_state,
)


model.eval()


print(
    "\nFinal best validation:",
    best_val_accuracy
)


# ============================================================
# 32. Calibrate second-pass threshold
#
# 1차 logits는 한 번만 계산.
#
# 애매한 문제만 high-resolution second pass.
#
# threshold 후보를 validation에서 자동 선택.
# ============================================================

def log_probabilities(
    logits,
):

    return F.log_softmax(
        logits,
        dim=1,
    )


def calibrate_second_pass(
    df,
):

    gold = gold_tensor(
        df
    )


    first_logits = (
        predict_logits_df(
            df,
            VALID_BATCH_SIZE,
            processor,
            desc="Pass1 calibration",
        )
    )


    first_logp = (
        log_probabilities(
            first_logits
        )
    )


    margins = (
        confidence_margin(
            first_logits
        )
    )


    first_top2 = torch.topk(
        first_logits,
        k=2,
        dim=1,
    ).indices


    max_threshold = max(
        SECOND_PASS_THRESHOLDS
    )


    uncertain_indices = (
        torch.where(
            margins
            <= max_threshold
        )[0]
        .tolist()
    )


    # second-pass logp 기본값은 pass1
    second_logp_full = (
        first_logp.clone()
    )


    if uncertain_indices:

        uncertain_df = (
            df.iloc[
                uncertain_indices
            ]
            .reset_index(drop=True)
        )


        focus_pairs = []


        for original_idx in (
            uncertain_indices
        ):

            top_pair_idx = (
                first_top2[
                    original_idx
                ]
                .tolist()
            )


            focus_pairs.append(
                (
                    CHOICES[
                        top_pair_idx[0]
                    ],
                    CHOICES[
                        top_pair_idx[1]
                    ],
                )
            )


        second_logits = (
            predict_logits_df(
                uncertain_df,
                VALID_BATCH_SIZE,
                high_processor,
                focus_pairs=(
                    focus_pairs
                ),
                desc="Pass2 calibration",
            )
        )


        second_logp = (
            log_probabilities(
                second_logits
            )
        )


        for local_idx, original_idx in enumerate(
            uncertain_indices
        ):

            second_logp_full[
                original_idx
            ] = second_logp[
                local_idx
            ]


    best_threshold = 0.0

    best_accuracy = -1.0


    results = []


    for threshold in (
        SECOND_PASS_THRESHOLDS
    ):

        use_second = (
            margins <= threshold
        )


        fused = (
            first_logp.clone()
        )


        if use_second.any():

            fused[
                use_second
            ] = (
                (
                    1.0
                    - SECOND_PASS_WEIGHT
                )
                * first_logp[
                    use_second
                ]
                +
                SECOND_PASS_WEIGHT
                * second_logp_full[
                    use_second
                ]
            )


        (
            acc,
            _,
        ) = accuracy_from_logits(
            fused,
            gold,
        )


        results.append(
            (
                threshold,
                acc,
                int(
                    use_second
                    .sum()
                    .item()
                ),
            )
        )


        print(
            f"threshold={threshold:.2f} "
            f"acc={acc:.4f} "
            f"second_pass="
            f"{int(use_second.sum())}"
        )


        if acc > best_accuracy:

            best_accuracy = acc

            best_threshold = (
                threshold
            )


    return (
        best_threshold,
        best_accuracy,
        first_logits,
        second_logp_full,
        margins,
        results,
    )


if USE_SECOND_PASS:

    (
        best_second_threshold,
        calibrated_accuracy,
        _,
        _,
        _,
        second_pass_results,
    ) = calibrate_second_pass(
        valid_subset
    )


    print(
        "\nBest second-pass threshold:",
        best_second_threshold
    )

    print(
        "Calibrated validation accuracy:",
        calibrated_accuracy
    )

else:

    best_second_threshold = 0.0


# ============================================================
# 33. Final validation predictions
# ============================================================

val_first_logits = (
    predict_logits_df(
        valid_subset,
        VALID_BATCH_SIZE,
        processor,
        desc="Final validation",
    )
)


val_first_logp = (
    log_probabilities(
        val_first_logits
    )
)


val_margins = (
    confidence_margin(
        val_first_logits
    )
)


val_final_logp = (
    val_first_logp.clone()
)


if (
    USE_SECOND_PASS
    and best_second_threshold > 0
):

    uncertain = torch.where(
        val_margins
        <= best_second_threshold
    )[0].tolist()


    if uncertain:

        uncertain_df = (
            valid_subset.iloc[
                uncertain
            ]
            .reset_index(drop=True)
        )


        top2 = torch.topk(
            val_first_logits,
            2,
            dim=1,
        ).indices


        focus_pairs = []


        for idx in uncertain:

            pair = (
                top2[
                    idx
                ].tolist()
            )

            focus_pairs.append(
                (
                    CHOICES[
                        pair[0]
                    ],
                    CHOICES[
                        pair[1]
                    ],
                )
            )


        second_logits = (
            predict_logits_df(
                uncertain_df,
                VALID_BATCH_SIZE,
                high_processor,
                focus_pairs=(
                    focus_pairs
                ),
                desc=(
                    "Final validation "
                    "second pass"
                ),
            )
        )


        second_logp = (
            log_probabilities(
                second_logits
            )
        )


        for local_idx, idx in enumerate(
            uncertain
        ):

            val_final_logp[
                idx
            ] = (
                (
                    1
                    - SECOND_PASS_WEIGHT
                )
                * val_first_logp[
                    idx
                ]
                +
                SECOND_PASS_WEIGHT
                * second_logp[
                    local_idx
                ]
            )


val_gold = gold_tensor(
    valid_subset
)


(
    final_val_accuracy,
    val_pred_idx,
) = accuracy_from_logits(
    val_final_logp,
    val_gold,
)


print(
    "\n===== FINAL VALIDATION ====="
)

print(
    "Accuracy:",
    final_val_accuracy
)


# ============================================================
# 34. Save validation predictions
# ============================================================

SUBMISSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


validation_rows = []


for idx in range(
    len(valid_subset)
):

    row = valid_subset.iloc[
        idx
    ]


    validation_rows.append(
        {
            "id": row["id"],
            "gold": (
                str(
                    row["answer"]
                )
                .strip()
                .lower()
            ),
            "pred": (
                CHOICES[
                    int(
                        val_pred_idx[
                            idx
                        ].item()
                    )
                ]
            ),
            "margin": float(
                val_margins[
                    idx
                ].item()
            ),
            "correct": (
                int(
                    val_pred_idx[
                        idx
                    ].item()
                )
                ==
                int(
                    val_gold[
                        idx
                    ].item()
                )
            ),
        }
    )


pd.DataFrame(
    validation_rows
).to_csv(
    VALIDATION_PREDICTION_PATH,
    index=False,
)


# ============================================================
# 35. Save final model
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


pd.DataFrame(
    history
).to_csv(
    SAVE_DIR
    / "training_history.csv",
    index=False,
)


print(
    "Saved model:",
    SAVE_DIR
)


# ============================================================
# 36. TEST PASS 1
# ============================================================

test_first_logits = (
    predict_logits_df(
        test_df,
        TEST_BATCH_SIZE,
        processor,
        desc="Test pass1",
    )
)


test_first_logp = (
    log_probabilities(
        test_first_logits
    )
)


test_margins = (
    confidence_margin(
        test_first_logits
    )
)


test_final_logp = (
    test_first_logp.clone()
)


used_second_pass = torch.zeros(
    len(test_df),
    dtype=torch.bool,
)


# ============================================================
# 37. TEST PASS 2
# ============================================================

if (
    USE_SECOND_PASS
    and best_second_threshold > 0
):

    uncertain = torch.where(
        test_margins
        <= best_second_threshold
    )[0].tolist()


    print(
        "Test second-pass samples:",
        len(uncertain)
    )


    if uncertain:

        uncertain_df = (
            test_df.iloc[
                uncertain
            ]
            .reset_index(drop=True)
        )


        top2 = torch.topk(
            test_first_logits,
            2,
            dim=1,
        ).indices


        focus_pairs = []


        for idx in uncertain:

            pair = (
                top2[
                    idx
                ].tolist()
            )


            focus_pairs.append(
                (
                    CHOICES[
                        pair[0]
                    ],
                    CHOICES[
                        pair[1]
                    ],
                )
            )


        second_logits = (
            predict_logits_df(
                uncertain_df,
                TEST_BATCH_SIZE,
                high_processor,
                focus_pairs=(
                    focus_pairs
                ),
                desc="Test pass2",
            )
        )


        second_logp = (
            log_probabilities(
                second_logits
            )
        )


        for local_idx, idx in enumerate(
            uncertain
        ):

            test_final_logp[
                idx
            ] = (
                (
                    1.0
                    - SECOND_PASS_WEIGHT
                )
                * test_first_logp[
                    idx
                ]
                +
                SECOND_PASS_WEIGHT
                * second_logp[
                    local_idx
                ]
            )


            used_second_pass[
                idx
            ] = True


# ============================================================
# 38. Final test prediction
# ============================================================

test_pred_idx = (
    test_final_logp.argmax(
        dim=1
    )
)


preds = [
    CHOICES[
        int(idx)
    ]
    for idx in test_pred_idx.tolist()
]


# ============================================================
# 39. Diagnostics
# ============================================================

print(
    "\n===== TEST DISTRIBUTION ====="
)


print(
    pd.Series(
        preds
    )
    .value_counts()
    .sort_index()
)


print(
    "\n===== TEST RATIO ====="
)


print(
    pd.Series(
        preds
    )
    .value_counts(
        normalize=True
    )
    .sort_index()
)


# ============================================================
# 40. Submission
# ============================================================

submission = pd.DataFrame(
    {
        "id": test_df[
            "id"
        ],
        "answer": preds,
    }
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
        "Invalid submission answer."
    )


submission.to_csv(
    SUBMISSION_PATH,
    index=False,
)


# ============================================================
# 41. Save logits / diagnostics
# ============================================================

logit_rows = []


for idx in range(
    len(test_df)
):

    logit_rows.append(
        {
            "id": (
                test_df.iloc[
                    idx
                ]["id"]
            ),

            "pred": preds[
                idx
            ],

            "margin_pass1": float(
                test_margins[
                    idx
                ].item()
            ),

            "used_second_pass": bool(
                used_second_pass[
                    idx
                ].item()
            ),

            "score_a": float(
                test_final_logp[
                    idx,
                    0
                ].item()
            ),

            "score_b": float(
                test_final_logp[
                    idx,
                    1
                ].item()
            ),

            "score_c": float(
                test_final_logp[
                    idx,
                    2
                ].item()
            ),

            "score_d": float(
                test_final_logp[
                    idx,
                    3
                ].item()
            ),
        }
    )


pd.DataFrame(
    logit_rows
).to_csv(
    TEST_LOGIT_PATH,
    index=False,
)


print(
    "\n===== DONE ====="
)

print(
    "Validation accuracy:",
    final_val_accuracy
)

print(
    "Second-pass threshold:",
    best_second_threshold
)

print(
    "Submission:",
    SUBMISSION_PATH
)

print(
    "Logits:",
    TEST_LOGIT_PATH
)

print(
    submission.head(10)
)