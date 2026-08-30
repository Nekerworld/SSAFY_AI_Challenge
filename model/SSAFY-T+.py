# ============================================================
# SSAFY-T+ Ensemble
# Surprise-Sensitive Adaptive Framework for Your Answers
# with Titans
#
# Qwen3-VL-2B + Qwen3-VL-4B
#
# ------------------------------------------------------------
# Existing SSAFY-T
# ------------------------------------------------------------
# 1. Qwen3-VL 2B + 4B sequential training
# 2. 4-bit NF4 QLoRA
# 3. Language LoRA + last-N Vision LoRA
# 4. Direct 4-choice Cross Entropy
# 5. Hard-negative margin loss
# 6. Titans-inspired Surprise Replay
# 7. Random option permutation training
# 8. Multi-permutation inference ensemble
# 9. Confidence-based high-resolution second pass
# 10. DEV human/teacher pseudo-label Stage 2
# 11. Validation-calibrated 2B/4B ensemble
#
# ------------------------------------------------------------
# Added in SSAFY-T+
# ------------------------------------------------------------
# 12. Train-only weak image augmentation
#     - brightness / contrast
#     - tiny crop
#     - JPEG noise
#     - weak blur
#
# 13. Semantic Candidate Scorer
#     - actual option text
#     - candidate verification via yes/no logits
#     - position-independent semantic scoring
#
# 14. Validation-calibrated
#     letter-score + semantic-score fusion
#
# 15. FINAL GOLD 100% RETRAINING
#     - validation used only for model selection
#     - once all hyperparameters are fixed:
#       reload pretrained model
#       train on 100% gold
#       optionally apply selected pseudo stage
#       infer test
#     - validation is NOT inspected again
#
# ============================================================


# ============================================================
# 0. Imports
# ============================================================

import gc
import io
import math
import random
import re

from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
)

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import (
    Dataset,
    DataLoader,
    Subset,
)

from sklearn.model_selection import (
    train_test_split,
)

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
# 1. Project
# ============================================================

try:
    PROJECT_DIR = Path(__file__).resolve().parent
except NameError:
    PROJECT_DIR = Path.cwd().resolve()


# ============================================================
# 2. Global config
# ============================================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# 3. Data
# ============================================================

TRAIN_CSV = PROJECT_DIR / "train.csv"
DEV_CSV = PROJECT_DIR / "dev.csv"
TEST_CSV = PROJECT_DIR / "test.csv"

TRAIN_LIMIT = None

VALID_RATIO = 0.10


# ============================================================
# 4. Image resolution
# ============================================================

LOW_MIN_PIXELS = 224 * 224
LOW_MAX_PIXELS = 448 * 448

HIGH_MIN_PIXELS = 256 * 28 * 28
HIGH_MAX_PIXELS = 768 * 28 * 28


# ============================================================
# 5. Models
# ============================================================

MODEL_CONFIGS = [
    {
        "name": "2b",
        "model_id": "Qwen/Qwen3-VL-2B-Instruct",

        "train_batch_size": 8,
        "eval_batch_size": 8,

        # semantic scorer expands each example ×4
        "semantic_batch_size": 2,

        "grad_accum": 2,

        "language_lr": 1e-4,
        "vision_lr": 1e-5,

        "vision_lora_blocks": 4,
    },

    {
        "name": "4b",
        "model_id": "Qwen/Qwen3-VL-4B-Instruct",

        "train_batch_size": 4,
        "eval_batch_size": 4,

        "semantic_batch_size": 1,

        "grad_accum": 4,

        "language_lr": 8e-5,
        "vision_lr": 8e-6,

        "vision_lora_blocks": 4,
    },
]


# ============================================================
# 6. Stage 1
# ============================================================

NUM_EPOCHS = 8
EVAL_EVERY = 2


# ============================================================
# 7. Optimizer
# ============================================================

WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
WARMUP_RATIO = 0.03


# ============================================================
# 8. LoRA
# ============================================================

LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05


# ============================================================
# 9. MC objective
# ============================================================

MC_CE_WEIGHT = 1.0

USE_MARGIN_LOSS = True

MARGIN = 1.0
MARGIN_LOSS_WEIGHT = 0.20


# ============================================================
# 10. Surprise Replay
# ============================================================

USE_SURPRISE_REPLAY = True

SURPRISE_REPLAY_RATIO = 0.25
SURPRISE_EMA_BETA = 0.80


# ============================================================
# 11. Option permutation
# ============================================================

USE_TRAIN_PERMUTATION = True
TRAIN_PERMUTATION_PROB = 1.0

USE_PERMUTATION_INFERENCE = True

PERMUTATIONS = [
    (0, 1, 2, 3),
    (1, 2, 3, 0),
    (2, 3, 0, 1),
    (3, 0, 1, 2),
]


# ============================================================
# 12. Weak Image Augmentation
# ============================================================

USE_AUGMENTATION = True

# 전체 augmentation 적용 여부
AUGMENT_PROB = 0.70

BRIGHTNESS_PROB = 0.35
CONTRAST_PROB = 0.35

TINY_CROP_PROB = 0.25

JPEG_PROB = 0.20
BLUR_PROB = 0.12

BRIGHTNESS_RANGE = (
    0.94,
    1.06,
)

CONTRAST_RANGE = (
    0.94,
    1.06,
)

# 가장자리 최대 3%
TINY_CROP_MAX_RATIO = 0.03

JPEG_QUALITY_RANGE = (
    90,
    98,
)

BLUR_RADIUS_RANGE = (
    0.10,
    0.55,
)


# 색상 질문에서는 brightness/contrast 최소화
COLOR_KEYWORDS = [
    "색",
    "색상",
    "무슨 색",
    "color",
]

# 위치/개수 질문에서는 crop 금지
SPATIAL_COUNT_KEYWORDS = [
    "왼쪽",
    "오른쪽",
    "위쪽",
    "아래쪽",
    "좌측",
    "우측",
    "몇 개",
    "몇개",
    "개수",
    "수량",
    "left",
    "right",
    "how many",
    "number of",
]


# ============================================================
# 13. Second Pass
# ============================================================

USE_SECOND_PASS = True

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

SECOND_PASS_WEIGHT = 0.70


# ============================================================
# 14. DEV pseudo stage
# ============================================================

USE_DEV_PSEUDO_STAGE = True

PSEUDO_MIN_HUMAN_CONF = 0.80
PSEUDO_MIN_TEACHER_MARGIN = 1.0

PSEUDO_BASE_WEIGHT = 0.50

PSEUDO_EPOCHS = 2

PSEUDO_LR_FACTOR = 0.20


# ============================================================
# 15. Semantic candidate scorer
# ============================================================

USE_SEMANTIC_SCORER = True

# semantic weight:
#
# final =
# (1 - semantic_weight) * letter_logp
# +
# semantic_weight * semantic_logp

SEMANTIC_WEIGHTS = [
    0.00,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
]


# ============================================================
# 16. 2B/4B ensemble
# ============================================================

ENSEMBLE_WEIGHTS_2B = [
    x / 20
    for x in range(21)
]


# ============================================================
# 17. Final Gold 100% retraining
# ============================================================

USE_FINAL_FULL_GOLD_RETRAIN = True

# final retraining에서도
# Stage1 best epoch 수를 그대로 사용.
#
# validation은 절대 다시 보지 않는다.
FINAL_USE_SELECTED_PSEUDO_STAGE = True


# ============================================================
# 18. Paths
# ============================================================

MODEL_ROOT = (
    PROJECT_DIR
    / "model"
)

SUBMISSION_DIR = (
    PROJECT_DIR
    / "submission"
)

FINAL_SUBMISSION_PATH = (
    SUBMISSION_DIR
    / "submission_ssafy_t_plus.csv"
)

FINAL_LOGIT_PATH = (
    SUBMISSION_DIR
    / "ssafy_t_plus_logits.csv"
)

ENSEMBLE_VALIDATION_PATH = (
    SUBMISSION_DIR
    / "ssafy_t_plus_validation.csv"
)

Image.MAX_IMAGE_PIXELS = None


# ============================================================
# 19. CUDA
# ============================================================

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
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "dtype:",
    COMPUTE_DTYPE
)


# ============================================================
# 20. Choice helpers
# ============================================================

CHOICES = (
    "a",
    "b",
    "c",
    "d",
)

CHOICE_TO_INDEX = {
    c: i
    for i, c
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


# ============================================================
# 21. File helpers
# ============================================================

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

    with Image.open(path) as img:

        return img.convert(
            "RGB"
        )


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
            f"{name} missing images:\n"
            + "\n".join(missing)
        )


def sanitize_inputs(
    inputs,
):

    inputs.pop(
        "token_type_ids",
        None,
    )

    return inputs


def autocast_context():

    return torch.autocast(
        device_type="cuda",
        dtype=COMPUTE_DTYPE,
    )


# ============================================================
# 22. Augmentation
# ============================================================

def contains_keyword(
    question,
    keywords,
):

    text = str(
        question
    ).lower()

    return any(
        keyword.lower() in text
        for keyword
        in keywords
    )


def tiny_crop(
    image,
):

    width, height = (
        image.size
    )

    max_crop_x = max(
        1,
        int(
            width
            * TINY_CROP_MAX_RATIO
        ),
    )

    max_crop_y = max(
        1,
        int(
            height
            * TINY_CROP_MAX_RATIO
        ),
    )

    left = random.randint(
        0,
        max_crop_x,
    )

    right = random.randint(
        0,
        max_crop_x,
    )

    top = random.randint(
        0,
        max_crop_y,
    )

    bottom = random.randint(
        0,
        max_crop_y,
    )

    crop_right = max(
        left + 2,
        width - right,
    )

    crop_bottom = max(
        top + 2,
        height - bottom,
    )

    cropped = image.crop(
        (
            left,
            top,
            crop_right,
            crop_bottom,
        )
    )

    return cropped.resize(
        (
            width,
            height,
        ),
        resample=(
            Image.Resampling.BICUBIC
        ),
    )


def jpeg_noise(
    image,
):

    quality = random.randint(
        JPEG_QUALITY_RANGE[0],
        JPEG_QUALITY_RANGE[1],
    )

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=quality,
    )

    buffer.seek(0)

    with Image.open(
        buffer
    ) as decoded:

        output = decoded.convert(
            "RGB"
        ).copy()

    buffer.close()

    return output


def augment_training_image(
    image,
    question,
):

    if not USE_AUGMENTATION:

        return image

    if random.random() > AUGMENT_PROB:

        return image

    img = image.copy()

    color_sensitive = (
        contains_keyword(
            question,
            COLOR_KEYWORDS,
        )
    )

    spatial_sensitive = (
        contains_keyword(
            question,
            SPATIAL_COUNT_KEYWORDS,
        )
    )

    # --------------------------------------------------------
    # Brightness
    # --------------------------------------------------------

    brightness_prob = (
        BRIGHTNESS_PROB
        * (
            0.25
            if color_sensitive
            else 1.0
        )
    )

    if random.random() < brightness_prob:

        factor = random.uniform(
            *BRIGHTNESS_RANGE
        )

        img = (
            ImageEnhance
            .Brightness(img)
            .enhance(factor)
        )


    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast_prob = (
        CONTRAST_PROB
        * (
            0.40
            if color_sensitive
            else 1.0
        )
    )

    if random.random() < contrast_prob:

        factor = random.uniform(
            *CONTRAST_RANGE
        )

        img = (
            ImageEnhance
            .Contrast(img)
            .enhance(factor)
        )


    # --------------------------------------------------------
    # Tiny crop
    # --------------------------------------------------------

    if (
        not spatial_sensitive
        and random.random()
        < TINY_CROP_PROB
    ):

        img = tiny_crop(
            img
        )


    # --------------------------------------------------------
    # JPEG noise
    # --------------------------------------------------------

    if random.random() < JPEG_PROB:

        img = jpeg_noise(
            img
        )


    # --------------------------------------------------------
    # Blur
    # --------------------------------------------------------

    if random.random() < BLUR_PROB:

        radius = random.uniform(
            *BLUR_RADIUS_RANGE
        )

        img = img.filter(
            ImageFilter.GaussianBlur(
                radius=radius
            )
        )

    return img


# ============================================================
# 23. Load CSV
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
    "train",
)

validate_image_paths(
    dev_df,
    "dev",
)

validate_image_paths(
    test_df,
    "test",
)


# ============================================================
# 24. Fixed development split
# ============================================================

train_pool, valid_subset = (
    train_test_split(
        train_df,

        test_size=VALID_RATIO,

        random_state=SEED,

        stratify=train_df[
            "answer"
        ],
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


if (
    TRAIN_LIMIT is not None
    and TRAIN_LIMIT
    < len(train_pool)
):

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


train_subset[
    "sample_weight"
] = 1.0


# final training data
full_gold_df = (
    train_df
    .reset_index(drop=True)
    .copy()
)

full_gold_df[
    "sample_weight"
] = 1.0


print(
    "\n===== DATA ====="
)

print(
    "Gold total:",
    len(train_df)
)

print(
    "Dev train:",
    len(train_subset)
)

print(
    "Validation:",
    len(valid_subset)
)

print(
    "DEV:",
    len(dev_df)
)

print(
    "Test:",
    len(test_df)
)


# ============================================================
# 25. Option permutation
# ============================================================

def random_permutation():

    values = list(
        range(4)
    )

    random.shuffle(
        values
    )

    return tuple(
        values
    )


def permute_row(
    row,
    permutation,
    has_answer=True,
):

    record = (
        row.to_dict()
        if hasattr(
            row,
            "to_dict",
        )
        else dict(row)
    )

    original_options = [
        str(
            record[c]
        )
        for c in CHOICES
    ]

    result = dict(
        record
    )

    for new_position, original_position in enumerate(
        permutation
    ):

        result[
            CHOICES[
                new_position
            ]
        ] = original_options[
            original_position
        ]

    if has_answer:

        old_answer = (
            str(
                record[
                    "answer"
                ]
            )
            .strip()
            .lower()
        )

        old_index = (
            CHOICE_TO_INDEX[
                old_answer
            ]
        )

        new_index = (
            permutation.index(
                old_index
            )
        )

        result[
            "answer"
        ] = CHOICES[
            new_index
        ]

    return result


def restore_logits_to_original_order(
    logits,
    permutation,
):

    restored = torch.empty_like(
        logits
    )

    for new_position, original_position in enumerate(
        permutation
    ):

        restored[
            :,
            original_position
        ] = logits[
            :,
            new_position
        ]

    return restored


# ============================================================
# 26. Letter prompt
# ============================================================

LETTER_SYSTEM = (
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

    text = (
        f"{question}\n\n"
        f"(a) {a}\n"
        f"(b) {b}\n"
        f"(c) {c}\n"
        f"(d) {d}\n"
    )

    if focus_choices is not None:

        x, y = (
            focus_choices
        )

        text += (
            "\nThe first evaluation was uncertain. "
            f"Compare options {x} and {y} "
            "especially carefully. "
            "Inspect the image again.\n"
        )

    text += (
        "\nReturn exactly one lowercase "
        "letter: a, b, c, or d."
    )

    return text


def build_prompt_messages(
    row,
    image,
    focus_choices=None,
):

    text = build_mc_prompt(
        row["question"],
        row["a"],
        row["b"],
        row["c"],
        row["d"],
        focus_choices,
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": LETTER_SYSTEM,
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
                    "text": text,
                },
            ],
        },
    ]


# ============================================================
# 27. Semantic verifier prompt
# ============================================================

SEMANTIC_SYSTEM = (
    "You are a visual question answering verifier. "
    "Given an image, a question, and one candidate answer, "
    "decide whether that candidate is the best correct answer. "
    "Reply exactly yes or no. "
    "Do not explain."
)


def build_semantic_messages(
    row,
    image,
    candidate_text,
):

    text = (
        f"Question:\n"
        f"{row['question']}\n\n"
        f"Candidate answer:\n"
        f"{candidate_text}\n\n"
        "Is this candidate the correct answer "
        "according to the image?"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": SEMANTIC_SYSTEM,
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
                    "text": text,
                },
            ],
        },
    ]


# ============================================================
# 28. LoRA target discovery
# ============================================================

LANGUAGE_SUFFIXES = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",

    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

VISION_SUFFIXES = (
    "qkv",
    "proj",

    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",

    "fc1",
    "fc2",

    "gate_proj",
    "up_proj",
    "down_proj",
)


def is_linear_like(
    module,
):

    if isinstance(
        module,
        nn.Linear,
    ):

        return True

    return (
        "linear"
        in module.__class__
        .__name__
        .lower()
    )


def get_vision_block_index(
    name,
):

    patterns = [
        r"(?:blocks)\.(\d+)\.",
        r"(?:layers)\.(\d+)\.",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            name,
        )

        if match:

            return int(
                match.group(1)
            )

    return None


def collect_lora_targets(
    base_model,
    vision_last_n,
):

    vision_blocks = []

    for name, _ in (
        base_model
        .named_modules()
    ):

        lower = name.lower()

        if (
            "vision" not in lower
            and "visual" not in lower
        ):
            continue

        index = (
            get_vision_block_index(
                name
            )
        )

        if index is not None:

            vision_blocks.append(
                index
            )

    vision_blocks = sorted(
        set(
            vision_blocks
        )
    )

    selected_vision = set(
        vision_blocks[
            -vision_last_n:
        ]
    )

    language_targets = []
    vision_targets = []

    for name, module in (
        base_model
        .named_modules()
    ):

        if not is_linear_like(
            module
        ):
            continue

        lower = name.lower()

        is_vision = (
            "vision" in lower
            or "visual" in lower
        )

        if not is_vision:

            if any(
                name.endswith(
                    suffix
                )
                for suffix
                in LANGUAGE_SUFFIXES
            ):

                language_targets.append(
                    name
                )

            continue

        block_index = (
            get_vision_block_index(
                name
            )
        )

        if (
            block_index
            not in selected_vision
        ):
            continue

        if any(
            name.endswith(
                suffix
            )
            for suffix
            in VISION_SUFFIXES
        ):

            vision_targets.append(
                name
            )

    language_targets = sorted(
        set(
            language_targets
        )
    )

    vision_targets = sorted(
        set(
            vision_targets
        )
    )

    targets = sorted(
        set(
            language_targets
            + vision_targets
        )
    )

    if not language_targets:

        raise RuntimeError(
            "Language LoRA targets not found."
        )

    print(
        "Vision blocks:",
        vision_blocks
    )

    print(
        "Selected vision blocks:",
        sorted(
            selected_vision
        )
    )

    print(
        "Language targets:",
        len(
            language_targets
        )
    )

    print(
        "Vision targets:",
        len(
            vision_targets
        )
    )

    return (
        targets,
        language_targets,
        vision_targets,
    )


# ============================================================
# 29. Main Runner
# ============================================================

class SSAFYTRunner:

    def __init__(
        self,
        cfg,
        final_mode=False,
        tuning_plan=None,
    ):

        self.cfg = cfg

        self.name = (
            cfg["name"]
        )

        self.model_id = (
            cfg["model_id"]
        )

        self.train_batch_size = (
            cfg[
                "train_batch_size"
            ]
        )

        self.eval_batch_size = (
            cfg[
                "eval_batch_size"
            ]
        )

        self.semantic_batch_size = (
            cfg[
                "semantic_batch_size"
            ]
        )

        self.grad_accum = (
            cfg[
                "grad_accum"
            ]
        )

        self.language_lr = (
            cfg[
                "language_lr"
            ]
        )

        self.vision_lr = (
            cfg[
                "vision_lr"
            ]
        )

        self.final_mode = (
            final_mode
        )

        self.tuning_plan = (
            tuning_plan
        )

        suffix = (
            "final"
            if final_mode
            else "tuning"
        )

        self.save_dir = (
            MODEL_ROOT
            / f"ssafy_t_plus_{self.name}_{suffix}"
        )

        self.model = None
        self.processor = None
        self.high_processor = None

        self.choice_token_tensor = None

        self.yes_no_token_tensor = None

        self.best_state = None
        self.best_val_accuracy = -1.0

        self.best_epoch = -1

        self.best_pseudo_epoch = 0

        self.best_pseudo_df = None

        self.history = []


    # ========================================================
    # Load model
    # ========================================================

    def load(
        self,
    ):

        print(
            "\n"
            + "=" * 70
        )

        print(
            "LOAD:",
            self.model_id,
        )

        print(
            "FINAL MODE:",
            self.final_mode,
        )

        print(
            "=" * 70
        )

        self.processor = (
            AutoProcessor
            .from_pretrained(
                self.model_id,

                min_pixels=(
                    LOW_MIN_PIXELS
                ),

                max_pixels=(
                    LOW_MAX_PIXELS
                ),

                trust_remote_code=True,
            )
        )

        self.high_processor = (
            AutoProcessor
            .from_pretrained(
                self.model_id,

                min_pixels=(
                    HIGH_MIN_PIXELS
                ),

                max_pixels=(
                    HIGH_MAX_PIXELS
                ),

                trust_remote_code=True,
            )
        )

        for processor in [
            self.processor,
            self.high_processor,
        ]:

            processor.tokenizer.padding_side = (
                "left"
            )

            if (
                processor
                .tokenizer
                .pad_token_id
                is None
            ):

                processor.tokenizer.pad_token = (
                    processor
                    .tokenizer
                    .eos_token
                )

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
                self.model_id,

                quantization_config=(
                    bnb_config
                ),

                device_map={
                    "": 0
                },

                trust_remote_code=True,

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

        targets, _, _ = (
            collect_lora_targets(
                base_model,

                self.cfg[
                    "vision_lora_blocks"
                ],
            )
        )

        lora_config = (
            LoraConfig(
                r=LORA_R,

                lora_alpha=(
                    LORA_ALPHA
                ),

                lora_dropout=(
                    LORA_DROPOUT
                ),

                bias="none",

                target_modules=(
                    targets
                ),

                task_type=(
                    "CAUSAL_LM"
                ),
            )
        )

        self.model = (
            get_peft_model(
                base_model,
                lora_config,
            )
        )

        self.model.print_trainable_parameters()

        self.choice_token_tensor = (
            self.discover_choice_tokens()
        )

        self.yes_no_token_tensor = (
            self.discover_yes_no_tokens()
        )


    # ========================================================
    # Discover a/b/c/d token IDs
    # ========================================================

    def discover_choice_tokens(
        self,
    ):

        dummy = Image.new(
            "RGB",
            (28, 28),
        )

        row = {
            "question":
                "Choose one answer.",

            "a": "A",
            "b": "B",
            "c": "C",
            "d": "D",
        }

        messages = (
            build_prompt_messages(
                row,
                dummy,
            )
        )

        prompt_text = (
            self.processor
            .apply_chat_template(
                messages,

                tokenize=False,

                add_generation_prompt=True,
            )
        )

        prompt_ids = (
            self.processor
            .tokenizer(
                prompt_text,

                add_special_tokens=False,
            )["input_ids"]
        )

        result = {}

        for choice in CHOICES:

            full_messages = (
                messages
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
                self.processor
                .apply_chat_template(
                    full_messages,

                    tokenize=False,

                    add_generation_prompt=False,
                )
            )

            full_ids = (
                self.processor
                .tokenizer(
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
                    "Choice token prefix mismatch."
                )

            suffix = (
                full_ids[
                    len(prompt_ids):
                ]
            )

            result[
                choice
            ] = suffix[0]

        ids = [
            result[c]
            for c in CHOICES
        ]

        if len(set(ids)) != 4:

            raise RuntimeError(
                "Choice token IDs are not unique."
            )

        print(
            "Choice token IDs:",
            result
        )

        return torch.tensor(
            ids,

            dtype=torch.long,

            device=DEVICE,
        )


    # ========================================================
    # Discover yes/no token IDs
    # ========================================================

    def discover_yes_no_tokens(
        self,
    ):

        dummy = Image.new(
            "RGB",
            (28, 28),
        )

        row = {
            "question":
                "Is the object recyclable?"
        }

        messages = (
            build_semantic_messages(
                row,
                dummy,
                "plastic",
            )
        )

        prompt_text = (
            self.processor
            .apply_chat_template(
                messages,

                tokenize=False,

                add_generation_prompt=True,
            )
        )

        prompt_ids = (
            self.processor
            .tokenizer(
                prompt_text,

                add_special_tokens=False,
            )["input_ids"]
        )

        result = {}

        for word in [
            "yes",
            "no",
        ]:

            full = (
                messages
                + [
                    {
                        "role":
                            "assistant",

                        "content": [
                            {
                                "type":
                                    "text",

                                "text":
                                    word,
                            }
                        ],
                    }
                ]
            )

            full_text = (
                self.processor
                .apply_chat_template(
                    full,

                    tokenize=False,

                    add_generation_prompt=False,
                )
            )

            full_ids = (
                self.processor
                .tokenizer(
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
                    "yes/no token prefix mismatch."
                )

            suffix = (
                full_ids[
                    len(prompt_ids):
                ]
            )

            result[
                word
            ] = suffix[0]

        if (
            result["yes"]
            == result["no"]
        ):

            raise RuntimeError(
                "yes/no tokens are identical."
            )

        print(
            "Semantic token IDs:",
            result
        )

        return torch.tensor(
            [
                result["yes"],
                result["no"],
            ],

            dtype=torch.long,

            device=DEVICE,
        )


    # ========================================================
    # Dataset
    # ========================================================

    class DatasetImpl(
        Dataset
    ):

        def __init__(
            self,
            df,
            train_mode=True,
        ):

            self.df = (
                df
                .reset_index(
                    drop=True
                )
            )

            self.train_mode = (
                train_mode
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

            if (
                self.train_mode
                and
                USE_TRAIN_PERMUTATION
                and
                random.random()
                < TRAIN_PERMUTATION_PROB
            ):

                row_data = (
                    permute_row(
                        row,

                        random_permutation(),

                        has_answer=True,
                    )
                )

            else:

                row_data = (
                    row.to_dict()
                )

            image = load_rgb_image(
                row_data[
                    "path"
                ]
            )

            # ------------------------------------------------
            # TRAIN-ONLY AUGMENTATION
            # ------------------------------------------------

            if self.train_mode:

                image = (
                    augment_training_image(
                        image,

                        row_data[
                            "question"
                        ],
                    )
                )

            gold = (
                str(
                    row_data[
                        "answer"
                    ]
                )
                .strip()
                .lower()
            )

            weight = float(
                row_data.get(
                    "sample_weight",
                    1.0,
                )
            )

            return {
                "row":
                    row_data,

                "image":
                    image,

                "target":
                    CHOICE_TO_INDEX[
                        gold
                    ],

                "sample_idx":
                    idx,

                "sample_weight":
                    weight,
            }


    # ========================================================
    # Collator
    # ========================================================

    @dataclass
    class Collator:

        processor: object


        def __call__(
            self,
            batch,
        ):

            texts = []
            images = []

            targets = []
            indices = []
            weights = []

            for sample in batch:

                messages = (
                    build_prompt_messages(
                        sample[
                            "row"
                        ],

                        sample[
                            "image"
                        ],
                    )
                )

                text = (
                    self.processor
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
                    sample[
                        "image"
                    ]
                )

                targets.append(
                    sample[
                        "target"
                    ]
                )

                indices.append(
                    sample[
                        "sample_idx"
                    ]
                )

                weights.append(
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

            enc[
                "mc_target"
            ] = torch.tensor(
                targets,
                dtype=torch.long,
            )

            enc[
                "sample_idx"
            ] = torch.tensor(
                indices,
                dtype=torch.long,
            )

            enc[
                "sample_weight"
            ] = torch.tensor(
                weights,
                dtype=torch.float32,
            )

            return enc


    # ========================================================
    # Training loader
    # ========================================================

    def build_loader(
        self,
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
            and
            surprise is not None
        ):

            extra_count = int(
                len(dataset)
                * SURPRISE_REPLAY_RATIO
            )

            weights = np.maximum(
                np.asarray(
                    surprise,
                    dtype=np.float64,
                ),
                1e-6,
            )

            replay = (
                random.choices(
                    base_indices,

                    weights=(
                        weights.tolist()
                    ),

                    k=extra_count,
                )
            )

            indices = (
                base_indices
                + replay
            )

        else:

            indices = (
                base_indices
            )

        subset = Subset(
            dataset,
            indices,
        )

        return DataLoader(
            subset,

            batch_size=(
                self.train_batch_size
            ),

            shuffle=True,

            collate_fn=(
                self.Collator(
                    self.processor
                )
            ),

            num_workers=0,

            pin_memory=False,
        )


    # ========================================================
    # Loss
    # ========================================================

    @staticmethod
    def compute_loss(
        logits,
        target,
        sample_weight,
    ):

        ce_sample = (
            F.cross_entropy(
                logits,
                target,
                reduction="none",
            )
        )

        correct = (
            logits.gather(
                1,
                target.unsqueeze(1),
            )
            .squeeze(1)
        )

        mask = (
            torch.ones_like(
                logits,
                dtype=torch.bool,
            )
        )

        mask.scatter_(
            1,
            target.unsqueeze(1),
            False,
        )

        hardest_wrong = (
            logits
            .masked_fill(
                ~mask,
                float("-inf"),
            )
            .max(dim=1)
            .values
        )

        margin_sample = (
            F.relu(
                MARGIN
                - correct
                + hardest_wrong
            )
        )

        ce = (
            ce_sample
            * sample_weight
        ).mean()

        margin_loss = (
            margin_sample
            * sample_weight
        ).mean()

        total = (
            MC_CE_WEIGHT
            * ce
        )

        if USE_MARGIN_LOSS:

            total = (
                total
                + MARGIN_LOSS_WEIGHT
                * margin_loss
            )

        return (
            total,
            ce_sample,
            ce,
            margin_loss,
        )


    # ========================================================
    # Optimizer
    # ========================================================

    def make_optimizer(
        self,
        epochs,
        loader_len,
        pseudo=False,
    ):

        language_params = []
        vision_params = []

        for name, parameter in (
            self.model
            .named_parameters()
        ):

            if not parameter.requires_grad:

                continue

            lower = name.lower()

            if (
                "vision" in lower
                or "visual" in lower
            ):

                vision_params.append(
                    parameter
                )

            else:

                language_params.append(
                    parameter
                )

        factor = (
            PSEUDO_LR_FACTOR
            if pseudo
            else 1.0
        )

        groups = []

        if language_params:

            groups.append(
                {
                    "params":
                        language_params,

                    "lr":
                        self.language_lr
                        * factor,
                }
            )

        if vision_params:

            groups.append(
                {
                    "params":
                        vision_params,

                    "lr":
                        self.vision_lr
                        * factor,
                }
            )

        optimizer = (
            torch.optim.AdamW(
                groups,

                weight_decay=(
                    WEIGHT_DECAY
                ),
            )
        )

        steps_per_epoch = (
            math.ceil(
                loader_len
                / self.grad_accum
            )
        )

        total_steps = max(
            1,
            epochs
            * steps_per_epoch,
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


    # ========================================================
    # Train one epoch
    # ========================================================

    def train_one_epoch(
        self,
        epoch,
        loader,
        optimizer,
        scheduler,
        surprise,
        stage,
    ):

        self.model.train()

        optimizer.zero_grad(
            set_to_none=True
        )

        scaler = (
            torch.amp.GradScaler(
                "cuda",
                enabled=USE_SCALER,
            )
        )

        seen = 0
        correct_count = 0

        loss_sum = 0.0

        surprise_sum = {}
        surprise_count = {}

        num_batches = len(
            loader
        )

        bar = tqdm(
            loader,

            desc=(
                f"{self.name} "
                f"{stage} E{epoch}"
            ),

            unit="batch",
        )

        for step, batch in enumerate(
            bar,
            start=1,
        ):

            target = (
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
                k: v.to(DEVICE)
                for k, v
                in batch.items()
            }

            group_start = (
                (
                    (step - 1)
                    // self.grad_accum
                )
                * self.grad_accum
                + 1
            )

            divisor = min(
                self.grad_accum,

                num_batches
                - group_start
                + 1,
            )

            with autocast_context():

                output = (
                    self.model(
                        **batch,

                        use_cache=False,

                        logits_to_keep=1,
                    )
                )

                vocab_logits = (
                    output.logits[
                        :,
                        -1,
                        :
                    ]
                )

                choice_logits = (
                    vocab_logits
                    .index_select(
                        -1,

                        self.choice_token_tensor,
                    )
                )

                (
                    raw_loss,
                    ce_sample,
                    _,
                    _,
                ) = self.compute_loss(
                    choice_logits,
                    target,
                    sample_weight,
                )

                loss = (
                    raw_loss
                    / divisor
                )

            if USE_SCALER:

                scaler.scale(
                    loss
                ).backward()

            else:

                loss.backward()

            predictions = (
                choice_logits
                .detach()
                .argmax(dim=1)
            )

            batch_size = (
                target.size(0)
            )

            seen += batch_size

            correct_count += (
                predictions
                .eq(target)
                .sum()
                .item()
            )

            loss_sum += (
                raw_loss
                .detach()
                .float()
                .item()
                * batch_size
            )

            # ------------------------------------------------
            # Surprise memory
            # ------------------------------------------------

            if surprise is not None:

                losses = (
                    ce_sample
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                )

                indices = (
                    sample_idx
                    .cpu()
                    .numpy()
                )

                for index, value in zip(
                    indices,
                    losses,
                ):

                    index = int(
                        index
                    )

                    surprise_sum[
                        index
                    ] = (
                        surprise_sum.get(
                            index,
                            0.0,
                        )
                        + float(value)
                    )

                    surprise_count[
                        index
                    ] = (
                        surprise_count.get(
                            index,
                            0,
                        )
                        + 1
                    )

            should_step = (
                step
                % self.grad_accum
                == 0

                or
                step == num_batches
            )

            if should_step:

                if USE_SCALER:

                    scaler.unscale_(
                        optimizer
                    )

                torch.nn.utils.clip_grad_norm_(
                    [
                        p
                        for p
                        in self.model.parameters()
                        if p.requires_grad
                    ],
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

            bar.set_postfix(
                loss=(
                    f"{loss_sum / seen:.4f}"
                ),

                acc=(
                    f"{correct_count / seen:.4f}"
                ),

                lr=(
                    f"{scheduler.get_last_lr()[0]:.1e}"
                ),
            )

        if surprise is not None:

            for index in surprise_sum:

                current = (
                    surprise_sum[index]
                    /
                    surprise_count[index]
                )

                surprise[
                    index
                ] = (
                    SURPRISE_EMA_BETA
                    * surprise[index]

                    +

                    (
                        1
                        - SURPRISE_EMA_BETA
                    )
                    * current
                )

        return {
            "loss":
                loss_sum / seen,

            "accuracy":
                correct_count / seen,
        }


    # ========================================================
    # Letter scorer — one permutation
    # ========================================================

    def predict_permutation_logits(
        self,
        df,
        permutation,
        processor_obj,
        focus_pairs=None,
        desc="Inference",
    ):

        self.model.eval()

        results = []

        for start in tqdm(
            range(
                0,
                len(df),
                self.eval_batch_size,
            ),

            desc=desc,

            unit="batch",
        ):

            end = min(
                start
                + self.eval_batch_size,
                len(df),
            )

            part = (
                df.iloc[
                    start:end
                ]
            )

            images = []
            texts = []

            for local_index, (
                _,
                row
            ) in enumerate(
                part.iterrows()
            ):

                row_perm = (
                    permute_row(
                        row,
                        permutation,

                        has_answer=(
                            "answer"
                            in df.columns
                        ),
                    )
                )

                image = load_rgb_image(
                    row_perm[
                        "path"
                    ]
                )

                focus = None

                if focus_pairs is not None:

                    original_focus = (
                        focus_pairs[
                            start
                            + local_index
                        ]
                    )

                    converted = []

                    for letter in (
                        original_focus
                    ):

                        original_index = (
                            CHOICE_TO_INDEX[
                                letter
                            ]
                        )

                        new_index = (
                            permutation.index(
                                original_index
                            )
                        )

                        converted.append(
                            CHOICES[
                                new_index
                            ]
                        )

                    focus = tuple(
                        converted
                    )

                messages = (
                    build_prompt_messages(
                        row_perm,
                        image,
                        focus,
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

                texts.append(
                    text
                )

                images.append(
                    image
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
                k: v.to(DEVICE)
                for k, v
                in inputs.items()
            }

            with (
                torch.inference_mode(),
                autocast_context(),
            ):

                output = self.model(
                    **inputs,

                    use_cache=False,

                    logits_to_keep=1,
                )

                vocab = (
                    output.logits[
                        :,
                        -1,
                        :
                    ]
                    .float()
                )

                choice = (
                    vocab.index_select(
                        -1,

                        self.choice_token_tensor,
                    )
                )

                restored = (
                    restore_logits_to_original_order(
                        choice,
                        permutation,
                    )
                )

            results.append(
                restored.cpu()
            )

        return torch.cat(
            results,
            dim=0,
        )


    # ========================================================
    # Letter permutation ensemble
    # ========================================================

    def predict_letter_logp(
        self,
        df,
        processor_obj=None,
        focus_pairs=None,
        use_permutation=True,
        desc="Inference",
    ):

        if processor_obj is None:

            processor_obj = (
                self.processor
            )

        if (
            use_permutation
            and
            USE_PERMUTATION_INFERENCE
        ):

            permutations = (
                PERMUTATIONS
            )

        else:

            permutations = [
                PERMUTATIONS[0]
            ]

        outputs = []

        for index, permutation in enumerate(
            permutations
        ):

            logits = (
                self.predict_permutation_logits(
                    df,
                    permutation,
                    processor_obj,
                    focus_pairs,

                    desc=(
                        f"{desc} P{index + 1}"
                    ),
                )
            )

            outputs.append(
                F.log_softmax(
                    logits,
                    dim=1,
                )
            )

        return (
            torch.stack(
                outputs,
                dim=0,
            )
            .mean(dim=0)
        )


    # ========================================================
    # Semantic Candidate Scorer
    #
    # Candidate A:
    # image + question + actual text of option A
    #
    # model scores:
    # yes / no
    #
    # repeated for all 4 candidates.
    # ========================================================

    def predict_semantic_logp(
        self,
        df,
        desc="Semantic",
    ):

        if not USE_SEMANTIC_SCORER:

            return torch.zeros(
                (
                    len(df),
                    4,
                ),
                dtype=torch.float32,
            )

        self.model.eval()

        outputs = []

        for start in tqdm(
            range(
                0,
                len(df),
                self.semantic_batch_size,
            ),

            desc=desc,

            unit="batch",
        ):

            end = min(
                start
                + self.semantic_batch_size,
                len(df),
            )

            rows = (
                df.iloc[
                    start:end
                ]
            )

            texts = []
            images = []

            for _, row in (
                rows.iterrows()
            ):

                image = load_rgb_image(
                    row["path"]
                )

                for choice in CHOICES:

                    candidate = str(
                        row[
                            choice
                        ]
                    )

                    messages = (
                        build_semantic_messages(
                            row,
                            image,
                            candidate,
                        )
                    )

                    text = (
                        self.processor
                        .apply_chat_template(
                            messages,

                            tokenize=False,

                            add_generation_prompt=True,
                        )
                    )

                    texts.append(
                        text
                    )

                    # four candidate prompts
                    # reference same PIL image
                    images.append(
                        image
                    )

            inputs = (
                self.processor(
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
                k: v.to(DEVICE)
                for k, v
                in inputs.items()
            }

            with (
                torch.inference_mode(),
                autocast_context(),
            ):

                result = (
                    self.model(
                        **inputs,

                        use_cache=False,

                        logits_to_keep=1,
                    )
                )

                vocab_logits = (
                    result.logits[
                        :,
                        -1,
                        :
                    ]
                    .float()
                )

                binary_logits = (
                    vocab_logits
                    .index_select(
                        -1,

                        self.yes_no_token_tensor,
                    )
                )

                binary_logp = (
                    F.log_softmax(
                        binary_logits,
                        dim=1,
                    )
                )

                # column 0 = log P(yes)
                yes_scores = (
                    binary_logp[
                        :,
                        0
                    ]
                )

                batch_rows = (
                    end - start
                )

                yes_scores = (
                    yes_scores.reshape(
                        batch_rows,
                        4,
                    )
                )

                # Normalize candidates against each other
                candidate_logp = (
                    F.log_softmax(
                        yes_scores,
                        dim=1,
                    )
                )

            outputs.append(
                candidate_logp.cpu()
            )

        return torch.cat(
            outputs,
            dim=0,
        )


    # ========================================================
    # Metrics
    # ========================================================

    @staticmethod
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
                for answer in df[
                    "answer"
                ]
            ],
            dtype=torch.long,
        )


    @staticmethod
    def accuracy(
        logp,
        gold,
    ):

        pred = (
            logp.argmax(
                dim=1
            )
        )

        acc = (
            pred
            .eq(gold)
            .float()
            .mean()
            .item()
        )

        return (
            acc,
            pred,
        )


    @staticmethod
    def confidence_margin(
        logp,
    ):

        values = (
            torch.topk(
                logp,
                2,
                dim=1,
            )
            .values
        )

        return (
            values[
                :,
                0
            ]
            -
            values[
                :,
                1
            ]
        )


    # ========================================================
    # Clone adapter
    # ========================================================

    def clone_state(
        self,
    ):

        state = (
            get_peft_model_state_dict(
                self.model
            )
        )

        return {
            key:
                value.detach()
                .cpu()
                .clone()

            for key, value
            in state.items()
        }


    # ========================================================
    # Tune Stage 1
    # ========================================================

    def tune_stage1(
        self,
    ):

        dataset = (
            self.DatasetImpl(
                train_subset,
                train_mode=True,
            )
        )

        surprise = np.ones(
            len(dataset),
            dtype=np.float32,
        )

        first_loader = (
            self.build_loader(
                dataset,
                surprise,
            )
        )

        optimizer, scheduler = (
            self.make_optimizer(
                NUM_EPOCHS,
                len(first_loader),
                pseudo=False,
            )
        )

        for epoch in range(
            1,
            NUM_EPOCHS + 1,
        ):

            loader = (
                self.build_loader(
                    dataset,
                    surprise,
                )
            )

            metrics = (
                self.train_one_epoch(
                    epoch,
                    loader,
                    optimizer,
                    scheduler,
                    surprise,
                    "Stage1",
                )
            )

            val_acc = None

            should_eval = (
                epoch % EVAL_EVERY == 0
                or
                epoch == NUM_EPOCHS
            )

            if should_eval:

                val_logp = (
                    self.predict_letter_logp(
                        valid_subset,

                        use_permutation=True,

                        desc=(
                            f"{self.name} "
                            f"Val E{epoch}"
                        ),
                    )
                )

                gold = (
                    self.gold_tensor(
                        valid_subset
                    )
                )

                (
                    val_acc,
                    _,
                ) = self.accuracy(
                    val_logp,
                    gold,
                )

                print(
                    f"\n{self.name} "
                    f"E{epoch} "
                    f"val={val_acc:.5f}"
                )

                if (
                    val_acc
                    > self.best_val_accuracy
                ):

                    self.best_val_accuracy = (
                        val_acc
                    )

                    self.best_epoch = (
                        epoch
                    )

                    self.best_state = (
                        self.clone_state()
                    )

                    print(
                        "★ New Stage1 best"
                    )

            self.history.append(
                {
                    "stage":
                        "stage1",

                    "epoch":
                        epoch,

                    "train_loss":
                        metrics["loss"],

                    "train_accuracy":
                        metrics[
                            "accuracy"
                        ],

                    "valid_accuracy":
                        val_acc,
                }
            )

        if self.best_state is None:

            raise RuntimeError(
                "No best Stage1 state."
            )

        set_peft_model_state_dict(
            self.model,
            self.best_state,
        )


    # ========================================================
    # Human majority
    # ========================================================

    @staticmethod
    def human_majority(
        row,
    ):

        answers = []

        for column in (
            DEV_ANSWER_COLS
        ):

            value = (
                row[
                    column
                ]
            )

            if pd.isna(
                value
            ):

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
            choice:
                answers.count(
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

        if len(winners) != 1:

            return (
                None,
                0.0,
            )

        return (
            winners[0],

            max_count
            / len(answers),
        )


    # ========================================================
    # Generate pseudo labels
    # ========================================================

    def generate_pseudo_df(
        self,
    ):

        if not USE_DEV_PSEUDO_STAGE:

            return None

        teacher_logp = (
            self.predict_letter_logp(
                dev_df,

                use_permutation=False,

                desc=(
                    f"{self.name} "
                    "DEV teacher"
                ),
            )
        )

        predictions = (
            teacher_logp.argmax(
                dim=1
            )
        )

        margins = (
            self.confidence_margin(
                teacher_logp
            )
        )

        records = []

        for index in range(
            len(dev_df)
        ):

            row = (
                dev_df.iloc[
                    index
                ]
            )

            (
                human_answer,
                human_conf,
            ) = self.human_majority(
                row
            )

            if human_answer is None:

                continue

            if (
                human_conf
                <
                PSEUDO_MIN_HUMAN_CONF
            ):

                continue

            teacher_answer = (
                CHOICES[
                    int(
                        predictions[
                            index
                        ].item()
                    )
                ]
            )

            if (
                teacher_answer
                != human_answer
            ):

                continue

            teacher_margin = float(
                margins[
                    index
                ].item()
            )

            if (
                teacher_margin
                <
                PSEUDO_MIN_TEACHER_MARGIN
            ):

                continue

            record = (
                row.to_dict()
            )

            record[
                "answer"
            ] = human_answer

            record[
                "sample_weight"
            ] = (
                PSEUDO_BASE_WEIGHT
                * human_conf
            )

            records.append(
                record
            )

        print(
            self.name,
            "pseudo count:",
            len(records),
        )

        if not records:

            return None

        return pd.DataFrame(
            records
        )


    # ========================================================
    # Tune pseudo Stage2
    # ========================================================

    def tune_pseudo_stage(
        self,
    ):

        pseudo_df = (
            self.generate_pseudo_df()
        )

        if (
            pseudo_df is None
            or len(pseudo_df) == 0
        ):

            return

        stage1_state = (
            self.clone_state()
        )

        gold = (
            train_subset.copy()
        )

        gold[
            "sample_weight"
        ] = 1.0

        pseudo_train = (
            pseudo_df[
                gold.columns
            ]
            .copy()
        )

        stage2 = pd.concat(
            [
                gold,
                pseudo_train,
            ],
            ignore_index=True,
        )

        dataset = (
            self.DatasetImpl(
                stage2,
                train_mode=True,
            )
        )

        surprise = np.ones(
            len(dataset),
            dtype=np.float32,
        )

        loader = (
            self.build_loader(
                dataset,
                surprise,
            )
        )

        optimizer, scheduler = (
            self.make_optimizer(
                PSEUDO_EPOCHS,
                len(loader),
                pseudo=True,
            )
        )

        best_acc = (
            self.best_val_accuracy
        )

        best_state = (
            stage1_state
        )

        best_epoch = 0

        for epoch in range(
            1,
            PSEUDO_EPOCHS + 1,
        ):

            loader = (
                self.build_loader(
                    dataset,
                    surprise,
                )
            )

            metrics = (
                self.train_one_epoch(
                    epoch,
                    loader,
                    optimizer,
                    scheduler,
                    surprise,
                    "Pseudo",
                )
            )

            val_logp = (
                self.predict_letter_logp(
                    valid_subset,
                    use_permutation=True,

                    desc=(
                        f"{self.name} "
                        f"Pseudo Val E{epoch}"
                    ),
                )
            )

            gold_tensor = (
                self.gold_tensor(
                    valid_subset
                )
            )

            (
                acc,
                _,
            ) = self.accuracy(
                val_logp,
                gold_tensor,
            )

            print(
                f"{self.name} "
                f"pseudo E{epoch}: "
                f"{acc:.5f}"
            )

            self.history.append(
                {
                    "stage":
                        "pseudo",

                    "epoch":
                        epoch,

                    "train_loss":
                        metrics["loss"],

                    "train_accuracy":
                        metrics[
                            "accuracy"
                        ],

                    "valid_accuracy":
                        acc,
                }
            )

            if acc > best_acc:

                best_acc = (
                    acc
                )

                best_state = (
                    self.clone_state()
                )

                best_epoch = (
                    epoch
                )

        if best_epoch > 0:

            self.best_val_accuracy = (
                best_acc
            )

            self.best_state = (
                best_state
            )

            self.best_pseudo_epoch = (
                best_epoch
            )

            self.best_pseudo_df = (
                pseudo_df.copy()
            )

        else:

            self.best_state = (
                stage1_state
            )

        set_peft_model_state_dict(
            self.model,
            self.best_state,
        )


    # ========================================================
    # Second-pass calibration
    # ========================================================

    def calibrate_second_pass(
        self,
        base_logp,
    ):

        if not USE_SECOND_PASS:

            return 0.0

        gold = (
            self.gold_tensor(
                valid_subset
            )
        )

        margins = (
            self.confidence_margin(
                base_logp
            )
        )

        max_threshold = max(
            SECOND_PASS_THRESHOLDS
        )

        uncertain = (
            torch.where(
                margins
                <= max_threshold
            )[0]
            .tolist()
        )

        if not uncertain:

            return 0.0

        subset = (
            valid_subset.iloc[
                uncertain
            ]
            .reset_index(drop=True)
        )

        top2 = (
            torch.topk(
                base_logp,
                2,
                dim=1,
            )
            .indices
        )

        focus = []

        for index in uncertain:

            pair = (
                top2[
                    index
                ].tolist()
            )

            focus.append(
                (
                    CHOICES[
                        pair[0]
                    ],
                    CHOICES[
                        pair[1]
                    ],
                )
            )

        second_logp = (
            self.predict_letter_logp(
                subset,

                processor_obj=(
                    self.high_processor
                ),

                focus_pairs=(
                    focus
                ),

                use_permutation=False,

                desc=(
                    f"{self.name} "
                    "HighRes Val"
                ),
            )
        )

        second_full = (
            base_logp.clone()
        )

        for local, original in enumerate(
            uncertain
        ):

            second_full[
                original
            ] = (
                second_logp[
                    local
                ]
            )

        best_threshold = 0.0
        best_accuracy = -1.0

        for threshold in (
            SECOND_PASS_THRESHOLDS
        ):

            use_second = (
                margins
                <= threshold
            )

            fused = (
                base_logp.clone()
            )

            if use_second.any():

                fused[
                    use_second
                ] = (
                    (
                        1
                        - SECOND_PASS_WEIGHT
                    )
                    * base_logp[
                        use_second
                    ]

                    +

                    SECOND_PASS_WEIGHT
                    * second_full[
                        use_second
                    ]
                )

            (
                accuracy,
                _,
            ) = self.accuracy(
                fused,
                gold,
            )

            print(
                self.name,
                "threshold",
                threshold,
                "accuracy",
                accuracy,
            )

            if accuracy > best_accuracy:

                best_accuracy = (
                    accuracy
                )

                best_threshold = (
                    threshold
                )

        return best_threshold


    # ========================================================
    # Apply second pass
    # ========================================================

    def apply_second_pass(
        self,
        df,
        base_logp,
        threshold,
        desc,
    ):

        if (
            not USE_SECOND_PASS
            or threshold <= 0
        ):

            return base_logp

        margins = (
            self.confidence_margin(
                base_logp
            )
        )

        uncertain = (
            torch.where(
                margins
                <= threshold
            )[0]
            .tolist()
        )

        if not uncertain:

            return base_logp

        subset = (
            df.iloc[
                uncertain
            ]
            .reset_index(drop=True)
        )

        top2 = (
            torch.topk(
                base_logp,
                2,
                dim=1,
            )
            .indices
        )

        focus = []

        for index in uncertain:

            pair = (
                top2[
                    index
                ].tolist()
            )

            focus.append(
                (
                    CHOICES[
                        pair[0]
                    ],
                    CHOICES[
                        pair[1]
                    ],
                )
            )

        second = (
            self.predict_letter_logp(
                subset,

                processor_obj=(
                    self.high_processor
                ),

                focus_pairs=focus,

                use_permutation=False,

                desc=desc,
            )
        )

        final = (
            base_logp.clone()
        )

        for local, original in enumerate(
            uncertain
        ):

            final[
                original
            ] = (
                (
                    1
                    - SECOND_PASS_WEIGHT
                )
                * base_logp[
                    original
                ]

                +

                SECOND_PASS_WEIGHT
                * second[
                    local
                ]
            )

        return final


    # ========================================================
    # Semantic weight calibration
    # ========================================================

    def calibrate_semantic_weight(
        self,
        letter_logp,
        semantic_logp,
    ):

        if not USE_SEMANTIC_SCORER:

            return (
                0.0,
                letter_logp,
            )

        gold = (
            self.gold_tensor(
                valid_subset
            )
        )

        best_weight = 0.0
        best_acc = -1.0
        best_fused = None

        print(
            f"\n===== {self.name} "
            "SEMANTIC CALIBRATION ====="
        )

        for weight in (
            SEMANTIC_WEIGHTS
        ):

            fused = (
                (
                    1.0
                    - weight
                )
                * letter_logp

                +

                weight
                * semantic_logp
            )

            (
                acc,
                _,
            ) = self.accuracy(
                fused,
                gold,
            )

            print(
                f"semantic={weight:.2f} "
                f"acc={acc:.5f}"
            )

            if acc > best_acc:

                best_acc = (
                    acc
                )

                best_weight = (
                    weight
                )

                best_fused = (
                    fused.clone()
                )

        return (
            best_weight,
            best_fused,
        )


    # ========================================================
    # Tuning phase
    # ========================================================

    def run_tuning(
        self,
    ):

        self.load()

        self.tune_stage1()

        self.tune_pseudo_stage()

        if self.best_state is not None:

            set_peft_model_state_dict(
                self.model,
                self.best_state,
            )

        self.model.eval()

        gold = (
            self.gold_tensor(
                valid_subset
            )
        )

        # ----------------------------------------------------
        # identity
        # ----------------------------------------------------

        identity = (
            self.predict_letter_logp(
                valid_subset,

                use_permutation=False,

                desc=(
                    f"{self.name} "
                    "Identity"
                ),
            )
        )

        (
            identity_acc,
            _,
        ) = self.accuracy(
            identity,
            gold,
        )

        # ----------------------------------------------------
        # permutation
        # ----------------------------------------------------

        perm = (
            self.predict_letter_logp(
                valid_subset,

                use_permutation=True,

                desc=(
                    f"{self.name} "
                    "Permutation"
                ),
            )
        )

        (
            perm_acc,
            _,
        ) = self.accuracy(
            perm,
            gold,
        )

        use_permutation = (
            perm_acc
            >= identity_acc
        )

        letter_base = (
            perm
            if use_permutation
            else identity
        )

        print(
            self.name,
            "identity:",
            identity_acc,
        )

        print(
            self.name,
            "permutation:",
            perm_acc,
        )

        # ----------------------------------------------------
        # second pass
        # ----------------------------------------------------

        second_threshold = (
            self.calibrate_second_pass(
                letter_base
            )
        )

        letter_final = (
            self.apply_second_pass(
                valid_subset,

                letter_base,

                second_threshold,

                desc=(
                    f"{self.name} "
                    "Final HighRes Val"
                ),
            )
        )

        # ----------------------------------------------------
        # semantic scorer
        # ----------------------------------------------------

        semantic_logp = (
            self.predict_semantic_logp(
                valid_subset,

                desc=(
                    f"{self.name} "
                    "Semantic Val"
                ),
            )
        )

        (
            semantic_weight,
            fused_val,
        ) = (
            self.calibrate_semantic_weight(
                letter_final,
                semantic_logp,
            )
        )

        (
            final_val_acc,
            _,
        ) = self.accuracy(
            fused_val,
            gold,
        )

        print(
            f"\n===== {self.name} "
            f"TUNING COMPLETE ====="
        )

        print(
            "best stage1 epoch:",
            self.best_epoch,
        )

        print(
            "best pseudo epoch:",
            self.best_pseudo_epoch,
        )

        print(
            "use permutation:",
            use_permutation,
        )

        print(
            "second threshold:",
            second_threshold,
        )

        print(
            "semantic weight:",
            semantic_weight,
        )

        print(
            "validation:",
            final_val_acc,
        )

        plan = {
            "name":
                self.name,

            "best_epoch":
                self.best_epoch,

            "best_pseudo_epoch":
                self.best_pseudo_epoch,

            "pseudo_df":
                (
                    self.best_pseudo_df.copy()
                    if self.best_pseudo_df
                    is not None
                    else None
                ),

            "use_permutation":
                use_permutation,

            "second_threshold":
                second_threshold,

            "semantic_weight":
                semantic_weight,

            "validation_logp":
                fused_val.cpu(),

            "validation_accuracy":
                final_val_acc,
        }

        return plan


    # ========================================================
    # FINAL Stage1
    #
    # Gold 100%
    # No validation.
    # ========================================================

    def final_gold_stage1(
        self,
    ):

        epochs = int(
            self.tuning_plan[
                "best_epoch"
            ]
        )

        print(
            f"\n===== {self.name} "
            "FINAL GOLD 100% ====="
        )

        print(
            "Epochs:",
            epochs,
        )

        dataset = (
            self.DatasetImpl(
                full_gold_df,

                train_mode=True,
            )
        )

        surprise = np.ones(
            len(dataset),
            dtype=np.float32,
        )

        loader = (
            self.build_loader(
                dataset,
                surprise,
            )
        )

        optimizer, scheduler = (
            self.make_optimizer(
                epochs,
                len(loader),
                pseudo=False,
            )
        )

        for epoch in range(
            1,
            epochs + 1,
        ):

            loader = (
                self.build_loader(
                    dataset,
                    surprise,
                )
            )

            self.train_one_epoch(
                epoch,
                loader,
                optimizer,
                scheduler,
                surprise,
                "FINAL-GOLD",
            )


    # ========================================================
    # FINAL pseudo stage
    #
    # Uses pseudo configuration selected during tuning.
    # No validation.
    # ========================================================

    def final_pseudo_stage(
        self,
    ):

        if not (
            USE_DEV_PSEUDO_STAGE
            and
            FINAL_USE_SELECTED_PSEUDO_STAGE
        ):

            return

        pseudo_epochs = int(
            self.tuning_plan[
                "best_pseudo_epoch"
            ]
        )

        pseudo_df = (
            self.tuning_plan[
                "pseudo_df"
            ]
        )

        if (
            pseudo_epochs <= 0
            or
            pseudo_df is None
            or
            len(pseudo_df) == 0
        ):

            return

        print(
            f"\n===== {self.name} "
            "FINAL PSEUDO ====="
        )

        print(
            "Pseudo epochs:",
            pseudo_epochs,
        )

        gold = (
            full_gold_df.copy()
        )

        pseudo_train = (
            pseudo_df[
                gold.columns
            ]
            .copy()
        )

        combined = pd.concat(
            [
                gold,
                pseudo_train,
            ],
            ignore_index=True,
        )

        dataset = (
            self.DatasetImpl(
                combined,
                train_mode=True,
            )
        )

        surprise = np.ones(
            len(dataset),
            dtype=np.float32,
        )

        loader = (
            self.build_loader(
                dataset,
                surprise,
            )
        )

        optimizer, scheduler = (
            self.make_optimizer(
                pseudo_epochs,
                len(loader),
                pseudo=True,
            )
        )

        for epoch in range(
            1,
            pseudo_epochs + 1,
        ):

            loader = (
                self.build_loader(
                    dataset,
                    surprise,
                )
            )

            self.train_one_epoch(
                epoch,
                loader,
                optimizer,
                scheduler,
                surprise,
                "FINAL-PSEUDO",
            )


    # ========================================================
    # FINAL test inference
    # ========================================================

    def run_final(
        self,
    ):

        if self.tuning_plan is None:

            raise RuntimeError(
                "Final runner requires tuning_plan."
            )

        self.load()

        # ====================================================
        # 100% Gold retraining
        # ====================================================

        self.final_gold_stage1()

        # ====================================================
        # Optional selected pseudo stage
        # ====================================================

        self.final_pseudo_stage()

        self.model.eval()

        use_permutation = (
            self.tuning_plan[
                "use_permutation"
            ]
        )

        threshold = (
            self.tuning_plan[
                "second_threshold"
            ]
        )

        semantic_weight = (
            self.tuning_plan[
                "semantic_weight"
            ]
        )

        # ====================================================
        # letter score
        # ====================================================

        letter = (
            self.predict_letter_logp(
                test_df,

                use_permutation=(
                    use_permutation
                ),

                desc=(
                    f"{self.name} "
                    "FINAL Test Letter"
                ),
            )
        )

        letter = (
            self.apply_second_pass(
                test_df,

                letter,

                threshold,

                desc=(
                    f"{self.name} "
                    "FINAL Test HighRes"
                ),
            )
        )

        # ====================================================
        # semantic score
        # ====================================================

        if (
            USE_SEMANTIC_SCORER
            and
            semantic_weight > 0
        ):

            semantic = (
                self.predict_semantic_logp(
                    test_df,

                    desc=(
                        f"{self.name} "
                        "FINAL Test Semantic"
                    ),
                )
            )

            final_logp = (
                (
                    1.0
                    - semantic_weight
                )
                * letter

                +

                semantic_weight
                * semantic
            )

        else:

            final_logp = (
                letter
            )

        # ====================================================
        # Save final adapter
        # ====================================================

        self.save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.model.save_pretrained(
            self.save_dir
        )

        self.processor.save_pretrained(
            self.save_dir
        )

        return (
            final_logp.cpu()
        )


    # ========================================================
    # Unload
    # ========================================================

    def unload(
        self,
    ):

        if self.model is not None:

            del self.model

        if self.processor is not None:

            del self.processor

        if self.high_processor is not None:

            del self.high_processor

        self.model = None

        self.processor = None

        self.high_processor = None

        gc.collect()

        torch.cuda.empty_cache()

        torch.cuda.synchronize()


# ============================================================
# 30. Output folders
# ============================================================

MODEL_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)

SUBMISSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 31. PHASE A
#
# DEVELOPMENT / TUNING
#
# Gold 90%
# Validation 10%
#
# Determine:
# - best epoch
# - pseudo stage
# - permutation
# - second pass threshold
# - semantic weight
# ============================================================

tuning_plans = {}


for cfg in MODEL_CONFIGS:

    runner = (
        SSAFYTRunner(
            cfg,
            final_mode=False,
        )
    )

    plan = (
        runner.run_tuning()
    )

    tuning_plans[
        cfg["name"]
    ] = plan

    runner.unload()

    del runner

    gc.collect()

    torch.cuda.empty_cache()


# ============================================================
# 32. Calibrate 2B / 4B ensemble
#
# Still development phase.
# ============================================================

val_2b = (
    tuning_plans[
        "2b"
    ][
        "validation_logp"
    ]
)

val_4b = (
    tuning_plans[
        "4b"
    ][
        "validation_logp"
    ]
)

validation_gold = torch.tensor(
    [
        CHOICE_TO_INDEX[
            str(answer)
            .strip()
            .lower()
        ]

        for answer
        in valid_subset[
            "answer"
        ]
    ],
    dtype=torch.long,
)


best_2b_weight = None
best_ensemble_acc = -1.0
best_validation_fused = None


print(
    "\n===== 2B/4B ENSEMBLE CALIBRATION ====="
)


for weight_2b in (
    ENSEMBLE_WEIGHTS_2B
):

    weight_4b = (
        1.0
        - weight_2b
    )

    fused = (
        weight_2b
        * val_2b

        +

        weight_4b
        * val_4b
    )

    prediction = (
        fused.argmax(
            dim=1
        )
    )

    accuracy = (
        prediction
        .eq(
            validation_gold
        )
        .float()
        .mean()
        .item()
    )

    print(
        f"2B={weight_2b:.2f} "
        f"4B={weight_4b:.2f} "
        f"acc={accuracy:.5f}"
    )

    if accuracy > best_ensemble_acc:

        best_ensemble_acc = (
            accuracy
        )

        best_2b_weight = (
            weight_2b
        )

        best_validation_fused = (
            fused.clone()
        )


best_4b_weight = (
    1.0
    - best_2b_weight
)


print(
    "\n===== DEVELOPMENT COMPLETE ====="
)

print(
    "2B validation:",
    tuning_plans[
        "2b"
    ][
        "validation_accuracy"
    ]
)

print(
    "4B validation:",
    tuning_plans[
        "4b"
    ][
        "validation_accuracy"
    ]
)

print(
    "Ensemble validation:",
    best_ensemble_acc,
)

print(
    "2B weight:",
    best_2b_weight,
)

print(
    "4B weight:",
    best_4b_weight,
)


# ============================================================
# 33. Save validation diagnostics
# ============================================================

ensemble_val_pred = (
    best_validation_fused
    .argmax(dim=1)
)

validation_rows = []

for index in range(
    len(valid_subset)
):

    validation_rows.append(
        {
            "id":
                valid_subset.iloc[
                    index
                ]["id"],

            "gold":
                str(
                    valid_subset.iloc[
                        index
                    ]["answer"]
                )
                .strip()
                .lower(),

            "pred":
                CHOICES[
                    int(
                        ensemble_val_pred[
                            index
                        ].item()
                    )
                ],

            "correct":
                int(
                    ensemble_val_pred[
                        index
                    ].item()
                )
                ==
                int(
                    validation_gold[
                        index
                    ].item()
                ),
        }
    )

pd.DataFrame(
    validation_rows
).to_csv(
    ENSEMBLE_VALIDATION_PATH,
    index=False,
)


# ============================================================
# IMPORTANT
#
# FROM THIS POINT ON:
#
# valid_subset is NEVER used again.
#
# All model architecture/hyperparameters
# have already been fixed.
#
# Now:
#
# pretrained model
#       ↓
# gold 100%
#       ↓
# selected pseudo stage
#       ↓
# test
#
# ============================================================


# ============================================================
# 34. PHASE B
#
# FINAL GOLD 100% RETRAINING
# ============================================================

final_model_test_logp = {}


if USE_FINAL_FULL_GOLD_RETRAIN:

    for cfg in MODEL_CONFIGS:

        name = (
            cfg["name"]
        )

        print(
            "\n"
            + "#" * 80
        )

        print(
            "FINAL 100% GOLD RETRAIN:",
            name,
        )

        print(
            "#" * 80
        )

        runner = (
            SSAFYTRunner(
                cfg,

                final_mode=True,

                tuning_plan=(
                    tuning_plans[
                        name
                    ]
                ),
            )
        )

        test_logp = (
            runner.run_final()
        )

        final_model_test_logp[
            name
        ] = (
            test_logp
        )

        runner.unload()

        del runner

        gc.collect()

        torch.cuda.empty_cache()

else:

    raise RuntimeError(
        "This script expects "
        "USE_FINAL_FULL_GOLD_RETRAIN=True."
    )


# ============================================================
# 35. Final 2B / 4B ensemble
#
# Uses weight selected BEFORE full-gold retraining.
# No validation access here.
# ============================================================

test_2b = (
    final_model_test_logp[
        "2b"
    ]
)

test_4b = (
    final_model_test_logp[
        "4b"
    ]
)

final_test_logp = (
    best_2b_weight
    * test_2b

    +

    best_4b_weight
    * test_4b
)


# ============================================================
# 36. Final predictions
# ============================================================

final_prediction_idx = (
    final_test_logp.argmax(
        dim=1
    )
)

final_predictions = [
    CHOICES[
        int(index)
    ]

    for index
    in final_prediction_idx.tolist()
]


# ============================================================
# 37. Submission
# ============================================================

submission = (
    pd.DataFrame(
        {
            "id":
                test_df[
                    "id"
                ],

            "answer":
                final_predictions,
        }
    )
)

if (
    len(submission)
    != len(test_df)
):

    raise RuntimeError(
        "Submission length mismatch."
    )

if not set(
    submission[
        "answer"
    ].unique()
).issubset(
    VALID_CHOICES
):

    raise RuntimeError(
        "Invalid prediction detected."
    )

submission.to_csv(
    FINAL_SUBMISSION_PATH,
    index=False,
)


# ============================================================
# 38. Save diagnostics
# ============================================================

logit_rows = []

for index in range(
    len(test_df)
):

    logit_rows.append(
        {
            "id":
                test_df.iloc[
                    index
                ]["id"],

            "answer":
                final_predictions[
                    index
                ],

            "score_a":
                float(
                    final_test_logp[
                        index,
                        0
                    ].item()
                ),

            "score_b":
                float(
                    final_test_logp[
                        index,
                        1
                    ].item()
                ),

            "score_c":
                float(
                    final_test_logp[
                        index,
                        2
                    ].item()
                ),

            "score_d":
                float(
                    final_test_logp[
                        index,
                        3
                    ].item()
                ),

            "pred_2b":
                CHOICES[
                    int(
                        test_2b[
                            index
                        ]
                        .argmax()
                        .item()
                    )
                ],

            "pred_4b":
                CHOICES[
                    int(
                        test_4b[
                            index
                        ]
                        .argmax()
                        .item()
                    )
                ],
        }
    )

pd.DataFrame(
    logit_rows
).to_csv(
    FINAL_LOGIT_PATH,
    index=False,
)


# ============================================================
# 39. Final report
# ============================================================

print(
    "\n===== SSAFY-T+ COMPLETE ====="
)

print(
    "\nDevelopment validation:"
)

print(
    "2B:",
    tuning_plans[
        "2b"
    ][
        "validation_accuracy"
    ],
)

print(
    "4B:",
    tuning_plans[
        "4b"
    ][
        "validation_accuracy"
    ],
)

print(
    "2B/4B ensemble:",
    best_ensemble_acc,
)

print(
    "\nSelected 2B settings:"
)

print(
    tuning_plans[
        "2b"
    ]
)

print(
    "\nSelected 4B settings:"
)

print(
    tuning_plans[
        "4b"
    ]
)

print(
    "\nEnsemble weights:"
)

print(
    "2B:",
    best_2b_weight,
)

print(
    "4B:",
    best_4b_weight,
)

print(
    "\nTest distribution:"
)

print(
    pd.Series(
        final_predictions
    )
    .value_counts()
    .sort_index()
)

print(
    "\nTest ratio:"
)

print(
    pd.Series(
        final_predictions
    )
    .value_counts(
        normalize=True
    )
    .sort_index()
)

print(
    "\nSubmission:"
)

print(
    FINAL_SUBMISSION_PATH
)

print(
    "\nLogits:"
)

print(
    FINAL_LOGIT_PATH
)

print(
    "\nValidation diagnostics:"
)

print(
    ENSEMBLE_VALIDATION_PATH
)

print(
    "\nPreview:"
)

print(
    submission.head(10)
)