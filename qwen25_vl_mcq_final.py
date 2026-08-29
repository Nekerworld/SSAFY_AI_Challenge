# ============================================================
# SSAFY-Agent v0
#
# InternVL3.5-4B + Rule Router + YOLO Visual Tools
#
# ------------------------------------------------------------
# MAIN MODEL
# ------------------------------------------------------------
# - OpenGVLab/InternVL3_5-4B
# - 4-bit NF4 QLoRA
# - Language LoRA only
# - r=16 / alpha=32
# - LR=1e-4
# - 1 epoch
# - Gold train 90% / validation 10%
#
# ------------------------------------------------------------
# TOOLS — inference only
# ------------------------------------------------------------
# 1. General YOLO
#    - object
#    - bbox
#    - count
#    - position
#
# 2. Recycling Material YOLO
#    - plastic
#    - glass
#    - metal
#    - paper
#    - etc.
#
# 3. Crop / Zoom
#    - YOLO bbox 확대
#
# 4. Rule Router
#    - question type
#    - InternVL confidence
#
# ------------------------------------------------------------
# PIPELINE
# ------------------------------------------------------------
#
# Image + Question
#        ↓
# InternVL First Pass
#        ↓
# confidence + question
#        ↓
# Rule Router
#        ↓
# YOLO tools (if needed)
#        ↓
# structured evidence + crop
#        ↓
# InternVL Second Pass
#        ↓
# a / b / c / d
#
# ============================================================


# ============================================================
# 0. Required packages
# ============================================================
#
# pip install -U transformers accelerate bitsandbytes peft
# pip install -U ultralytics huggingface_hub
# pip install -U scikit-learn pandas pillow tqdm
#
# ============================================================


import gc
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from PIL import Image

import torch

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from sklearn.model_selection import (
    train_test_split,
)

from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    BitsAndBytesConfig,
    get_linear_schedule_with_warmup,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from ultralytics import YOLO

from huggingface_hub import (
    hf_hub_download,
    list_repo_files,
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
# 2. Core config
# ============================================================

SEED = 42

MODEL_ID = "OpenGVLab/InternVL3_5-4B"


# ============================================================
# 3. Training config
# ============================================================

NUM_EPOCHS = 1

LR = 1e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.03

MAX_GRAD_NORM = 1.0


# ------------------------------------------------------------
# 3090 Ti 24GB
# ------------------------------------------------------------

TRAIN_BATCH_SIZE = 2

GRAD_ACCUM = 8

EVAL_BATCH_SIZE = 4


# ============================================================
# 4. LoRA
# ============================================================

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05


# ============================================================
# 5. Tool config
# ============================================================

USE_GENERAL_YOLO = True

USE_MATERIAL_YOLO = True

USE_CROP_TOOL = True


# ------------------------------------------------------------
# General YOLO
#
# 최초 실행 시 Ultralytics weight 다운로드 가능.
# 추론 자체는 로컬 GPU에서 수행.
# ------------------------------------------------------------

GENERAL_YOLO_MODEL = "yolov8l.pt"

GENERAL_YOLO_CONF = 0.25

GENERAL_YOLO_IMGSZ = 640


# ------------------------------------------------------------
# Material YOLO
#
# Hugging Face의 공개 recycling/material YOLO 저장소.
#
# repo 내 .pt checkpoint를 자동 탐색한다.
#
# 저장소를 다른 recycling YOLO로 교체해도 됨.
# ------------------------------------------------------------

MATERIAL_YOLO_REPO = (
    "CatSat/yolov11-litter-materials"
)

MATERIAL_YOLO_CONF = 0.20

MATERIAL_YOLO_IMGSZ = 640


# ------------------------------------------------------------
# Router
#
# first-pass top1 - top2 logit margin
#
# 낮으면 tool을 사용.
# ------------------------------------------------------------

ROUTER_MARGIN_THRESHOLD = 0.80


# ------------------------------------------------------------
# validation에서 threshold 탐색할 후보
# ------------------------------------------------------------

CALIBRATE_ROUTER_THRESHOLD = True

ROUTER_THRESHOLD_CANDIDATES = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
]


# ------------------------------------------------------------
# Crop
#
# bbox 주변을 조금 넓혀 자른다.
# ------------------------------------------------------------

CROP_PADDING_RATIO = 0.12

MAX_CROPS_PER_SAMPLE = 2


# ============================================================
# 6. Paths
# ============================================================

TRAIN_CSV = PROJECT_DIR / "train.csv"

TEST_CSV = PROJECT_DIR / "test.csv"


SAVE_DIR = (
    PROJECT_DIR
    / "model"
    / "internvl3_5_4b_agent"
)


SUBMISSION_DIR = (
    PROJECT_DIR
    / "submission"
)


SUBMISSION_PATH = (
    SUBMISSION_DIR
    / "submission_internvl_agent.csv"
)


BASELINE_SUBMISSION_PATH = (
    SUBMISSION_DIR
    / "submission_internvl_baseline.csv"
)


VALIDATION_PATH = (
    SUBMISSION_DIR
    / "internvl_agent_validation.csv"
)


TOOL_LOG_PATH = (
    SUBMISSION_DIR
    / "internvl_agent_tool_log.csv"
)


Image.MAX_IMAGE_PIXELS = None


# ============================================================
# 7. Reproducibility / CUDA
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)


if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU가 필요합니다."
    )


torch.cuda.manual_seed_all(
    SEED
)


DEVICE = torch.device(
    "cuda:0"
)


COMPUTE_DTYPE = (
    torch.bfloat16
    if torch.cuda.is_bf16_supported()
    else torch.float16
)


USE_SCALER = (
    COMPUTE_DTYPE
    == torch.float16
)


print(
    "Project:",
    PROJECT_DIR
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
# 8. Choices
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


# ============================================================
# 9. Image helpers
# ============================================================

def resolve_image_path(
    value,
):

    path = Path(
        str(value)
    )

    if not path.is_absolute():

        path = (
            PROJECT_DIR
            / path
        )

    return path


def load_rgb_image(
    value,
):

    path = resolve_image_path(
        value
    )

    with Image.open(
        path
    ) as image:

        return image.convert(
            "RGB"
        )


def autocast_context():

    return torch.autocast(
        device_type="cuda",
        dtype=COMPUTE_DTYPE,
    )


def sanitize_inputs(
    inputs,
):

    inputs.pop(
        "token_type_ids",
        None,
    )

    return inputs


# ============================================================
# 10. Dataset
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
            f"{name}: missing "
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


# ============================================================
# 11. Split
# ============================================================

train_subset, valid_subset = (
    train_test_split(
        train_df,

        test_size=0.10,

        random_state=SEED,

        stratify=train_df[
            "answer"
        ],
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


print(
    "\n===== DATA ====="
)

print(
    "Train:",
    len(train_subset)
)

print(
    "Valid:",
    len(valid_subset)
)

print(
    "Test:",
    len(test_df)
)


# ============================================================
# 12. InternVL Processor
# ============================================================

processor = (
    AutoProcessor.from_pretrained(
        MODEL_ID,

        trust_remote_code=True,
    )
)


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


# ============================================================
# 13. InternVL 4-bit model
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


model = (
    AutoModelForImageTextToText
    .from_pretrained(

        MODEL_ID,

        quantization_config=(
            bnb_config
        ),

        device_map={
            "": 0
        },

        trust_remote_code=True,
    )
)


model.config.use_cache = False


model = (
    prepare_model_for_kbit_training(

        model,

        use_gradient_checkpointing=True,
    )
)


# ============================================================
# 14. Language-only LoRA
# ============================================================

TARGET_SUFFIXES = (

    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",

    "gate_proj",
    "up_proj",
    "down_proj",
)


target_modules = []


for name, module in (
    model.named_modules()
):

    lower = (
        name.lower()
    )


    # vision 계층 제외
    if (
        "vision" in lower
        or
        "visual" in lower
        or
        "intern_vit" in lower
    ):

        continue


    if any(
        name.endswith(
            suffix
        )

        for suffix
        in TARGET_SUFFIXES
    ):

        # Linear 계열만
        if (
            "linear"
            in module.__class__
            .__name__
            .lower()
        ):

            target_modules.append(
                name
            )


target_modules = sorted(
    set(
        target_modules
    )
)


if not target_modules:

    raise RuntimeError(
        "Language LoRA target을 "
        "찾지 못했습니다."
    )


print(
    "\nLoRA targets:",
    len(target_modules)
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
            target_modules
        ),

        task_type="CAUSAL_LM",
    )
)


model = get_peft_model(
    model,
    lora_config,
)


model.print_trainable_parameters()


# ============================================================
# 15. Prompts
# ============================================================

SYSTEM_INSTRUCT = (
    "You are a visual multiple-choice "
    "question answering assistant. "
    "Inspect the image carefully and answer "
    "using exactly one lowercase letter: "
    "a, b, c, or d. "
    "Do not explain."
)


def build_question_text(
    row,
):

    return (
        f"{row['question']}\n\n"
        f"(a) {row['a']}\n"
        f"(b) {row['b']}\n"
        f"(c) {row['c']}\n"
        f"(d) {row['d']}\n\n"
        "Return exactly one lowercase letter: "
        "a, b, c, or d."
    )


def build_messages(
    row,
    images,
    extra_context=None,
):

    content = []


    # 원본 + crop 모두 이미지로 전달
    for image in images:

        content.append(
            {
                "type": "image",
                "image": image,
            }
        )


    text = build_question_text(
        row
    )


    if extra_context:

        text = (
            "External visual tools produced "
            "the following observations.\n\n"

            + extra_context

            + "\n\n"
            "The tool outputs may contain errors. "
            "Use them only as supporting evidence. "
            "Inspect the image yourself and make "
            "the final decision.\n\n"

            + text
        )


    content.append(
        {
            "type": "text",
            "text": text,
        }
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
            "content": content,
        },
    ]


# ============================================================
# 16. Training Dataset
# ============================================================

class InternDataset(
    Dataset
):

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


        messages = build_messages(
            row,
            [image],
        )


        full_messages = (
            messages
            + [
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
        )


        return {
            "image": image,
            "messages": messages,
            "full_messages": full_messages,
        }


# ============================================================
# 17. Collator
# ============================================================

@dataclass
class TrainCollator:

    processor: Any


    def __call__(
        self,
        batch,
    ):

        images = [
            sample["image"]
            for sample
            in batch
        ]


        prompt_texts = []

        full_texts = []


        for sample in batch:

            prompt_texts.append(
                self.processor
                .apply_chat_template(

                    sample[
                        "messages"
                    ],

                    tokenize=False,

                    add_generation_prompt=True,
                )
            )


            full_texts.append(
                self.processor
                .apply_chat_template(

                    sample[
                        "full_messages"
                    ],

                    tokenize=False,

                    add_generation_prompt=False,
                )
            )


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


        full_enc = sanitize_inputs(
            full_enc
        )


        prompt_enc = sanitize_inputs(
            prompt_enc
        )


        labels = torch.full_like(

            full_enc[
                "input_ids"
            ],

            -100,
        )


        for i in range(
            len(batch)
        ):

            full_pos = (
                full_enc[
                    "attention_mask"
                ][i]
                .nonzero(
                    as_tuple=False
                )
                .squeeze(-1)
            )


            prompt_pos = (
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
                    full_pos
                ]
            )


            prompt_ids = (
                prompt_enc[
                    "input_ids"
                ][
                    i,
                    prompt_pos
                ]
            )


            prompt_len = int(
                prompt_ids.numel()
            )


            if not torch.equal(

                full_ids[
                    :prompt_len
                ].cpu(),

                prompt_ids.cpu(),
            ):

                raise RuntimeError(
                    "Prompt/full prefix mismatch."
                )


            answer_pos = (
                full_pos[
                    prompt_len
                ]
            )


            labels[
                i,
                answer_pos
            ] = (
                full_enc[
                    "input_ids"
                ][
                    i,
                    answer_pos
                ]
            )


        full_enc[
            "labels"
        ] = labels


        return full_enc


# ============================================================
# 18. Train Loader
# ============================================================

train_loader = DataLoader(

    InternDataset(
        train_subset
    ),

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
)


# ============================================================
# 19. Discover a/b/c/d tokens
# ============================================================

def discover_choice_tokens():

    dummy = Image.new(
        "RGB",
        (448, 448),
    )


    row = {
        "question":
            "Choose the correct answer.",

        "a": "one",
        "b": "two",
        "c": "three",
        "d": "four",
    }


    messages = build_messages(
        row,
        [dummy],
    )


    prompt = (
        processor
        .apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )


    prompt_ids = (
        processor.tokenizer(

            prompt,

            add_special_tokens=False,
        )[
            "input_ids"
        ]
    )


    result = {}


    for choice in CHOICES:

        full = (
            messages
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
            processor
            .apply_chat_template(

                full,

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


    if (
        len(
            set(
                result.values()
            )
        )
        != 4
    ):

        raise RuntimeError(
            "Choice token IDs not unique."
        )


    return result


choice_token_ids = (
    discover_choice_tokens()
)


CHOICE_TOKEN_TENSOR = (
    torch.tensor(

        [
            choice_token_ids[c]
            for c in CHOICES
        ],

        dtype=torch.long,

        device=DEVICE,
    )
)


print(
    "Choice tokens:",
    choice_token_ids
)


# ============================================================
# 20. Optimizer
# ============================================================

trainable_parameters = [

    p

    for p
    in model.parameters()

    if p.requires_grad
]


optimizer = (
    torch.optim.AdamW(

        trainable_parameters,

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
    * steps_per_epoch
)


warmup_steps = int(
    total_steps
    * WARMUP_RATIO
)


scheduler = (
    get_linear_schedule_with_warmup(

        optimizer,

        warmup_steps,

        total_steps,
    )
)


scaler = (
    torch.amp.GradScaler(

        "cuda",

        enabled=USE_SCALER,
    )
)


# ============================================================
# 21. Train InternVL
# ============================================================

optimizer.zero_grad(
    set_to_none=True
)


for epoch in range(
    NUM_EPOCHS
):

    model.train()


    total_loss = 0.0

    count = 0


    bar = tqdm(
        train_loader,
        desc="InternVL training",
    )


    for step, batch in enumerate(
        bar,
        1,
    ):

        batch = sanitize_inputs(
            batch
        )


        batch = {
            k: v.to(DEVICE)

            for k, v
            in batch.items()
        }


        with autocast_context():

            out = model(
                **batch,
                use_cache=False,
            )


            raw_loss = (
                out.loss
            )


            loss = (
                raw_loss
                /
                GRAD_ACCUM
            )


        if USE_SCALER:

            scaler.scale(
                loss
            ).backward()

        else:

            loss.backward()


        total_loss += float(
            raw_loss
            .detach()
            .float()
        )


        count += 1


        if (
            step % GRAD_ACCUM == 0
            or step == len(
                train_loader
            )
        ):

            if USE_SCALER:

                scaler.unscale_(
                    optimizer
                )


            torch.nn.utils.clip_grad_norm_(
                trainable_parameters,
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
                f"{total_loss / count:.4f}"
            )
        )


# ============================================================
# 22. Save InternVL adapter
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


# ============================================================
# 23. First-pass InternVL
# ============================================================

def internvl_predict(
    row,
    images,
    extra_context=None,
):

    model.eval()


    messages = build_messages(

        row,

        images,

        extra_context=(
            extra_context
        ),
    )


    text = (
        processor
        .apply_chat_template(

            messages,

            tokenize=False,

            add_generation_prompt=True,
        )
    )


    inputs = processor(

        text=[text],

        images=images,

        return_tensors="pt",
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

        out = model(
            **inputs,
            use_cache=False,
        )


        logits = (
            out.logits[
                0,
                -1,
            ]
            .float()
        )


        choice_logits = (
            logits
            .index_select(
                0,
                CHOICE_TOKEN_TENSOR,
            )
        )


    top2 = torch.topk(
        choice_logits,
        2,
    )


    pred_index = int(
        top2.indices[0]
    )


    margin = float(
        top2.values[0]
        - top2.values[1]
    )


    return {
        "pred":
            CHOICES[
                pred_index
            ],

        "margin":
            margin,

        "logits":
            choice_logits
            .cpu(),
    }


# ============================================================
# 24. Rule Router
# ============================================================

OBJECT_KEYWORDS = [

    "몇 개",
    "몇개",
    "개수",
    "수량",

    "왼쪽",
    "오른쪽",
    "중앙",
    "가운데",

    "위",
    "아래",

    "어디",
    "위치",

    "보이는",
    "들어 있는",

    "병",
    "캔",
    "컵",
    "용기",
]


MATERIAL_KEYWORDS = [

    "재질",
    "소재",

    "플라스틱",
    "비닐",
    "유리",
    "금속",
    "종이",
    "캔",

    "plastic",
    "glass",
    "metal",
    "paper",
]


def route_tools(
    question,
    margin,
    threshold,
):

    q = str(
        question
    ).lower()


    object_match = any(
        keyword in q

        for keyword
        in OBJECT_KEYWORDS
    )


    material_match = any(
        keyword in q

        for keyword
        in MATERIAL_KEYWORDS
    )


    low_confidence = (
        margin
        <= threshold
    )


    tools = []


    if (
        USE_GENERAL_YOLO
        and
        (
            object_match
            or
            low_confidence
        )
    ):

        tools.append(
            "general_yolo"
        )


    if (
        USE_MATERIAL_YOLO
        and
        (
            material_match
            or
            low_confidence
        )
    ):

        tools.append(
            "material_yolo"
        )


    if (
        USE_CROP_TOOL
        and tools
    ):

        tools.append(
            "crop"
        )


    return tools


# ============================================================
# 25. Load YOLO tools
# ============================================================

general_yolo = None

material_yolo = None


if USE_GENERAL_YOLO:

    print(
        "\nLoading General YOLO..."
    )

    general_yolo = YOLO(
        GENERAL_YOLO_MODEL
    )


# ============================================================
# Material checkpoint automatic discovery
# ============================================================

def find_hf_yolo_checkpoint(
    repo_id,
):

    files = list_repo_files(
        repo_id
    )


    pt_files = [
        name
        for name in files
        if name.lower().endswith(
            ".pt"
        )
    ]


    if not pt_files:

        raise RuntimeError(
            f"No .pt YOLO checkpoint "
            f"found in {repo_id}"
        )


    # best.pt 우선
    preferred = [

        name

        for name
        in pt_files

        if Path(
            name
        ).name.lower()
        == "best.pt"
    ]


    filename = (
        preferred[0]
        if preferred
        else pt_files[0]
    )


    print(
        "Material YOLO checkpoint:",
        filename
    )


    return hf_hub_download(

        repo_id=repo_id,

        filename=filename,
    )


if USE_MATERIAL_YOLO:

    print(
        "\nLoading Material YOLO..."
    )


    material_checkpoint = (
        find_hf_yolo_checkpoint(
            MATERIAL_YOLO_REPO
        )
    )


    material_yolo = YOLO(
        material_checkpoint
    )


# ============================================================
# 26. YOLO result helpers
# ============================================================

def box_position(
    bbox,
    width,
    height,
):

    x1, y1, x2, y2 = bbox


    cx = (
        x1 + x2
    ) / 2


    cy = (
        y1 + y2
    ) / 2


    horizontal = (
        "left"
        if cx < width / 3

        else
        "right"
        if cx > width * 2 / 3

        else
        "center"
    )


    vertical = (
        "top"
        if cy < height / 3

        else
        "bottom"
        if cy > height * 2 / 3

        else
        "middle"
    )


    return (
        f"{horizontal}-{vertical}"
    )


def parse_yolo_results(
    result,
    image,
):

    output = []


    if (
        result is None
        or
        result.boxes is None
    ):

        return output


    width, height = (
        image.size
    )


    boxes = (
        result.boxes.xyxy
        .detach()
        .cpu()
        .tolist()
    )


    confs = (
        result.boxes.conf
        .detach()
        .cpu()
        .tolist()
    )


    classes = (
        result.boxes.cls
        .detach()
        .cpu()
        .tolist()
    )


    names = (
        result.names
    )


    for bbox, conf, cls in zip(
        boxes,
        confs,
        classes,
    ):

        class_id = int(
            cls
        )


        label = (
            names[
                class_id
            ]

            if isinstance(
                names,
                dict
            )

            else names[
                class_id
            ]
        )


        output.append(
            {
                "label":
                    str(label),

                "confidence":
                    float(conf),

                "bbox":
                    [
                        float(x)
                        for x in bbox
                    ],

                "position":
                    box_position(
                        bbox,
                        width,
                        height,
                    ),
            }
        )


    output.sort(
        key=lambda x:
            x["confidence"],

        reverse=True,
    )


    return output


# ============================================================
# 27. General YOLO Tool
# ============================================================

def run_general_yolo(
    image,
):

    if general_yolo is None:

        return []


    result = (
        general_yolo.predict(

            source=image,

            conf=(
                GENERAL_YOLO_CONF
            ),

            imgsz=(
                GENERAL_YOLO_IMGSZ
            ),

            verbose=False,

            device=0,
        )[0]
    )


    return parse_yolo_results(
        result,
        image,
    )


# ============================================================
# 28. Material YOLO Tool
# ============================================================

def run_material_yolo(
    image,
):

    if material_yolo is None:

        return []


    result = (
        material_yolo.predict(

            source=image,

            conf=(
                MATERIAL_YOLO_CONF
            ),

            imgsz=(
                MATERIAL_YOLO_IMGSZ
            ),

            verbose=False,

            device=0,
        )[0]
    )


    return parse_yolo_results(
        result,
        image,
    )


# ============================================================
# 29. Crop / Zoom Tool
# ============================================================

def crop_bbox(
    image,
    bbox,
):

    width, height = (
        image.size
    )


    x1, y1, x2, y2 = bbox


    box_w = (
        x2 - x1
    )


    box_h = (
        y2 - y1
    )


    pad_x = (
        box_w
        * CROP_PADDING_RATIO
    )


    pad_y = (
        box_h
        * CROP_PADDING_RATIO
    )


    x1 = max(
        0,
        int(
            x1 - pad_x
        ),
    )


    y1 = max(
        0,
        int(
            y1 - pad_y
        ),
    )


    x2 = min(
        width,
        int(
            x2 + pad_x
        ),
    )


    y2 = min(
        height,
        int(
            y2 + pad_y
        ),
    )


    if (
        x2 <= x1
        or
        y2 <= y1
    ):

        return None


    crop = image.crop(
        (
            x1,
            y1,
            x2,
            y2,
        )
    )


    # 작은 crop 확대
    min_side = min(
        crop.size
    )


    if min_side < 448:

        scale = (
            448
            /
            max(
                min_side,
                1
            )
        )


        new_size = (

            int(
                crop.width
                * scale
            ),

            int(
                crop.height
                * scale
            ),
        )


        crop = crop.resize(
            new_size,

            Image.Resampling.BICUBIC,
        )


    return crop


# ============================================================
# 30. Tool evidence formatter
# ============================================================

def format_detections(
    title,
    detections,
    limit=8,
):

    if not detections:

        return (
            f"{title}:\n"
            "- no reliable detections"
        )


    lines = [
        f"{title}:"
    ]


    for det in (
        detections[
            :limit
        ]
    ):

        lines.append(

            "- "
            f"{det['label']} "
            f"(confidence "
            f"{det['confidence']:.2f}, "
            f"position "
            f"{det['position']})"
        )


    return "\n".join(
        lines
    )


def build_tool_context(
    general_results,
    material_results,
):

    sections = []


    if general_results is not None:

        sections.append(

            format_detections(
                "General object detector",

                general_results,
            )
        )


    if material_results is not None:

        sections.append(

            format_detections(
                "Recycling material detector",

                material_results,
            )
        )


    return "\n\n".join(
        sections
    )


# ============================================================
# 31. Run Tool Layer
# ============================================================

def run_visual_tools(
    image,
    tools,
):

    general_results = None

    material_results = None


    if (
        "general_yolo"
        in tools
    ):

        general_results = (
            run_general_yolo(
                image
            )
        )


    if (
        "material_yolo"
        in tools
    ):

        material_results = (
            run_material_yolo(
                image
            )
        )


    crops = []


    if (
        "crop"
        in tools
    ):

        # Material detection 우선
        candidates = []


        if material_results:

            candidates.extend(
                material_results
            )


        if general_results:

            candidates.extend(
                general_results
            )


        # confidence 높은 순
        candidates.sort(

            key=lambda x:
                x[
                    "confidence"
                ],

            reverse=True,
        )


        for candidate in (
            candidates[
                :MAX_CROPS_PER_SAMPLE
            ]
        ):

            crop = crop_bbox(

                image,

                candidate[
                    "bbox"
                ],
            )


            if crop is not None:

                crops.append(
                    crop
                )


    context = (
        build_tool_context(

            general_results,

            material_results,
        )
    )


    return {
        "general":
            general_results,

        "material":
            material_results,

        "crops":
            crops,

        "context":
            context,
    }


# ============================================================
# 32. Agent prediction
# ============================================================

def agent_predict(
    row,
    threshold,
):

    image = load_rgb_image(
        row["path"]
    )


    # --------------------------------------------------------
    # Pass 1
    # --------------------------------------------------------

    first = internvl_predict(

        row,

        [image],
    )


    # --------------------------------------------------------
    # Router
    # --------------------------------------------------------

    tools = route_tools(

        row[
            "question"
        ],

        first[
            "margin"
        ],

        threshold,
    )


    # no tool
    if not tools:

        return {
            "first_pred":
                first["pred"],

            "final_pred":
                first["pred"],

            "margin":
                first["margin"],

            "tools":
                [],

            "context":
                "",

            "used_second_pass":
                False,
        }


    # --------------------------------------------------------
    # Tools
    # --------------------------------------------------------

    tool_output = (
        run_visual_tools(
            image,
            tools,
        )
    )


    # Original image always first.
    # Crops follow.
    images = (
        [image]
        +
        tool_output[
            "crops"
        ]
    )


    # --------------------------------------------------------
    # Pass 2
    # --------------------------------------------------------

    second = internvl_predict(

        row,

        images,

        extra_context=(
            tool_output[
                "context"
            ]
        ),
    )


    return {
        "first_pred":
            first["pred"],

        "final_pred":
            second["pred"],

        "margin":
            first["margin"],

        "tools":
            tools,

        "context":
            tool_output[
                "context"
            ],

        "used_second_pass":
            True,
    }


# ============================================================
# 33. Baseline validation
# ============================================================

print(
    "\n===== BASELINE VALIDATION ====="
)


baseline_rows = []


for idx in tqdm(
    range(
        len(valid_subset)
    ),
    desc="Baseline validation",
):

    row = (
        valid_subset.iloc[
            idx
        ]
    )


    image = load_rgb_image(
        row["path"]
    )


    pred = internvl_predict(

        row,

        [image],
    )


    gold = (
        str(
            row["answer"]
        )
        .strip()
        .lower()
    )


    baseline_rows.append(
        {
            "id":
                row["id"],

            "gold":
                gold,

            "pred":
                pred["pred"],

            "margin":
                pred["margin"],

            "correct":
                pred["pred"]
                == gold,
        }
    )


baseline_df = pd.DataFrame(
    baseline_rows
)


baseline_accuracy = (
    baseline_df[
        "correct"
    ]
    .mean()
)


print(
    "InternVL baseline accuracy:",
    baseline_accuracy
)


# ============================================================
# 34. Tool Validation
#
# Maximum threshold를 사용해 candidate들에 대한
# second pass를 미리 계산.
# ============================================================

max_threshold = max(
    ROUTER_THRESHOLD_CANDIDATES
)


tool_validation_cache = []


for idx in tqdm(

    range(
        len(valid_subset)
    ),

    desc="Tool validation",
):

    row = (
        valid_subset.iloc[
            idx
        ]
    )


    first_data = (
        baseline_df.iloc[
            idx
        ]
    )


    question = (
        row[
            "question"
        ]
    )


    tools = route_tools(

        question,

        float(
            first_data[
                "margin"
            ]
        ),

        max_threshold,
    )


    gold = (
        str(
            row[
                "answer"
            ]
        )
        .strip()
        .lower()
    )


    if not tools:

        tool_validation_cache.append(
            {
                "id":
                    row["id"],

                "gold":
                    gold,

                "first_pred":
                    first_data[
                        "pred"
                    ],

                "second_pred":
                    first_data[
                        "pred"
                    ],

                "margin":
                    float(
                        first_data[
                            "margin"
                        ]
                    ),

                "tools":
                    "",

                "context":
                    "",
            }
        )

        continue


    image = load_rgb_image(
        row["path"]
    )


    output = run_visual_tools(
        image,
        tools,
    )


    images = (
        [image]
        +
        output[
            "crops"
        ]
    )


    second = internvl_predict(

        row,

        images,

        extra_context=(
            output[
                "context"
            ]
        ),
    )


    tool_validation_cache.append(
        {
            "id":
                row["id"],

            "gold":
                gold,

            "first_pred":
                first_data[
                    "pred"
                ],

            "second_pred":
                second[
                    "pred"
                ],

            "margin":
                float(
                    first_data[
                        "margin"
                    ]
                ),

            "tools":
                ",".join(
                    tools
                ),

            "context":
                output[
                    "context"
                ],
        }
    )


tool_val_df = pd.DataFrame(
    tool_validation_cache
)


# ============================================================
# 35. Router threshold calibration
# ============================================================

best_threshold = (
    ROUTER_MARGIN_THRESHOLD
)


best_accuracy = (
    baseline_accuracy
)


if CALIBRATE_ROUTER_THRESHOLD:

    print(
        "\n===== ROUTER CALIBRATION ====="
    )


    for threshold in (
        ROUTER_THRESHOLD_CANDIDATES
    ):

        final_predictions = []


        for idx in range(
            len(
                tool_val_df
            )
        ):

            item = (
                tool_val_df.iloc[
                    idx
                ]
            )


            question = (
                valid_subset.iloc[
                    idx
                ][
                    "question"
                ]
            )


            tools = route_tools(

                question,

                float(
                    item[
                        "margin"
                    ]
                ),

                threshold,
            )


            if tools:

                final_predictions.append(
                    item[
                        "second_pred"
                    ]
                )

            else:

                final_predictions.append(
                    item[
                        "first_pred"
                    ]
                )


        accuracy = sum(

            pred == gold

            for pred, gold
            in zip(

                final_predictions,

                tool_val_df[
                    "gold"
                ],
            )

        ) / len(
            final_predictions
        )


        print(
            f"threshold={threshold:.2f} "
            f"accuracy={accuracy:.5f}"
        )


        if accuracy > best_accuracy:

            best_accuracy = (
                accuracy
            )

            best_threshold = (
                threshold
            )


print(
    "\nBaseline:",
    baseline_accuracy
)


print(
    "Best Tool Accuracy:",
    best_accuracy
)


print(
    "Best threshold:",
    best_threshold
)


# ============================================================
# 36. Final validation diagnostics
# ============================================================

final_val_rows = []


for idx in range(
    len(
        tool_val_df
    )
):

    item = (
        tool_val_df.iloc[
            idx
        ]
    )


    question = (
        valid_subset.iloc[
            idx
        ][
            "question"
        ]
    )


    tools = route_tools(

        question,

        float(
            item[
                "margin"
            ]
        ),

        best_threshold,
    )


    final_pred = (

        item[
            "second_pred"
        ]

        if tools

        else item[
            "first_pred"
        ]
    )


    final_val_rows.append(
        {
            **item.to_dict(),

            "final_pred":
                final_pred,

            "correct":
                final_pred
                == item[
                    "gold"
                ],
        }
    )


pd.DataFrame(
    final_val_rows
).to_csv(

    VALIDATION_PATH,

    index=False,
)


# ============================================================
# 37. TEST — baseline + tools
# ============================================================

baseline_test_preds = []

agent_test_preds = []

tool_logs = []


for idx in tqdm(

    range(
        len(test_df)
    ),

    desc="Test Agent",
):

    row = (
        test_df.iloc[
            idx
        ]
    )


    image = load_rgb_image(
        row["path"]
    )


    first = internvl_predict(

        row,

        [image],
    )


    baseline_test_preds.append(
        first[
            "pred"
        ]
    )


    tools = route_tools(

        row[
            "question"
        ],

        first[
            "margin"
        ],

        best_threshold,
    )


    if not tools:

        final_pred = (
            first[
                "pred"
            ]
        )


        context = ""

        used_second = False


    else:

        tool_output = (
            run_visual_tools(
                image,
                tools,
            )
        )


        images = (
            [image]
            +
            tool_output[
                "crops"
            ]
        )


        second = (
            internvl_predict(

                row,

                images,

                extra_context=(
                    tool_output[
                        "context"
                    ]
                ),
            )
        )


        final_pred = (
            second[
                "pred"
            ]
        )


        context = (
            tool_output[
                "context"
            ]
        )


        used_second = True


    agent_test_preds.append(
        final_pred
    )


    tool_logs.append(
        {
            "id":
                row["id"],

            "first_pred":
                first["pred"],

            "first_margin":
                first["margin"],

            "tools":
                ",".join(
                    tools
                ),

            "used_second_pass":
                used_second,

            "final_pred":
                final_pred,

            "context":
                context,
        }
    )


# ============================================================
# 38. Submissions
# ============================================================

SUBMISSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


baseline_submission = (
    pd.DataFrame(
        {
            "id":
                test_df[
                    "id"
                ],

            "answer":
                baseline_test_preds,
        }
    )
)


baseline_submission.to_csv(

    BASELINE_SUBMISSION_PATH,

    index=False,
)


agent_submission = (
    pd.DataFrame(
        {
            "id":
                test_df[
                    "id"
                ],

            "answer":
                agent_test_preds,
        }
    )
)


agent_submission.to_csv(

    SUBMISSION_PATH,

    index=False,
)


pd.DataFrame(
    tool_logs
).to_csv(

    TOOL_LOG_PATH,

    index=False,
)


# ============================================================
# 39. Diagnostics
# ============================================================

changed = sum(

    a != b

    for a, b
    in zip(

        baseline_test_preds,

        agent_test_preds,
    )
)


print(
    "\n"
    + "=" * 70
)


print(
    "SSAFY-AGENT RESULT"
)


print(
    "=" * 70
)


print(
    "InternVL baseline validation:",
    baseline_accuracy
)


print(
    "Tool validation:",
    best_accuracy
)


print(
    "Selected threshold:",
    best_threshold
)


print(
    "Test predictions changed:",
    changed
)


print(
    "Change ratio:",
    changed
    /
    len(test_df)
)


print(
    "\nBaseline submission:"
)

print(
    BASELINE_SUBMISSION_PATH
)


print(
    "\nAgent submission:"
)

print(
    SUBMISSION_PATH
)


print(
    "\nTool diagnostics:"
)

print(
    TOOL_LOG_PATH
)


print(
    "\nFinal distribution:"
)


print(
    pd.Series(
        agent_test_preds
    )
    .value_counts()
    .sort_index()
)


print(
    "=" * 70
)