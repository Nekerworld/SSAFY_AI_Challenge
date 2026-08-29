# ============================================================
# SSAFY-Agent v1
#
# InternVL3.5-4B-Flash
# + Cached Vision Features
# + QLoRA on Language Model Only
# + Rule Router
# + General YOLO
# + Recycling Material YOLO
# + Crop / Zoom
#
# ============================================================
#
# TRAIN PIPELINE
#
# image
#   ↓
# InternVL-Flash Vision Encoder      [ONCE]
#   ↓
# vision embedding
#   ↓
# disk cache (.pt)
#
# ------------------------------------------------------------
#
# cached vision embedding
#       +
# question / options
#       ↓
# Qwen3 inner language model
#       ↓
# LoRA
#
# Vision Encoder is NOT executed during training.
#
# ------------------------------------------------------------
#
# INFERENCE
#
# Cached image feature
#       ↓
# InternVL First Pass
#       ↓
# confidence + question type
#       ↓
# Rule Router
#       ↓
# General YOLO / Material YOLO
#       ↓
# optional crops
#       ↓
# Vision encoder ONLY for new crops
#       ↓
# InternVL Second Pass
#
# ============================================================


# ============================================================
# 0. Requirements
# ============================================================
#
# python -m pip install -U \
# transformers accelerate bitsandbytes peft \
# torch torchvision einops timm \
# ultralytics huggingface_hub \
# scikit-learn pandas pillow tqdm
#
# InternVL3.5 recommends transformers >= 4.52.1
#
# ============================================================


# ============================================================
# 1. Imports
# ============================================================

import copy
import json
import math
import random
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List, Dict

import pandas as pd
from PIL import Image

import torch
import torch.nn as nn

import torchvision.transforms as T

from torchvision.transforms.functional import (
    InterpolationMode,
)

from torch.utils.data import (
    Dataset,
    DataLoader,
)

from sklearn.model_selection import (
    train_test_split,
)

from transformers import (
    AutoModel,
    AutoTokenizer,
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
)

from tqdm.auto import tqdm


# ============================================================
# 2. Project
# ============================================================

try:
    PROJECT_DIR = (
        Path(__file__)
        .resolve()
        .parent
    )

except NameError:
    PROJECT_DIR = (
        Path.cwd()
        .resolve()
    )


# ============================================================
# 3. Model
# ============================================================

MODEL_ID = (
    "OpenGVLab/InternVL3_5-4B-Flash"
)

SEED = 42


# ============================================================
# 4. Vision
# ============================================================

IMAGE_SIZE = 448

# 핵심:
# dynamic high-resolution을 사용하지 않고
# 우선 정확히 1 tile만 사용한다.
#
# 이전 실행:
# batch=2인데 tiles=10
#
# 현재:
# batch=2 → tiles=2

TRAIN_MAX_TILES = 1

INFER_MAX_TILES = 1

CROP_MAX_TILES = 1


# ============================================================
# 5. Feature Cache
# ============================================================

FEATURE_CACHE_DIR = (
    PROJECT_DIR
    / "feature_cache"
    / "internvl3_5_4b_flash_tile1"
)

CACHE_DTYPE = torch.float16

REBUILD_FEATURE_CACHE = False


# ============================================================
# 6. Training
# ============================================================

NUM_EPOCHS = 1

LR = 1e-4

WEIGHT_DECAY = 0.01

WARMUP_RATIO = 0.03

MAX_GRAD_NORM = 1.0


# ============================================================
# 7. Batch
# ============================================================

# Vision encoder가 학습 loop에서 사라지므로
# 이전보다 batch를 키울 여지가 있음.
#
# 우선 보수적으로 시작.

TRAIN_BATCH_SIZE = 4

GRAD_ACCUM = 4

# effective batch = 16


# ============================================================
# 8. LoRA
# ============================================================

LORA_R = 16

LORA_ALPHA = 32

LORA_DROPOUT = 0.05


# ============================================================
# 9. YOLO tools
# ============================================================

USE_GENERAL_YOLO = True

USE_MATERIAL_YOLO = True

USE_CROP_TOOL = True


GENERAL_YOLO_MODEL = (
    "yolov8l.pt"
)

GENERAL_YOLO_CONF = 0.25

GENERAL_YOLO_IMGSZ = 640


MATERIAL_REPO = (
    "Jeremy341/MIRA-AI"
)

MATERIAL_FILENAME = (
    "mira_exp019.pt"
)

MATERIAL_YOLO_CONF = 0.25

MATERIAL_YOLO_IMGSZ = 640


# ============================================================
# 10. Router
# ============================================================

ROUTER_THRESHOLDS = [
    0.00,
    0.25,
    0.50,
    0.75,
    1.00,
    1.50,
    2.00,
]


# ============================================================
# 11. Crop
# ============================================================

CROP_PADDING_RATIO = 0.12

MAX_CROPS = 2


# ============================================================
# 12. Paths
# ============================================================

TRAIN_CSV = (
    PROJECT_DIR
    / "train.csv"
)

TEST_CSV = (
    PROJECT_DIR
    / "test.csv"
)


SAVE_DIR = (
    PROJECT_DIR
    / "model"
    / "internvl3_5_4b_flash_agent"
)


SUBMISSION_DIR = (
    PROJECT_DIR
    / "submission"
)


BASELINE_SUBMISSION_PATH = (
    SUBMISSION_DIR
    / "submission_internvl_flash_baseline.csv"
)


AGENT_SUBMISSION_PATH = (
    SUBMISSION_DIR
    / "submission_internvl_flash_agent.csv"
)


VALIDATION_PATH = (
    SUBMISSION_DIR
    / "internvl_flash_validation.csv"
)


TOOL_LOG_PATH = (
    SUBMISSION_DIR
    / "internvl_flash_tool_log.csv"
)


# ============================================================
# 13. CUDA / Seed
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
    ==
    torch.float16
)


print(
    "Project:",
    PROJECT_DIR
)

print(
    "Model:",
    MODEL_ID
)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "Compute dtype:",
    COMPUTE_DTYPE
)


# ============================================================
# 14. Choices
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


# ============================================================
# 15. Image helpers
# ============================================================

Image.MAX_IMAGE_PIXELS = None


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


# ============================================================
# 16. InternVL Image Transform
# ============================================================

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)


IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)


IMAGE_TRANSFORM = T.Compose(
    [
        T.Lambda(
            lambda image:
                image.convert("RGB")
                if image.mode != "RGB"
                else image
        ),

        T.Resize(
            (
                IMAGE_SIZE,
                IMAGE_SIZE,
            ),

            interpolation=(
                InterpolationMode.BICUBIC
            ),
        ),

        T.ToTensor(),

        T.Normalize(
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ]
)


# ============================================================
# 17. Dynamic preprocessing
#
# tile=1에서는 사실상:
#
# original → 448x448
#
# 이지만 나중에 tile 값을 다시 늘릴 수 있게
# InternVL 형태는 유지한다.
# ============================================================

def find_closest_aspect_ratio(
    aspect_ratio,
    target_ratios,
    width,
    height,
    image_size,
):

    best_diff = float(
        "inf"
    )

    best_ratio = (
        1,
        1,
    )


    area = (
        width
        *
        height
    )


    for ratio in (
        target_ratios
    ):

        target_aspect = (
            ratio[0]
            /
            ratio[1]
        )


        diff = abs(
            aspect_ratio
            -
            target_aspect
        )


        if diff < best_diff:

            best_diff = (
                diff
            )

            best_ratio = (
                ratio
            )


        elif diff == best_diff:

            if (
                area
                >
                0.5
                *
                image_size
                *
                image_size
                *
                ratio[0]
                *
                ratio[1]
            ):

                best_ratio = (
                    ratio
                )


    return best_ratio


def dynamic_preprocess(
    image,
    min_num=1,
    max_num=1,
    image_size=448,
    use_thumbnail=True,
):

    width, height = (
        image.size
    )


    aspect_ratio = (
        width
        /
        height
    )


    target_ratios = set()


    for n in range(
        min_num,
        max_num + 1,
    ):

        for i in range(
            1,
            n + 1,
        ):

            for j in range(
                1,
                n + 1,
            ):

                blocks = (
                    i * j
                )


                if (
                    min_num
                    <= blocks
                    <= max_num
                ):

                    target_ratios.add(
                        (
                            i,
                            j,
                        )
                    )


    target_ratios = sorted(

        target_ratios,

        key=lambda x:
            x[0]
            *
            x[1],
    )


    target_ratio = (
        find_closest_aspect_ratio(

            aspect_ratio,

            target_ratios,

            width,

            height,

            image_size,
        )
    )


    target_width = (
        image_size
        *
        target_ratio[0]
    )


    target_height = (
        image_size
        *
        target_ratio[1]
    )


    blocks = (
        target_ratio[0]
        *
        target_ratio[1]
    )


    resized = image.resize(
        (
            target_width,
            target_height,
        ),

        Image.Resampling.BICUBIC,
    )


    tiles_per_row = (
        target_width
        //
        image_size
    )


    output = []


    for index in range(
        blocks
    ):

        x = (
            index
            %
            tiles_per_row
        )


        y = (
            index
            //
            tiles_per_row
        )


        box = (
            x * image_size,
            y * image_size,
            (x + 1) * image_size,
            (y + 1) * image_size,
        )


        output.append(
            resized.crop(
                box
            )
        )


    if (
        use_thumbnail
        and
        len(output) != 1
    ):

        output.append(

            image.resize(
                (
                    image_size,
                    image_size,
                ),

                Image.Resampling.BICUBIC,
            )
        )


    return output


def preprocess_image(
    image,
    max_num,
):

    tiles = dynamic_preprocess(

        image,

        min_num=1,

        max_num=max_num,

        image_size=IMAGE_SIZE,

        use_thumbnail=True,
    )


    return torch.stack(
        [
            IMAGE_TRANSFORM(
                tile
            )

            for tile in tiles
        ]
    )


# ============================================================
# 18. Dataset load
# ============================================================

train_df = pd.read_csv(
    TRAIN_CSV
)


test_df = pd.read_csv(
    TEST_CSV
)


TRAIN_COLUMNS = {
    "id",
    "path",
    "question",
    "a",
    "b",
    "c",
    "d",
    "answer",
}


TEST_COLUMNS = {
    "id",
    "path",
    "question",
    "a",
    "b",
    "c",
    "d",
}


def check_columns(
    df,
    required,
    name,
):

    missing = (
        required
        -
        set(
            df.columns
        )
    )


    if missing:

        raise RuntimeError(
            f"{name} missing columns: "
            f"{sorted(missing)}"
        )


check_columns(
    train_df,
    TRAIN_COLUMNS,
    "train.csv",
)


check_columns(
    test_df,
    TEST_COLUMNS,
    "test.csv",
)


answers = (

    train_df[
        "answer"
    ]

    .astype(str)

    .str.strip()

    .str.lower()
)


if not set(
    answers.unique()
).issubset(
    VALID_CHOICES
):

    raise RuntimeError(
        "Invalid train labels."
    )


# ============================================================
# 19. Split
# ============================================================

train_subset, valid_subset = (
    train_test_split(

        train_df,

        test_size=0.10,

        random_state=SEED,

        stratify=(
            train_df[
                "answer"
            ]
        ),
    )
)


train_subset = (
    train_subset
    .reset_index(
        drop=True
    )
)


valid_subset = (
    valid_subset
    .reset_index(
        drop=True
    )
)


test_df = (
    test_df
    .reset_index(
        drop=True
    )
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
# 20. Tokenizer
#
# Native InternVL tokenizer.
# No AutoProcessor.
# ============================================================

tokenizer = (
    AutoTokenizer
    .from_pretrained(

        MODEL_ID,

        trust_remote_code=True,

        use_fast=False,
    )
)


tokenizer.padding_side = (
    "left"
)


if (
    tokenizer.pad_token_id
    is None
):

    tokenizer.pad_token = (
        tokenizer.eos_token
    )


# ============================================================
# 21. Load Flash model
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
    AutoModel
    .from_pretrained(

        MODEL_ID,

        trust_remote_code=True,

        low_cpu_mem_usage=True,

        quantization_config=(
            bnb_config
        ),

        device_map={
            "": 0
        },

        use_flash_attn=False,
    )
)


# ============================================================
# 22. Structure checks
# ============================================================

for attr in [
    "vision_model",
    "language_model",
    "mlp1",
    "conv_template",
]:

    if not hasattr(
        model,
        attr,
    ):

        raise RuntimeError(
            f"Unexpected InternVL Flash structure: "
            f"missing {attr}"
        )


if not hasattr(
    model,
    "extract_feature",
):

    raise RuntimeError(
        "This InternVL Flash checkpoint "
        "does not expose extract_feature()."
    )


print(
    "\n===== INTERNVL FLASH ====="
)

print(
    "Outer:",
    type(
        model
    ).__name__
)

print(
    "Vision:",
    type(
        model.vision_model
    ).__name__
)

print(
    "Language:",
    type(
        model.language_model
    ).__name__
)

print(
    "extract_feature(): OK"
)


# ============================================================
# 23. Special tokens
# ============================================================

IMG_START_TOKEN = (
    "<img>"
)

IMG_END_TOKEN = (
    "</img>"
)

IMG_CONTEXT_TOKEN = (
    "<IMG_CONTEXT>"
)


IMG_CONTEXT_ID = (
    tokenizer.convert_tokens_to_ids(
        IMG_CONTEXT_TOKEN
    )
)


if (
    IMG_CONTEXT_ID is None

    or

    IMG_CONTEXT_ID
    ==
    tokenizer.unk_token_id
):

    raise RuntimeError(
        "IMG_CONTEXT token missing."
    )


print(
    "IMG_CONTEXT:",
    IMG_CONTEXT_ID
)


# ============================================================
# 24. Freeze vision components
# ============================================================

for parameter in (
    model.vision_model.parameters()
):

    parameter.requires_grad = (
        False
    )


for parameter in (
    model.mlp1.parameters()
):

    parameter.requires_grad = (
        False
    )


# Flash contains additional routing/gating components.
# These are part of the frozen visual pathway.

for name, parameter in (
    model.named_parameters()
):

    lower = (
        name.lower()
    )


    if (
        "gating"
        in lower

        or

        "router"
        in lower
    ):

        parameter.requires_grad = (
            False
        )


model.vision_model.eval()

model.mlp1.eval()


# ============================================================
# 25. Flash feature extraction helper
# ============================================================

def normalize_extracted_feature(
    output,
):

    # Standard InternVL:
    # Tensor [tiles, tokens, hidden]

    if torch.is_tensor(
        output
    ):

        tensor = (
            output
        )


    elif isinstance(
        output,
        (tuple, list),
    ):

        tensor = None


        for item in output:

            if (
                torch.is_tensor(
                    item
                )

                and

                item.ndim >= 2
            ):

                tensor = (
                    item
                )

                break


        if tensor is None:

            raise RuntimeError(
                "extract_feature() returned no tensor."
            )


    elif isinstance(
        output,
        dict,
    ):

        tensor = None


        for key in [
            "vit_embeds",
            "image_embeds",
            "features",
            "last_hidden_state",
        ]:

            value = output.get(
                key
            )


            if torch.is_tensor(
                value
            ):

                tensor = (
                    value
                )

                break


        if tensor is None:

            raise RuntimeError(
                "Cannot locate visual feature tensor."
            )


    else:

        raise RuntimeError(
            "Unknown extract_feature() output type: "
            f"{type(output)}"
        )


    # flatten:
    #
    # [tiles, tokens, hidden]
    #      →
    # [visual_tokens, hidden]

    if tensor.ndim == 3:

        tensor = tensor.reshape(
            -1,
            tensor.shape[-1],
        )


    elif tensor.ndim != 2:

        raise RuntimeError(
            "Unexpected visual feature shape: "
            f"{tuple(tensor.shape)}"
        )


    return tensor


def extract_visual_feature(
    image,
    max_tiles=1,
):

    pixel_values = (
        preprocess_image(

            image,

            max_tiles,
        )
    )


    pixel_values = (
        pixel_values
        .to(
            DEVICE,

            dtype=(
                COMPUTE_DTYPE
            ),
        )
    )


    model.vision_model.eval()

    model.mlp1.eval()


    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda",
            dtype=COMPUTE_DTYPE,
        ),
    ):

        feature_output = (
            model.extract_feature(
                pixel_values
            )
        )


    features = (
        normalize_extracted_feature(
            feature_output
        )
    )


    # Store cache in fp16 to save disk space.

    features = (

        features

        .detach()

        .to(
            "cpu",
            dtype=(
                CACHE_DTYPE
            ),
        )

        .contiguous()
    )


    if (
        features.numel()
        == 0
    ):

        raise RuntimeError(
            "Empty vision feature."
        )


    return features


# ============================================================
# 26. Cache helpers
# ============================================================

FEATURE_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def cache_file(
    split_name,
    sample_id,
):

    split_dir = (
        FEATURE_CACHE_DIR
        / split_name
    )


    split_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    safe_id = (
        str(
            sample_id
        )
        .replace(
            "/",
            "_",
        )
        .replace(
            "\\",
            "_",
        )
    )


    return (
        split_dir
        /
        f"{safe_id}.pt"
    )


def cache_dataframe(
    df,
    split_name,
):

    print(
        f"\n===== FEATURE CACHE: "
        f"{split_name} ====="
    )


    for index in tqdm(

        range(
            len(
                df
            )
        ),

        desc=(
            f"Vision cache "
            f"{split_name}"
        ),
    ):

        row = (
            df.iloc[
                index
            ]
        )


        path = cache_file(

            split_name,

            row[
                "id"
            ],
        )


        if (
            path.exists()

            and

            not REBUILD_FEATURE_CACHE
        ):

            continue


        image = load_rgb_image(
            row[
                "path"
            ]
        )


        features = (
            extract_visual_feature(

                image,

                max_tiles=(
                    TRAIN_MAX_TILES

                    if split_name
                    == "train"

                    else
                    INFER_MAX_TILES
                ),
            )
        )


        torch.save(
            {
                "features":
                    features,

                "model_id":
                    MODEL_ID,

                "image_size":
                    IMAGE_SIZE,

                "max_tiles":
                    (
                        TRAIN_MAX_TILES
                        if split_name == "train"
                        else INFER_MAX_TILES
                    ),
            },

            path,
        )


def load_cached_feature(
    split_name,
    sample_id,
):

    path = cache_file(

        split_name,

        sample_id,
    )


    if not path.exists():

        raise FileNotFoundError(
            f"Feature cache missing: "
            f"{path}"
        )


    data = torch.load(

        path,

        map_location="cpu",

        weights_only=False,
    )


    if (
        data.get(
            "model_id"
        )
        !=
        MODEL_ID
    ):

        raise RuntimeError(
            "Feature cache model mismatch."
        )


    return (
        data[
            "features"
        ]
    )


# ============================================================
# 27. Build all fixed visual caches
#
# IMPORTANT:
#
# This is the only time the vision encoder sees
# train/valid/test original images.
#
# ============================================================

cache_dataframe(
    train_subset,
    "train",
)


cache_dataframe(
    valid_subset,
    "valid",
)


cache_dataframe(
    test_df,
    "test",
)


# ============================================================
# 28. Feature sanity
# ============================================================

sample_feature = (
    load_cached_feature(

        "train",

        train_subset.iloc[0][
            "id"
        ],
    )
)


print(
    "\n===== FEATURE CHECK ====="
)

print(
    "Feature shape:",
    tuple(
        sample_feature.shape
    )
)

print(
    "dtype:",
    sample_feature.dtype
)


# ============================================================
# 29. Freeze everything except language LoRA
# ============================================================

for parameter in (
    model.parameters()
):

    parameter.requires_grad = (
        False
    )


# ============================================================
# 30. Prepare inner LLM for QLoRA
# ============================================================

model.language_model = (
    prepare_model_for_kbit_training(

        model.language_model,

        use_gradient_checkpointing=True,
    )
)


model.language_model.config.use_cache = (
    False
)


# ============================================================
# 31. Language LoRA
# ============================================================

LLM_TARGET_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
    "mlp.up_proj",
]


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
            LLM_TARGET_MODULES
        ),

        task_type="CAUSAL_LM",
    )
)


model.language_model = (
    get_peft_model(

        model.language_model,

        lora_config,
    )
)


if hasattr(
    model.language_model,
    "enable_input_require_grads",
):

    model.language_model.enable_input_require_grads()


model.language_model.print_trainable_parameters()


# ============================================================
# 32. Verify visual feature hidden size
# ============================================================

embedding_layer = (
    model.language_model
    .get_input_embeddings()
)


embedding_dim = (
    embedding_layer
    .weight
    .shape[-1]
)


if (
    sample_feature.shape[-1]
    !=
    embedding_dim
):

    raise RuntimeError(

        "Vision/LLM hidden size mismatch: "
        f"vision={sample_feature.shape[-1]}, "
        f"llm={embedding_dim}"
    )


print(
    "LLM hidden:",
    embedding_dim
)


# ============================================================
# 33. Prompt
# ============================================================

SYSTEM_INSTRUCTION = (
    "You are a visual multiple-choice "
    "question answering assistant. "
    "Inspect the image carefully and choose "
    "the best answer. "
    "Output exactly one lowercase letter: "
    "a, b, c, or d. "
    "Do not explain."
)


def build_mc_text(
    row,
):

    return (
        f"{row['question']}\n\n"
        f"(a) {row['a']}\n"
        f"(b) {row['b']}\n"
        f"(c) {row['c']}\n"
        f"(d) {row['d']}\n\n"
        "Answer with exactly one lowercase letter: "
        "a, b, c, or d."
    )


# ============================================================
# 34. InternVL native conversation
# ============================================================

def visual_placeholder(
    feature_count,
):

    return (

        IMG_START_TOKEN

        +

        IMG_CONTEXT_TOKEN
        *
        int(
            feature_count
        )

        +

        IMG_END_TOKEN
    )


def replace_image_placeholders(
    query,
    feature_counts,
):

    for count in (
        feature_counts
    ):

        if "<image>" not in query:

            raise RuntimeError(
                "Image placeholder mismatch."
            )


        query = query.replace(

            "<image>",

            visual_placeholder(
                count
            ),

            1,
        )


    if "<image>" in query:

        raise RuntimeError(
            "Unused <image> placeholder."
        )


    return query


def make_query(
    user_text,
    feature_counts,
    answer=None,
):

    template = copy.deepcopy(
        model.conv_template
    )


    template.system_message = (
        SYSTEM_INSTRUCTION
    )


    template.append_message(

        template.roles[0],

        user_text,
    )


    template.append_message(

        template.roles[1],

        answer,
    )


    query = (
        template.get_prompt()
    )


    return replace_image_placeholders(

        query,

        feature_counts,
    )


def build_first_user_text(
    row,
):

    return (
        "<image>\n"
        +
        build_mc_text(
            row
        )
    )


def build_tool_user_text(
    row,
    num_images,
    context,
):

    lines = [
        "Original image: <image>"
    ]


    for index in range(
        1,
        num_images,
    ):

        lines.append(
            f"Detected crop {index}: <image>"
        )


    return (

        "\n".join(
            lines
        )

        +

        "\n\nExternal visual-tool observations:\n"

        +

        context

        +

        "\n\nThese observations may contain errors. "
        "Treat them only as supporting evidence. "
        "Use the original image as the primary evidence.\n\n"

        +

        build_mc_text(
            row
        )
    )


# ============================================================
# 35. Training Dataset
# ============================================================

class CachedInternDataset(
    Dataset
):

    def __init__(
        self,
        df,
        split_name,
    ):

        self.df = (
            df.reset_index(
                drop=True
            )
        )


        self.split_name = (
            split_name
        )


    def __len__(
        self,
    ):

        return len(
            self.df
        )


    def __getitem__(
        self,
        index,
    ):

        row = (
            self.df.iloc[
                index
            ]
        )


        visual_feature = (
            load_cached_feature(

                self.split_name,

                row[
                    "id"
                ],
            )
        )


        feature_count = int(
            visual_feature.shape[0]
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


        user_text = (
            build_first_user_text(
                row
            )
        )


        prompt_query = (
            make_query(

                user_text,

                [
                    feature_count
                ],

                answer=None,
            )
        )


        full_query = (
            make_query(

                user_text,

                [
                    feature_count
                ],

                answer=gold,
            )
        )


        return {
            "features":
                visual_feature,

            "prompt":
                prompt_query,

            "full":
                full_query,
        }


# ============================================================
# 36. Training Collator
# ============================================================

@dataclass
class CachedCollator:

    tokenizer: Any


    def __call__(
        self,
        batch,
    ):

        features = [

            sample[
                "features"
            ]

            for sample
            in batch
        ]


        prompts = [

            sample[
                "prompt"
            ]

            for sample
            in batch
        ]


        fulls = [

            sample[
                "full"
            ]

            for sample
            in batch
        ]


        prompt_enc = (
            self.tokenizer(

                prompts,

                padding=True,

                return_tensors="pt",

                add_special_tokens=True,
            )
        )


        full_enc = (
            self.tokenizer(

                fulls,

                padding=True,

                return_tensors="pt",

                add_special_tokens=True,
            )
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

            prompt_pos = (

                prompt_enc[
                    "attention_mask"
                ][i]

                .nonzero(
                    as_tuple=False
                )

                .squeeze(-1)
            )


            full_pos = (

                full_enc[
                    "attention_mask"
                ][i]

                .nonzero(
                    as_tuple=False
                )

                .squeeze(-1)
            )


            prompt_ids = (
                prompt_enc[
                    "input_ids"
                ][
                    i,
                    prompt_pos
                ]
            )


            full_ids = (
                full_enc[
                    "input_ids"
                ][
                    i,
                    full_pos
                ]
            )


            prompt_len = int(
                prompt_ids.numel()
            )


            if not torch.equal(

                full_ids[
                    :prompt_len
                ],

                prompt_ids,
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


        # ----------------------------------------------------
        # Visual-token sanity
        # ----------------------------------------------------

        expected = sum(

            int(
                feature.shape[0]
            )

            for feature
            in features
        )


        actual = int(

            (
                full_enc[
                    "input_ids"
                ]
                ==
                IMG_CONTEXT_ID
            )

            .sum()

            .item()
        )


        if (
            expected
            != actual
        ):

            raise RuntimeError(

                "Visual token mismatch: "
                f"features={expected}, "
                f"context_tokens={actual}"
            )


        return {
            "features":
                features,

            "input_ids":
                full_enc[
                    "input_ids"
                ],

            "attention_mask":
                full_enc[
                    "attention_mask"
                ],

            "labels":
                labels,
        }


# ============================================================
# 37. Loader
# ============================================================

train_loader = DataLoader(

    CachedInternDataset(
        train_subset,
        "train",
    ),

    batch_size=(
        TRAIN_BATCH_SIZE
    ),

    shuffle=True,

    collate_fn=(
        CachedCollator(
            tokenizer
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
# 38. Inject cached visual features
# ============================================================

def inject_visual_features(
    input_ids,
    visual_features,
):

    input_embeds = (

        model.language_model
        .get_input_embeddings()
        (
            input_ids
        )
    )


    input_embeds = (
        input_embeds.clone()
    )


    visual_mask = (
        input_ids
        ==
        IMG_CONTEXT_ID
    )


    flat_features = torch.cat(

        [
            feature.to(
                DEVICE,

                dtype=(
                    input_embeds.dtype
                ),
            )

            for feature
            in visual_features
        ],

        dim=0,
    )


    selected_count = int(

        visual_mask.sum().item()
    )


    if (
        selected_count
        !=
        flat_features.shape[0]
    ):

        raise RuntimeError(

            "Feature injection mismatch: "
            f"tokens={selected_count}, "
            f"features={flat_features.shape[0]}"
        )


    input_embeds[
        visual_mask
    ] = (
        flat_features
    )


    return input_embeds


# ============================================================
# 39. Choice token discovery
# ============================================================

def discover_choice_tokens():

    user_text = (
        "<image>\n"
        "Choose the correct answer.\n\n"
        "(a) one\n"
        "(b) two\n"
        "(c) three\n"
        "(d) four\n\n"
        "Answer with exactly one lowercase letter: "
        "a, b, c, or d."
    )


    prompt = make_query(

        user_text,

        [1],

        answer=None,
    )


    prompt_ids = tokenizer(

        prompt,

        add_special_tokens=True,
    )[
        "input_ids"
    ]


    result = {}


    for choice in (
        CHOICES
    ):

        full = make_query(

            user_text,

            [1],

            answer=choice,
        )


        full_ids = tokenizer(

            full,

            add_special_tokens=True,
        )[
            "input_ids"
        ]


        if (
            full_ids[
                :len(prompt_ids)
            ]
            != prompt_ids
        ):

            raise RuntimeError(
                "Choice prefix mismatch."
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
        ] = (
            suffix[0]
        )


        print(
            f"{choice}: "
            f"{suffix[0]} "
            f"{tokenizer.decode([suffix[0]])!r}"
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
            "Choice tokens not unique."
        )


    return result


CHOICE_TOKEN_IDS = (
    discover_choice_tokens()
)


CHOICE_TOKEN_TENSOR = (
    torch.tensor(

        [
            CHOICE_TOKEN_IDS[
                c
            ]

            for c
            in CHOICES
        ],

        dtype=torch.long,

        device=DEVICE,
    )
)


# ============================================================
# 40. Training preflight
# ============================================================

debug_batch = next(
    iter(
        train_loader
    )
)


debug_target = (

    debug_batch[
        "labels"
    ][0]

    [
        debug_batch[
            "labels"
        ][0]
        != -100
    ]
)


print(
    "\n===== TRAIN PREFLIGHT ====="
)

print(
    "Target:",
    repr(
        tokenizer.decode(
            debug_target
        )
    )
)

print(
    "Cached feature counts:",
    [
        tuple(
            feature.shape
        )

        for feature
        in debug_batch[
            "features"
        ]
    ]
)


# ============================================================
# 41. Optimizer
# ============================================================

trainable_parameters = [

    parameter

    for parameter
    in model.parameters()

    if parameter.requires_grad
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


scaler = torch.amp.GradScaler(

    "cuda",

    enabled=(
        USE_SCALER
    ),
)


# ============================================================
# 42. Training
#
# IMPORTANT:
#
# model.extract_feature()
# model.vision_model()
#
# ARE NEVER CALLED HERE.
# ============================================================

optimizer.zero_grad(
    set_to_none=True
)


for epoch in range(
    NUM_EPOCHS
):

    model.language_model.train()

    running_loss = 0.0

    batch_count = 0

    num_batches = len(
        train_loader
    )


    progress = tqdm(

        train_loader,

        desc=(
            f"Flash cached "
            f"{epoch + 1}/"
            f"{NUM_EPOCHS}"
        ),

        unit="batch",
    )


    for step, batch in enumerate(
        progress,
        1,
    ):

        input_ids = (
            batch[
                "input_ids"
            ]
            .to(
                DEVICE
            )
        )


        attention_mask = (
            batch[
                "attention_mask"
            ]
            .to(
                DEVICE
            )
        )


        labels = (
            batch[
                "labels"
            ]
            .to(
                DEVICE
            )
        )


        input_embeds = (
            inject_visual_features(

                input_ids,

                batch[
                    "features"
                ],
            )
        )


        group_start = (

            (
                (step - 1)
                //
                GRAD_ACCUM
            )

            *
            GRAD_ACCUM

            + 1
        )


        accum_divisor = min(

            GRAD_ACCUM,

            num_batches
            -
            group_start
            + 1,
        )


        with torch.autocast(

            device_type="cuda",

            dtype=(
                COMPUTE_DTYPE
            ),
        ):

            outputs = (
                model.language_model(

                    inputs_embeds=(
                        input_embeds
                    ),

                    attention_mask=(
                        attention_mask
                    ),

                    labels=(
                        labels
                    ),

                    use_cache=False,
                )
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
                f"Non-finite loss "
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
            %
            GRAD_ACCUM
            == 0

            or

            step
            ==
            num_batches
        )


        if should_step:

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


        avg_loss = (
            running_loss
            /
            batch_count
        )


        progress.set_postfix(

            loss=(
                f"{avg_loss:.4f}"
            ),

            lr=(
                f"{scheduler.get_last_lr()[0]:.2e}"
            ),
        )


# ============================================================
# 43. Save LoRA
# ============================================================

SAVE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


model.language_model.save_pretrained(

    SAVE_DIR
    / "language_lora"
)


tokenizer.save_pretrained(
    SAVE_DIR
)


# ============================================================
# 44. Cached prediction
# ============================================================

def predict_from_features(
    row,
    visual_features,
    tool_context=None,
):

    model.language_model.eval()


    feature_counts = [

        int(
            feature.shape[0]
        )

        for feature
        in visual_features
    ]


    if tool_context is None:

        user_text = (
            build_first_user_text(
                row
            )
        )

    else:

        user_text = (
            build_tool_user_text(

                row,

                len(
                    visual_features
                ),

                tool_context,
            )
        )


    query = make_query(

        user_text,

        feature_counts,

        answer=None,
    )


    encoded = tokenizer(

        query,

        return_tensors="pt",

        add_special_tokens=True,
    )


    input_ids = (
        encoded[
            "input_ids"
        ]
        .to(
            DEVICE
        )
    )


    attention_mask = (
        encoded[
            "attention_mask"
        ]
        .to(
            DEVICE
        )
    )


    input_embeds = (
        inject_visual_features(

            input_ids,

            visual_features,
        )
    )


    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda",
            dtype=COMPUTE_DTYPE,
        ),
    ):

        outputs = (
            model.language_model(

                inputs_embeds=(
                    input_embeds
                ),

                attention_mask=(
                    attention_mask
                ),

                use_cache=False,
            )
        )


        next_logits = (
            outputs.logits[
                0,
                -1,
            ]
            .float()
        )


        choice_logits = (
            next_logits
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
        .item()
    )


    margin = float(

        (
            top2.values[0]
            -
            top2.values[1]
        )
        .item()
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
# 45. Router
# ============================================================

OBJECT_KEYWORDS = [
    "몇 개",
    "몇개",
    "개수",
    "수량",
    "왼쪽",
    "오른쪽",
    "좌측",
    "우측",
    "중앙",
    "가운데",
    "위치",
    "어디",
]


MATERIAL_KEYWORDS = [
    "재질",
    "소재",
    "플라스틱",
    "유리",
    "금속",
    "종이",
    "비닐",
    "plastic",
    "glass",
    "metal",
    "paper",
]


def route_tools(
    question,
    margin,
    threshold,
    enabled=True,
):

    if not enabled:

        return []


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
            "general"
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
            "material"
        )


    if (
        USE_CROP_TOOL

        and

        tools
    ):

        tools.append(
            "crop"
        )


    return tools


# ============================================================
# 46. Load YOLO models
# ============================================================

print(
    "\n===== LOAD YOLO TOOLS ====="
)


general_yolo = None

material_yolo = None


if USE_GENERAL_YOLO:

    general_yolo = YOLO(
        GENERAL_YOLO_MODEL
    )


if USE_MATERIAL_YOLO:

    material_path = (
        hf_hub_download(

            repo_id=(
                MATERIAL_REPO
            ),

            filename=(
                MATERIAL_FILENAME
            ),
        )
    )


    material_yolo = YOLO(
        material_path
    )


# ============================================================
# 47. Detection helpers
# ============================================================

def get_position(
    bbox,
    width,
    height,
):

    x1, y1, x2, y2 = (
        bbox
    )


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


def parse_yolo(
    result,
    image,
):

    output = []


    if (
        result is None
        or
        result.boxes is None
        or
        len(
            result.boxes
        )
        == 0
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


    for (
        bbox,
        confidence,
        cls,
    ) in zip(
        boxes,
        confs,
        classes,
    ):

        class_id = int(
            cls
        )


        output.append(
            {
                "label":
                    str(
                        names[
                            class_id
                        ]
                    ),

                "confidence":
                    float(
                        confidence
                    ),

                "bbox":
                    [
                        float(x)
                        for x
                        in bbox
                    ],

                "position":
                    get_position(

                        bbox,

                        width,

                        height,
                    ),
            }
        )


    output.sort(

        key=lambda item:
            item[
                "confidence"
            ],

        reverse=True,
    )


    return output


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


    return parse_yolo(
        result,
        image,
    )


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


    return parse_yolo(
        result,
        image,
    )


# ============================================================
# 48. Crop
# ============================================================

def crop_bbox(
    image,
    bbox,
):

    width, height = (
        image.size
    )


    x1, y1, x2, y2 = (
        bbox
    )


    w = (
        x2 - x1
    )


    h = (
        y2 - y1
    )


    pad_x = (
        w
        *
        CROP_PADDING_RATIO
    )


    pad_y = (
        h
        *
        CROP_PADDING_RATIO
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


    if min(
        crop.size
    ) < IMAGE_SIZE:

        scale = (
            IMAGE_SIZE
            /
            max(
                min(
                    crop.size
                ),
                1,
            )
        )


        crop = crop.resize(

            (
                int(
                    crop.width
                    * scale
                ),

                int(
                    crop.height
                    * scale
                ),
            ),

            Image.Resampling.BICUBIC,
        )


    return crop


# ============================================================
# 49. Tool context
# ============================================================

def format_detections(
    title,
    detections,
):

    lines = [
        f"{title}:"
    ]


    if not detections:

        lines.append(
            "- no reliable detection"
        )


        return "\n".join(
            lines
        )


    for item in (
        detections[
            :8
        ]
    ):

        lines.append(

            "- "
            f"{item['label']} "
            f"(confidence={item['confidence']:.2f}, "
            f"position={item['position']})"
        )


    return "\n".join(
        lines
    )


def run_tools(
    image,
    tools,
):

    general = []

    material = []


    if (
        "general"
        in tools
    ):

        general = (
            run_general_yolo(
                image
            )
        )


    if (
        "material"
        in tools
    ):

        material = (
            run_material_yolo(
                image
            )
        )


    sections = []


    if "general" in tools:

        sections.append(

            format_detections(

                "General detector",

                general,
            )
        )


    if "material" in tools:

        sections.append(

            format_detections(

                "Material detector",

                material,
            )
        )


    context = "\n\n".join(
        sections
    )


    crops = []


    if "crop" in tools:

        candidates = (
            material
            +
            general
        )


        candidates.sort(

            key=lambda x:
                x[
                    "confidence"
                ],

            reverse=True,
        )


        centers = []


        for candidate in (
            candidates
        ):

            bbox = (
                candidate[
                    "bbox"
                ]
            )


            center = (
                (
                    bbox[0]
                    +
                    bbox[2]
                )
                / 2,

                (
                    bbox[1]
                    +
                    bbox[3]
                )
                / 2,
            )


            duplicate = any(

                (
                    (
                        center[0]
                        -
                        old[0]
                    )
                    ** 2

                    +

                    (
                        center[1]
                        -
                        old[1]
                    )
                    ** 2
                )
                ** 0.5
                < 30

                for old
                in centers
            )


            if duplicate:

                continue


            crop = crop_bbox(

                image,

                bbox,
            )


            if crop is not None:

                crops.append(
                    crop
                )


                centers.append(
                    center
                )


            if (
                len(crops)
                >= MAX_CROPS
            ):

                break


    return {
        "general":
            general,

        "material":
            material,

        "crops":
            crops,

        "context":
            context,

        "evidence":
            bool(
                general
                or material
                or crops
            ),
    }


# ============================================================
# 50. Crop feature extraction
#
# Only tool-routed samples need this.
# ============================================================

def extract_crop_features(
    crops,
):

    features = []


    for crop in crops:

        features.append(

            extract_visual_feature(

                crop,

                max_tiles=(
                    CROP_MAX_TILES
                ),
            )
        )


    return features


# ============================================================
# 51. Baseline Validation
# ============================================================

print(
    "\n===== BASELINE VALIDATION ====="
)


baseline_rows = []


for index in tqdm(

    range(
        len(
            valid_subset
        )
    ),

    desc="Flash baseline",
):

    row = (
        valid_subset.iloc[
            index
        ]
    )


    feature = (
        load_cached_feature(

            "valid",

            row[
                "id"
            ],
        )
    )


    result = (
        predict_from_features(

            row,

            [feature],
        )
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


    baseline_rows.append(
        {
            "id":
                row[
                    "id"
                ],

            "gold":
                gold,

            "pred":
                result[
                    "pred"
                ],

            "margin":
                result[
                    "margin"
                ],

            "correct":
                result[
                    "pred"
                ]
                ==
                gold,
        }
    )


baseline_df = pd.DataFrame(
    baseline_rows
)


baseline_accuracy = float(

    baseline_df[
        "correct"
    ]
    .mean()
)


print(
    "Baseline accuracy:",
    baseline_accuracy
)


# ============================================================
# 52. Tool Validation Cache
# ============================================================

MAX_THRESHOLD = max(
    ROUTER_THRESHOLDS
)


tool_rows = []


for index in tqdm(

    range(
        len(
            valid_subset
        )
    ),

    desc="YOLO validation",
):

    row = (
        valid_subset.iloc[
            index
        ]
    )


    base = (
        baseline_df.iloc[
            index
        ]
    )


    tools = route_tools(

        row[
            "question"
        ],

        float(
            base[
                "margin"
            ]
        ),

        MAX_THRESHOLD,

        True,
    )


    if not tools:

        tool_rows.append(
            {
                "second_pred":
                    base[
                        "pred"
                    ],

                "tools":
                    "",

                "evidence":
                    False,

                "context":
                    "",
            }
        )


        continue


    image = load_rgb_image(
        row[
            "path"
        ]
    )


    tool_output = run_tools(

        image,

        tools,
    )


    if not tool_output[
        "evidence"
    ]:

        tool_rows.append(
            {
                "second_pred":
                    base[
                        "pred"
                    ],

                "tools":
                    ",".join(
                        tools
                    ),

                "evidence":
                    False,

                "context":
                    tool_output[
                        "context"
                    ],
            }
        )


        continue


    original_feature = (
        load_cached_feature(

            "valid",

            row[
                "id"
            ],
        )
    )


    crop_features = (
        extract_crop_features(

            tool_output[
                "crops"
            ]
        )
    )


    all_features = (
        [original_feature]
        +
        crop_features
    )


    second = predict_from_features(

        row,

        all_features,

        tool_context=(
            tool_output[
                "context"
            ]
        ),
    )


    tool_rows.append(
        {
            "second_pred":
                second[
                    "pred"
                ],

            "tools":
                ",".join(
                    tools
                ),

            "evidence":
                True,

            "context":
                tool_output[
                    "context"
                ],
        }
    )


tool_df = pd.DataFrame(
    tool_rows
)


# ============================================================
# 53. Router Calibration
# ============================================================

best_accuracy = (
    baseline_accuracy
)


best_threshold = None

agent_enabled = False


print(
    "\n===== ROUTER ====="
)


print(
    f"OFF = "
    f"{baseline_accuracy:.5f}"
)


for threshold in (
    ROUTER_THRESHOLDS
):

    predictions = []


    for index in range(
        len(
            valid_subset
        )
    ):

        row = (
            valid_subset.iloc[
                index
            ]
        )


        base = (
            baseline_df.iloc[
                index
            ]
        )


        cache = (
            tool_df.iloc[
                index
            ]
        )


        tools = route_tools(

            row[
                "question"
            ],

            float(
                base[
                    "margin"
                ]
            ),

            threshold,

            True,
        )


        use_tool = (

            bool(
                tools
            )

            and

            bool(
                cache[
                    "evidence"
                ]
            )
        )


        predictions.append(

            cache[
                "second_pred"
            ]

            if use_tool

            else base[
                "pred"
            ]
        )


    accuracy = sum(

        p == g

        for p, g
        in zip(

            predictions,

            baseline_df[
                "gold"
            ],
        )

    ) / len(
        predictions
    )


    print(
        f"{threshold:.2f}: "
        f"{accuracy:.5f}"
    )


    if (
        accuracy
        >
        best_accuracy
    ):

        best_accuracy = (
            accuracy
        )


        best_threshold = (
            threshold
        )


        agent_enabled = (
            True
        )


print(
    "Agent enabled:",
    agent_enabled
)

print(
    "Threshold:",
    best_threshold
)

print(
    "Best:",
    best_accuracy
)


# ============================================================
# 54. Validation diagnostics
# ============================================================

validation_rows = []


for index in range(
    len(
        valid_subset
    )
):

    row = (
        valid_subset.iloc[
            index
        ]
    )


    base = (
        baseline_df.iloc[
            index
        ]
    )


    cache = (
        tool_df.iloc[
            index
        ]
    )


    tools = route_tools(

        row[
            "question"
        ],

        float(
            base[
                "margin"
            ]
        ),

        (
            best_threshold
            if best_threshold is not None
            else 0.0
        ),

        agent_enabled,
    )


    use_tool = (

        bool(
            tools
        )

        and

        bool(
            cache[
                "evidence"
            ]
        )
    )


    final_pred = (

        cache[
            "second_pred"
        ]

        if use_tool

        else base[
            "pred"
        ]
    )


    validation_rows.append(
        {
            "id":
                row[
                    "id"
                ],

            "gold":
                base[
                    "gold"
                ],

            "first_pred":
                base[
                    "pred"
                ],

            "margin":
                base[
                    "margin"
                ],

            "tools":
                cache[
                    "tools"
                ],

            "evidence":
                cache[
                    "evidence"
                ],

            "second_pred":
                cache[
                    "second_pred"
                ],

            "final_pred":
                final_pred,

            "first_correct":
                base[
                    "correct"
                ],

            "final_correct":
                final_pred
                ==
                base[
                    "gold"
                ],

            "context":
                cache[
                    "context"
                ],
        }
    )


validation_df = pd.DataFrame(
    validation_rows
)


SUBMISSION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


validation_df.to_csv(

    VALIDATION_PATH,

    index=False,
)


fixed = int(

    (
        (~validation_df["first_correct"])
        &
        validation_df["final_correct"]
    )
    .sum()
)


broken = int(

    (
        validation_df["first_correct"]
        &
        (~validation_df["final_correct"])
    )
    .sum()
)


print(
    "\nWrong -> Correct:",
    fixed
)

print(
    "Correct -> Wrong:",
    broken
)

print(
    "Net:",
    fixed - broken
)


# ============================================================
# 55. Test inference
# ============================================================

baseline_predictions = []

agent_predictions = []

tool_logs = []


for index in tqdm(

    range(
        len(
            test_df
        )
    ),

    desc="Test",
):

    row = (
        test_df.iloc[
            index
        ]
    )


    original_feature = (
        load_cached_feature(

            "test",

            row[
                "id"
            ],
        )
    )


    first = predict_from_features(

        row,

        [original_feature],
    )


    baseline_pred = (
        first[
            "pred"
        ]
    )


    baseline_predictions.append(
        baseline_pred
    )


    tools = route_tools(

        row[
            "question"
        ],

        first[
            "margin"
        ],

        (
            best_threshold
            if best_threshold is not None
            else 0.0
        ),

        agent_enabled,
    )


    final_pred = (
        baseline_pred
    )


    context = ""

    evidence = False


    if tools:

        image = load_rgb_image(
            row[
                "path"
            ]
        )


        tool_output = run_tools(

            image,

            tools,
        )


        context = (
            tool_output[
                "context"
            ]
        )


        evidence = bool(
            tool_output[
                "evidence"
            ]
        )


        if evidence:

            crop_features = (
                extract_crop_features(

                    tool_output[
                        "crops"
                    ]
                )
            )


            all_features = (

                [original_feature]

                +

                crop_features
            )


            second = (
                predict_from_features(

                    row,

                    all_features,

                    tool_context=(
                        context
                    ),
                )
            )


            final_pred = (
                second[
                    "pred"
                ]
            )


    agent_predictions.append(
        final_pred
    )


    tool_logs.append(
        {
            "id":
                row[
                    "id"
                ],

            "first_pred":
                baseline_pred,

            "margin":
                first[
                    "margin"
                ],

            "tools":
                ",".join(
                    tools
                ),

            "evidence":
                evidence,

            "final_pred":
                final_pred,

            "changed":
                final_pred
                !=
                baseline_pred,

            "context":
                context,
        }
    )


# ============================================================
# 56. Save submissions
# ============================================================

baseline_submission = pd.DataFrame(
    {
        "id":
            test_df[
                "id"
            ],

        "answer":
            baseline_predictions,
    }
)


agent_submission = pd.DataFrame(
    {
        "id":
            test_df[
                "id"
            ],

        "answer":
            agent_predictions,
    }
)


baseline_submission.to_csv(

    BASELINE_SUBMISSION_PATH,

    index=False,
)


agent_submission.to_csv(

    AGENT_SUBMISSION_PATH,

    index=False,
)


tool_log_df = pd.DataFrame(
    tool_logs
)


tool_log_df.to_csv(

    TOOL_LOG_PATH,

    index=False,
)


# ============================================================
# 57. Final checks
# ============================================================

if (
    len(
        agent_submission
    )
    !=
    len(
        test_df
    )
):

    raise RuntimeError(
        "Submission row count mismatch."
    )


if not set(

    agent_submission[
        "answer"
    ].unique()

).issubset(
    VALID_CHOICES
):

    raise RuntimeError(
        "Invalid prediction."
    )


# ============================================================
# 58. Final report
# ============================================================

print(
    "\n"
    +
    "=" * 70
)


print(
    "SSAFY-Agent v3"
)


print(
    "=" * 70
)


print(
    "Model:",
    MODEL_ID
)


print(
    "Vision mode:"
)

print(
    "  Flash"
)

print(
    "  tile=1"
)

print(
    "  cached"
)


print(
    "Train:",
    len(
        train_subset
    )
)


print(
    "Valid:",
    len(
        valid_subset
    )
)


print(
    "LoRA:",
    f"r={LORA_R}, "
    f"alpha={LORA_ALPHA}"
)


print(
    "Baseline validation:",
    f"{baseline_accuracy:.5f}"
)


print(
    "Agent validation:",
    f"{best_accuracy:.5f}"
)


print(
    "Agent enabled:",
    agent_enabled
)


print(
    "Router threshold:",
    best_threshold
)


print(
    "Wrong -> Correct:",
    fixed
)


print(
    "Correct -> Wrong:",
    broken
)


print(
    "Net tool gain:",
    fixed - broken
)


print(
    "Baseline submission:",
    BASELINE_SUBMISSION_PATH
)


print(
    "Agent submission:",
    AGENT_SUBMISSION_PATH
)


print(
    "Feature cache:",
    FEATURE_CACHE_DIR
)


print(
    "=" * 70
)