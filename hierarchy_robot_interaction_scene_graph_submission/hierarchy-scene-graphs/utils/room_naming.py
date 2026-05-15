import re

import numpy as np
import openai

from utils.clip_utils import get_text_feats_multiple_templates
from utils.conceptgraph_utils import normalize_object_name, should_skip_object_name


DEFAULT_ROOM_KEYWORDS = {
    "bathroom": [
        "bath",
        "bathtub",
        "bath tub",
        "bathroom",
        "sink",
        "basin",
        "sink/basin",
        "toilet",
        "shower",
        "towel",
        "mirror",
        "vanity",
    ],
    "bedroom": [
        "bed",
        "nightstand",
        "bedside",
        "dresser",
        "wardrobe",
        "closet",
        "pillow",
        "mattress",
        "bed base",
        "bed frame",
        "lamp",
        "chest",
    ],
    "dining room": [
        "dining",
        "dining table",
        "table",
        "buffet",
        "sideboard",
    ],
    "kitchen": [
        "kitchen",
        "fridge",
        "refrigerator",
        "microwave",
        "oven",
        "stove",
        "cooktop",
        "dishwasher",
        "countertop",
        "cabinet",
        "pantry",
        "sink",
        "basin",
    ],
    "laundry room": [
        "laundry",
        "washer",
        "dryer",
        "detergent",
        "hamper",
        "ironing",
    ],
    "library": [
        "bookshelf",
        "books",
        "book",
        "library",
        "desk",
        "office chair",
    ],
    "living room": [
        "living",
        "sofa",
        "couch",
        "coffee table",
        "tv",
        "television",
        "armchair",
        "chaise",
        "ottoman",
        "leg rest",
    ],
    "corridor": [
        "corridor",
        "hallway",
        "hall",
        "door",
        "door screen",
        "panel screen",
    ],
}


def _get_openai_key():
    openai_key = openai.api_key or __import__("os").environ.get("OPENAI_KEY") or __import__("os").environ.get("OPENAI_API_KEY")
    if not openai_key:
        raise RuntimeError(
            "OPENAI_KEY (or OPENAI_API_KEY) environment variable is required for GPT-based room naming."
        )
    return openai_key


def _normalize_choice(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", text.lower()).strip()


def infer_room_type_from_object_list_chat(object_list, default_room_types=None):
    openai_key = _get_openai_key()
    client = openai.OpenAI(api_key=openai_key)
    room_types = ""
    if default_room_types is not None:
        room_types = ", ".join(default_room_types)
        room_types = (
            "Please pick the most matching room type from the following list: "
            + room_types
            + "."
        )

    objects = ", ".join(object_list)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "You are a room type detector. You can infer a room type based on a list of objects.",
            },
            {
                "role": "user",
                "content": "The list of objects contained in this room are: bed, wardrobe, chair, sofa. What is the room type? Please just answer the room name.",
            },
            {"role": "assistant", "content": "bedroom"},
            {
                "role": "user",
                "content": "The list of objects contained in this room are: tv, table, chair, sofa. Please pick the most matching room type from the following list: living room, bedroom, bathroom, kitchen. What is the room type? Please just answer the room name.",
            },
            {"role": "assistant", "content": "living room"},
            {
                "role": "user",
                "content": f"The list of objects contained in this room are: {objects}. {room_types} What is the room type? Please just answer the room name.",
            },
        ],
    )
    result = response.choices[0].message.content.strip()
    if default_room_types is not None:
        normalized_default = {
            _normalize_choice(room_type): room_type for room_type in default_room_types
        }
        normalized_result = _normalize_choice(result)
        if normalized_result in normalized_default:
            result = normalized_default[normalized_result]
    return result


def infer_room_type_from_object_labels(object_names, default_room_types, keyword_map=None):
    keyword_map = keyword_map or DEFAULT_ROOM_KEYWORDS
    filtered_names = [
        normalize_object_name(name)
        for name in object_names
        if name and not should_skip_object_name(name)
    ]
    scores = {room_type: 0 for room_type in default_room_types}
    if len(filtered_names) == 0:
        return None, scores
    for room_type in default_room_types:
        room_keywords = keyword_map.get(room_type.lower(), [])
        for object_name in filtered_names:
            scores[room_type] += sum(1 for keyword in room_keywords if keyword in object_name)
    best_room_type = max(scores, key=scores.get)
    if scores[best_room_type] <= 0:
        return None, scores
    return best_room_type, scores


def infer_room_type_from_view_embeddings(room, default_room_types, clip_model, clip_feat_dim):
    if len(room.embeddings) == 0:
        return None
    text_feats = get_text_feats_multiple_templates(default_room_types, clip_model, clip_feat_dim)
    embeddings = np.array(room.embeddings)
    sim_mat = np.dot(embeddings, text_feats.T)
    col_ids = np.argmax(sim_mat, axis=1)
    unique, counts = np.unique(col_ids, return_counts=True)
    type_id = unique[np.argmax(counts)]
    return default_room_types[type_id]


def name_rooms(
    rooms,
    method,
    default_room_types,
    fallback_method="none",
    clip_model=None,
    clip_feat_dim=None,
):
    for room in rooms:
        if method == "none":
            continue
        if method == "view_embedding":
            room_type = infer_room_type_from_view_embeddings(
                room, default_room_types, clip_model, clip_feat_dim
            )
            if room_type is not None:
                room.name = room_type
            continue
        if method == "gpt_label":
            object_names = [obj.name for obj in room.objects if not should_skip_object_name(obj.name)]
            try:
                if object_names:
                    room.name = infer_room_type_from_object_list_chat(
                        object_names, default_room_types=default_room_types
                    )
                    continue
            except Exception as exc:
                print(f"GPT room naming failed for {room.room_id}: {exc}")
            if fallback_method == "view_embedding":
                room_type = infer_room_type_from_view_embeddings(
                    room, default_room_types, clip_model, clip_feat_dim
                )
                if room_type is not None:
                    room.name = room_type
            continue
        if method == "heuristic_label":
            room_type, scores = infer_room_type_from_object_labels(
                [obj.name for obj in room.objects],
                default_room_types=default_room_types,
            )
            print(f"room {room.room_id} label scores: {scores}")
            if room_type is not None:
                room.name = room_type
                continue
            if fallback_method == "view_embedding":
                room_type = infer_room_type_from_view_embeddings(
                    room, default_room_types, clip_model, clip_feat_dim
                )
                if room_type is not None:
                    room.name = room_type
            continue
        raise ValueError(f"Unsupported room naming method: {method}")

