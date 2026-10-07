from __future__ import annotations

from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import RLock
from urllib.parse import quote

import gc
import hashlib
import json
import os
import re
import sqlite3
import sys
import unicodedata
import uuid

import bm25s
import faiss
import numpy as np
import pymupdf
import requests
import torch

from bm25s.tokenization import Tokenizer
from dotenv import load_dotenv
from flask import Flask, abort, jsonify, render_template, request, send_file
from sentence_transformers import CrossEncoder, SentenceTransformer
from transformers import AutoModel, AutoProcessor, AutoTokenizer


# ============================================================
# 0. Paths + Configuration
# ============================================================

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

load_dotenv(PROJECT_ROOT / ".env")

OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "ragscholar_full"
DOCUMENTS_ROOT = OUTPUT_ROOT / "documents"
CORPUS_ROOT = OUTPUT_ROOT / "corpus"

DB_PATH = APP_DIR / "ragscholar_chat.db"


EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
RERANK_MODEL = "Qwen/Qwen3-Reranker-0.6B"
SIGLIP_MODEL = "google/siglip2-base-patch16-224"


GROQ_API = "https://api.groq.com/openai/v1"

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY",
    "",
).strip()

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "openai/gpt-oss-120b",
).strip()

GROQ_REASONING_EFFORT = os.getenv(
    "GROQ_REASONING_EFFORT",
    "medium",
).strip().lower()

GROQ_MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "GROQ_MAX_OUTPUT_TOKENS",
        "2400",
    )
)

RAGSCHOLAR_TOP_DOCS = int(
    os.getenv(
        "RAGSCHOLAR_TOP_DOCS",
        "5",
    )
)

RAGSCHOLAR_MAX_EVIDENCE = int(
    os.getenv(
        "RAGSCHOLAR_MAX_EVIDENCE",
        "10",
    )
)

RAGSCHOLAR_EVIDENCE_CHARS = int(
    os.getenv(
        "RAGSCHOLAR_EVIDENCE_CHARS",
        "24000",
    )
)

RAGSCHOLAR_MAX_VISUAL_RESULTS = int(
    os.getenv(
        "RAGSCHOLAR_MAX_VISUAL_RESULTS",
        "5",
    )
)

PORT = int(
    os.getenv(
        "PORT",
        "5000",
    )
)


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

SIGLIP_QUERY_DEVICE = "cpu"

MAX_RUNTIME_DOCS = 1
GLOBAL_RERANK_POOL = 80


TEXT_CFG = {
    "dense_k": 30,
    "bm25_k": 30,
    "rrf_k": 60,
    "rrf_top_k": 30,
    "anchor_budget": 5,
    "neighbor_budget": 4,
}


MATH_CFG = {
    "dense_k": 40,
    "bm25_k": 40,
    "rrf_k": 60,
    "rerank_pool": 40,
    "evidence_k": 3,
}


VISUAL_CFG = {
    "candidate_k": 24,
    "evidence_k": 3,
    "rrf_k": 60,
    "rerank_pool": 10,

    "semantic_weights": {
        "qwen_source": 1.0,
        "bm25_source": 1.0,
        "ocr_lexical": 0.25,
    },

    "visual_weights": {
        "qwen_source": 1.0,
        "siglip_multiview": 1.0,
        "ocr_lexical": 0.5,
        "bm25_source": 0.25,
    },
}


TABLE_CFG = {
    "table_k": 16,
    "row_k": 24,
    "rrf_k": 60,
    "evidence_k": 3,
}


ROUTER_CFG = {
    "dense_k": 15,
    "bm25_k": 15,
    "rrf_k": 60,
    "candidate_k": 12,
}


# ============================================================
# 1. Utilities
# ============================================================

def clean_text(value):
    return re.sub(
        r"\s+",
        " ",
        str(value or ""),
    ).strip()


def short_text(value, n=1200):
    text = clean_text(value)

    if len(text) <= n:
        return text

    return text[:n].rstrip() + "..."


def stable_hash(*parts, n=12):
    payload = json.dumps(
        parts,
        ensure_ascii=False,
        sort_keys=True,
        default=str,
    ).encode("utf-8")

    return hashlib.sha1(
        payload
    ).hexdigest()[:n]


def read_jsonl(path):
    path = Path(path)

    if not path.is_file():
        return []

    return [
        json.loads(line)
        for line in path.read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]


TECH_TOKEN_RE = re.compile(
    r"[^\W\s]+(?:[./×+\-][^\W\s]+)*",
    re.UNICODE,
)

TECH_SPLIT_RE = re.compile(
    r"[./×+\-]"
)


def technical_split(text):
    out = []

    for token in TECH_TOKEN_RE.findall(
        str(text).lower()
    ):
        out.append(token)

        if TECH_SPLIT_RE.search(token):
            out.extend(
                p
                for p in TECH_SPLIT_RE.split(token)
                if p and (
                    len(p) >= 2
                    or p.isdigit()
                )
            )

    return out


def weighted_rrf(
    lanes,
    weights,
    k=60,
    top_k=30,
    id_key="id",
):
    scores = defaultdict(float)
    provenance = defaultdict(dict)

    for lane, rows in lanes.items():
        weight = float(
            weights.get(
                lane,
                1.0,
            )
        )

        for row in rows:
            rid = row[id_key]
            rank = int(
                row["rank"]
            )

            scores[rid] += (
                weight
                / (k + rank)
            )

            provenance[rid][lane] = {
                "rank": rank,

                "score": float(
                    row.get(
                        "score",
                        0.0,
                    )
                ),

                "weight": weight,
            }

    ranked = sorted(
        scores.items(),
        key=lambda x: (
            -x[1],
            x[0],
        ),
    )[:top_k]

    return [
        {
            id_key: rid,
            "rank": i,
            "score": float(score),
            "lanes": provenance[rid],
        }

        for i, (rid, score)
        in enumerate(
            ranked,
            1,
        )
    ]


def release_memory():
    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# 2. Models
# ============================================================

TEXT_TOKENIZER = None
EMBEDDER = None
ENCODE_DOCS = None

RERANKER = None

SIGLIP_PROCESSOR = None
SIGLIP_MODEL_OBJ = None


TEXT_RERANK_PROMPT = (
    "Rank academic passages by how directly and accurately "
    "they provide evidence for the question. "
    "Prefer explicit source evidence over merely related text."
)


MATH_RERANK_PROMPT = (
    "Rank equations by whether the equation and its local "
    "source context satisfy the exact mathematical role requested. "
    "Distinguish definitions from related formulas and "
    "forward from inverse relations."
)


def init_models():
    global TEXT_TOKENIZER
    global EMBEDDER
    global ENCODE_DOCS

    if EMBEDDER is not None:
        return

    print(
        f"[RAGScholar] Loading embedding model "
        f"on {DEVICE}: {EMBED_MODEL}"
    )

    TEXT_TOKENIZER = (
        AutoTokenizer.from_pretrained(
            EMBED_MODEL,
            use_fast=True,
        )
    )

    EMBEDDER = SentenceTransformer(
        EMBED_MODEL,
        device=DEVICE,
    )

    if hasattr(
        EMBEDDER,
        "encode_document",
    ):
        ENCODE_DOCS = (
            EMBEDDER.encode_document
        )

    else:
        ENCODE_DOCS = (
            EMBEDDER.encode
        )


def get_reranker():
    global RERANKER

    init_models()

    if RERANKER is None:

        if (
            DEVICE == "cuda"
            and torch.cuda.is_bf16_supported()
        ):
            dtype = torch.bfloat16

        elif DEVICE == "cuda":
            dtype = torch.float16

        else:
            dtype = torch.float32

        print(
            f"[RAGScholar] Loading reranker "
            f"on {DEVICE}: {RERANK_MODEL}"
        )

        RERANKER = CrossEncoder(
            RERANK_MODEL,
            device=DEVICE,

            model_kwargs={
                "torch_dtype": dtype,
            },

            prompts={
                "research":
                    TEXT_RERANK_PROMPT,

                "math_equation":
                    MATH_RERANK_PROMPT,
            },

            default_prompt_name="research",
        )

    return RERANKER


def get_siglip(device=None):
    global SIGLIP_PROCESSOR
    global SIGLIP_MODEL_OBJ

    device = (
        device
        or SIGLIP_QUERY_DEVICE
    )

    if SIGLIP_PROCESSOR is None:

        print(
            "[RAGScholar] Loading SigLIP processor:",
            SIGLIP_MODEL,
        )

        SIGLIP_PROCESSOR = (
            AutoProcessor.from_pretrained(
                SIGLIP_MODEL
            )
        )

    if SIGLIP_MODEL_OBJ is None:

        print(
            f"[RAGScholar] Loading SigLIP "
            f"on {device}: {SIGLIP_MODEL}"
        )

        SIGLIP_MODEL_OBJ = (
            AutoModel.from_pretrained(
                SIGLIP_MODEL
            ).eval()
        )

    SIGLIP_MODEL_OBJ = (
        SIGLIP_MODEL_OBJ.to(device)
    )

    return (
        SIGLIP_PROCESSOR,
        SIGLIP_MODEL_OBJ,
    )


def encode_query(query):
    init_models()

    if hasattr(
        EMBEDDER,
        "encode_query",
    ):
        fn = EMBEDDER.encode_query
        kwargs = {}

    else:
        fn = EMBEDDER.encode

        kwargs = {
            "prompt_name":
                "query"
        }

    vec = fn(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        **kwargs,
    ).astype("float32")

    vec = np.ascontiguousarray(
        vec
    )

    faiss.normalize_L2(
        vec
    )

    return vec


# ============================================================
# 3. Corpus Router
# ============================================================

DOC_RECORDS = []
DOC_BY_ROW = {}
DOC_BY_ID = {}

DOC_DENSE_INDEX = None

DOC_BM25_TOKENIZER = None
DOC_BM25 = None


def build_bm25_runtime(texts):
    if not texts:
        return None, None

    tok = Tokenizer(
        splitter=technical_split,
        stopwords=[],
        stemmer=None,
    )

    tokens = tok.tokenize(
        texts,
        return_as="tuple",
    )

    bm = bm25s.BM25(
        method="lucene",
        k1=1.2,
        b=0.75,
        backend="numpy",
    )

    bm.index(
        tokens
    )

    return bm, tok


def init_corpus_router():
    global DOC_RECORDS
    global DOC_BY_ROW
    global DOC_BY_ID
    global DOC_DENSE_INDEX
    global DOC_BM25_TOKENIZER
    global DOC_BM25

    docs_path = (
        CORPUS_ROOT
        / "documents.jsonl"
    )

    index_path = (
        CORPUS_ROOT
        / "documents.faiss"
    )

    if not docs_path.is_file():
        raise FileNotFoundError(
            "Missing corpus router metadata:\n"
            f"{docs_path}\n\n"
            "Run the RAGScholar notebook first."
        )

    if not index_path.is_file():
        raise FileNotFoundError(
            "Missing corpus router FAISS index:\n"
            f"{index_path}\n\n"
            "Run the RAGScholar notebook first."
        )

    DOC_RECORDS = read_jsonl(
        docs_path
    )

    DOC_BY_ROW = {
        int(r["row_id"]): r
        for r in DOC_RECORDS
    }

    DOC_BY_ID = {
        r["document_id"]: r
        for r in DOC_RECORDS
    }

    DOC_DENSE_INDEX = (
        faiss.read_index(
            str(index_path)
        )
    )

    DOC_BM25_TOKENIZER = Tokenizer(
        splitter=technical_split,
        stopwords=[],
        stemmer=None,
    )

    doc_tokens = (
        DOC_BM25_TOKENIZER.tokenize(
            [
                r["router_text"]
                for r in DOC_RECORDS
            ],
            return_as="tuple",
        )
    )

    DOC_BM25 = bm25s.BM25(
        method="lucene",
        k1=1.2,
        b=0.75,
        backend="numpy",
    )

    DOC_BM25.index(
        doc_tokens
    )

    if (
        DOC_DENSE_INDEX.ntotal
        != len(DOC_RECORDS)
    ):
        raise RuntimeError(
            "Router artifact mismatch: "
            f"documents.faiss has "
            f"{DOC_DENSE_INDEX.ntotal} vectors "
            f"but documents.jsonl has "
            f"{len(DOC_RECORDS)} rows."
        )

    print(
        "[RAGScholar] Corpus router ready:",
        len(DOC_RECORDS),
        "documents",
    )


def route_documents(
    query,
    top_k=RAGSCHOLAR_TOP_DOCS,
):
    if not DOC_RECORDS:
        return []

    k_dense = min(
        ROUTER_CFG["dense_k"],
        len(DOC_RECORDS),
    )

    scores, rows = (
        DOC_DENSE_INDEX.search(
            encode_query(query),
            k_dense,
        )
    )

    dense = [
        {
            "id":
                DOC_BY_ROW[
                    int(row)
                ]["document_id"],

            "rank": i,

            "score":
                float(score),
        }

        for i, (row, score)
        in enumerate(
            zip(
                rows[0],
                scores[0],
            ),
            1,
        )

        if row >= 0
    ]

    qtok = (
        DOC_BM25_TOKENIZER.tokenize(
            [query],
            update_vocab=False,
            return_as="tuple",
        )
    )

    brow, bscores = (
        DOC_BM25.retrieve(
            qtok,

            k=min(
                ROUTER_CFG["bm25_k"],
                len(DOC_RECORDS),
            ),

            return_as="tuple",
            show_progress=False,
        )
    )

    lexical = [
        {
            "id":
                DOC_BY_ROW[
                    int(row)
                ]["document_id"],

            "score":
                float(score),
        }

        for row, score
        in zip(
            brow[0],
            bscores[0],
        )

        if (
            row >= 0
            and float(score) > 0
        )
    ]

    for i, hit in enumerate(
        lexical,
        1,
    ):
        hit["rank"] = i

    fused = weighted_rrf(
        {
            "dense": dense,
            "bm25": lexical,
        },

        {
            "dense": 1.0,
            "bm25": 1.0,
        },

        ROUTER_CFG["rrf_k"],
        ROUTER_CFG["candidate_k"],
    )

    pool = fused[
        :min(
            len(fused),
            10,
        )
    ]

    if pool:

        reranker = get_reranker()

        pairs = [
            (
                query,

                DOC_BY_ID[
                    x["id"]
                ]["router_text"],
            )

            for x
            in pool
        ]

        rr_scores = np.asarray(
            reranker.predict(
                pairs,

                prompt_name="research",

                batch_size=4,

                show_progress_bar=False,

                convert_to_numpy=True,
            )
        ).reshape(-1)

        pool = [
            dict(
                item,
                rerank_score=float(
                    score
                ),
            )

            for item, score
            in zip(
                pool,
                rr_scores,
            )
        ]

        pool.sort(
            key=lambda x: (
                -x["rerank_score"],
                x["rank"],
            )
        )

    return [
        {
            **DOC_BY_ID[
                x["id"]
            ],

            "router_score":
                x.get(
                    "rerank_score",
                    x["score"],
                ),
        }

        for x in pool[:top_k]
    ]


# ============================================================
# 4. Lazy Document Runtime
# ============================================================

RUNTIME_POOL = OrderedDict()


def load_document_runtime(
    document_id,
):
    if document_id in RUNTIME_POOL:

        runtime = (
            RUNTIME_POOL.pop(
                document_id
            )
        )

        RUNTIME_POOL[
            document_id
        ] = runtime

        return runtime

    root = (
        DOCUMENTS_ROOT
        / document_id
    )

    manifest_path = (
        root
        / "manifest.json"
    )

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Document manifest not found: "
            f"{manifest_path}"
        )

    manifest = json.loads(
        manifest_path.read_text(
            encoding="utf-8"
        )
    )

    labels_path = (
        root
        / "canonical"
        / "labels.json"
    )

    if labels_path.is_file():

        labels = json.loads(
            labels_path.read_text(
                encoding="utf-8"
            )
        )

    else:
        labels = {}

    text_records = read_jsonl(
        root
        / "text"
        / "records.jsonl"
    )

    math_cards = read_jsonl(
        root
        / "math"
        / "equation_cards.jsonl"
    )

    math_edges = read_jsonl(
        root
        / "math"
        / "source_graph.jsonl"
    )

    visuals = read_jsonl(
        root
        / "visual"
        / "records.jsonl"
    )

    tables = read_jsonl(
        root
        / "table"
        / "tables.jsonl"
    )

    table_rows = read_jsonl(
        root
        / "table"
        / "rows.jsonl"
    )

    table_cells = read_jsonl(
        root
        / "table"
        / "cells.jsonl"
    )

    text_bm, text_tok = (
        build_bm25_runtime(
            [
                r["bm25_text"]
                for r in text_records
            ]
        )
    )

    math_bm, math_tok = (
        build_bm25_runtime(
            [
                r["lexical_text"]
                for r in math_cards
            ]
        )
    )

    visual_text = [
        "\n".join(
            x
            for x in (
                r.get("label")
                or "",

                " > ".join(
                    r.get(
                        "section_path"
                    )
                    or []
                ),

                r.get(
                    "source_text"
                )
                or "",

                r.get(
                    "visible_text"
                )
                or "",
            )

            if x
        )

        for r in visuals
    ]

    visual_bm, visual_tok = (
        build_bm25_runtime(
            visual_text
        )
    )

    table_bm, table_tok = (
        build_bm25_runtime(
            [
                r["table_text"]
                for r in tables
            ]
        )
    )

    row_bm, row_tok = (
        build_bm25_runtime(
            [
                r["text"]
                for r in table_rows
            ]
        )
    )

    runtime = {
        "document_id":
            document_id,

        "root":
            root,

        "manifest":
            manifest,

        "labels":
            labels,

        "text_records":
            text_records,

        "text_by_chunk": {
            r["chunk_id"]: r
            for r in text_records
        },

        "text_index":
            faiss.read_index(
                str(
                    root
                    / "text"
                    / "dense.faiss"
                )
            )
            if (
                root
                / "text"
                / "dense.faiss"
            ).is_file()
            else None,

        "text_bm":
            text_bm,

        "text_tok":
            text_tok,

        "math_cards":
            math_cards,

        "math_by_object": {
            r["object_id"]: r
            for r in math_cards
        },

        "math_by_label": {
            r["equation_label"].lower():
                r

            for r in math_cards

            if r.get(
                "equation_label"
            )
        },

        "math_edges":
            math_edges,

        "math_index":
            faiss.read_index(
                str(
                    root
                    / "math"
                    / "dense.faiss"
                )
            )
            if (
                root
                / "math"
                / "dense.faiss"
            ).is_file()
            else None,

        "math_bm":
            math_bm,

        "math_tok":
            math_tok,

        "visuals":
            visuals,

        "visual_by_id": {
            r["visual_id"]: r
            for r in visuals
        },

        "visual_index":
            faiss.read_index(
                str(
                    root
                    / "visual"
                    / "qwen.faiss"
                )
            )
            if (
                root
                / "visual"
                / "qwen.faiss"
            ).is_file()
            else None,

        "visual_bm":
            visual_bm,

        "visual_tok":
            visual_tok,

        "visual_view_meta":
            read_jsonl(
                root
                / "visual"
                / "siglip_multiview_meta.jsonl"
            ),

        "visual_view_vectors":
            np.load(
                root
                / "visual"
                / "siglip_multiview.npy"
            )
            if (
                root
                / "visual"
                / "siglip_multiview.npy"
            ).is_file()
            else np.empty(
                (0, 0),
                dtype="float32",
            ),

        "tables":
            tables,

        "table_by_id": {
            r["table_id"]: r
            for r in tables
        },

        "table_bm":
            table_bm,

        "table_tok":
            table_tok,

        "table_index":
            faiss.read_index(
                str(
                    root
                    / "table"
                    / "tables.faiss"
                )
            )
            if (
                root
                / "table"
                / "tables.faiss"
            ).is_file()
            else None,

        "table_rows":
            table_rows,

        "table_row_by_id": {
            r["row_id"]: r
            for r in table_rows
        },

        "row_bm":
            row_bm,

        "row_tok":
            row_tok,

        "row_index":
            faiss.read_index(
                str(
                    root
                    / "table"
                    / "rows.faiss"
                )
            )
            if (
                root
                / "table"
                / "rows.faiss"
            ).is_file()
            else None,

        "table_cells":
            table_cells,
    }

    section_records = defaultdict(
        list
    )

    for r in text_records:

        section_records[
            tuple(
                r["section_path"]
            )
        ].append(r)

    runtime[
        "section_records"
    ] = section_records

    runtime[
        "section_position"
    ] = {
        r["chunk_id"]:
            (
                section,
                i,
            )

        for section, rows
        in section_records.items()

        for i, r
        in enumerate(rows)
    }

    ref_index = defaultdict(
        list
    )

    for r in visuals:

        if r.get(
            "reference_key"
        ):
            ref_index[
                r["reference_key"]
            ].append(
                r["visual_id"]
            )

    runtime[
        "visual_reference_index"
    ] = ref_index

    RUNTIME_POOL[
        document_id
    ] = runtime

    while (
        len(RUNTIME_POOL)
        > MAX_RUNTIME_DOCS
    ):
        RUNTIME_POOL.popitem(
            last=False
        )

    return runtime


# ============================================================
# 5. Query Planner
# ============================================================

NUMERIC_OPERATION_RE = re.compile(
    r"\b(?:average|mean|sum|total|percentage|percent|difference|ratio|calculate|compute)\b",
    re.I,
)

FIGURE_WORD_RE = re.compile(
    r"\b(?:figure|fig\.?|diagram|plot|graph|chart|architecture|image|images|picture|pictures|photo|visual|schematic)\b",
    re.I,
)

TABLE_WORD_RE = re.compile(
    r"\b(?:table|row|column|cell|coverage)\b",
    re.I,
)

MATH_WORD_RE = re.compile(
    r"\b(?:eq(?:uation)?s?\.?|formula|mathematical expression|derive|derivative|jacobian|softmax|sigmoid|logits?|attention)\b",
    re.I,
)

TEXT_WORD_RE = re.compile(
    r"\b(?:why|explain|describe|discuss|what|how|meaning|represent|compare|summarize)\b",
    re.I,
)

PAGE_REFERENCE_RE = re.compile(
    r"\b(?:pdf\s*)?page\s*(\d+)\b",
    re.I,
)

EXPLICIT_EQ_RE = re.compile(
    r"\b(?:eq(?:uation)?s?\.?|formula)\s*\(?\s*([A-Za-z]\.?\d+|\d+(?:\.\d+)+)\s*\)?",
    re.I,
)

EXACT_VISUAL_RE = re.compile(
    r"\b(?:figure|fig\.?|chart)\s*[:#]?\s*(\d+(?:\.\d+)*[a-z]?)\b",
    re.I,
)

TABLE_REFERENCE_RE = re.compile(
    r"\btable\s*(\d+(?:\.\d+)*)",
    re.I,
)

VISUAL_CUE_RE = re.compile(
    r"\b(?:diagram|architecture|plot|graph|curve|axis|legend|arrow|panel|image|picture|photo|visual|schematic|dashed|dotted|blue|red|green|left|right|top|bottom)\b",
    re.I,
)


def detect_modalities(query):
    mods = []

    if TABLE_WORD_RE.search(
        query
    ):
        mods.append(
            "table"
        )

    if FIGURE_WORD_RE.search(
        query
    ):
        mods.append(
            "visual"
        )

    if MATH_WORD_RE.search(
        query
    ):
        mods.append(
            "math"
        )

    if (
        TEXT_WORD_RE.search(
            query
        )
        or not mods
    ):
        mods.append(
            "text"
        )

    return list(
        dict.fromkeys(
            mods
        )
    )


def plan_query(query):
    query = clean_text(
        query
    )

    if not query:
        return {
            "status":
                "invalid_query",

            "query":
                query,

            "modalities":
                [],
        }

    modalities = (
        detect_modalities(
            query
        )
    )

    if (
        NUMERIC_OPERATION_RE.search(
            query
        )
        and TABLE_WORD_RE.search(
            query
        )
    ):
        return {
            "status":
                "numeric_engine_required",

            "query":
                query,

            "modalities":
                ["table"],

            "pdf_pages":
                [],
        }

    return {
        "status":
            "plan_ready",

        "query":
            query,

        "modalities":
            modalities,

        "pdf_pages": [
            int(x)

            for x in
            PAGE_REFERENCE_RE.findall(
                query
            )
        ],
    }


# ============================================================
# 6. Retrieval Helpers
# ============================================================

def dense_hits(
    index,
    records,
    query,
    id_field,
    k,
):
    if (
        index is None
        or not records
    ):
        return []

    scores, rows = (
        index.search(
            encode_query(
                query
            ),

            min(
                k,
                len(records),
            ),
        )
    )

    return [
        {
            "id":
                records[
                    int(row)
                ][id_field],

            "rank": i,

            "score":
                float(score),

            "record":
                records[
                    int(row)
                ],
        }

        for i, (row, score)
        in enumerate(
            zip(
                rows[0],
                scores[0],
            ),
            1,
        )

        if row >= 0
    ]


def bm25_hits(
    bm,
    tok,
    records,
    query,
    id_field,
    k,
):
    if (
        bm is None
        or tok is None
        or not records
    ):
        return []

    qtok = tok.tokenize(
        [query],
        update_vocab=False,
        return_as="tuple",
    )

    rows, scores = (
        bm.retrieve(
            qtok,

            k=min(
                k,
                len(records),
            ),

            return_as="tuple",
            show_progress=False,
        )
    )

    hits = [
        {
            "id":
                records[
                    int(row)
                ][id_field],

            "score":
                float(score),

            "record":
                records[
                    int(row)
                ],
        }

        for row, score
        in zip(
            rows[0],
            scores[0],
        )

        if (
            row >= 0
            and float(score) > 0
        )
    ]

    for i, hit in enumerate(
        hits,
        1,
    ):
        hit[
            "rank"
        ] = i

    return hits


def bind_evidence(
    runtime,
    item,
    score=0.0,
):
    source = (
        runtime[
            "manifest"
        ]["source"]
    )

    out = dict(
        item
    )

    out[
        "document_id"
    ] = runtime[
        "document_id"
    ]

    out[
        "source_title"
    ] = (
        runtime[
            "manifest"
        ].get(
            "title"
        )
        or source.get(
            "filename"
        )
    )

    out[
        "source_category"
    ] = runtime[
        "manifest"
    ].get(
        "category"
    )

    out[
        "source_pdf_sha256"
    ] = source.get(
        "source_sha256"
    )

    out[
        "retrieval_score"
    ] = float(
        score
    )

    out[
        "evidence_id"
    ] = (
        f"{runtime['document_id']}:"
        f"{item['modality']}:"
        f"{stable_hash(item['source_object_id'], item['pdf_page_number'], n=12)}"
    )

    return out


# ============================================================
# 7. Text Retrieval
# ============================================================

def retrieve_text_doc(
    runtime,
    query,
    top_k=5,
):
    dense = dense_hits(
        runtime[
            "text_index"
        ],

        runtime[
            "text_records"
        ],

        query,

        "chunk_id",

        TEXT_CFG[
            "dense_k"
        ],
    )

    lexical = bm25_hits(
        runtime[
            "text_bm"
        ],

        runtime[
            "text_tok"
        ],

        runtime[
            "text_records"
        ],

        query,

        "chunk_id",

        TEXT_CFG[
            "bm25_k"
        ],
    )

    fused = weighted_rrf(
        {
            "dense":
                dense,

            "bm25":
                lexical,
        },

        {
            "dense":
                1.0,

            "bm25":
                1.0,
        },

        TEXT_CFG[
            "rrf_k"
        ],

        TEXT_CFG[
            "rrf_top_k"
        ],
    )

    pool = {
        x["id"]:
            x["record"]

        for x in (
            dense
            + lexical
        )
    }

    candidates = [
        dict(
            x,
            record=pool[
                x["id"]
            ],
        )

        for x in fused

        if x["id"]
        in pool
    ]

    if not candidates:
        return []

    reranker = get_reranker()

    scores = np.asarray(
        reranker.predict(
            [
                (
                    query,

                    c[
                        "record"
                    ][
                        "dense_text"
                    ],
                )

                for c
                in candidates
            ],

            prompt_name="research",

            batch_size=4,

            show_progress_bar=False,

            convert_to_numpy=True,
        )
    ).reshape(-1)

    ranked = [
        dict(
            c,

            rerank_score=float(
                score
            ),
        )

        for c, score
        in zip(
            candidates,
            scores,
        )
    ]

    ranked.sort(
        key=lambda x: (
            -x[
                "rerank_score"
            ],

            x[
                "rank"
            ],
        )
    )

    anchors = ranked[
        :min(
            TEXT_CFG[
                "anchor_budget"
            ],

            len(
                ranked
            ),
        )
    ]

    neighbors = {}

    anchor_ids = {
        a[
            "record"
        ][
            "chunk_id"
        ]

        for a
        in anchors
    }

    for anchor in anchors:

        position = (
            runtime[
                "section_position"
            ].get(
                anchor[
                    "record"
                ][
                    "chunk_id"
                ]
            )
        )

        if not position:
            continue

        section, i = position

        rows = runtime[
            "section_records"
        ][section]

        for j in (
            i - 1,
            i + 1,
        ):
            if (
                0 <= j < len(rows)

                and rows[j][
                    "chunk_id"
                ]
                not in anchor_ids
            ):
                neighbors[
                    rows[j][
                        "chunk_id"
                    ]
                ] = rows[j]

    selected = (
        [
            a[
                "record"
            ]
            for a in anchors
        ]

        + list(
            neighbors.values()
        )[
            :TEXT_CFG[
                "neighbor_budget"
            ]
        ]
    )

    seen = set()
    evidence = []

    for row in selected:

        key = clean_text(
            unicodedata.normalize(
                "NFKC",

                row[
                    "source_text"
                ],
            )
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        item = {
            "source_object_id":
                row[
                    "chunk_id"
                ],

            "pdf_page_number":
                row[
                    "page_start"
                ]
                + 1,

            "modality":
                "text",

            "content": {
                "text":
                    row[
                        "source_text"
                    ],

                "section_path":
                    row[
                        "section_path"
                    ],
            },

            "provenance": {
                "source_object_ids":
                    row[
                        "source_object_ids"
                    ],

                "page_range": [
                    row[
                        "page_start"
                    ]
                    + 1,

                    row[
                        "page_end"
                    ]
                    + 1,
                ],
            },
        }

        evidence.append(
            bind_evidence(
                runtime,
                item,
            )
        )

        if (
            len(evidence)
            >= top_k
        ):
            break

    return evidence


# ============================================================
# 8. Math Retrieval
# ============================================================

SOURCE_TAG_RE = re.compile(
    r"\\tag\*?\s*\{\s*([^{}]+?)\s*\}",
    re.I,
)


def strip_equation_tag(
    latex,
):
    return SOURCE_TAG_RE.sub(
        "",

        unicodedata.normalize(
            "NFKC",
            str(
                latex
                or ""
            ),
        ),

    ).strip()


def requested_math_roles(
    query,
):
    q = clean_text(
        query
    ).lower()

    required = set()
    forbidden = set()

    if (
        "softmax" in q

        and any(
            x in q

            for x in (
                "define",
                "definition",
                "standalone",
                "expression",
            )
        )
    ):
        required |= {
            "softmax",
            "standalone_softmax",
        }

        forbidden |= {
            "loss"
        }

    if (
        "softmax" in q

        and any(
            x in q

            for x in (
                "probability",
                "probabilities",
                "class logits",
                "normalized",
            )
        )
    ):
        required.add(
            "softmax"
        )

    if (
        "jacobian" in q
        and "determinant" in q
    ):
        required.add(
            "jacobian_determinant"
        )

        forbidden.add(
            "jacobian_matrix"
        )

    elif (
        "jacobian" in q
        or "derivative" in q
    ):
        required.add(
            "jacobian_matrix"
        )

        forbidden.add(
            "jacobian_determinant"
        )

    if (
        "inverse" in q
        and "mapping" in q
    ):
        required.add(
            "inverse_mapping"
        )

        forbidden.add(
            "forward_mapping"
        )

    elif (
        "forward" in q
        and "mapping" in q
    ):
        required.add(
            "forward_mapping"
        )

        forbidden.add(
            "inverse_mapping"
        )

    if (
        "attention" in q
        and "weight" in q
    ):
        required.add(
            "attention_weights"
        )

    if (
        "weighted sum" in q
        and "value" in q
    ):
        required.add(
            "weighted_value_sum"
        )

    return (
        required,
        forbidden,
    )


def retrieve_math_doc(
    runtime,
    query,
    top_k=3,
):
    explicit = [
        f"equation:{x}".lower()

        for x in
        EXPLICIT_EQ_RE.findall(
            query
        )
    ]

    if explicit:

        cards = [
            runtime[
                "math_by_label"
            ][x]

            for x in explicit

            if x in runtime[
                "math_by_label"
            ]
        ]

    else:

        dense = dense_hits(
            runtime[
                "math_index"
            ],

            runtime[
                "math_cards"
            ],

            query,

            "object_id",

            MATH_CFG[
                "dense_k"
            ],
        )

        lexical = bm25_hits(
            runtime[
                "math_bm"
            ],

            runtime[
                "math_tok"
            ],

            runtime[
                "math_cards"
            ],

            query,

            "object_id",

            MATH_CFG[
                "bm25_k"
            ],
        )

        fused = weighted_rrf(
            {
                "dense":
                    dense,

                "bm25":
                    lexical,
            },

            {
                "dense":
                    1.0,

                "bm25":
                    1.0,
            },

            MATH_CFG[
                "rrf_k"
            ],

            MATH_CFG[
                "rerank_pool"
            ],
        )

        pool = {
            x["id"]:
                x["record"]

            for x in (
                dense
                + lexical
            )
        }

        candidates = [
            dict(
                x,

                card=pool[
                    x["id"]
                ],
            )

            for x in fused

            if x["id"]
            in pool
        ]

        if candidates:

            reranker = (
                get_reranker()
            )

            pairs = []

            for candidate in candidates:

                card = (
                    candidate[
                        "card"
                    ]
                )

                pairs.append(
                    (
                        query,

                        "\n".join(
                            [
                                "Section: "
                                + " > ".join(
                                    card[
                                        "section_path"
                                    ]
                                ),

                                "Context: "
                                + " ".join(
                                    card.get(
                                        "source_contexts",
                                        [],
                                    )[:2]
                                ),

                                "Formula: "
                                + strip_equation_tag(
                                    card[
                                        "original_latex"
                                    ]
                                ),

                                "Roles: "
                                + ", ".join(
                                    card.get(
                                        "formula_roles",
                                        [],
                                    )
                                ),
                            ]
                        ),
                    )
                )

            scores = np.asarray(
                reranker.predict(
                    pairs,

                    prompt_name=
                        "math_equation",

                    batch_size=4,

                    show_progress_bar=False,

                    convert_to_numpy=True,
                )
            ).reshape(-1)

            candidates = [
                dict(
                    candidate,

                    rerank_score=float(
                        score
                    ),
                )

                for candidate, score
                in zip(
                    candidates,
                    scores,
                )
            ]

            candidates.sort(
                key=lambda x: (
                    -x[
                        "rerank_score"
                    ],

                    x[
                        "rank"
                    ],
                )
            )

        required, forbidden = (
            requested_math_roles(
                query
            )
        )

        cards = []

        for item in candidates:

            roles = set(
                item[
                    "card"
                ].get(
                    "formula_roles",
                    [],
                )
            )

            if (
                forbidden
                & roles
            ):
                continue

            if (
                required

                and not required.issubset(
                    roles
                )
            ):
                continue

            cards.append(
                item[
                    "card"
                ]
            )

            if (
                len(cards)
                >= top_k
            ):
                break

        if (
            not cards
            and not required
            and candidates
        ):
            cards = [
                candidates[
                    0
                ][
                    "card"
                ]
            ]

    evidence = []

    for card in cards[
        :top_k
    ]:

        item = {
            "source_object_id":
                card[
                    "object_id"
                ],

            "pdf_page_number":
                card[
                    "pdf_page_number"
                ],

            "modality":
                "math",

            "content": {
                "latex":
                    card[
                        "original_latex"
                    ],

                "equation_label":
                    card.get(
                        "equation_label"
                    ),

                "section_path":
                    card[
                        "section_path"
                    ],
            },

            "provenance": {
                "bbox_pdf":
                    card.get(
                        "bbox_pdf"
                    ),

                "source_contexts":
                    card.get(
                        "source_contexts",
                        [],
                    )[:2],
            },
        }

        evidence.append(
            bind_evidence(
                runtime,
                item,
            )
        )

    return evidence


# ============================================================
# 9. Visual Retrieval
# ============================================================

def encode_siglip_query(
    query,
):
    processor, model = (
        get_siglip(
            SIGLIP_QUERY_DEVICE
        )
    )

    inputs = processor(
        text=[query],
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    text_inputs = {
        k:
            v.to(
                SIGLIP_QUERY_DEVICE
            )

        for k, v
        in inputs.items()

        if k in {
            "input_ids",
            "attention_mask",
        }
    }

    with torch.inference_mode():

        feats = (
            model.get_text_features(
                **text_inputs
            )
        )

        if hasattr(
            feats,
            "pooler_output",
        ):
            feats = (
                feats.pooler_output
            )

        feats = (
            torch.nn.functional.normalize(
                feats.float(),
                dim=-1,
            )
        )

    return (
        feats.cpu()
        .numpy()[0]
        .astype("float32")
    )


def retrieve_visual_doc(
    runtime,
    query,
    top_k=5,
):
    records = runtime[
        "visuals"
    ]

    if not records:
        return []

    exact = (
        EXACT_VISUAL_RE.search(
            query
        )
    )

    if exact:

        ids = runtime[
            "visual_reference_index"
        ].get(
            exact.group(1).lower(),
            [],
        )

        ranked = [
            {
                "id":
                    visual_id,

                "rank": i,

                "score":
                    1.0,
            }

            for i, visual_id
            in enumerate(
                ids,
                1,
            )
        ]

    else:

        dense = dense_hits(
            runtime[
                "visual_index"
            ],

            records,

            query,

            "visual_id",

            VISUAL_CFG[
                "candidate_k"
            ],
        )

        lexical = bm25_hits(
            runtime[
                "visual_bm"
            ],

            runtime[
                "visual_tok"
            ],

            records,

            query,

            "visual_id",

            VISUAL_CFG[
                "candidate_k"
            ],
        )

        lanes = {
            "qwen_source":
                dense,

            "bm25_source":
                lexical,
        }

        weights = (
            VISUAL_CFG[
                "semantic_weights"
            ]
        )

        # SigLIP is only activated when the query itself
        # has explicit visual language. For normal queries
        # the gallery still works using semantic/lexical
        # visual metadata retrieval.
        if (
            VISUAL_CUE_RE.search(
                query
            )

            and runtime[
                "visual_view_vectors"
            ].size
        ):
            q = (
                encode_siglip_query(
                    query
                )
            )

            scores = (
                runtime[
                    "visual_view_vectors"
                ]
                @ q
            )

            grouped = defaultdict(
                list
            )

            for i, meta in enumerate(
                runtime[
                    "visual_view_meta"
                ]
            ):
                grouped[
                    meta[
                        "visual_id"
                    ]
                ].append(i)

            rows = []

            for (
                visual_id,
                idxs,
            ) in grouped.items():

                best = max(
                    idxs,
                    key=lambda i:
                        scores[i],
                )

                rows.append(
                    (
                        visual_id,
                        float(
                            scores[best]
                        ),
                    )
                )

            rows.sort(
                key=lambda x: (
                    -x[1],
                    x[0],
                )
            )

            lanes[
                "siglip_multiview"
            ] = [
                {
                    "id":
                        visual_id,

                    "rank":
                        i,

                    "score":
                        score,
                }

                for i, (
                    visual_id,
                    score,
                )
                in enumerate(
                    rows[
                        :VISUAL_CFG[
                            "candidate_k"
                        ]
                    ],
                    1,
                )
            ]

            weights = (
                VISUAL_CFG[
                    "visual_weights"
                ]
            )

        ranked = weighted_rrf(
            lanes,
            weights,

            VISUAL_CFG[
                "rrf_k"
            ],

            VISUAL_CFG[
                "candidate_k"
            ],
        )

        pool = ranked[
            :VISUAL_CFG[
                "rerank_pool"
            ]
        ]

        if pool:

            reranker = (
                get_reranker()
            )

            pairs = []

            for item in pool:

                row = runtime[
                    "visual_by_id"
                ][
                    item[
                        "id"
                    ]
                ]

                text = "\n".join(
                    x

                    for x in (
                        row.get(
                            "label"
                        )
                        or "",

                        row.get(
                            "caption"
                        )
                        or "",

                        row.get(
                            "source_text"
                        )
                        or "",

                        row.get(
                            "visible_text"
                        )
                        or "",
                    )

                    if x
                )

                pairs.append(
                    (
                        query,
                        text,
                    )
                )

            scores = np.asarray(
                reranker.predict(
                    pairs,

                    prompt_name="research",

                    batch_size=4,

                    show_progress_bar=False,

                    convert_to_numpy=True,
                )
            ).reshape(-1)

            reranked = [
                dict(
                    item,

                    relevance_score=
                        float(
                            score
                        ),
                )

                for item, score
                in zip(
                    pool,
                    scores,
                )
            ]

            reranked.sort(
                key=lambda x: (
                    -x[
                        "relevance_score"
                    ],

                    x[
                        "rank"
                    ],
                )
            )

            reranked_ids = {
                x["id"]
                for x in reranked
            }

            ranked = (
                reranked

                + [
                    x
                    for x in ranked
                    if x[
                        "id"
                    ] not in reranked_ids
                ]
            )

    selected = []
    used_groups = set()

    for item in ranked:

        row = runtime[
            "visual_by_id"
        ][
            item[
                "id"
            ]
        ]

        group = (
            row.get(
                "visual_group_id"
            )

            or row[
                "visual_id"
            ]
        )

        if group in used_groups:
            continue

        used_groups.add(
            group
        )

        evidence = {
            "source_object_id":
                row[
                    "visual_id"
                ],

            "pdf_page_number":
                row[
                    "pdf_page_number"
                ],

            "modality":
                "visual",

            "content": {
                "label":
                    row.get(
                        "label"
                    ),

                "caption":
                    row.get(
                        "caption"
                    ),

                "visual_type":
                    row.get(
                        "visual_type"
                    ),

                "crop_path":
                    row.get(
                        "crop_path"
                    ),

                "visible_text":
                    row.get(
                        "visible_text"
                    ),
            },

            "provenance": {
                "bbox_pdf":
                    row.get(
                        "bbox_pdf"
                    ),

                "crop_sha256":
                    row.get(
                        "crop_sha256"
                    ),
            },
        }

        selected.append(
            bind_evidence(
                runtime,

                evidence,

                item.get(
                    "relevance_score",

                    item.get(
                        "score",
                        0.0,
                    ),
                ),
            )
        )

        if (
            len(selected)
            >= top_k
        ):
            break

    return selected


# ============================================================
# 10. Table Retrieval
# ============================================================

def hybrid_runtime_search(
    query,
    records,
    dense_index,
    bm,
    tok,
    id_field,
    k,
):
    dense = dense_hits(
        dense_index,
        records,
        query,
        id_field,
        k,
    )

    lexical = bm25_hits(
        bm,
        tok,
        records,
        query,
        id_field,
        k,
    )

    return weighted_rrf(
        {
            "dense":
                dense,

            "bm25":
                lexical,
        },

        {
            "dense":
                1.0,

            "bm25":
                1.0,
        },

        TABLE_CFG[
            "rrf_k"
        ],

        k,
    )


def retrieve_table_doc(
    runtime,
    query,
    top_k=3,
    page_constraint=None,
):
    tables = runtime[
        "tables"
    ]

    if not tables:
        return []

    explicit = (
        TABLE_REFERENCE_RE.search(
            query
        )
    )

    if explicit:

        target = (
            f"table:"
            f"{explicit.group(1)}"
        ).lower()

        selected_tables = [
            table

            for table in tables

            if (
                table.get(
                    "label"
                )
                or ""
            ).lower() == target
        ]

    elif page_constraint:

        selected_tables = [
            table

            for table in tables

            if table[
                "pdf_page_number"
            ] == page_constraint
        ]

    else:

        ranked = (
            hybrid_runtime_search(
                query,
                tables,

                runtime[
                    "table_index"
                ],

                runtime[
                    "table_bm"
                ],

                runtime[
                    "table_tok"
                ],

                "table_id",

                TABLE_CFG[
                    "table_k"
                ],
            )
        )

        selected_tables = [
            runtime[
                "table_by_id"
            ][
                x[
                    "id"
                ]
            ]

            for x in ranked
        ]

    row_ranked = (
        hybrid_runtime_search(
            query,

            runtime[
                "table_rows"
            ],

            runtime[
                "row_index"
            ],

            runtime[
                "row_bm"
            ],

            runtime[
                "row_tok"
            ],

            "row_id",

            TABLE_CFG[
                "row_k"
            ],
        )
    )

    row_by_table = defaultdict(
        list
    )

    for item in row_ranked:

        row = runtime[
            "table_row_by_id"
        ][
            item[
                "id"
            ]
        ]

        row_by_table[
            row[
                "table_id"
            ]
        ].append(
            row
        )

    evidence = []

    for table in selected_tables:

        rows = (
            row_by_table.get(
                table[
                    "table_id"
                ],
                [],
            )[:3]
        )

        if not rows:

            rows = [
                runtime[
                    "table_row_by_id"
                ][row_id]

                for row_id in
                table[
                    "row_ids"
                ][:3]

                if row_id
                in runtime[
                    "table_row_by_id"
                ]
            ]

        row_numbers = {
            row[
                "row"
            ]
            for row in rows
        }

        cells = [
            cell

            for cell in runtime[
                "table_cells"
            ]

            if (
                cell[
                    "table_id"
                ]
                == table[
                    "table_id"
                ]

                and cell[
                    "row"
                ]
                in row_numbers
            )
        ]

        item = {
            "source_object_id":
                table[
                    "table_id"
                ],

            "pdf_page_number":
                table[
                    "pdf_page_number"
                ],

            "modality":
                "table",

            "content": {
                "label":
                    table.get(
                        "label"
                    ),

                "caption":
                    table.get(
                        "caption"
                    ),

                "rows":
                    rows,

                "selected_cells": [
                    {
                        "cell_id":
                            cell[
                                "cell_id"
                            ],

                        "row":
                            cell[
                                "row"
                            ],

                        "col":
                            cell[
                                "col"
                            ],

                        "header":
                            cell.get(
                                "header"
                            ),

                        "raw_value":
                            cell[
                                "text"
                            ],

                        "typed_value":
                            cell.get(
                                "typed_value"
                            ),
                    }

                    for cell
                    in cells
                ],
            },

            "provenance": {
                "bbox_pdf":
                    table.get(
                        "bbox_pdf"
                    ),

                "reconstruction":
                    table.get(
                        "reconstruction"
                    ),
            },
        }

        evidence.append(
            bind_evidence(
                runtime,
                item,
            )
        )

        if (
            len(evidence)
            >= top_k
        ):
            break

    return evidence


# ============================================================
# 11. Global Reranking
# ============================================================

def evidence_text(item):
    content = (
        item.get(
            "content",
            {},
        )
    )

    modality = (
        item[
            "modality"
        ]
    )

    if modality == "text":

        body = content.get(
            "text",
            "",
        )

    elif modality == "math":

        body = "\n".join(
            [
                content.get(
                    "equation_label"
                )
                or "",

                content.get(
                    "latex"
                )
                or "",

                " ".join(
                    item.get(
                        "provenance",
                        {},
                    ).get(
                        "source_contexts",
                        [],
                    )
                ),
            ]
        )

    elif modality == "visual":

        body = "\n".join(
            x

            for x in (
                content.get(
                    "label"
                )
                or "",

                content.get(
                    "caption"
                )
                or "",

                content.get(
                    "visible_text"
                )
                or "",
            )

            if x
        )

    elif modality == "table":

        rows = "\n".join(
            row.get(
                "text",
                "",
            )

            for row in content.get(
                "rows",
                [],
            )
        )

        body = "\n".join(
            x

            for x in (
                content.get(
                    "label"
                )
                or "",

                content.get(
                    "caption"
                )
                or "",

                rows,
            )

            if x
        )

    else:

        body = str(
            content
        )

    return (
        f"Source: "
        f"{item.get('source_title')}\n"

        f"Page: "
        f"{item.get('pdf_page_number')}\n"

        f"Modality: "
        f"{item.get('modality')}\n"

        f"{body}"
    )


def global_rerank(
    query,
    evidence,
    top_k=RAGSCHOLAR_MAX_EVIDENCE,
):
    if not evidence:
        return []

    unique = {}

    for item in evidence:

        unique[
            (
                item[
                    "document_id"
                ],

                item[
                    "modality"
                ],

                item[
                    "source_object_id"
                ],

                item[
                    "pdf_page_number"
                ],
            )
        ] = item

    rows = list(
        unique.values()
    )[
        :GLOBAL_RERANK_POOL
    ]

    reranker = get_reranker()

    scores = np.asarray(
        reranker.predict(
            [
                (
                    query,
                    evidence_text(
                        item
                    ),
                )

                for item
                in rows
            ],

            prompt_name=
                "research",

            batch_size=4,

            show_progress_bar=False,

            convert_to_numpy=True,
        )
    ).reshape(-1)

    ranked = [
        dict(
            item,

            global_score=
                float(
                    score
                ),
        )

        for item, score
        in zip(
            rows,
            scores,
        )
    ]

    ranked.sort(
        key=lambda x:
            -x[
                "global_score"
            ]
    )

    requested = (
        detect_modalities(
            query
        )
    )

    selected = []
    used = set()

    for modality in requested:

        match = next(
            (
                item

                for item in ranked

                if (
                    item[
                        "modality"
                    ] == modality

                    and item[
                        "evidence_id"
                    ] not in used
                )
            ),

            None,
        )

        if match:

            selected.append(
                match
            )

            used.add(
                match[
                    "evidence_id"
                ]
            )

    for item in ranked:

        if (
            item[
                "evidence_id"
            ]
            in used
        ):
            continue

        selected.append(
            item
        )

        used.add(
            item[
                "evidence_id"
            ]
        )

        if (
            len(selected)
            >= top_k
        ):
            break

    return selected[
        :top_k
    ]


# ============================================================
# 12. Visual Gallery Global Ranking
# ============================================================

def rank_visual_results(
    query,
    evidence,
    top_k=RAGSCHOLAR_MAX_VISUAL_RESULTS,
):
    if not evidence:
        return []

    unique = {}

    for item in evidence:

        key = (
            item.get(
                "document_id"
            ),

            item.get(
                "source_object_id"
            ),
        )

        current = (
            unique.get(
                key
            )
        )

        if current is None:

            unique[
                key
            ] = item

            continue

        old_score = float(
            current.get(
                "retrieval_score",
                0.0,
            )
        )

        new_score = float(
            item.get(
                "retrieval_score",
                0.0,
            )
        )

        if new_score > old_score:

            unique[
                key
            ] = item

    candidates = list(
        unique.values()
    )

    if len(candidates) <= 1:

        return candidates[
            :top_k
        ]

    reranker = get_reranker()

    scores = np.asarray(
        reranker.predict(
            [
                (
                    query,
                    evidence_text(
                        item
                    ),
                )

                for item
                in candidates
            ],

            prompt_name=
                "research",

            batch_size=4,

            show_progress_bar=False,

            convert_to_numpy=True,
        )
    ).reshape(-1)

    ranked = [
        dict(
            item,

            visual_gallery_score=
                float(
                    score
                ),
        )

        for item, score
        in zip(
            candidates,
            scores,
        )
    ]

    ranked.sort(
        key=lambda x: (
            -x[
                "visual_gallery_score"
            ],

            -float(
                x.get(
                    "retrieval_score",
                    0.0,
                )
            ),
        )
    )

    return ranked[
        :top_k
    ]


# ============================================================
# 13. Corpus Retrieval
# ============================================================

def retrieve_corpus(
    query,
    top_docs=RAGSCHOLAR_TOP_DOCS,
    top_evidence=RAGSCHOLAR_MAX_EVIDENCE,
):
    plan = plan_query(
        query
    )

    if (
        plan[
            "status"
        ]
        != "plan_ready"
    ):
        return {
            "query":
                query,

            "status":
                plan[
                    "status"
                ],

            "documents":
                [],

            "modalities":
                plan.get(
                    "modalities",
                    [],
                ),

            "evidence":
                [],

            "visual_evidence":
                [],
        }

    routed = (
        route_documents(
            query,
            top_docs,
        )
    )

    collected = []

    # IMPORTANT:
    # visual_collected is independent from the answer modalities.
    # Therefore ANY query can return a visual gallery if the routed
    # documents contain relevant figures.
    visual_collected = []

    for doc in routed:

        runtime = (
            load_document_runtime(
                doc[
                    "document_id"
                ]
            )
        )

        page_constraint = (
            plan[
                "pdf_pages"
            ][0]

            if (
                len(
                    plan[
                        "pdf_pages"
                    ]
                )
                == 1
            )

            else None
        )

        # ----------------------------------------------------
        # Always search visual records for gallery rendering.
        # This does NOT automatically feed visual evidence into
        # the answer. It is purely the direct-image result strip.
        # ----------------------------------------------------

        visual_hits = []

        if runtime.get(
            "visuals"
        ):
            try:

                visual_hits = (
                    retrieve_visual_doc(
                        runtime,
                        query,

                        top_k=max(
                            RAGSCHOLAR_MAX_VISUAL_RESULTS,
                            VISUAL_CFG[
                                "evidence_k"
                            ],
                        ),
                    )
                )

                visual_collected.extend(
                    visual_hits
                )

            except Exception as exc:

                print(
                    "[RAGScholar] "
                    f"Visual gallery retrieval failed for "
                    f"{doc['document_id']}: {exc}",
                    file=sys.stderr,
                )

        # ----------------------------------------------------
        # Main answer retrieval follows query modalities.
        # ----------------------------------------------------

        for modality in (
            plan[
                "modalities"
            ]
        ):

            if modality == "text":

                collected.extend(
                    retrieve_text_doc(
                        runtime,
                        query,
                    )
                )

            elif modality == "math":

                collected.extend(
                    retrieve_math_doc(
                        runtime,
                        query,
                    )
                )

            elif modality == "visual":

                # Reuse the already-computed gallery hits,
                # preserving normal answer evidence budget.
                collected.extend(
                    visual_hits[
                        :VISUAL_CFG[
                            "evidence_k"
                        ]
                    ]
                )

            elif modality == "table":

                collected.extend(
                    retrieve_table_doc(
                        runtime,
                        query,

                        page_constraint=
                            page_constraint,
                    )
                )

    evidence = global_rerank(
        query,
        collected,
        top_evidence,
    )

    # Visual results are globally reranked across all routed docs,
    # regardless of the user's primary query modality.
    visual_evidence = (
        rank_visual_results(
            query,

            visual_collected,

            RAGSCHOLAR_MAX_VISUAL_RESULTS,
        )
    )

    return {
        "query":
            query,

        "status":
            (
                "provisional_evidence"

                if evidence

                else "no_evidence"
            ),

        "documents": [
            {
                "document_id":
                    doc[
                        "document_id"
                    ],

                "title":
                    doc.get(
                        "title"
                    ),

                "category":
                    doc.get(
                        "category"
                    ),

                "router_score":
                    doc.get(
                        "router_score"
                    ),
            }

            for doc
            in routed
        ],

        "modalities":
            plan[
                "modalities"
            ],

        "evidence":
            evidence,

        "visual_evidence":
            visual_evidence,
    }


# ============================================================
# 14. Source Metadata
# ============================================================

def get_manifest(
    document_id,
):
    path = (
        DOCUMENTS_ROOT
        / document_id
        / "manifest.json"
    )

    if not path.is_file():
        return {}

    try:

        return json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

    except Exception:

        return {}


def display_title(item):
    manifest = get_manifest(
        item.get(
            "document_id"
        )
    )

    source = manifest.get(
        "source",
        {},
    )

    meta = (
        source.get(
            "pdf_metadata"
        )
        or {}
    )

    choices = [
        item.get(
            "source_title"
        ),

        manifest.get(
            "title"
        ),

        meta.get(
            "title"
        ),

        Path(
            source.get(
                "filename"
            )
            or ""
        ).stem,
    ]

    router = str(
        manifest.get(
            "router_text"
        )
        or ""
    ).splitlines()

    if router:

        choices.insert(
            0,
            router[
                0
            ].strip(),
        )

    internal = re.compile(
        r"(?:landmark|survey|textbooks?|paper|research|thesis)[_\-\s]*\d+",
        re.I,
    )

    for value in choices:

        value = clean_text(
            value
        )

        if (
            value
            and len(value) > 3

            and not internal.fullmatch(
                value
            )
        ):
            return value

    return clean_text(
        item.get(
            "source_title"
        )

        or manifest.get(
            "title"
        )

        or "Academic source"
    )


def source_author(item):
    manifest = get_manifest(
        item.get(
            "document_id"
        )
    )

    meta = (
        manifest.get(
            "source",
            {},
        ).get(
            "pdf_metadata"
        )
        or {}
    )

    return clean_text(
        meta.get(
            "author"
        )
    )[:160]


def source_excerpt(item):
    content = (
        item.get(
            "content"
        )
        or {}
    )

    modality = (
        item.get(
            "modality"
        )
    )

    if modality == "text":

        return short_text(
            content.get(
                "text"
            ),
            3200,
        )

    if modality == "math":

        context = " ".join(
            (
                item.get(
                    "provenance"
                )
                or {}
            ).get(
                "source_contexts",
                [],
            )[:2]
        )

        return short_text(
            context,
            2400,
        )

    if modality == "visual":

        return short_text(
            " ".join(
                x

                for x in (
                    content.get(
                        "caption"
                    )
                    or "",

                    content.get(
                        "visible_text"
                    )
                    or "",
                )

                if x
            ),

            2400,
        )

    if modality == "table":

        rows = "\n".join(
            str(
                row.get(
                    "text"
                )
                or ""
            )

            for row in (
                content.get(
                    "rows"
                )
                or []
            )[:6]

            if isinstance(
                row,
                dict,
            )
        )

        return short_text(
            rows,
            2600,
        )

    return short_text(
        str(
            content
        ),
        2400,
    )


def source_record(
    item,
    ordinal,
):
    content = (
        item.get(
            "content"
        )
        or {}
    )

    modality = (
        item.get(
            "modality"
        )
    )

    section = (
        content.get(
            "section_path"
        )

        or item.get(
            "section_path"
        )

        or []
    )

    document_id = (
        item.get(
            "document_id"
        )
    )

    object_id = (
        item.get(
            "source_object_id"
        )
    )

    asset_url = None

    if (
        modality == "visual"
        and document_id
        and object_id
    ):
        asset_url = (
            "/api/asset/"
            + quote(
                str(
                    document_id
                ),
                safe="",
            )
            + "/"
            + quote(
                str(
                    object_id
                ),
                safe="",
            )
        )

    page = item.get(
        "pdf_page_number"
    )

    if (
        document_id
        and page
    ):
        page_url = (
            "/api/page/"
            + quote(
                str(
                    document_id
                ),
                safe="",
            )
            + "/"
            + str(
                int(
                    page
                )
            )
        )

    else:
        page_url = None

    return {
        "number":
            ordinal,

        "evidence_id":
            f"E{ordinal}",

        "original_evidence_id":
            item.get(
                "evidence_id"
            ),

        "document_id":
            document_id,

        "source_object_id":
            object_id,

        "title":
            display_title(
                item
            ),

        "author":
            source_author(
                item
            ),

        "category":
            item.get(
                "source_category"
            )
            or "Academic source",

        "page":
            page,

        "modality":
            modality,

        "section":
            (
                " > ".join(
                    section
                )

                if isinstance(
                    section,
                    list,
                )

                else clean_text(
                    section
                )
            ),

        "excerpt":
            source_excerpt(
                item
            ),

        "latex":
            (
                content.get(
                    "latex"
                )

                if modality
                == "math"

                else None
            ),

        "label":
            (
                content.get(
                    "label"
                )

                or content.get(
                    "equation_label"
                )
            ),

        "caption":
            content.get(
                "caption"
            ),

        "asset_url":
            asset_url,

        "page_url":
            page_url,

        "rows":
            (
                content.get(
                    "rows",
                    [],
                )[:6]

                if modality
                == "table"

                else []
            ),

        "cells":
            (
                content.get(
                    "selected_cells",
                    [],
                )[:24]

                if modality
                == "table"

                else []
            ),

        "score":
            item.get(
                "visual_gallery_score",

                item.get(
                    "global_score",

                    item.get(
                        "retrieval_score"
                    ),
                ),
            ),
    }


def resolve_pdf_path(
    document_id,
):
    manifest = get_manifest(
        document_id
    )

    source = manifest.get(
        "source",
        {},
    )

    raw = source.get(
        "source_path"
    )

    if raw:

        path = Path(
            raw
        )

        if path.is_file():
            return path

    filename = source.get(
        "filename"
    )

    category = (
        manifest.get(
            "category"
        )

        or source.get(
            "category"
        )
    )

    if (
        filename
        and category
    ):
        candidate = (
            PROJECT_ROOT
            / "data"
            / str(category)
            / str(filename)
        )

        if candidate.is_file():
            return candidate

    if filename:

        matches = list(
            (
                PROJECT_ROOT
                / "data"
            ).rglob(
                str(filename)
            )
        )

        if matches:
            return matches[
                0
            ]

    return None


def resolve_visual_path(
    document_id,
    visual_id,
):
    root = (
        DOCUMENTS_ROOT
        / document_id
    )

    records = read_jsonl(
        root
        / "visual"
        / "records.jsonl"
    )

    row = next(
        (
            record

            for record
            in records

            if str(
                record.get(
                    "visual_id"
                )
            )
            == str(
                visual_id
            )
        ),

        None,
    )

    if not row:
        return None

    raw = row.get(
        "crop_path"
    )

    if raw:

        path = Path(
            raw
        )

        if path.is_file():
            return path

        fallback = (
            root
            / "visual"
            / "crops"
            / path.name
        )

        if fallback.is_file():
            return fallback

    return None


# ============================================================
# 15. GPT-OSS / Groq
# ============================================================

STYLE_RULES = {
    "concise":
        "Answer directly and compactly. "
        "Prefer roughly 150-350 words unless more is required.",

    "balanced":
        "Give a clear and well-structured academic explanation "
        "with enough detail to understand the mechanism.",

    "detailed":
        "Give a thorough academic explanation. "
        "Explain mechanisms and relationships clearly without filler.",
}


SYSTEM_PROMPT = r"""
You are RAGScholar, a source-grounded academic assistant.

Use ONLY the supplied evidence for factual claims.
Do not silently add factual claims from pretrained knowledge.

If evidence is incomplete, say so clearly.
If sources disagree, explain the disagreement.

Formatting:
- Write polished Markdown inside the answer field.
- Use short paragraphs.
- Use headings when helpful.
- Use bullet or numbered lists when helpful.
- Inline math must use \(...\).
- Display math must use $$...$$.
- Preserve retrieved equations faithfully.
- Cite factual claims using evidence IDs exactly like [E1], [E2].
- Never invent citations.
- Never invent source titles.
- Never invent page numbers.
- Never invent equations.
- Never invent table values.
- Never invent visual details.
- NEVER use Markdown image syntax.
- The UI renders retrieved figures separately.

Return ONLY valid JSON:

{
  "answer": "Markdown answer",
  "citations": ["E1", "E2"],
  "confidence": "high|medium|low",
  "status": "answered|insufficient_evidence"
}
"""


def evidence_packet(
    item,
    idx,
):
    content = (
        item.get(
            "content"
        )
        or {}
    )

    modality = (
        item.get(
            "modality"
        )
    )

    section = (
        content.get(
            "section_path"
        )

        or item.get(
            "section_path"
        )

        or []
    )

    base = {
        "evidence_id":
            f"E{idx}",

        "document":
            display_title(
                item
            ),

        "category":
            item.get(
                "source_category"
            ),

        "page":
            item.get(
                "pdf_page_number"
            ),

        "modality":
            modality,

        "section":
            (
                " > ".join(
                    section
                )

                if isinstance(
                    section,
                    list,
                )

                else str(
                    section
                    or ""
                )
            ),
    }

    if modality == "text":

        base[
            "content"
        ] = short_text(
            content.get(
                "text"
            ),
            2600,
        )

    elif modality == "math":

        base[
            "content"
        ] = {
            "label":
                content.get(
                    "equation_label"
                ),

            "latex":
                content.get(
                    "latex"
                ),

            "context": [
                short_text(
                    x,
                    800,
                )

                for x in (
                    item.get(
                        "provenance"
                    )
                    or {}
                ).get(
                    "source_contexts",
                    [],
                )[:2]
            ],
        }

    elif modality == "visual":

        base[
            "content"
        ] = {
            "label":
                content.get(
                    "label"
                ),

            "caption":
                short_text(
                    content.get(
                        "caption"
                    ),
                    1200,
                ),

            "visible_text":
                short_text(
                    content.get(
                        "visible_text"
                    ),
                    900,
                ),
        }

    elif modality == "table":

        base[
            "content"
        ] = {
            "label":
                content.get(
                    "label"
                ),

            "caption":
                short_text(
                    content.get(
                        "caption"
                    ),
                    800,
                ),

            "rows": [
                short_text(
                    row.get(
                        "text"
                    ),
                    650,
                )

                for row in (
                    content.get(
                        "rows"
                    )
                    or []
                )[:6]

                if isinstance(
                    row,
                    dict,
                )
            ],

            "cells":
                (
                    content.get(
                        "selected_cells"
                    )
                    or []
                )[:24],
        }

    return base


def fit_evidence_packet(
    evidence,
):
    packet = []
    used = 0

    for i, item in enumerate(
        evidence,
        1,
    ):

        row = evidence_packet(
            item,
            i,
        )

        serialized = json.dumps(
            row,
            ensure_ascii=False,
        )

        if (
            packet
            and (
                used
                + len(serialized)
            )
            > RAGSCHOLAR_EVIDENCE_CHARS
        ):
            break

        packet.append(
            row
        )

        used += len(
            serialized
        )

    return packet


def parse_json_result(text):
    text = str(
        text
        or ""
    ).strip()

    text = re.sub(
        r"^```(?:json)?\s*|\s*```$",
        "",
        text,
        flags=re.I | re.S,
    )

    try:

        return json.loads(
            text
        )

    except Exception:
        pass

    match = re.search(
        r"\{[\s\S]*\}",
        text,
    )

    if match:

        try:

            return json.loads(
                match.group(
                    0
                )
            )

        except Exception:
            pass

    return {
        "answer":
            text,

        "citations":
            [],

        "confidence":
            "medium",

        "status":
            "answered",
    }


def groq_models():
    fallback = [
        {
            "id":
                GROQ_MODEL,

            "label":
                (
                    "GPT-OSS 120B"

                    if (
                        "gpt-oss-120b"
                        in GROQ_MODEL
                    )

                    else GROQ_MODEL
                ),

            "description":
                "Configured Groq model",

            "available":
                True,
        }
    ]

    if not GROQ_API_KEY:
        return fallback

    try:

        response = requests.get(
            f"{GROQ_API}/models",

            headers={
                "Authorization":
                    f"Bearer {GROQ_API_KEY}"
            },

            timeout=12,
        )

        response.raise_for_status()

        ids = sorted(
            {
                item[
                    "id"
                ]

                for item in
                response.json().get(
                    "data",
                    [],
                )

                if item.get(
                    "id"
                )
            }
        )

        preferred = []

        def add_match(
            needle,
            label,
            description,
        ):
            match = next(
                (
                    model_id

                    for model_id
                    in ids

                    if needle
                    in model_id.lower()
                ),

                None,
            )

            if (
                match
                and all(
                    item[
                        "id"
                    ]
                    != match

                    for item
                    in preferred
                )
            ):
                preferred.append(
                    {
                        "id":
                            match,

                        "label":
                            label(
                                match
                            ),

                        "description":
                            description,

                        "available":
                            True,
                    }
                )

        add_match(
            "gpt-oss-120b",

            lambda _:
                "GPT-OSS 120B",

            "Academic synthesis and reasoning",
        )

        add_match(
            "qwen",

            lambda x:
                "Qwen · "
                + x.split(
                    "/"
                )[-1],

            "Technical reasoning alternative",
        )

        add_match(
            "deepseek",

            lambda x:
                "DeepSeek · "
                + x.split(
                    "/"
                )[-1],

            "Reasoning alternative",
        )

        add_match(
            "llama",

            lambda x:
                "Llama · "
                + x.split(
                    "/"
                )[-1],

            "General-purpose alternative",
        )

        if (
            GROQ_MODEL in ids

            and all(
                item[
                    "id"
                ]
                != GROQ_MODEL

                for item
                in preferred
            )
        ):
            preferred.insert(
                0,

                {
                    "id":
                        GROQ_MODEL,

                    "label":
                        (
                            "GPT-OSS 120B"

                            if (
                                "gpt-oss-120b"
                                in GROQ_MODEL
                            )

                            else GROQ_MODEL
                        ),

                    "description":
                        "Configured default",

                    "available":
                        True,
                },
            )

        return (
            preferred[
                :4
            ]
            or fallback
        )

    except Exception:

        return fallback


def _groq_post(
    payload,
):
    return requests.post(
        f"{GROQ_API}/chat/completions",

        headers={
            "Authorization":
                f"Bearer {GROQ_API_KEY}",

            "Content-Type":
                "application/json",
        },

        json=payload,

        timeout=240,
    )


def _print_groq_error(
    response,
    payload,
    attempt,
):
    print(
        "\n"
        "========== GROQ REQUEST ERROR =========="
    )

    print(
        "Attempt:",
        attempt,
    )

    print(
        "Status:",
        response.status_code,
    )

    print(
        "Model:",
        payload.get(
            "model"
        ),
    )

    print(
        "Payload keys:",
        list(
            payload.keys()
        ),
    )

    print(
        "Response:",
        response.text,
    )

    print(
        "========================================\n"
    )


def groq_answer(
    query,
    evidence,
    model,
    style,
    reasoning,
    history,
):
    if not GROQ_API_KEY:

        raise RuntimeError(
            "GROQ_API_KEY is missing from .env."
        )

    selected_model = (
        model
        or GROQ_MODEL
    )

    packet = (
        fit_evidence_packet(
            evidence
        )
    )

    history_messages = []

    for row in (
        history
        or []
    )[-6:]:

        role = row.get(
            "role"
        )

        content = short_text(
            row.get(
                "content"
            ),
            1200,
        )

        if (
            role
            in {
                "user",
                "assistant",
            }
            and content
        ):
            history_messages.append(
                {
                    "role":
                        role,

                    "content":
                        content,
                }
            )

    prompt = (
        "QUESTION:\n"
        + query

        + "\n\nANSWER STYLE:\n"
        + STYLE_RULES.get(
            style,

            STYLE_RULES[
                "balanced"
            ],
        )

        + "\n\nEVIDENCE:\n"
        + json.dumps(
            packet,
            ensure_ascii=False,
            indent=2,
        )

        + "\n\nReturn only the requested JSON object."
    )

    messages = [
        {
            "role":
                "system",

            "content":
                SYSTEM_PROMPT,
        },

        *history_messages,

        {
            "role":
                "user",

            "content":
                prompt,
        },
    ]

    payload = {
        "model":
            selected_model,

        "messages":
            messages,

        "temperature":
            0.1,

        "max_completion_tokens":
            GROQ_MAX_OUTPUT_TOKENS,

        "response_format": {
            "type":
                "json_object"
        },
    }

    is_gpt_oss = (
        "gpt-oss"
        in selected_model.lower()
    )

    effort = (
        reasoning

        if reasoning
        in {
            "low",
            "medium",
            "high",
        }

        else GROQ_REASONING_EFFORT
    )

    # Groq GPT-OSS supports low / medium / high.
    # Suppress separate reasoning output because the web app
    # only needs the final JSON response.
    if is_gpt_oss:

        payload[
            "include_reasoning"
        ] = False

        if effort in {
            "low",
            "medium",
            "high",
        }:
            payload[
                "reasoning_effort"
            ] = effort

    # --------------------------------------------------------
    # Attempt 1:
    # Normal JSON-mode request.
    # --------------------------------------------------------

    response = _groq_post(
        payload
    )

    # --------------------------------------------------------
    # Attempt 2:
    # If Groq rejects reasoning-specific parameters,
    # remove them and retry.
    # --------------------------------------------------------

    if response.status_code == 400:

        _print_groq_error(
            response,
            payload,
            "1 / normal",
        )

        retry_payload = dict(
            payload
        )

        retry_payload.pop(
            "reasoning_effort",
            None,
        )

        retry_payload.pop(
            "include_reasoning",
            None,
        )

        response = _groq_post(
            retry_payload
        )

        payload = retry_payload

    # --------------------------------------------------------
    # Attempt 3:
    # If JSON mode itself is rejected by a selected alternative
    # model, remove response_format and rely on the prompt +
    # parse_json_result().
    # --------------------------------------------------------

    if response.status_code == 400:

        _print_groq_error(
            response,
            payload,
            "2 / without reasoning params",
        )

        retry_payload = dict(
            payload
        )

        retry_payload.pop(
            "response_format",
            None,
        )

        response = _groq_post(
            retry_payload
        )

        payload = retry_payload

    if not response.ok:

        _print_groq_error(
            response,
            payload,
            "final",
        )

        raise RuntimeError(
            f"Groq API {response.status_code}: "
            f"{response.text}"
        )

    response_data = (
        response.json()
    )

    try:

        content = (
            response_data[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ]
        )

    except Exception as exc:

        raise RuntimeError(
            "Groq returned an unexpected response: "
            f"{response_data}"
        ) from exc

    return parse_json_result(
        content
    )


def citation_order(
    answer,
    explicit,
    count,
):
    found = []

    for match in re.finditer(
        r"\bE(\d+)\b",
        str(
            answer
            or ""
        ),
    ):
        n = int(
            match.group(
                1
            )
        )

        if (
            1 <= n <= count
            and n not in found
        ):
            found.append(
                n
            )

    for value in (
        explicit
        or []
    ):

        match = re.search(
            r"E?(\d+)",
            str(
                value
            ),
        )

        if match:

            n = int(
                match.group(
                    1
                )
            )

            if (
                1 <= n <= count
                and n not in found
            ):
                found.append(
                    n
                )

    return found


def replace_citations(
    answer,
    order,
):
    mapping = {
        n:
            i + 1

        for i, n
        in enumerate(
            order
        )
    }

    def repl(match):

        n = int(
            match.group(
                1
            )
        )

        if n in mapping:

            return (
                f"[{mapping[n]}]"
            )

        return ""

    answer = re.sub(
        r"\[?E(\d+)\]?",
        repl,

        str(
            answer
            or ""
        ),
    )

    # Prevent accidental browser requests such as GET /E2
    answer = re.sub(
        r"!\[([^\]]*)\]\(\s*E?\d+\s*\)",
        r"\1",

        answer,

        flags=re.I,
    )

    return answer.strip()


def answer_query(
    query,
    model=None,
    style="balanced",
    reasoning="auto",
    history=None,
):
    query = clean_text(
        query
    )

    if not query:

        raise ValueError(
            "Question cannot be empty."
        )

    previous_user = next(
        (
            item.get(
                "content"
            )

            for item
            in reversed(
                history
                or []
            )

            if (
                item.get(
                    "role"
                )
                == "user"

                and item.get(
                    "content"
                )
            )
        ),

        None,
    )

    if (
        previous_user

        and len(
            query.split()
        )
        <= 6
    ):
        retrieval_query = (
            f"{previous_user}\n"
            f"Follow-up question: "
            f"{query}"
        )

    else:
        retrieval_query = query

    with RETRIEVAL_LOCK:

        retrieval = (
            retrieve_corpus(
                retrieval_query,

                top_docs=
                    RAGSCHOLAR_TOP_DOCS,

                top_evidence=
                    RAGSCHOLAR_MAX_EVIDENCE,
            )
        )

    # --------------------------------------------------------
    # Visual gallery is independent from answer citations.
    # This now works for ANY query.
    # --------------------------------------------------------

    visual_results = [
        source_record(
            item,
            i,
        )

        for i, item
        in enumerate(
            retrieval.get(
                "visual_evidence",
                [],
            ),
            1,
        )
    ]

    if (
        retrieval[
            "status"
        ]
        == "numeric_engine_required"
    ):
        return {
            "status":
                "numeric_engine_required",

            "answer":
                (
                    "This query requires a numerical table "
                    "computation engine. RAGScholar can retrieve "
                    "the relevant table evidence, but the current "
                    "runtime does not perform that calculation."
                ),

            "confidence":
                "high",

            "sources":
                [],

            "visual_results":
                visual_results,

            "modalities":
                retrieval.get(
                    "modalities",
                    [],
                ),

            "documents":
                retrieval.get(
                    "documents",
                    [],
                ),

            "model":
                model
                or GROQ_MODEL,
        }

    evidence = (
        retrieval.get(
            "evidence"
        )
        or []
    )

    if not evidence:

        return {
            "status":
                retrieval.get(
                    "status",
                    "no_evidence",
                ),

            "answer":
                (
                    "I couldn't find enough source-grounded "
                    "evidence in the indexed corpus to answer "
                    "this reliably."
                ),

            "confidence":
                "low",

            "sources":
                [],

            # Even when textual/math evidence is insufficient,
            # relevant retrieved source images can still appear.
            "visual_results":
                visual_results,

            "modalities":
                retrieval.get(
                    "modalities",
                    [],
                ),

            "documents":
                retrieval.get(
                    "documents",
                    [],
                ),

            "model":
                model
                or GROQ_MODEL,
        }

    generated = groq_answer(
        query=query,

        evidence=evidence,

        model=(
            model
            or GROQ_MODEL
        ),

        style=style,

        reasoning=reasoning,

        history=history,
    )

    order = citation_order(
        generated.get(
            "answer"
        ),

        generated.get(
            "citations"
        ),

        len(
            evidence
        ),
    )

    if not order:

        order = list(
            range(
                1,

                min(
                    4,
                    len(
                        evidence
                    ),
                )
                + 1,
            )
        )

    answer = (
        replace_citations(
            generated.get(
                "answer"
            ),

            order,
        )
    )

    sources = [
        source_record(
            evidence[
                n - 1
            ],

            i,
        )

        for i, n
        in enumerate(
            order,
            1,
        )
    ]

    return {
        "status":
            generated.get(
                "status",
                "answered",
            ),

        "answer":
            answer,

        "confidence":
            generated.get(
                "confidence",
                "medium",
            ),

        "sources":
            sources,

        "visual_results":
            visual_results,

        "modalities":
            retrieval.get(
                "modalities",
                [],
            ),

        "documents":
            retrieval.get(
                "documents",
                [],
            ),

        "model":
            model
            or GROQ_MODEL,
    }


# ============================================================
# 16. SQLite
# ============================================================

def utc_now():
    return datetime.now(
        timezone.utc
    ).isoformat()


def db_conn():
    conn = sqlite3.connect(
        DB_PATH
    )

    conn.row_factory = (
        sqlite3.Row
    )

    conn.execute(
        "PRAGMA foreign_keys = ON"
    )

    return conn


def init_db():
    with db_conn() as conn:

        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS conversations(
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources_json TEXT NOT NULL DEFAULT '[]',
                meta_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                FOREIGN KEY(conversation_id)
                    REFERENCES conversations(id)
                    ON DELETE CASCADE
            );
            """
        )


def create_conversation(
    title="New chat",
):
    conversation_id = (
        uuid.uuid4().hex
    )

    now = utc_now()

    with db_conn() as conn:

        conn.execute(
            """
            INSERT INTO conversations(
                id,
                title,
                created_at,
                updated_at
            )
            VALUES(?,?,?,?)
            """,

            (
                conversation_id,
                title,
                now,
                now,
            ),
        )

    return conversation_id


def save_message(
    conversation_id,
    role,
    content,
    sources=None,
    meta=None,
):
    now = utc_now()

    with db_conn() as conn:

        conn.execute(
            """
            INSERT INTO messages(
                conversation_id,
                role,
                content,
                sources_json,
                meta_json,
                created_at
            )
            VALUES(?,?,?,?,?,?)
            """,

            (
                conversation_id,

                role,

                content,

                json.dumps(
                    sources
                    or [],

                    ensure_ascii=False,
                ),

                json.dumps(
                    meta
                    or {},

                    ensure_ascii=False,
                ),

                now,
            ),
        )

        conn.execute(
            """
            UPDATE conversations
            SET updated_at=?
            WHERE id=?
            """,

            (
                now,
                conversation_id,
            ),
        )


def conversation_history(
    conversation_id,
):
    with db_conn() as conn:

        rows = conn.execute(
            """
            SELECT
                role,
                content

            FROM messages

            WHERE conversation_id=?

            ORDER BY id
            """,

            (
                conversation_id,
            ),
        ).fetchall()

    return [
        {
            "role":
                row[
                    "role"
                ],

            "content":
                row[
                    "content"
                ],
        }

        for row in rows
    ]


def maybe_title_conversation(
    conversation_id,
    query,
):
    title = clean_text(
        query
    )

    if len(title) > 56:

        title = (
            title[
                :56
            ].rstrip()
            + "…"
        )

    with db_conn() as conn:

        row = conn.execute(
            """
            SELECT title
            FROM conversations
            WHERE id=?
            """,

            (
                conversation_id,
            ),
        ).fetchone()

        if (
            row

            and row[
                "title"
            ]
            in {
                "New chat",
                "Untitled chat",
            }
        ):
            conn.execute(
                """
                UPDATE conversations
                SET title=?
                WHERE id=?
                """,

                (
                    title
                    or "New chat",

                    conversation_id,
                ),
            )


# ============================================================
# 17. Flask App
# ============================================================

app = Flask(
    __name__
)

app.config[
    "JSON_SORT_KEYS"
] = False

RETRIEVAL_LOCK = (
    RLock()
)

RUNTIME_READY = False
RUNTIME_ERROR = None


@app.get("/")
def index():
    return render_template(
        "index.html"
    )


@app.get("/api/health")
def health():
    return jsonify(
        {
            "ok":
                RUNTIME_READY,

            "runtime_ready":
                RUNTIME_READY,

            "runtime_error":
                RUNTIME_ERROR,

            "documents":
                len(
                    DOC_RECORDS
                ),

            "device":
                DEVICE,

            "model":
                GROQ_MODEL,
        }
    )


@app.get("/api/models")
def api_models():
    return jsonify(
        {
            "models":
                groq_models()
        }
    )


@app.get("/api/conversations")
def list_conversations():
    with db_conn() as conn:

        rows = conn.execute(
            """
            SELECT
                c.id,
                c.title,
                c.created_at,
                c.updated_at,

                (
                    SELECT COUNT(*)
                    FROM messages m
                    WHERE
                        m.conversation_id
                        = c.id
                ) AS message_count

            FROM conversations c

            ORDER BY
                c.updated_at DESC
            """
        ).fetchall()

    return jsonify(
        {
            "conversations": [
                dict(
                    row
                )
                for row
                in rows
            ]
        }
    )


@app.post("/api/conversations")
def new_conversation():
    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    title = (
        clean_text(
            data.get(
                "title"
            )
        )
        or "New chat"
    )

    conversation_id = (
        create_conversation(
            title
        )
    )

    return jsonify(
        {
            "id":
                conversation_id,

            "title":
                title,
        }
    )


@app.get(
    "/api/conversations/<conversation_id>"
)
def get_conversation(
    conversation_id,
):
    with db_conn() as conn:

        conversation = (
            conn.execute(
                """
                SELECT
                    id,
                    title,
                    created_at,
                    updated_at

                FROM conversations

                WHERE id=?
                """,

                (
                    conversation_id,
                ),
            ).fetchone()
        )

        if not conversation:
            abort(
                404
            )

        rows = conn.execute(
            """
            SELECT
                id,
                role,
                content,
                sources_json,
                meta_json,
                created_at

            FROM messages

            WHERE conversation_id=?

            ORDER BY id
            """,

            (
                conversation_id,
            ),
        ).fetchall()

    messages = []

    for row in rows:

        messages.append(
            {
                "id":
                    row[
                        "id"
                    ],

                "role":
                    row[
                        "role"
                    ],

                "content":
                    row[
                        "content"
                    ],

                "sources":
                    json.loads(
                        row[
                            "sources_json"
                        ]
                        or "[]"
                    ),

                "meta":
                    json.loads(
                        row[
                            "meta_json"
                        ]
                        or "{}"
                    ),

                "created_at":
                    row[
                        "created_at"
                    ],
            }
        )

    return jsonify(
        {
            "conversation":
                dict(
                    conversation
                ),

            "messages":
                messages,
        }
    )


@app.delete(
    "/api/conversations/<conversation_id>"
)
def delete_conversation(
    conversation_id,
):
    with db_conn() as conn:

        conn.execute(
            """
            DELETE FROM messages
            WHERE conversation_id=?
            """,

            (
                conversation_id,
            ),
        )

        conn.execute(
            """
            DELETE FROM conversations
            WHERE id=?
            """,

            (
                conversation_id,
            ),
        )

    return jsonify(
        {
            "ok":
                True,
        }
    )


@app.post("/api/chat")
def chat():
    if not RUNTIME_READY:

        return jsonify(
            {
                "error":
                    (
                        RUNTIME_ERROR

                        or "RAGScholar runtime is not ready."
                    )
            }
        ), 503

    data = (
        request.get_json(
            silent=True
        )
        or {}
    )

    query = clean_text(
        data.get(
            "query"
        )
    )

    if not query:

        return jsonify(
            {
                "error":
                    "Question cannot be empty."
            }
        ), 400

    conversation_id = (
        clean_text(
            data.get(
                "conversation_id"
            )
        )

        or create_conversation(
            "New chat"
        )
    )

    with db_conn() as conn:

        exists = conn.execute(
            """
            SELECT 1
            FROM conversations
            WHERE id=?
            """,

            (
                conversation_id,
            ),
        ).fetchone()

    if not exists:

        conversation_id = (
            create_conversation(
                "New chat"
            )
        )

    history = (
        conversation_history(
            conversation_id
        )
    )

    save_message(
        conversation_id,
        "user",
        query,
    )

    maybe_title_conversation(
        conversation_id,
        query,
    )

    try:

        result = answer_query(
            query=query,

            model=(
                clean_text(
                    data.get(
                        "model"
                    )
                )

                or GROQ_MODEL
            ),

            style=(
                clean_text(
                    data.get(
                        "style"
                    )
                )

                or "balanced"
            ),

            reasoning=(
                clean_text(
                    data.get(
                        "reasoning"
                    )
                )

                or "auto"
            ),

            history=history,
        )

    except Exception as exc:

        print(
            "\n[RAGScholar] CHAT ERROR:",
            repr(
                exc
            ),
            "\n",

            file=sys.stderr,
        )

        return jsonify(
            {
                "error":
                    str(
                        exc
                    ),

                "conversation_id":
                    conversation_id,
            }
        ), 500

    meta = {
        "status":
            result.get(
                "status"
            ),

        "confidence":
            result.get(
                "confidence"
            ),

        "model":
            result.get(
                "model"
            ),

        "modalities":
            result.get(
                "modalities",
                [],
            ),

        "documents":
            result.get(
                "documents",
                [],
            ),

        # Persist visual gallery so old conversations also
        # display images when reopened.
        "visual_results":
            result.get(
                "visual_results",
                [],
            ),
    }

    save_message(
        conversation_id,

        "assistant",

        result.get(
            "answer",
            "",
        ),

        sources=result.get(
            "sources",
            [],
        ),

        meta=meta,
    )

    return jsonify(
        {
            "conversation_id":
                conversation_id,

            **result,
        }
    )


@app.get(
    "/api/asset/<document_id>/<path:visual_id>"
)
def asset(
    document_id,
    visual_id,
):
    path = resolve_visual_path(
        document_id,
        visual_id,
    )

    if (
        not path
        or not path.is_file()
    ):
        abort(
            404
        )

    return send_file(
        path,
        conditional=True,
        max_age=3600,
    )


@app.get(
    "/api/page/<document_id>/<int:page_number>"
)
def page_preview(
    document_id,
    page_number,
):
    pdf_path = resolve_pdf_path(
        document_id
    )

    if (
        not pdf_path
        or not pdf_path.is_file()
    ):
        abort(
            404
        )

    pdf = pymupdf.open(
        pdf_path
    )

    try:

        if (
            page_number < 1

            or page_number
            > pdf.page_count
        ):
            abort(
                404
            )

        page = pdf.load_page(
            page_number
            - 1
        )

        pix = page.get_pixmap(
            dpi=145,
            alpha=False,
        )

        data = pix.tobytes(
            "png"
        )

    finally:

        pdf.close()

    return send_file(
        BytesIO(
            data
        ),

        mimetype=
            "image/png",

        max_age=3600,
    )


# ============================================================
# 18. Startup
# ============================================================

def initialize_runtime():
    global RUNTIME_READY
    global RUNTIME_ERROR

    try:

        init_db()

        init_models()

        init_corpus_router()

        RUNTIME_READY = True
        RUNTIME_ERROR = None

    except Exception as exc:

        RUNTIME_READY = False

        RUNTIME_ERROR = str(
            exc
        )

        print(
            "[RAGScholar] Runtime initialization failed:",
            exc,

            file=sys.stderr,
        )


initialize_runtime()


if __name__ == "__main__":

    print()
    print(
        "[RAGScholar] Project   :",
        PROJECT_ROOT,
    )

    print(
        "[RAGScholar] Artifacts :",
        OUTPUT_ROOT,
    )

    print(
        "[RAGScholar] Device    :",
        DEVICE,
    )

    print(
        "[RAGScholar] Groq      :",
        GROQ_MODEL,
    )

    print(
        "[RAGScholar] Documents :",
        len(
            DOC_RECORDS
        ),
    )

    print(
        "[RAGScholar] Visual max:",
        RAGSCHOLAR_MAX_VISUAL_RESULTS,
    )

    print(
        "[RAGScholar] URL       :",
        f"http://127.0.0.1:{PORT}",
    )

    print()

    app.run(
        host="127.0.0.1",
        port=PORT,
        debug=False,
        threaded=True,
    )