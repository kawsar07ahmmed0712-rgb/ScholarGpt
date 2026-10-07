from pathlib import Path
from threading import RLock
from urllib.parse import quote
import importlib
import json
import os
import re

import pymupdf
import requests
from dotenv import load_dotenv


APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

DOCUMENTS_ROOT = PROJECT_ROOT / "artifacts" / "ragscholar_full" / "documents"
GROQ_API = "https://api.groq.com/openai/v1"
GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip() or "openai/gpt-oss-120b"

_RETRIEVE_FN = None
_LOCK = RLock()


def configure_runtime(retrieve_fn=None, documents_root=None):
    global _RETRIEVE_FN, DOCUMENTS_ROOT
    if retrieve_fn is not None:
        _RETRIEVE_FN = retrieve_fn
    if documents_root is not None:
        DOCUMENTS_ROOT = Path(documents_root)


def autoload_runtime():
    global _RETRIEVE_FN
    if _RETRIEVE_FN is not None:
        return True

    names = [
        os.getenv("RAGSCHOLAR_RUNTIME_MODULE", "").strip(),
        "ragscholar_runtime",
        "runtime",
    ]
    for name in filter(None, names):
        try:
            module = importlib.import_module(name)
            fn = getattr(module, "retrieve_corpus", None)
            if callable(fn):
                configure_runtime(fn, getattr(module, "DOCUMENTS_ROOT", None))
                return True
        except Exception:
            continue
    return False


def corpus_count():
    if not DOCUMENTS_ROOT.exists():
        return 0
    return len(list(DOCUMENTS_ROOT.glob("*/manifest.json")))


def _manifest(document_id):
    if not document_id:
        return {}
    path = DOCUMENTS_ROOT / str(document_id) / "manifest.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _internal_title(title):
    return bool(
        re.fullmatch(
            r"(?:landmark|survey|textbooks?|paper|research|thesis)[_\-\s]*\d+",
            str(title or ""),
            re.I,
        )
    )


def _clean_title(value):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    value = re.sub(r"^(?:title|document)\s*:\s*", "", value, flags=re.I).strip()
    return value


def display_title(item):
    manifest = _manifest(item.get("document_id"))
    source = manifest.get("source", {})
    meta = source.get("pdf_metadata") or {}

    choices = [
        item.get("source_title"),
        manifest.get("title"),
        meta.get("title"),
        Path(source.get("filename") or "").stem,
    ]

    router_lines = str(manifest.get("router_text") or "").splitlines()
    if router_lines:
        choices.insert(0, router_lines[0])

    for value in choices:
        value = _clean_title(value)
        if value and len(value) > 3 and not _internal_title(value):
            return value

    return _clean_title(item.get("source_title") or manifest.get("title") or "Academic source")


def source_author(item):
    manifest = _manifest(item.get("document_id"))
    meta = manifest.get("source", {}).get("pdf_metadata") or {}
    author = re.sub(r"\s+", " ", str(meta.get("author") or "")).strip()
    return author[:160]


def clip(value, n):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value[:n] + ("…" if len(value) > n else "")


def evidence_body(item):
    content = item.get("content") or {}
    modality = item.get("modality")

    if modality == "text":
        return content.get("text") or ""

    if modality == "math":
        contexts = (item.get("provenance") or {}).get("source_contexts", [])[:2]
        return "\n\n".join(
            x for x in [content.get("equation_label"), content.get("latex"), *contexts] if x
        )

    if modality == "visual":
        return "\n\n".join(
            x
            for x in [
                content.get("label"),
                content.get("caption"),
                content.get("visible_text"),
                content.get("source_text"),
            ]
            if x
        )

    if modality == "table":
        rows = content.get("rows") or []
        row_text = "\n".join(
            str(row.get("text") or "") for row in rows[:8] if isinstance(row, dict)
        )
        return "\n\n".join(
            x for x in [content.get("label"), content.get("caption"), row_text] if x
        )

    return str(content)


def evidence_packet(item, idx):
    content = item.get("content") or {}
    modality = item.get("modality")
    section = content.get("section_path") or item.get("section_path") or []

    base = {
        "evidence_id": f"E{idx}",
        "document": display_title(item),
        "category": item.get("source_category"),
        "page": item.get("pdf_page_number"),
        "modality": modality,
        "section": " > ".join(section) if isinstance(section, list) else str(section or ""),
    }

    if modality == "text":
        base["content"] = clip(content.get("text"), 2200)

    elif modality == "math":
        base["content"] = {
            "label": content.get("equation_label"),
            "latex": content.get("latex"),
            "context": [
                clip(x, 650)
                for x in (item.get("provenance") or {}).get("source_contexts", [])[:2]
            ],
        }

    elif modality == "visual":
        base["content"] = {
            "label": content.get("label"),
            "caption": clip(content.get("caption"), 1100),
            "visible_text": clip(content.get("visible_text"), 800),
            "context": clip(content.get("source_text"), 1000),
        }

    elif modality == "table":
        base["content"] = {
            "label": content.get("label"),
            "caption": clip(content.get("caption"), 700),
            "rows": [
                clip(row.get("text"), 500)
                for row in (content.get("rows") or [])[:8]
                if isinstance(row, dict)
            ],
            "cells": (content.get("selected_cells") or [])[:24],
        }

    return base


def _safe_doc_root(document_id):
    if not document_id:
        return None
    root = (DOCUMENTS_ROOT / str(document_id)).resolve()
    documents_root = DOCUMENTS_ROOT.resolve()
    try:
        root.relative_to(documents_root)
    except ValueError:
        return None
    return root


def resolve_asset(document_id, relative_path):
    root = _safe_doc_root(document_id)
    if root is None or not relative_path:
        return None

    candidate = (root / str(relative_path)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None

    return candidate if candidate.is_file() else None


def _asset_url(document_id, crop_path):
    if not document_id or not crop_path:
        return None

    root = _safe_doc_root(document_id)
    if root is None:
        return None

    crop = Path(str(crop_path))
    if not crop.is_absolute():
        crop = root / crop

    try:
        crop = crop.resolve()
        rel = crop.relative_to(root)
    except Exception:
        return None

    if not crop.is_file():
        return None

    doc = quote(str(document_id), safe="")
    rel_url = quote(rel.as_posix(), safe="/")
    return f"/api/asset/{doc}/{rel_url}"


def _page_url(document_id, page):
    if not document_id or not page:
        return None
    try:
        page = int(page)
    except (TypeError, ValueError):
        return None
    if page < 1:
        return None
    return f"/api/page/{quote(str(document_id), safe='')}/{page}"


def page_png(document_id, page, dpi=145):
    manifest = _manifest(document_id)
    source = manifest.get("source", {})
    pdf_path = Path(source.get("source_path") or "")

    if not pdf_path.is_file():
        return None

    try:
        page = int(page)
    except (TypeError, ValueError):
        return None

    pdf = pymupdf.open(pdf_path)
    try:
        if page < 1 or page > pdf.page_count:
            return None
        pix = pdf[page - 1].get_pixmap(dpi=dpi, alpha=False)
        return pix.tobytes("png")
    finally:
        pdf.close()


def source_record(item, eid):
    content = item.get("content") or {}
    modality = item.get("modality") or "text"
    section = content.get("section_path") or item.get("section_path") or []
    document_id = item.get("document_id")
    page = item.get("pdf_page_number")
    source_object_id = item.get("source_object_id") or item.get("evidence_id") or eid

    crop_path = None
    if modality == "visual":
        crop_path = (
            content.get("crop_path")
            or content.get("asset_path")
            or item.get("crop_path")
            or item.get("asset_path")
        )

    asset_url = _asset_url(document_id, crop_path)
    page_url = _page_url(document_id, page)
    latex = content.get("latex") if modality == "math" else None
    caption = content.get("caption") if modality == "visual" else content.get("caption")

    return {
        "evidence_id": eid,
        "cite_id": eid,
        "source_key": f"{document_id}:{modality}:{source_object_id}",
        "document_id": document_id,
        "source_object_id": source_object_id,
        "title": display_title(item),
        "author": source_author(item),
        "category": item.get("source_category") or "Academic source",
        "page": page,
        "modality": modality,
        "section": " > ".join(section) if isinstance(section, list) else str(section or ""),
        "excerpt": clip(evidence_body(item), 3000),
        "latex": latex,
        "equation": latex,
        "label": content.get("label") or content.get("equation_label"),
        "caption": caption,
        "asset_url": asset_url,
        "crop_url": asset_url,
        "page_url": page_url,
        "rows": content.get("rows", [])[:8] if modality == "table" else [],
        "cells": content.get("selected_cells", [])[:24] if modality == "table" else [],
    }


def groq_models():
    fallback = [
        {
            "id": DEFAULT_MODEL,
            "name": "GPT-OSS 120B" if "gpt-oss-120b" in DEFAULT_MODEL else DEFAULT_MODEL,
            "label": "GPT-OSS 120B" if "gpt-oss-120b" in DEFAULT_MODEL else DEFAULT_MODEL,
            "description": "Default Groq model",
            "available": True,
        }
    ]

    if not GROQ_KEY:
        return fallback

    try:
        response = requests.get(
            f"{GROQ_API}/models",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            timeout=15,
        )
        response.raise_for_status()
        ids = sorted({x["id"] for x in response.json().get("data", []) if x.get("id")})

        preferred = []

        def add(pattern, label_fn, description):
            match = next((x for x in ids if pattern in x.lower()), None)
            if match and all(x["id"] != match for x in preferred):
                label = label_fn(match)
                preferred.append(
                    {
                        "id": match,
                        "name": label,
                        "label": label,
                        "description": description,
                        "available": True,
                    }
                )

        add("gpt-oss-120b", lambda _: "GPT-OSS 120B", "Best default for academic synthesis")
        add("deepseek", lambda x: "DeepSeek · " + x.split("/")[-1], "Strong reasoning option")
        add("qwen", lambda x: "Qwen · " + x.split("/")[-1], "Strong technical and mathematical reasoning")
        add("llama", lambda x: "Llama · " + x.split("/")[-1], "General-purpose alternative")

        if DEFAULT_MODEL in ids and all(x["id"] != DEFAULT_MODEL for x in preferred):
            preferred.insert(
                0,
                {
                    "id": DEFAULT_MODEL,
                    "name": DEFAULT_MODEL,
                    "label": DEFAULT_MODEL,
                    "description": "Default model",
                    "available": True,
                },
            )

        return preferred[:4] or fallback
    except Exception:
        return fallback


STYLE_RULES = {
    "concise": "Keep the answer compact, usually 150-350 words unless the question genuinely needs more.",
    "balanced": "Give a well-structured academic explanation, usually 350-750 words when appropriate.",
    "detailed": "Give a thorough academic explanation, usually 650-1200 words when the evidence supports it. Explain mechanisms and experimental evidence clearly without filler.",
}


SYSTEM_PROMPT = r"""You are RAGScholar, a source-grounded academic research assistant.

Use ONLY the supplied evidence for factual claims. Do not silently add facts from pretrained knowledge.

Formatting requirements:
- Write polished Markdown.
- Begin with a direct answer, then use meaningful ## or ### sections when useful.
- Use bullet lists only when they improve clarity.
- Write mathematical expressions with valid LaTeX delimiters.
- Inline math MUST use \(...\).
- Display equations MUST use $$...$$ on separate lines.
- Never output raw LaTeX commands outside math delimiters.
- Cite factual claims inline with evidence IDs exactly like [E1], [E2].
- Never write citations as (E1, E2).
- NEVER create Markdown images such as ![caption](E2), ![caption](url), or HTML <img> tags.
- If the user asks for a figure or image, describe the relevant evidence and cite it with [E#]. The application itself renders the trusted source image.
- Preserve source equations faithfully.
- Never invent unseen visual details.
- Never invent page numbers, source names, equations, table values, experiment results, or citations.
- If sources disagree, explain the disagreement.
- If evidence is insufficient, say so clearly.

Return ONLY valid JSON:
{
  "answer": "Markdown answer",
  "citations": ["E1", "E2"],
  "confidence": "high|medium|low",
  "status": "answered|insufficient_evidence"
}
"""


def _json_result(text):
    text = str(text or "").strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I | re.S)

    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            pass

    return {
        "answer": text,
        "citations": [],
        "confidence": "medium",
        "status": "answered",
    }


def _groq(query, packet, model, style, reasoning, history=None):
    if not GROQ_KEY:
        raise RuntimeError("GROQ_API_KEY was not found in the project .env file.")

    style_rule = STYLE_RULES.get(style, STYLE_RULES["balanced"])
    history = history or []
    context = []

    for row in history[-6:]:
        role = row.get("role")
        text = clip(row.get("content"), 1000)
        if role in {"user", "assistant"} and text:
            context.append({"role": role, "content": text})

    prompt = f"""QUESTION:
{query}

ANSWER STYLE:
{style_rule}

EVIDENCE:
{json.dumps(packet, ensure_ascii=False, indent=2)}

Return the requested JSON only."""

    if "deepseek" in model.lower():
        messages = [{"role": "user", "content": SYSTEM_PROMPT + "\n\n" + prompt}]
    else:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            *context,
            {"role": "user", "content": prompt},
        ]

    max_tokens = {"concise": 800, "balanced": 1500, "detailed": 2400}.get(style, 1500)
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.15,
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }

    if reasoning in {"low", "medium", "high"} and (
        "gpt-oss" in model.lower() or "qwen" in model.lower()
    ):
        payload["reasoning_effort"] = reasoning

    headers = {
        "Authorization": f"Bearer {GROQ_KEY}",
        "Content-Type": "application/json",
    }
    response = requests.post(
        f"{GROQ_API}/chat/completions",
        headers=headers,
        json=payload,
        timeout=180,
    )

    if response.status_code >= 400 and "reasoning_effort" in payload:
        payload.pop("reasoning_effort", None)
        response = requests.post(
            f"{GROQ_API}/chat/completions",
            headers=headers,
            json=payload,
            timeout=180,
        )

    response.raise_for_status()
    return _json_result(response.json()["choices"][0]["message"]["content"])


def _sanitize_model_answer(answer):
    answer = str(answer or "")

    # Convert evidence-looking Markdown images to normal text + citation.
    answer = re.sub(
        r"!\[([^\]]*)\]\(\s*\[?E(\d+)\]?\s*\)",
        lambda m: f"{m.group(1).strip()} [E{m.group(2)}]".strip(),
        answer,
        flags=re.I,
    )

    # Remove any other model-generated image syntax. Trusted figures are rendered from source metadata.
    answer = re.sub(
        r"!\[([^\]]*)\]\([^)]*\)",
        lambda m: m.group(1).strip(),
        answer,
        flags=re.I,
    )
    answer = re.sub(r"<img\b[^>]*>", "", answer, flags=re.I)
    return answer.strip()


def _citation_order(answer, explicit, count):
    found = []

    for match in re.finditer(r"\bE(\d+)\b", str(answer or ""), flags=re.I):
        n = int(match.group(1))
        if 1 <= n <= count and n not in found:
            found.append(n)

    for value in explicit or []:
        match = re.search(r"E?(\d+)", str(value), flags=re.I)
        if match:
            n = int(match.group(1))
            if 1 <= n <= count and n not in found:
                found.append(n)

    return found


def _replace_citations(answer, order):
    mapping = {evidence_number: i + 1 for i, evidence_number in enumerate(order)}

    def replace_evidence(match):
        evidence_number = int(match.group(1))
        return f"[{mapping[evidence_number]}]" if evidence_number in mapping else ""

    answer = re.sub(r"\[\s*E(\d+)\s*\]", replace_evidence, str(answer or ""), flags=re.I)
    answer = re.sub(r"\bE(\d+)\b", replace_evidence, answer, flags=re.I)
    answer = re.sub(
        r"\(\s*(\[\d+\](?:\s*,\s*\[\d+\])*)\s*\)",
        lambda m: m.group(1).replace(",", ""),
        answer,
    )
    return answer.strip()


def ask(query, model=None, style="balanced", reasoning="medium", history=None, top_docs=5, top_evidence=10):
    if not autoload_runtime():
        raise RuntimeError(
            "RAGScholar retrieval runtime is not connected. Start the app from the notebook with "
            "start_app(retrieve_fn=retrieve_corpus), or expose retrieve_corpus from a runtime module."
        )

    query = re.sub(r"\s+", " ", str(query or "")).strip()
    if not query:
        raise ValueError("Question cannot be empty.")

    history = history or []
    previous_user = next(
        (
            row.get("content")
            for row in reversed(history)
            if row.get("role") == "user" and row.get("content")
        ),
        None,
    )
    retrieval_query = (
        f"{previous_user}\nFollow-up question: {query}"
        if previous_user and len(query.split()) <= 6
        else query
    )

    with _LOCK:
        retrieval = _RETRIEVE_FN(
            retrieval_query,
            top_docs=top_docs,
            top_evidence=top_evidence,
        )

    evidence = retrieval.get("evidence") or []
    model = model or DEFAULT_MODEL

    if not evidence:
        return {
            "status": retrieval.get("status", "no_evidence"),
            "answer": "I couldn't find enough source-grounded evidence in the indexed corpus to answer this reliably.",
            "confidence": "low",
            "citations": [],
            "sources": [],
            "visuals": [],
            "modalities": retrieval.get("modalities", []),
            "documents": retrieval.get("documents", []),
            "model": model,
        }

    packet = [evidence_packet(item, i) for i, item in enumerate(evidence, 1)]
    generated = _groq(query, packet, model, style, reasoning, history)

    raw_answer = _sanitize_model_answer(generated.get("answer"))
    order = _citation_order(raw_answer, generated.get("citations"), len(evidence))

    # If the model omitted citations entirely, keep a small evidence set available in the UI,
    # but do not inject unsupported citation markers into the prose.
    source_order = order or list(range(1, min(4, len(evidence)) + 1))
    answer = _replace_citations(raw_answer, order) if order else raw_answer

    sources = [source_record(evidence[n - 1], f"E{n}") for n in source_order]
    for i, source in enumerate(sources, 1):
        source["number"] = i

    visuals = [
        source
        for source in sources
        if source.get("modality") == "visual" and source.get("asset_url")
    ]

    return {
        "status": generated.get("status", "answered"),
        "answer": answer,
        "confidence": generated.get("confidence", "medium"),
        "citations": [str(i) for i in range(1, len(order) + 1)],
        "sources": sources,
        "visuals": visuals,
        "modalities": retrieval.get("modalities", []),
        "documents": retrieval.get("documents", []),
        "model": model,
    }
