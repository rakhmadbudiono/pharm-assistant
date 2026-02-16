import json
import re

import pandas as pd

from src.models.gemini_model import gemini_model

DEFAULT_RAW_NAME = "(Unnamed)"
DEFAULT_PRICE = 0.0
DEFAULT_DISCOUNT = 0.0


def safe_float(value: object, default: float = 0.0) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    if isinstance(value, (int, float)):
        return float(value) if not pd.isna(value) else default
    s = str(value).strip().upper()
    if s in ("", "NAN", "#VALUE!", "#REF!", "#DIV/0!", "NA", "-"):
        return default
    s = re.sub(r",(?=\d{3}(?:\.|$))", "", str(value).strip())
    try:
        return float(s)
    except (ValueError, TypeError):
        return default


def safe_discount(value: object) -> float:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return DEFAULT_DISCOUNT
    if isinstance(value, (int, float)):
        if pd.isna(value):
            return DEFAULT_DISCOUNT
        v = float(value)
        return v if 0 <= v <= 1 else (v / 100.0 if v > 1 else DEFAULT_DISCOUNT)
    s = str(value).strip()
    if not s or "netto" in s.lower():
        return DEFAULT_DISCOUNT
    if "%" in s:
        return safe_float(re.sub(r"%", "", s), DEFAULT_DISCOUNT) / 100.0
    return safe_float(s, DEFAULT_DISCOUNT)


def safe_str(value: object, default: str = DEFAULT_RAW_NAME) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return default
    s = str(value).strip()
    if s.upper() in ("#VALUE!", "#REF!", "#DIV/0!", "NAN", "NA"):
        return default
    return s if s else default


def _ensure_mapping_dict(mapping: object) -> dict:
    if isinstance(mapping, list) and len(mapping) > 0 and isinstance(mapping[0], dict):
        return mapping[0]
    return mapping if isinstance(mapping, dict) else {}


def _normalize_mapping(mapping: dict) -> dict:
    mapping = _ensure_mapping_dict(mapping)
    return {
        k: v
        for k, v in mapping.items()
        if k in ("raw_name", "base_price", "discount")
        and v is not None
        and str(v).strip()
    }


def _resolve_column(df: pd.DataFrame, header: str) -> str | None:
    if header in df.columns:
        return header
    h = str(header).strip()
    for c in df.columns:
        if str(c).strip() == h:
            return c
    return None


class CatalogParser:
    def __init__(self) -> None:
        self.model = gemini_model

    def get_headers(self, file: object) -> list:
        if getattr(file, "name", "").endswith((".xlsx", ".xls")):
            df = pd.read_excel(file, nrows=0)
        elif getattr(file, "name", "").endswith(".csv"):
            df = pd.read_csv(file, nrows=0)
        else:
            return []
        return df.columns.tolist()

    def get_intelligent_mapping(self, headers: list) -> dict:
        from src.services.prompts import COLUMN_MAPPING_PROMPT

        prompt = COLUMN_MAPPING_PROMPT.format(headers=headers)
        response = self.model.get_llm(0).invoke(prompt)
        clean = response.content.replace("```json", "").replace("```", "").strip()
        return _ensure_mapping_dict(json.loads(clean))

    def process_file(
        self,
        file: object,
        mapping: dict,
        distributor_id: int,
        catalog_id: int,
    ) -> list[dict]:
        file.seek(0)
        name = getattr(file, "name", "")
        if name.endswith((".xlsx", ".xls")):
            df = pd.read_excel(file)
        else:
            df = pd.read_csv(file)

        norm = _normalize_mapping(mapping)
        raw_name_col = _resolve_column(df, norm.get("raw_name") or "")
        base_price_col = _resolve_column(df, norm.get("base_price") or "")
        discount_col = (
            _resolve_column(df, norm.get("discount") or "")
            if norm.get("discount")
            else None
        )

        if not raw_name_col or not base_price_col:
            return []

        processed = []
        for _, row in df.iterrows():
            base_price = safe_float(row.get(base_price_col, DEFAULT_PRICE))
            discount_val = (
                safe_discount(row.get(discount_col, DEFAULT_DISCOUNT))
                if discount_col
                else DEFAULT_DISCOUNT
            )
            effective_price = base_price * (1.0 - discount_val)
            if effective_price < 0:
                effective_price = DEFAULT_PRICE
            processed.append(
                {
                    "raw_name": safe_str(row.get(raw_name_col, "")),
                    "base_price": base_price,
                    "discount": discount_val,
                    "effective_price": effective_price,
                    "catalog_id": catalog_id,
                }
            )
        return processed
