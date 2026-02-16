from __future__ import annotations

from typing import Literal, Optional, TypedDict

from rapidfuzz import fuzz, process
from sqlalchemy.orm import Session

from src.database.models import AliasMap, ProductMaster

CONFIDENCE_THRESHOLD = 60.0


class MatchResult(TypedDict):
    product_id: Optional[int]
    score: float
    match_type: Literal["Exact", "Fuzzy", "None"]


class ProductMatcher:
    def __init__(self, session: Session) -> None:
        self._session = session
        self._product_names: Optional[list[tuple[int, str]]] = None

    def _load_products(self) -> list[tuple[int, str]]:
        if self._product_names is None:
            rows = self._session.query(ProductMaster.id, ProductMaster.name).all()
            self._product_names = [(r.id, r.name) for r in rows]
        return self._product_names

    def find_match(self, raw_name: str) -> MatchResult:
        if not raw_name or not str(raw_name).strip():
            return {"product_id": None, "score": 0.0, "match_type": "None"}

        raw_name = str(raw_name).strip()

        alias = (
            self._session.query(AliasMap)
            .filter(AliasMap.raw_name_from_distributor == raw_name)
            .first()
        )
        if alias is not None:
            return {
                "product_id": alias.product_id,
                "score": 100.0,
                "match_type": "Exact",
            }

        products = self._load_products()
        if not products:
            return {"product_id": None, "score": 0.0, "match_type": "None"}

        names = [name for _, name in products]
        result = process.extractOne(raw_name, names, scorer=fuzz.WRatio)
        if result is None:
            return {"product_id": None, "score": 0.0, "match_type": "None"}

        _, score, index = result
        return {
            "product_id": products[index][0],
            "score": float(score),
            "match_type": "Fuzzy",
        }
