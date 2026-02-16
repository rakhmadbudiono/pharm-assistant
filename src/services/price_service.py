from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from src.database.models import (
    AliasMap,
    Catalog,
    Distributor,
    ProductCatalog,
    ProductMaster,
)

VAT_RATE = 1.11


def _price_with_vat(effective_price: float, includes_vat: bool) -> float:
    if includes_vat:
        return effective_price
    return effective_price * VAT_RATE


def get_price_comparison(db_session: Session) -> list[dict[str, Any]]:
    all_products = db_session.query(ProductMaster.id, ProductMaster.name).all()
    product_ids = {p.id for p in all_products}
    product_names = {p.id: p.name for p in all_products}

    q = (
        db_session.query(
            ProductMaster.id.label("product_id"),
            ProductMaster.name.label("product_name"),
            Distributor.id.label("distributor_id"),
            Distributor.name.label("distributor_name"),
            Distributor.includes_vat,
            Distributor.min_order_amount,
            ProductCatalog.effective_price,
            Catalog.uploaded_at,
        )
        .select_from(ProductMaster)
        .join(AliasMap, ProductMaster.id == AliasMap.product_id)
        .join(
            ProductCatalog,
            AliasMap.raw_name_from_distributor == ProductCatalog.raw_name,
        )
        .join(Catalog, ProductCatalog.catalog_id == Catalog.id)
        .join(Distributor, Catalog.distributor_id == Distributor.id)
    )
    rows = q.all()

    by_key: dict[tuple[int, int], Any] = {}
    for r in rows:
        key = (r.product_id, r.distributor_id)
        if key not in by_key or (
            r.uploaded_at and (by_key[key].uploaded_at or datetime.min) < r.uploaded_at
        ):
            by_key[key] = r

    out: list[dict[str, Any]] = []
    for (pid, did), r in by_key.items():
        actual = _price_with_vat(r.effective_price, r.includes_vat)
        out.append(
            {
                "product_id": pid,
                "product_name": r.product_name,
                "distributor_id": did,
                "distributor_name": r.distributor_name,
                "actual_price": actual,
                "uploaded_at": r.uploaded_at,
                "includes_vat": r.includes_vat,
                "min_order_amount": r.min_order_amount,
            }
        )

    for pid in product_ids:
        if pid not in {d["product_id"] for d in out}:
            out.append(
                {
                    "product_id": pid,
                    "product_name": product_names[pid],
                    "distributor_id": None,
                    "distributor_name": None,
                    "actual_price": None,
                    "uploaded_at": None,
                    "includes_vat": None,
                    "min_order_amount": None,
                }
            )

    return out


def to_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def format_rupiah(value: float | None) -> str:
    if value is None or (isinstance(value, float) and (value != value)):
        return "N/A"
    return f"Rp {int(round(value)):,}".replace(",", ".")
