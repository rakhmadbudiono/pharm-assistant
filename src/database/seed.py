import pandas as pd

from src.database.connection import SessionLocal
from src.database.models import ProductMaster


def seed_products(file_path: str) -> int:
    db = SessionLocal()
    try:
        df = pd.read_csv(file_path)
        added = 0
        for _, row in df.iterrows():
            if (
                db.query(ProductMaster)
                .filter(ProductMaster.name == row["name"])
                .first()
            ):
                continue
            db.add(ProductMaster(name=row["name"], sku=row.get("sku")))
            added += 1
        db.commit()
        return added
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    seed_products("initial_products.csv")
