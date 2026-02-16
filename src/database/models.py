from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Distributor(Base):
    __tablename__ = "distributors"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False)
    includes_vat = Column(Boolean, default=True)
    min_order_amount = Column(Float, default=0.0)
    last_catalog_update = Column(DateTime)
    catalogs = relationship("Catalog", back_populates="distributor")


class ProductMaster(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True, nullable=False, index=True)
    sku = Column(String, unique=True, nullable=True)
    aliases = relationship("AliasMap", back_populates="product")


class Catalog(Base):
    __tablename__ = "catalogs"
    id = Column(Integer, primary_key=True)
    distributor_id = Column(Integer, ForeignKey("distributors.id"))
    filename = Column(String)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    distributor = relationship("Distributor", back_populates="catalogs")
    items = relationship("ProductCatalog", back_populates="catalog")


class ProductCatalog(Base):
    __tablename__ = "product_catalog"
    id = Column(Integer, primary_key=True)
    catalog_id = Column(Integer, ForeignKey("catalogs.id"))
    raw_name = Column(String, nullable=False)
    base_price = Column(Float, nullable=False)
    discount = Column(Float, default=0.0)
    effective_price = Column(Float, default=0.0, index=True)
    catalog = relationship("Catalog", back_populates="items")


class AliasMap(Base):
    __tablename__ = "alias_map"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, ForeignKey("products.id"))
    raw_name_from_distributor = Column(String, unique=True, nullable=False, index=True)
    product = relationship("ProductMaster", back_populates="aliases")


Index("idx_product_price", ProductCatalog.raw_name, ProductCatalog.effective_price)
