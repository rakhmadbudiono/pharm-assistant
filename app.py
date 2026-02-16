from collections import defaultdict

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.chatbot.engine import create_rag_chain
from src.config import (
    EMBEDDING_MODEL_PROVIDER,
    MODEL_PROVIDER,
)
from src.database.connection import SessionLocal, init_db
from src.database.models import (
    AliasMap,
    Catalog,
    Distributor,
    ProductCatalog,
    ProductMaster,
)
from src.knowledge_base.vectorstore import VectorStoreManager
from src.models.gemini_model import gemini_model
from src.models.hf_model import hf_model
from src.models.openai_model import openai_model
from src.services.catalog_parser import CatalogParser
from src.services.matcher import CONFIDENCE_THRESHOLD, ProductMatcher
from src.services.price_service import format_rupiah, get_price_comparison

load_dotenv()
init_db()


def get_model():
    if MODEL_PROVIDER == "openai":
        return openai_model
    if MODEL_PROVIDER == "hugging-face":
        return hf_model
    return gemini_model


def get_model_embedding():
    if EMBEDDING_MODEL_PROVIDER == "openai":
        return openai_model.get_embeddings()
    if EMBEDDING_MODEL_PROVIDER == "gemini":
        return gemini_model.get_embeddings()
    return hf_model.get_embeddings()


def get_embeddings():
    if "embeddings" not in st.session_state or st.session_state.embeddings is None:
        if model.is_configured():
            st.session_state.embeddings = get_model_embedding()
    return st.session_state.embeddings


if "messages" not in st.session_state:
    st.session_state.messages = []

model = get_model()
vector_store_manager = VectorStoreManager()

if model.is_configured():
    embeddings = get_embeddings()
    if embeddings and vector_store_manager.vector_store is None:
        vector_store_manager.load_index(embeddings)

tabs = st.tabs(
    [
        "Chatbot",
        "CRM",
        "Catalog upload",
        "Product mapping",
        "Price comparison",
    ]
)

with tabs[0]:
    st.header("RAG Chatbot")

    if not model.is_configured():
        api_key_name = (
            "GOOGLE_API_KEY" if MODEL_PROVIDER == "gemini" else "OPENAI_API_KEY"
        )
        st.warning(f"Please set {api_key_name} in .env file")
    elif not vector_store_manager.vector_store:
        st.info("Upload documents to start chatting")
    else:
        if "chain" not in st.session_state:
            st.session_state.chain = create_rag_chain(
                model, vector_store_manager.vector_store
            )

        for msg in st.session_state.messages:
            st.chat_message(msg["role"]).write(msg["content"])

        if prompt := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            st.chat_message("user").write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    result = st.session_state.chain({"question": prompt})
                    answer = result["answer"]
                    st.write(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )

    with st.sidebar:
        st.header("Current Knowledge Base")

        if vector_store_manager.vector_store is not None:
            doc_metadata = vector_store_manager.vector_store.docstore._dict.values()
            sources = sorted(
                list(
                    set([doc.metadata.get("source", "Unknown") for doc in doc_metadata])
                )
            )

            if sources:
                for source in sources:
                    st.info(f"📄 {source}")
            else:
                st.write("No documents in index.")
        else:
            st.write("Knowledge base is empty.")

        st.divider()

        st.header("Upload Documents")
        uploaded_files = st.file_uploader(
            "Upload PDF or TXT files", type=["pdf", "txt"], accept_multiple_files=True
        )

        if uploaded_files and st.button("Process"):
            if not model.is_configured():
                api_key_name = (
                    "GOOGLE_API_KEY" if MODEL_PROVIDER == "gemini" else "OPENAI_API_KEY"
                )
                st.error(f"Please set {api_key_name} in .env file")
            else:
                with st.spinner("Processing..."):
                    embeddings = get_embeddings()
                    vector_store_manager.process_documents(uploaded_files, embeddings)

with tabs[1]:
    st.header("Distributor Management")
    db = SessionLocal()

    with st.expander("Add distributor"):
        with st.form("add_distributor_form"):
            name = st.text_input("Name")
            incl_vat = st.checkbox("Catalog prices include VAT (11%)", value=True)
            min_order = st.number_input(
                "Minimum order (IDR)", min_value=0.0, step=100000.0
            )
            submit = st.form_submit_button("Save")
            if submit:
                if name:
                    db.add(
                        Distributor(
                            name=name,
                            includes_vat=incl_vat,
                            min_order_amount=min_order,
                        )
                    )
                    db.commit()
                    st.success(f"Added {name}.")
                    st.rerun()
                else:
                    st.error("Name is required.")

    st.subheader("Registered distributors")
    distributors = db.query(Distributor).all()
    if not distributors:
        st.info("No distributors yet. Use the form above to add one.")
    else:
        data_list = [
            {
                "ID": d.id,
                "Name": d.name,
                "VAT incl.": "Yes" if d.includes_vat else "No",
                "Min. order": f"Rp {d.min_order_amount:,.0f}".replace(",", "."),
                "Last update": (
                    d.last_catalog_update.strftime("%Y-%m-%d %H:%M")
                    if d.last_catalog_update
                    else "—"
                ),
            }
            for d in distributors
        ]
        st.table(data_list)
    db.close()

with tabs[2]:
    st.header("Catalog upload")

    db = SessionLocal()
    try:
        distributors = db.query(Distributor).all()
        dist_options = {d.name: d.id for d in distributors}
        selected_dist = st.selectbox("Distributor", options=list(dist_options.keys()))
        uploaded_file = st.file_uploader(
            "File (Excel/CSV)", type=["xlsx", "xls", "csv"]
        )

        if uploaded_file:
            parser = CatalogParser()
            headers = parser.get_headers(uploaded_file)
            st.write("Headers:", headers)

            if st.button("Guess columns with AI"):
                with st.spinner("Analyzing…"):
                    mapping = parser.get_intelligent_mapping(headers)
                    st.session_state.current_mapping = mapping
                    st.json(mapping)

            if "current_mapping" in st.session_state and st.button("Save to database"):
                mapping = st.session_state.current_mapping
                distributor_id = dist_options[selected_dist]
                try:
                    catalog = Catalog(
                        distributor_id=distributor_id,
                        filename=uploaded_file.name,
                    )
                    db.add(catalog)
                    db.flush()
                    items = parser.process_file(
                        uploaded_file, mapping, distributor_id, catalog.id
                    )
                    if not items:
                        db.rollback()
                        st.warning(
                            "No rows parsed. Mapping must include 'raw_name' and 'base_price'."
                        )
                    else:
                        for item in items:
                            db.add(ProductCatalog(**item))
                        dist = db.get(Distributor, distributor_id)
                        if dist:
                            dist.last_catalog_update = catalog.uploaded_at
                        db.commit()
                        st.success(
                            f"Saved {len(items)} items from {uploaded_file.name}."
                        )
                        st.rerun()
                except Exception as e:
                    db.rollback()
                    st.error(f"Save failed: {e}")
    finally:
        db.close()

with tabs[3]:
    st.header("Product mapping")
    st.caption(
        "Link catalog raw names to master products. Approve to store in AliasMap."
    )

    db = SessionLocal()
    try:
        mapped_subq = db.query(AliasMap.raw_name_from_distributor)
        unmapped_rows = (
            db.query(ProductCatalog.raw_name)
            .distinct()
            .filter(ProductCatalog.raw_name.notin_(mapped_subq))
            .all()
        )
        unmapped_raw_names = [r[0] for r in unmapped_rows if r[0]]

        products = db.query(ProductMaster).order_by(ProductMaster.name).all()
        product_options: list[int | None] = [None]
        product_labels: list[str] = ["— Skip —"]
        for p in products:
            product_options.append(p.id)
            product_labels.append(p.name)

        if not unmapped_raw_names:
            st.info("No unmapped items, or no catalog data.")
        elif not products:
            st.warning("Add master products first (e.g. via seed).")
        else:
            matcher = ProductMatcher(db)
            for raw_name in unmapped_raw_names:
                match = matcher.find_match(raw_name)
                product_id = match["product_id"]
                score = match["score"]
                default_index = (
                    product_options.index(product_id)
                    if score >= CONFIDENCE_THRESHOLD
                    and product_id is not None
                    and product_id in product_options
                    else 0
                )

                col1, col2, col3 = st.columns([3, 2, 1])
                with col1:
                    st.text(raw_name)
                    if match["match_type"] == "Fuzzy" and score >= CONFIDENCE_THRESHOLD:
                        st.caption(f"Match score: {score:.0f}")
                with col2:
                    selected_label = st.selectbox(
                        "Map to",
                        product_labels,
                        index=default_index,
                        key=f"map_select_{raw_name}",
                        label_visibility="collapsed",
                    )
                with col3:
                    chosen_id = product_options[product_labels.index(selected_label)]
                    if st.button("Approve", key=f"map_btn_{raw_name}"):
                        if chosen_id is None:
                            st.warning("Pick a product or leave as Skip.")
                        else:
                            try:
                                db.add(
                                    AliasMap(
                                        raw_name_from_distributor=raw_name,
                                        product_id=chosen_id,
                                    )
                                )
                                db.commit()
                                st.success(f"Linked «{raw_name}» to {selected_label}.")
                                st.rerun()
                            except Exception:
                                db.rollback()
                                st.error("Save failed.")
    finally:
        db.close()

with tabs[4]:
    st.header("Price comparison")
    st.caption("Per-product distributor prices. Cheapest row highlighted.")

    db = SessionLocal()
    try:
        rows = get_price_comparison(db)
        if not rows:
            st.info(
                "No products or catalog data. Add products and upload catalogs first."
            )
        else:
            search = st.text_input(
                "Search by product name",
                placeholder="Filter…",
                key="price_search",
            )
            search_lower = (search or "").strip().lower()

            by_product: dict[int, list[dict]] = defaultdict(list)
            for r in rows:
                by_product[r["product_id"]].append(r)

            product_ids = sorted(
                by_product.keys(),
                key=lambda pid: (by_product[pid][0]["product_name"] or ""),
            )
            if search_lower:
                product_ids = [
                    pid
                    for pid in product_ids
                    if search_lower
                    in (by_product[pid][0]["product_name"] or "").lower()
                ]

            if not product_ids:
                st.warning("No products match the search.")
            else:
                for product_id in product_ids:
                    price_rows = [
                        r
                        for r in by_product[product_id]
                        if r["distributor_id"] is not None
                    ]
                    product_name = by_product[product_id][0]["product_name"]
                    if not price_rows:
                        st.subheader(product_name)
                        st.caption("No prices mapped.")
                        st.divider()
                        continue

                    price_rows = sorted(
                        price_rows,
                        key=lambda x: (x["actual_price"] or float("inf")),
                    )
                    min_price = price_rows[0]["actual_price"]
                    max_price = price_rows[-1]["actual_price"]
                    savings = (
                        (max_price - min_price)
                        if max_price and min_price and max_price > min_price
                        else 0
                    )
                    for r in price_rows:
                        r["_is_cheapest"] = r["actual_price"] == min_price

                    tbl = []
                    for r in price_rows:
                        last_updated = (
                            r["uploaded_at"].strftime("%Y-%m-%d %H:%M")
                            if r.get("uploaded_at")
                            else "N/A"
                        )
                        min_order = r.get("min_order_amount") or 0
                        tbl.append(
                            {
                                "Distributor": r["distributor_name"],
                                "Price": format_rupiah(r["actual_price"]),
                                "Last updated": last_updated,
                                "Min order": (
                                    format_rupiah(min_order) if min_order else "—"
                                ),
                                "_is_cheapest": r["_is_cheapest"],
                            }
                        )
                    df = pd.DataFrame(tbl)
                    display_cols = ["Distributor", "Price", "Last updated", "Min order"]
                    df_display = df[display_cols].copy()
                    is_cheapest_series = df["_is_cheapest"]

                    def _highlight(row, _ic=is_cheapest_series):
                        idx = row.name
                        flag = _ic.loc[idx] if idx in _ic.index else False
                        return [
                            (
                                "background-color: #d4edda; font-weight: 500;"
                                if flag
                                else ""
                            )
                            for _ in row
                        ]

                    styled = df_display.style.apply(_highlight, axis=1)
                    st.subheader(product_name)
                    col_a, col_b = st.columns([2, 1])
                    with col_a:
                        st.dataframe(styled, use_container_width=True, hide_index=True)
                    with col_b:
                        st.metric("Potential savings", format_rupiah(savings))
                        cheapest_dist = price_rows[0]
                        if cheapest_dist.get("min_order_amount"):
                            st.caption(
                                f"Min order ({cheapest_dist['distributor_name']}): "
                                f"{format_rupiah(cheapest_dist['min_order_amount'])}"
                            )
                    st.divider()
    finally:
        db.close()
