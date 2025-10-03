import json
import uuid

import pytest


def test_qdrant_basic_search_integration():
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qm
    from fastembed import TextEmbedding
    from clinical_rag.config import load_settings

    s = load_settings()
    client = QdrantClient(url=s.qdrant_url)

    collection = f"tm_integ_{uuid.uuid4().hex[:8]}"
    model = s.embedding_model_name

    # Try to create collection; skip test if server not reachable
    try:
        dim = client.get_embedding_size(model)
        if client.collection_exists(collection_name=collection):
            client.delete_collection(collection_name=collection)
        client.create_collection(
            collection_name=collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )
    except Exception as e:
        pytest.skip(f"Qdrant not available or embed size unsupported: {e}")

    try:
        # Prepare two simple points (inclusion bullets)
        payloads = [
            {
                "nct_id": "TRIAL_ANAL",
                "section": "eligibility_inclusion",
                "min_age": 18,
                "gender": "ALL",
                "text": "biopsy verified localized squamous cell anal cancer",
            },
            {
                "nct_id": "TRIAL_SCD",
                "section": "eligibility_inclusion",
                "min_age": 2,
                "gender": "ALL",
                "text": "clinical diagnosis of sickle cell disease",
            },
        ]

        ids = [str(uuid.uuid4()), str(uuid.uuid4())]
        embedder = TextEmbedding(model_name=model)
        vecs = list(embedder.embed([p["text"] for p in payloads]))

        client.upsert(
            collection_name=collection,
            points=[
                qm.PointStruct(id=ids[i], vector=vecs[i], payload=payloads[i])
                for i in range(2)
            ],
            wait=True,
        )

        # Build a filter: inclusion-only and min_age <= 30
        flt = qm.Filter(
            must=[
                qm.FieldCondition(
                    key="section", match=qm.MatchValue(value="eligibility_inclusion")
                ),
                qm.FieldCondition(key="min_age", range=qm.Range(lte=30)),
            ]
        )

        # Query for anal cancer (vector-based, non-deprecated API)
        q_vec = list(embedder.embed(["anal cancer"]))[0]
        resp = client.query_points(
            collection_name=collection,
            query=q_vec,
            query_filter=flt,
            limit=3,
            with_payload=True,
            with_vectors=False,
        )
        hits = resp.points or []
        assert hits, "Expected at least one hit"
        top_nct = (hits[0].payload or {}).get("nct_id")
        assert top_nct == "TRIAL_ANAL"
    finally:
        try:
            client.delete_collection(collection)
        except Exception:
            pass
