"""
Trial-level ingestion entrypoint.

Usage:
  uv run python -m ingest.main

What it does:
- Streams studies from ClinicalTrials.gov v2 within the configured date window
- Builds one vector per trial (NOT per bullet)
- Embedding text = trial title + top conditions + top interventions
- Upserts to Qdrant with rich payload for filters and judge (incl/excl bullets)
"""

from dotenv import load_dotenv
from clinical_rag.config import load_settings
from ingest import iter_study_fields
from typing import Optional, List, Dict, Iterator
from datetime import date
import argparse
import os
import json
from pathlib import Path
import uuid

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels  # for Document, VectorParams, Distance
from qdrant_client.http import models as http_models  # for payload index schema types


def main() -> None:
    # Load variables from .env if present (no-op if missing)
    load_dotenv()

    # Settings
    settings = load_settings()
    print(f"Qdrant URL: {settings.qdrant_url}")
    print(f"CTGov API: {settings.ctgov_api_base}")

    parser = argparse.ArgumentParser(
        description="ClinicalTrials.gov v2 ingestion (trial-level)"
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Small sample: upsert from API sample and write trials.jsonl",
    )
    parser.add_argument(
        "--resume-from",
        type=int,
        default=0,
        help="Resume upsert from this trial offset",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Only write trials.jsonl (and snapshot if set); do not upsert",
    )
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Create a timestamped snapshot after run",
    )
    parser.add_argument(
        "--upsert-from",
        type=str,
        default=None,
        help="Path to a snapshot dir or trials.jsonl",
    )
    parser.add_argument(
        "--upsert-from-snapshot",
        nargs="?",
        const="LATEST",
        default=None,
        help="Re-embed and upsert from a snapshot directory (or latest if no value)",
    )
    parser.add_argument(
        "--page-size", type=int, default=200, help="CTGov API page size"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Upload batch size override (0 = auto)",
    )
    parser.add_argument(
        "--parallel", type=int, default=0, help="Upload parallelism override (0 = auto)"
    )
    args = parser.parse_args()

    start: Optional[date] = settings.data_start_date
    end: Optional[date] = settings.data_end_date
    collection = settings.collection_name or "clinical_trials"
    model_name = settings.embedding_model_name
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    def _lower_unique(items):
        out = []
        seen = set()
        for it in items or []:
            if not it:
                continue
            val = str(it).strip().lower()
            if val and val not in seen:
                out.append(val)
                seen.add(val)
        return out

    # Auto-tuned defaults; overridable via flags
    cpu = os.cpu_count() or 4
    batch_size = (
        (4096 if cpu >= 8 else 2048) if args.batch_size <= 0 else args.batch_size
    )
    upload_parallel = (8 if cpu >= 8 else 4) if args.parallel <= 0 else args.parallel
    texts: List[str] = []
    ids: List[str] = []
    metadata: List[Dict] = []

    def flush_batch():
        # This is re-bound below after client is created; placeholder to avoid NameError
        pass

    def _trial_embedding_text(rec: Dict) -> str:
        title = (rec.get("trial_title") or "").strip()
        conditions = [c for c in (rec.get("conditions") or []) if c][:3]
        interventions = [i for i in (rec.get("interventions") or []) if i][:3]
        parts: List[str] = []
        if title:
            parts.append(title)
        if conditions:
            parts.append("conditions: " + ", ".join(conditions))
        if interventions:
            parts.append("interventions: " + ", ".join(interventions))
        return "\n\n".join(parts) if parts else title

    trials_path = out_dir / "trials.jsonl"

    def _iter_trials_from_api(sample: bool = False) -> Iterator[Dict]:
        print(f"Downloading trials (page_size={50 if sample else args.page_size})...")
        produced = 0
        study_count = 0
        for rec in iter_study_fields(
            start=start,
            end=end,
            page_size=(50 if sample else args.page_size),
            base_url=settings.ctgov_api_base,
        ):
            study_count += 1
            if study_count % 50 == 0:
                print(f"  fetched {study_count} studies ...")
            yield rec
            produced += 1
            if sample and produced >= 50:
                break
        print(f"Finished fetching {study_count} studies.")

    # Source selection
    trial_source: Iterator[Dict]
    if args.upsert_from_snapshot is not None:
        # choose snapshot dir
        snap_dir: Path
        if args.upsert_from_snapshot == "LATEST":
            base = Path("data") / "snapshots"
            latest = None
            if base.exists():
                for d in base.iterdir():
                    if d.is_dir() and (d / "trials.jsonl").exists():
                        ts = d.stat().st_mtime
                        latest = (
                            d
                            if (latest is None or ts > latest.stat().st_mtime)
                            else latest
                        )
            if not latest:
                raise FileNotFoundError("No suitable snapshot in data/snapshots")
            snap_dir = latest
        else:
            snap_dir = Path(args.upsert_from_snapshot)
        if (snap_dir / "trials.jsonl").exists():
            trial_source = (
                json.loads(line)
                for line in (snap_dir / "trials.jsonl").open("r", encoding="utf-8")
            )
        else:
            raise FileNotFoundError("Snapshot missing trials.jsonl")
    elif args.upsert_from:
        p = Path(args.upsert_from)
        if p.is_dir():
            if (p / "trials.jsonl").exists():
                trial_source = (
                    json.loads(line)
                    for line in (p / "trials.jsonl").open("r", encoding="utf-8")
                )
            else:
                raise FileNotFoundError("Directory lacks trials.jsonl")
        else:
            if p.name.endswith("trials.jsonl"):
                trial_source = (
                    json.loads(line) for line in p.open("r", encoding="utf-8")
                )
            else:
                raise FileNotFoundError("Provide trials.jsonl")
    elif args.demo:
        trial_source = _iter_trials_from_api(sample=True)
    else:
        trial_source = _iter_trials_from_api(sample=False)

    # No chunks-only mode; trial-level only

    # If we are streaming from API, also write trials.jsonl for snapshotting
    if not (args.upsert_from or args.upsert_from_snapshot):
        trials_file = trials_path.open("w", encoding="utf-8")
    else:
        trials_file = None

    # Prepare-only mode: build trials.jsonl (and snapshot) without upserting
    if args.ingest_only:
        count = 0
        for rec in trial_source:
            if trials_file:
                trials_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
        if trials_file:
            trials_file.close()
        print(f"Prepared {count} trials to {trials_path}")
        if args.snapshot:
            from datetime import datetime
            import tarfile, shutil

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_dir = Path("data") / "snapshots"
            snap_name = f"{ts}_{collection}"
            snap_dir = base_dir / snap_name
            snap_dir.mkdir(parents=True, exist_ok=True)
            if trials_path.exists():
                try:
                    shutil.copy2(str(trials_path), str(snap_dir / trials_path.name))
                except Exception:
                    pass
            manifest = {
                "snapshot": snap_name,
                "collection": collection,
                "embedding_model": model_name,
                "created_at": ts,
                "data_start_date": start.isoformat() if start else None,
                "data_end_date": end.isoformat() if end else None,
                "trial_count": count,
                "qdrant_point_count": None,
            }
            with (snap_dir / "manifest.json").open("w", encoding="utf-8") as mf:
                mf.write(json.dumps(manifest, ensure_ascii=False, indent=2))
            archive_path = base_dir / f"{snap_name}.tar.gz"
            with tarfile.open(archive_path, "w:gz") as tar:
                tar.add(snap_dir, arcname=snap_dir.name)
            print(f"Snapshot written: {snap_dir}\nArchive: {archive_path}")
        return

    # Initialize Qdrant only when we actually upsert (not for --chunks-only)
    print("Upserting trials to Qdrant (FastEmbed via client.add)...")
    client = QdrantClient(url=settings.qdrant_url, prefer_grpc=True)

    # Create collection if missing
    if not client.collection_exists(collection_name=collection):
        dim = client.get_embedding_size(model_name)
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(
                size=dim, distance=qmodels.Distance.COSINE
            ),
        )

    # Create payload indexes for common filters (idempotent)
    try:
        client.create_payload_index(
            collection,
            field_name="overall_status",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection,
            field_name="gender",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection,
            field_name="nct_id",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection,
            field_name="location_cities",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection,
            field_name="location_states",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection,
            field_name="location_countries",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass

    def flush_batch():
        if not texts:
            return
        docs = [qmodels.Document(text=t, model=model_name) for t in texts]
        try:
            client.upload_collection(
                collection_name=collection,
                vectors=docs,
                ids=ids,
                payload=metadata,
                batch_size=batch_size,
                parallel=upload_parallel,
            )
        except Exception as e:
            print(f"Upload batch failed: {e}")
            raise
        texts.clear()
        ids.clear()
        metadata.clear()

    total = 0
    for rec in trial_source:
        if args.resume_from and total < args.resume_from:
            total += 1
            continue
        nct_id = rec.get("nct_id")
        if not nct_id:
            continue
        if trials_file:
            trials_file.write(json.dumps(rec, ensure_ascii=False) + "\n")
        texts.append(_trial_embedding_text(rec))
        # Deterministic UUID for idempotent upserts
        ids.append(str(uuid.uuid5(uuid.NAMESPACE_URL, f"ctgov-trial:{nct_id}")))

        locs = rec.get("locations") or []
        location_cities = _lower_unique(
            [l.get("city") for l in locs if isinstance(l, dict)]
        )
        location_states = _lower_unique(
            [l.get("state") for l in locs if isinstance(l, dict)]
        )
        location_countries = _lower_unique(
            [l.get("country") for l in locs if isinstance(l, dict)]
        )

        incl = rec.get("inclusion_criteria") or []
        excl = rec.get("exclusion_criteria") or []

        metadata.append(
            {
                "nct_id": nct_id,
                "trial_title": rec.get("trial_title"),
                "overall_status": rec.get("overall_status"),
                "gender": rec.get("sex"),
                "min_age": rec.get("min_age_years"),
                "max_age": rec.get("max_age_years"),
                "conditions": rec.get("conditions") or [],
                "interventions": rec.get("interventions") or [],
                "phase": rec.get("phase"),
                "study_type": rec.get("study_type"),
                "mesh_terms": rec.get("mesh_terms") or [],
                "url": rec.get("url"),
                "locations": locs,
                "location_cities": location_cities,
                "location_states": location_states,
                "location_countries": location_countries,
                "inclusion_criteria": incl,
                "exclusion_criteria": excl,
                "inclusion_count": len(incl),
                "exclusion_count": len(excl),
            }
        )

        total += 1
        if len(texts) >= batch_size:
            flush_batch()
        if total % 500 == 0:
            print(f"  prepared {total} trials...")

    # close trials file if opened
    if trials_file:
        trials_file.close()

    flush_batch()

    print("Finalizing payload indexes ...")
    try:
        client.create_payload_index(
            collection,
            field_name="overall_status",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass
    try:
        client.create_payload_index(
            collection,
            field_name="gender",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass
    try:
        client.create_payload_index(
            collection,
            field_name="nct_id",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass
    try:
        client.create_payload_index(
            collection,
            field_name="location_cities",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection,
            field_name="location_states",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
        client.create_payload_index(
            collection,
            field_name="location_countries",
            field_schema=http_models.PayloadSchemaType.KEYWORD,
        )
    except Exception:
        pass

    q_count = None
    try:
        total_points = QdrantClient(url=settings.qdrant_url).count(
            collection, exact=True
        )
        q_count = (
            getattr(total_points, "count", None)
            if hasattr(total_points, "count")
            else total_points
        )
        print(f"Ingestion complete. Qdrant points total: {q_count}")
    except Exception:
        print("Ingestion pipeline complete: trials added via FastEmbed.")

    # Optional snapshot after upsert
    if args.snapshot:
        from datetime import datetime
        import tarfile, shutil

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_dir = Path("data") / "snapshots"
        snap_name = f"{ts}_{collection}"
        snap_dir = base_dir / snap_name
        snap_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts if present
        if trials_path.exists():
            try:
                shutil.copy2(str(trials_path), str(snap_dir / trials_path.name))
            except Exception:
                pass

        manifest = {
            "snapshot": snap_name,
            "collection": collection,
            "embedding_model": model_name,
            "created_at": ts,
            "data_start_date": start.isoformat() if start else None,
            "data_end_date": end.isoformat() if end else None,
            "trial_count": total,
            "qdrant_point_count": int(q_count) if isinstance(q_count, int) else q_count,
        }
        with (snap_dir / "manifest.json").open("w", encoding="utf-8") as mf:
            mf.write(json.dumps(manifest, ensure_ascii=False, indent=2))

        archive_path = base_dir / f"{snap_name}.tar.gz"
        with tarfile.open(archive_path, "w:gz") as tar:
            tar.add(snap_dir, arcname=snap_dir.name)
        print(f"Snapshot written: {snap_dir}\nArchive: {archive_path}")


if __name__ == "__main__":
    main()
