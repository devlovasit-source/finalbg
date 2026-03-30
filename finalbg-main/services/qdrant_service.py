import os
from dotenv import load_dotenv

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams

# Load env variables
load_dotenv()


class QdrantService:

    def __init__(self):
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")
        self.collection = os.getenv("QDRANT_COLLECTION", "wardrobe")
        self.memory_collection = os.getenv("QDRANT_MEMORY_COLLECTION", "outfit_memory")
        self.vector_size = 384
        self.memory_vector_size = 8
        self._initialized = False
        self.client = None
        self._vector_name_cache = {}
        self._vector_dim_cache = {}

        if self.url:
            try:
                self.client = QdrantClient(url=self.url, api_key=self.api_key)
            except Exception as e:
                print("Qdrant client init failed:", str(e))
                self.client = None

    def enabled(self) -> bool:
        return self.client is not None

    def _ensure_ready(self) -> bool:
        if not self.enabled():
            return False
        if not self._initialized:
            self.init()
        return True

    # -------------------------
    # INIT (SAFE - CALL ON STARTUP)
    # -------------------------
    def init(self):
        if not self.enabled():
            print("Qdrant disabled: missing QDRANT_URL or client init failed")
            return

        if self._initialized:
            return

        print("Initializing Qdrant...")
        self._ensure_collection()
        self._ensure_memory_collection()
        self._refresh_vector_name_cache()
        self._initialized = True

    # -------------------------
    # CREATE COLLECTIONS
    # -------------------------
    def _ensure_collection(self):
        if not self.enabled():
            return
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]

            if self.collection not in names:
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                )
        except Exception as e:
            print("Qdrant init error:", str(e))

    def _ensure_memory_collection(self):
        if not self.enabled():
            return
        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]

            if self.memory_collection not in names:
                self.client.create_collection(
                    collection_name=self.memory_collection,
                    vectors_config=VectorParams(size=self.memory_vector_size, distance=Distance.COSINE),
                )
        except Exception as e:
            print("Memory collection init error:", str(e))

    def _refresh_vector_name_cache(self):
        self._vector_name_cache = {
            self.collection: self._detect_vector_name(self.collection),
            self.memory_collection: self._detect_vector_name(self.memory_collection),
        }
        self._vector_dim_cache = {
            self.collection: self._detect_vector_dim(self.collection),
            self.memory_collection: self._detect_vector_dim(self.memory_collection),
        }

    def _detect_vector_name(self, collection_name: str):
        if not self.enabled():
            return None
        try:
            info = self.client.get_collection(collection_name)
            params = getattr(getattr(info, "config", None), "params", None)
            vectors = getattr(params, "vectors", None)

            # Unnamed single-vector collection.
            if isinstance(vectors, VectorParams):
                return None

            # Named-vector collection.
            if isinstance(vectors, dict) and vectors:
                return next(iter(vectors.keys()))
        except Exception:
            return None
        return None

    def _detect_vector_dim(self, collection_name: str):
        if not self.enabled():
            return None
        try:
            info = self.client.get_collection(collection_name)
            params = getattr(getattr(info, "config", None), "params", None)
            vectors = getattr(params, "vectors", None)
            name = self._vector_name_cache.get(collection_name)

            if isinstance(vectors, VectorParams):
                return int(vectors.size)

            if isinstance(vectors, dict) and vectors:
                if name and name in vectors:
                    cfg = vectors[name]
                else:
                    cfg = next(iter(vectors.values()))
                size = getattr(cfg, "size", None)
                if size is not None:
                    return int(size)
        except Exception:
            return None
        return None

    def _adapt_vector_dim(self, collection_name: str, vector: list):
        expected = self._vector_dim_cache.get(collection_name)
        if not expected:
            return vector
        current = len(vector or [])
        if current == expected:
            return vector
        if current > expected:
            return vector[:expected]
        return (vector or []) + ([0.0] * (expected - current))

    def _point_vector(self, collection_name: str, vector: list):
        vector = self._adapt_vector_dim(collection_name, vector)
        name = self._vector_name_cache.get(collection_name)
        if name:
            return {name: vector}
        return vector

    def _query_vector(self, collection_name: str, vector: list):
        vector = self._adapt_vector_dim(collection_name, vector)
        name = self._vector_name_cache.get(collection_name)
        if name:
            return (name, vector)
        return vector

    # -------------------------
    # UPSERT
    # -------------------------
    def upsert_item(self, item_id: str, vector: list, payload: dict):
        if not self._ensure_ready():
            return
        if not str((payload or {}).get("userId", "")).strip():
            print("Upsert skipped: missing userId in payload")
            return
        try:
            self.client.upsert(
                collection_name=self.collection,
                points=[PointStruct(id=item_id, vector=self._point_vector(self.collection, vector), payload=payload)],
            )
        except Exception as e:
            print("Upsert failed:", str(e))

    def upsert_memory_vector(self, point_id: str, vector: list, payload: dict):
        if not self._ensure_ready():
            return
        if not str((payload or {}).get("userId", "")).strip():
            print("Memory upsert skipped: missing userId in payload")
            return
        try:
            self.client.upsert(
                collection_name=self.memory_collection,
                points=[PointStruct(id=point_id, vector=self._point_vector(self.memory_collection, vector), payload=payload)],
            )
        except Exception as e:
            print("Memory upsert failed:", str(e))

    # -------------------------
    # SEARCH
    # -------------------------
    def search_similar(self, vector: list, user_id: str, limit: int = 5):
        if not self._ensure_ready():
            return []
        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=self._query_vector(self.collection, vector),
                limit=limit,
                query_filter={
                    "must": [{"key": "userId", "match": {"value": user_id}}],
                    "must_not": [{"key": "feedback", "match": {"value": "down"}}],
                },
            )

            return [
                {"id": str(r.id), "score": self._boost_score(float(r.score), r.payload), "payload": r.payload or {}}
                for r in results
            ]

        except Exception as e:
            print("Search failed:", str(e))
            return []

    def semantic_retrieve(self, vector: list, user_id: str, limit: int = 40):
        if not self._ensure_ready():
            return []
        try:
            results = self.client.search(
                collection_name=self.collection,
                query_vector=self._query_vector(self.collection, vector),
                limit=limit,
                query_filter={"must": [{"key": "userId", "match": {"value": user_id}}]},
            )
            return [
                {"id": str(r.id), "score": float(r.score), "payload": r.payload or {}}
                for r in results
            ]
        except Exception as e:
            print("Semantic retrieve failed:", str(e))
            return []

    # -------------------------
    # FEEDBACK UPDATE
    # -------------------------
    def update_feedback(self, item_id: str, feedback: str):
        if not self._ensure_ready():
            return
        try:
            self.client.set_payload(
                collection_name=self.collection,
                payload={"feedback": feedback},
                points=[item_id],
            )
        except Exception as e:
            print("Feedback update failed:", str(e))

    # -------------------------
    # BOOST SCORES
    # -------------------------
    def _boost_score(self, score: float, payload: dict):
        if not payload:
            return score

        feedback = payload.get("feedback")

        if feedback == "up":
            return score + 0.05

        if feedback == "down":
            return score - 0.2

        return score

    # -------------------------
    # DUPLICATE CHECK
    # -------------------------
    def is_duplicate(self, vector: list, user_id: str, threshold: float = 0.97):
        if not self._ensure_ready():
            return False
        try:
            results = self.search_similar(vector, user_id, limit=1)

            if results and results[0]["score"] > threshold:
                return True

        except Exception as e:
            print("Duplicate check failed:", str(e))

        return False

    def status(self) -> dict:
        if not self.enabled():
            return {"enabled": False, "initialized": False, "url_configured": bool(self.url)}

        details = {
            "enabled": True,
            "initialized": self._initialized,
            "url_configured": bool(self.url),
            "collection": self.collection,
            "memory_collection": self.memory_collection,
        }

        try:
            collections = self.client.get_collections().collections
            names = [c.name for c in collections]
            details["collections"] = names
            details["collection_ready"] = self.collection in names
            details["memory_collection_ready"] = self.memory_collection in names
        except Exception as e:
            details["error"] = str(e)

        return details

    def close(self) -> None:
        if not self.enabled():
            return
        try:
            if hasattr(self.client, "close"):
                self.client.close()
        except Exception as e:
            print("Qdrant close failed:", str(e))
        finally:
            self.client = None
            self._initialized = False
            self._vector_name_cache = {}
            self._vector_dim_cache = {}


# -------------------------
# SINGLETON INSTANCE
# -------------------------
qdrant_service = QdrantService()
