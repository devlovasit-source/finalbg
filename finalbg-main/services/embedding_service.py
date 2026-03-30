from sentence_transformers import SentenceTransformer

# -------------------------
# GLOBAL MODEL (LAZY LOAD)
# -------------------------
_model = None


def get_model():
    global _model

    if _model is None:
        print("🔠 Loading embedding model...")
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model


# -------------------------
# BUILD TEXT FROM METADATA
# -------------------------
def build_text(data: dict) -> str:
    return " ".join([
        data.get("category", ""),
        data.get("sub_category", ""),
        data.get("color_code", ""),
        data.get("pattern", ""),
        " ".join(data.get("occasions", []))
    ])


# -------------------------
# MAIN FUNCTION
# -------------------------
def encode_metadata(data: dict) -> list:
    try:
        model = get_model()

        text = build_text(data)

        embedding = model.encode(text)

        return embedding.tolist()

    except Exception as e:
        print("Embedding error:", str(e))
        return [0.0] * 384