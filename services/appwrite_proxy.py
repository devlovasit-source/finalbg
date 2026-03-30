import os
from typing import Any, Dict, List, Optional

import requests
from requests.exceptions import RequestException


class AppwriteProxyError(Exception):
    pass


def _load_local_env() -> None:
    cwd = os.getcwd()
    parent = os.path.dirname(cwd)
    candidate_paths = [
        os.path.join(cwd, ".env"),
        os.path.join(parent, ".env"),
        os.path.join(parent, "backend", ".env"),
        os.path.join(parent, "backend-master", ".env"),
        os.path.join(parent, "ahvi", ".env"),
    ]

    for env_path in candidate_paths:
        if not os.path.exists(env_path):
            continue
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip().strip('"').strip("'")
                    if key and key not in os.environ:
                        os.environ[key] = value
        except Exception:
            continue


class AppwriteProxy:
    def __init__(self) -> None:
        _load_local_env()
        self.endpoint = (
            os.getenv("APPWRITE_ENDPOINT", "")
            or os.getenv("EXPO_PUBLIC_APPWRITE_ENDPOINT", "")
        ).rstrip("/")
        self.project_id = (
            os.getenv("APPWRITE_PROJECT_ID", "")
            or os.getenv("EXPO_PUBLIC_APPWRITE_PROJECT_ID", "")
        )
        self.database_id = (
            os.getenv("APPWRITE_DATABASE_ID", "")
            or os.getenv("EXPO_PUBLIC_APPWRITE_DATABASE_ID", "")
        )
        self.api_key = (
            os.getenv("APPWRITE_API_KEY", "")
            or os.getenv("EXPO_PUBLIC_APPWRITE_API_KEY", "")
            or os.getenv("APPWRITE_KEY", "")
        )

        self.collection_map = {
            "outfits": os.getenv("APPWRITE_COLLECTION_OUTFITS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_OUTFITS", ""),
            "users": os.getenv("APPWRITE_COLLECTION_USERS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_USERS", ""),
            "plans": os.getenv("APPWRITE_COLLECTION_PLANS", "") or os.getenv("PLANS_COLLECTION_ID", ""),
            "saved_boards": os.getenv("APPWRITE_COLLECTION_SAVED_BOARDS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_SAVED_BOARDS", ""),
            "skincare": os.getenv("APPWRITE_COLLECTION_SKINCARE", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_SKINCARE", ""),
            "workout_outfits": os.getenv("APPWRITE_COLLECTION_WORKOUT_OUTFITS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_WORKOUT_OUTFITS", ""),
            "bills": os.getenv("APPWRITE_COLLECTION_BILLS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_BILLS", ""),
            "coupons": os.getenv("APPWRITE_COLLECTION_COUPONS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_COUPONS", ""),
            "meds": os.getenv("APPWRITE_COLLECTION_MEDS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_MEDS", ""),
            "med_logs": os.getenv("APPWRITE_COLLECTION_MED_LOGS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_MED_LOGS", ""),
            "meal_plans": os.getenv("APPWRITE_COLLECTION_MEAL_PLANS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_MEAL_PLANS", ""),
            "life_goals": os.getenv("APPWRITE_COLLECTION_LIFE_GOALS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_LIFE_GOALS", ""),
            "life_boards": os.getenv("APPWRITE_COLLECTION_LIFE_BOARDS", "") or os.getenv("EXPO_PUBLIC_APPWRITE_COLLECTION_LIFE_BOARDS", ""),
        }

        self.user_field_map = {
            "outfits": "userId",
            "users": None,
            "plans": "userId",
            "saved_boards": "userId",
            "skincare": "userId",
            "workout_outfits": "userId",
            "bills": "userId",
            "coupons": "userId",
            "meds": "userId",
            "med_logs": "userId",
            "meal_plans": "userId",
            "life_goals": "userId",
            "life_boards": "userId",
        }

        self.order_query_map = {
            "outfits": 'orderDesc("$createdAt")',
            "plans": "",
            "saved_boards": 'orderDesc("$createdAt")',
            "skincare": "",
            "workout_outfits": 'orderDesc("$createdAt")',
            "bills": 'orderDesc("$createdAt")',
            "coupons": 'orderDesc("$createdAt")',
            "meds": 'orderDesc("$createdAt")',
            "med_logs": 'orderDesc("time")',
            "meal_plans": 'orderDesc("$createdAt")',
            "life_goals": 'orderDesc("$createdAt")',
            "life_boards": 'orderDesc("$createdAt")',
        }

    def _ensure_config(self) -> None:
        missing = []
        if not self.endpoint:
            missing.append("APPWRITE_ENDPOINT/EXPO_PUBLIC_APPWRITE_ENDPOINT")
        if not self.project_id:
            missing.append("APPWRITE_PROJECT_ID/EXPO_PUBLIC_APPWRITE_PROJECT_ID")
        if not self.database_id:
            missing.append("APPWRITE_DATABASE_ID/EXPO_PUBLIC_APPWRITE_DATABASE_ID")
        if not self.api_key:
            missing.append("APPWRITE_API_KEY/EXPO_PUBLIC_APPWRITE_API_KEY/APPWRITE_KEY")

        if missing:
            raise AppwriteProxyError(
                "Missing Appwrite backend configuration: " + ", ".join(missing)
            )

    def _collection_id(self, resource: str) -> str:
        collection_id = self.collection_map.get(resource, "")
        if not collection_id:
            env_suffix = resource.upper()
            collection_id = (
                os.getenv(f"APPWRITE_COLLECTION_{env_suffix}", "")
                or os.getenv(f"EXPO_PUBLIC_APPWRITE_COLLECTION_{env_suffix}", "")
            )
        if not collection_id:
            # Final fallback: allow resource name to be used directly as collection id.
            collection_id = resource
        if not collection_id:
            raise AppwriteProxyError(f"Unknown or unconfigured resource: {resource}")
        return collection_id

    def _headers(self) -> Dict[str, str]:
        return {
            "X-Appwrite-Project": self.project_id,
            "X-Appwrite-Key": self.api_key,
            "Content-Type": "application/json",
        }

    def _url(self, collection_id: str, document_id: Optional[str] = None) -> str:
        base = f"{self.endpoint}/databases/{self.database_id}/collections/{collection_id}/documents"
        if document_id:
            return f"{base}/{document_id}"
        return base

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        self._ensure_config()
        try:
            timeout_seconds = float(os.getenv("APPWRITE_TIMEOUT_SECONDS", "8"))
            response = requests.request(
                method=method,
                url=url,
                headers=self._headers(),
                params=params,
                json=payload,
                timeout=timeout_seconds,
            )
        except RequestException as exc:
            raise AppwriteProxyError(f"Appwrite connection failed: {exc}") from exc
        if response.status_code >= 400:
            raise AppwriteProxyError(
                f"Appwrite request failed ({response.status_code}): {response.text}"
            )
        if not response.text:
            return {}
        try:
            return response.json()
        except Exception as exc:
            raise AppwriteProxyError("Appwrite returned invalid JSON response.") from exc

    @staticmethod
    def _equal_query(field: str, value: str) -> str:
        safe_value = value.replace('"', '\\"')
        return f'equal("{field}", "{safe_value}")'

    def list_documents(
        self,
        resource: str,
        *,
        user_id: Optional[str] = None,
        occasion: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        collection_id = self._collection_id(resource)
        user_field = self.user_field_map.get(resource)
        safe_limit = max(1, min(int(limit), 100))
        data = self._request(
            "GET",
            self._url(collection_id),
        )
        docs = data.get("documents", [])

        # Apply filters in backend to avoid REST query syntax incompatibilities.
        if user_field and user_id:
            docs = [d for d in docs if str(d.get(user_field, "")) == str(user_id)]
        if occasion:
            docs = [d for d in docs if str(d.get("occasion", "")) == str(occasion)]

        if resource == "med_logs":
            docs.sort(key=lambda d: d.get("time", ""), reverse=True)
        else:
            docs.sort(key=lambda d: d.get("$createdAt", ""), reverse=True)

        return docs[:safe_limit]

    def get_document(self, resource: str, document_id: str) -> Dict[str, Any]:
        collection_id = self._collection_id(resource)
        return self._request("GET", self._url(collection_id, document_id))

    def create_document(self, resource: str, data: Dict[str, Any], document_id: str = "unique()") -> Dict[str, Any]:
        collection_id = self._collection_id(resource)
        payload = {"documentId": document_id, "data": data}
        return self._request("POST", self._url(collection_id), payload=payload)

    def update_document(self, resource: str, document_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        collection_id = self._collection_id(resource)
        payload = {"data": data}
        return self._request("PATCH", self._url(collection_id, document_id), payload=payload)

    def delete_document(self, resource: str, document_id: str) -> None:
        collection_id = self._collection_id(resource)
        self._request("DELETE", self._url(collection_id, document_id))
