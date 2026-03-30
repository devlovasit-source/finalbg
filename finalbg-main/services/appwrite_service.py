import os

from appwrite.client import Client
from appwrite.services.account import Account
from appwrite.services.databases import Databases
from dotenv import load_dotenv


def _env_first(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        value = os.getenv(key)
        if value:
            return value
    return default


load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

client = Client()
client.set_endpoint(
    _env_first(
        "APPWRITE_ENDPOINT",
        "EXPO_PUBLIC_APPWRITE_ENDPOINT",
        default="https://cloud.appwrite.io/v1",
    )
)
client.set_project(
    _env_first(
        "APPWRITE_PROJECT_ID",
        "APPWRITE_PROJECT",
        "EXPO_PUBLIC_APPWRITE_PROJECT_ID",
        default="69958f25003190519213",
    )
)

_appwrite_api_key = _env_first("APPWRITE_API_KEY", "APPWRITE_KEY")
if _appwrite_api_key:
    client.set_key(_appwrite_api_key)

account = Account(client)
databases = Databases(client)
