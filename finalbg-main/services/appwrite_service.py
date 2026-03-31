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


def _configure_base_client(client: Client) -> Client:
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
    return client


def build_appwrite_client(*, jwt_token: str | None = None) -> Client:
    client = _configure_base_client(Client())
    if jwt_token:
        client.set_jwt(jwt_token)
        return client

    api_key = _env_first("APPWRITE_API_KEY", "APPWRITE_KEY")
    if api_key:
        client.set_key(api_key)
    return client


def build_request_account(jwt_token: str) -> Account:
    return Account(build_appwrite_client(jwt_token=jwt_token))


load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env"))

client = build_appwrite_client()

account = Account(client)
databases = Databases(client)
