from pymongo import MongoClient
import certifi
from typing import Any, Dict, Iterable, List, Literal, Optional, Sequence, Union


class MongoDatabase:
    """Wrapper around a mongo database."""

    def __init__(
            self,
            connectionString: str,
            db_name: str,
            sample_rows_in_table_info: int = 3):
        self.client = MongoClient(connectionString, tlsCAFile=certifi.where())
        self.db_name = db_name
        self.db = self.client[db_name]
        self.sample_rows_in_table_info = sample_rows_in_table_info

    def get_collection_names(self) -> Iterable[str]:
        return self.db.list_collection_names()