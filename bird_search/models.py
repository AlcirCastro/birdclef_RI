from dataclasses import dataclass
from pathlib import Path


@dataclass
class Record:
    item_id: int
    primary_label: str
    scientific_name: str
    common_name: str
    class_name: str
    rating: float
    rel_path: str
    local_path: Path
