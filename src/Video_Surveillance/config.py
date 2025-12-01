from typing import Optional
from dataclasses import dataclass

@dataclass
class AppConfig:
    video: Optional[str]
    min_area: int
    save_path: Optional[str]
    direction: str
    limit_line_rate: int
    frame_size: int
    freq: int
    computer_no: int