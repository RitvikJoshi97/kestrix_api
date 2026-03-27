from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    data_dir: Path = Path("./data")
    output_dir: Path = Path("./outputs")
    default_palette: str = "IRON"
    default_low_percentile: float = 2.0
    default_high_percentile: float = 98.0

    @field_validator("default_low_percentile", "default_high_percentile")
    @classmethod
    def _check_percentile(cls, v: float) -> float:
        if not (0.0 <= v <= 100.0):
            raise ValueError("Percentile must be between 0 and 100")
        return v

    @property
    def index_csv(self) -> Path:
        return self.data_dir / "image_index.csv"

    @property
    def temps_dir(self) -> Path:
        return self.data_dir / "thermal_temps"

    @property
    def images_dir(self) -> Path:
        return self.data_dir / "thermal_images"


settings = Settings()
