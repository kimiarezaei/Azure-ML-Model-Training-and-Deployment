from pathlib import Path
import logging
import yaml


# Logger
def setup_logger(log_path: Path = None, log_file: str = "logs_file.log"):
    log_path = Path(log_path or Path(__file__).parent.parent / "logs")
    log_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
            logging.StreamHandler(),                  
            logging.FileHandler(log_path / log_file),     
        ]
    )


def load_config(path: Path):
    """Load config (paths and train params)"""
    path = Path(path)  
    with open(path, "r") as f:
        config = yaml.safe_load(f)
    return config