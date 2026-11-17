from pathlib import Path

# Screen dimensions
WIDTH, HEIGHT = 900, 600
FPS = 60

# Colors (RGB)
WHITE = (245, 245, 245)
GRAY = (70, 70, 70)
BLUE = (50, 120, 220)
GREEN = (46, 204, 113)
RED = (231, 76, 60)
LASER_GREEN = (0, 255, 120)

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DATA_DIR = ROOT_DIR / "data"