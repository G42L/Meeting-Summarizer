from pathlib import Path
import sys

def resource_path(*parts):
    if getattr(sys, "frozen", False):
        # Running from a PyInstaller bundle
        base = Path(sys._MEIPASS)
    else:
        # Running from source
        base = Path(__file__).resolve().parent.parent

    return base.joinpath(*parts)