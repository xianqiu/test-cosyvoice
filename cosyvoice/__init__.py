import sys
from pathlib import Path


def load_third_party():
    """ Automatically load third party packages.
    E.g., put `Matcha-TTS`, `WeTextProcessing` in folder `cosyvoice/third_party`,
    then no need to pip-install the two packages (but their dependencies should be installed).
    @xianqiu
    """
    third_party_directory = Path.cwd() / "cosyvoice" / "third_party"
    if not third_party_directory.exists():
        return
    packages = [str(d) for d in third_party_directory.iterdir() if d.is_dir()]
    for package in packages:
        sys.path.append(package)


load_third_party()

