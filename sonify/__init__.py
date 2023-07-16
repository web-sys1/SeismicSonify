import subprocess
from pathlib import Path

try:
 __version__ = subprocess.run(
    [
        'git',
        '-C',
        Path(__file__).resolve().parent,
        'rev-parse',
        '--short=7',
        'HEAD',
    ],
    capture_output=True,
    text=True,
 ).stdout.strip()
except:
 __version_info__ = (1,0,0)
 __version__ = '.'.join(map(str,__version_info__))

del subprocess
del Path

from .sonify import sonify
