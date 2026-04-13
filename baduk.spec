# baduk.spec
# ─────────────────────────────────────────────────────────────────────────────
# PyInstaller spec for the Baduk release build.
#
# Usage (from the project root, with RELEASE_MODE = True set in config.py):
#
#   pip install pyinstaller
#   pyinstaller baduk.spec
#
# Output: dist/Baduk/  — a self-contained folder.
#   Zip dist/Baduk/ and distribute. Users extract and double-click Baduk.exe.
#
# Requirements for the build machine:
#   pip install tensorflow-cpu pygame numpy
#   (tensorflow-cpu avoids bundling ~1 GB of CUDA DLLs; inference at 25
#    MCTS iterations is fast enough on CPU for comfortable play.)
#
# ─────────────────────────────────────────────────────────────────────────────

import os
import glob
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

# ── Locate the project root (same directory as this spec file) ───────────────
ROOT = os.path.dirname(os.path.abspath(SPEC))

# ── Bundle any .h5 weight files found in the project root ────────────────────
# Place trained model files (e.g. model.weights.h5, shusaku.weights.h5) next to
# run.py before building. They will land beside Baduk.exe in the output folder.
h5_files = [(f, '.') for f in glob.glob(os.path.join(ROOT, '*.h5'))]

# ── Data files ───────────────────────────────────────────────────────────────
datas = []

# pygame font/data files
datas += collect_data_files('pygame')

# any tensorflow/keras data files (protocol buffers, ops registry, etc.)
datas += collect_data_files('tensorflow', includes=['**/*.json',
                                                     '**/*.pb',
                                                     '**/*.pbtxt',
                                                     '**/*.so',
                                                     '**/*.dll'])
datas += collect_data_files('keras')

# the GoGame source package itself
datas += [(os.path.join(ROOT, 'src', 'GoGame'), os.path.join('src', 'GoGame'))]

# bundled .h5 weight files
datas += h5_files

# ── Hidden imports ────────────────────────────────────────────────────────────
# TensorFlow uses __import__ and importlib extensively; these won't be detected
# by PyInstaller's static analysis.
hidden_imports = [
    # TensorFlow core
    'tensorflow',
    'tensorflow.python',
    'tensorflow.python.keras',
    'tensorflow.python.framework',
    'tensorflow.python.ops',
    'tensorflow.python.platform',
    'tensorflow.python.util',
    'tensorflow.python.eager',
    'tensorflow.python.saved_model',
    'tensorflow.lite.python',
    'tensorflow.compiler',
    # Keras
    'keras',
    'keras.src',
    'keras.src.layers',
    'keras.src.models',
    'keras.src.optimizers',
    'keras.src.losses',
    'keras.src.metrics',
    'keras.src.callbacks',
    # numpy / scipy used by TF
    'numpy',
    'numpy.core',
    'numpy.lib',
    # multiprocessing (spawn mode needs these)
    'multiprocessing',
    'multiprocessing.spawn',
    'multiprocessing.managers',
    'multiprocessing.pool',
    'multiprocessing.queues',
    'multiprocessing.synchronize',
    # pygame
    'pygame',
    'pygame.font',
    'pygame.gfxdraw',
    'pygame.mixer',
    # GoGame modules
    'GoGame',
    'GoGame.main',
    'GoGame.config',
    'GoGame.goclasses',
    'GoGame.uifunctions',
    'GoGame.pygame_ui',
    'GoGame.stone_renderer',
    'GoGame.game_initialization',
    'GoGame.player',
    'GoGame.turn_options',
    'GoGame.saving_loading',
    'GoGame.remove_dead',
    'GoGame.scoringboard',
    'GoGame.undoing',
    'GoGame.handicap',
    'GoGame.botnormalgo',
    'GoGame.neuralnet',
    'GoGame.neuralnetboard',
    'GoGame.nnmcst',
    'GoGame.mcst',
    'GoGame.sgf_to_training',
    # misc stdlib used at runtime
    'pickle',
    'json',
    'math',
    'copy',
    're',
]

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    [os.path.join(ROOT, 'run.py')],
    pathex=[ROOT, os.path.join(ROOT, 'src')],
    binaries=collect_dynamic_libs('tensorflow'),
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,      # --onedir: binaries live alongside the exe
    name='Baduk',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,                # don't strip symbols — can break TF on Windows
    upx=False,                  # UPX compression breaks TF DLLs on Windows
    console=False,              # no console window; set True temporarily to debug
    icon=None,                  # replace with 'icon.ico' if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name='Baduk',
)
