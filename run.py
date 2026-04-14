import sys, os

# Suppress TensorFlow info/warning messages and SDL audio init noise.
# Must be set before any TF or pygame imports.
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')       # suppress TF C++ logs (0=all, 3=errors only)
os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '0')      # suppress oneDNN floating-point notice
os.environ.setdefault('TF_TRT_LOGGER_VERBOSITY', '0')    # suppress TF-TRT warnings
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')         # suppress SDL audio init noise if no audio

# When running as a PyInstaller bundle, _MEIPASS holds the unpacked temp directory.
# We need to add it to sys.path so GoGame can be found, and set the working directory
# to the exe's own folder so relative paths (h5 files, pklfiles/, etc.) resolve correctly.
if getattr(sys, 'frozen', False):
    _base = os.path.dirname(sys.executable)
    os.chdir(_base)
    sys.path.insert(0, os.path.join(sys._MEIPASS, 'src'))
else:
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()   # required for PyInstaller + spawn on Windows
    multiprocessing.set_start_method('spawn')
    import GoGame.main as m
    m.play_game_main()
