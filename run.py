import sys, os

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
