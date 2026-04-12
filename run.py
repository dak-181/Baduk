import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')
    import GoGame.main as m
    m.play_game_main()
