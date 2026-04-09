import pygame
import GoGame.game_initialization as start
import GoGame.uifunctions as ui


def play_game_main():
    '''
    Main menu loop.  Replaced sg.Window with ui.start_game_menu() which
    returns the chosen action as a string — same values as the old sg events.
    '''
    while True:
        event = ui.start_game_menu()

        if event == "Choose File":
            from GoGame.saving_loading import choose_file
            choose_file()
            break

        elif event == "New Game From Custom":
            board_size = ui.start_game()
            start.initializing_game(board_size, defaults=False)

        elif event == "New Game From Default":
            start.initializing_game(9, True)

        elif event == "Play Against AI":
            start.initializing_game(9, True, vs_bot=True)

        elif event == "AI SelfPlay":
            from GoGame.neuralnet import training_cycle
            import cProfile, pstats
            with cProfile.Profile() as pr:
                training_cycle(5)
                stats = pstats.Stats(pr)
                stats.sort_stats(pstats.SortKey.TIME)
                stats.dump_stats(filename="5000x30testingv3.prof")

        elif event == "AI Training":
            from GoGame.neuralnet import loading_file_for_training
            loading_file_for_training(epochs=10, size_of_batch=32)

        elif event == "Exit Game":
            break

    pygame.quit()


if __name__ == "__main__":
    pygame.init()
    play_game_main()
