

class BaseBot:
    def __init__(self):
        raise NotImplementedError("Subclasses must implement choose_move method")

    def choose_move(self, board):
        raise NotImplementedError("Subclasses must implement choose_move method")