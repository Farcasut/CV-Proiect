"""
Ingenious Board — Simple Implementation
"""

# 6 colour symbols at the corners of the inner area
SYMBOLS = {(0, -5): "R", (5, -5): "G", (-5, 0): "P", (5, 0): "B", (-5, 5): "Y", (0, 5): "O", }

# 6 directions as (dq, dr)
E = (+1, 0)
W = (-1, 0)
SE = (0, +1)
NW = (0, -1)
NE = (+1, -1)
SW = (-1, +1)

DIRECTIONS = [E, W, SE, NW, NE, SW]


class Hexboard:

    def __init__(self):
        # board[(q, r)] = colour string, or None if empty
        self.board = {}
        self.index = {}
        self._build()

    def _build(self):
        R = 5
        cells = []
        for q in range(-R, R + 1):
            for r in range(max(-R, -q - R), min(R, -q + R) + 1):
                self.board[(q, r)] = SYMBOLS.get((q, r))
                cells.append((q, r))

        playable = sorted((pos for pos in cells if pos not in SYMBOLS), key=lambda pos: (pos[1], pos[0]))
        for i, pos in enumerate(playable, start=1):
            self.index[i] = pos

    def get(self, q, r):
        return self.board.get((q, r))

    def place(self, pieces):
        for index, color in pieces.items():
            q, r = self.pos(index)
            if (q, r) not in self.board:
                raise ValueError(f"({q}, {r}) is not on the board")
            if self.board[(q, r)] is not None:
                raise ValueError(f"({q}, {r}) is already occupied")
            self.board[(q, r)] = color

    def score_position(self, q, r, color):
        """
        Walk all 6 directions from (q, r).
        For each direction count consecutive cells matching the color
        at (q, r) — stopping at the board edge, an empty cell, or a
        different color.
        Returns total points scored (int).
        """
        color = color
        if color is None:
            return 0

        points = 0
        for dq, dr in DIRECTIONS:
            cq, cr = q + dq, r + dr
            while (cq, cr) in self.board:
                if self.board[(cq, cr)] == color:
                    points += 1
                    cq += dq
                    cr += dr
                else:
                    break
        return points

    def pos(self, index):
        # Return the (q, r) of a playable hex by its index (1-85)
        return self.index[index]
