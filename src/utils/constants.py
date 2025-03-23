from typing import Dict

# Window dimensions
WINDOW_WIDTH = 1000
WINDOW_HEIGHT = 800
BOARD_SIZE = 600
SQUARE_SIZE = BOARD_SIZE // 8
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
LIGHT_SQUARE = (240, 217, 181)
DARK_SQUARE = (181, 136, 99)
HOVER_COLOR = (255, 255, 0, 128)
SELECTED_COLOR = (255, 255, 0)
LEGAL_MOVE_COLOR = (144, 238, 144, 128)

# UI Constants
PANEL_WIDTH = 300
PANEL_PADDING = 20
FONT_SIZE = 24
SMALL_FONT_SIZE = 18

# Game Constants
MAX_MOVES_HISTORY = 10

# Board Coordinates
FILES = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
RANKS = ['8', '7', '6', '5', '4', '3', '2', '1']

# Piece Images
PIECE_IMAGES: Dict[str, str] = {
    'wP': 'assets/images/wP.png',
    'wR': 'assets/images/wR.png',
    'wN': 'assets/images/wN.png',
    'wB': 'assets/images/wB.png',
    'wQ': 'assets/images/wQ.png',
    'wK': 'assets/images/wK.png',
    'bP': 'assets/images/bP.png',
    'bR': 'assets/images/bR.png',
    'bN': 'assets/images/bN.png',
    'bB': 'assets/images/bB.png',
    'bQ': 'assets/images/bQ.png',
    'bK': 'assets/images/bK.png'
}

# Model Path
MODEL_PATH = 'assets/models/chess_model.pkl' 