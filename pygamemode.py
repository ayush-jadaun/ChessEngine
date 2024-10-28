import pygame
import chess

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 800  # No extra height needed for the header
SQUARE_SIZE = WIDTH // 8

# Colors
LIGHT_SQUARES = pygame.Color("#f0d9b5")  # Beige color
DARK_SQUARES = pygame.Color("#b58863")  # Soft green color
WHITE_PIECE_COLOR = pygame.Color("white")
BLACK_PIECE_COLOR = pygame.Color("black")
RED = pygame.Color("red")
BLUE = pygame.Color("blue")

# Initialize board
board = chess.Board()

# Draw chess board
def draw_board(screen):
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQUARES if (r + c) % 2 == 0 else DARK_SQUARES
            pygame.draw.rect(screen, color, pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Draw chess pieces directly on the board
def draw_pieces(screen, board):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = RED if piece.color == chess.WHITE else BLUE
            center = (chess.square_file(square) * SQUARE_SIZE + SQUARE_SIZE // 2, HEIGHT - (chess.square_rank(square) + 1) * SQUARE_SIZE + SQUARE_SIZE // 2)
            pygame.draw.circle(screen, color, center, SQUARE_SIZE // 2 - 10)
            text = piece.symbol().upper() if piece.color == chess.WHITE else piece.symbol().lower()
            font = pygame.font.SysFont(None, 36)
            img = font.render(text, True, BLACK_PIECE_COLOR if piece.color == chess.WHITE else WHITE_PIECE_COLOR)
            screen.blit(img, img.get_rect(center=center))

# Handle user input and moves
def handle_click(pos):
    file = pos[0] // SQUARE_SIZE
    rank = 7 - pos[1] // SQUARE_SIZE
    if 0 <= rank < 8:
        square = chess.square(file, rank)
        return square
    return None

# Main game loop
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Chess.com Styled Game')

selected_square = None

running = True
while running:
    draw_board(screen)
    draw_pieces(screen, board)
    pygame.display.flip()
   
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            square = handle_click(event.pos)
            if square is not None:
                if selected_square is None:
                    if board.piece_at(square):
                        selected_square = square
                else:
                    move = chess.Move(selected_square, square)
                    if move in board.legal_moves:
                        board.push(move)
                    selected_square = None

pygame.quit()
