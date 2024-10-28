import pygame
import chess
import joblib
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 800, 900  # Height for the chessboard
SQUARE_SIZE = WIDTH // 8

# Colors
LIGHT_SQUARES = pygame.Color("#f0d9b5")
DARK_SQUARES = pygame.Color("#b58863")
SELECTED_COLOR = pygame.Color("yellow")
WHITE_PIECE_COLOR = pygame.Color("white")
BLACK_PIECE_COLOR = pygame.Color("black")
DARK_GRAY = pygame.Color("darkgray")
RED = pygame.Color("red")
BLUE = pygame.Color("blue")

# Load the pre-trained model
model = joblib.load('chess_engine_model.pkl')

# Flatten the piece_map dictionary into a feature vector for model compatibility
def flatten_piece_map(piece_map):
    flattened = [0] * 64 * 6 * 2  # 64 squares, 6 pieces, 2 colors
    for square, piece in piece_map.items():
        piece_type_index = (piece.piece_type - 1) * 2
        if piece.color == chess.BLACK:
            piece_type_index += 1
        flattened[square * 12 + piece_type_index] = 1
    return flattened

# Function to predict advantage
def predict_advantage(board):
    piece_map = board.piece_map()
    features = flatten_piece_map(piece_map)
    return model.predict([features])[0]

# Draw chess board
def draw_board(screen):
    for r in range(8):
        for c in range(8):
            color = LIGHT_SQUARES if (r + c) % 2 == 0 else DARK_SQUARES
            pygame.draw.rect(screen, color, pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

# Draw geometric shapes for chess pieces and add letters
def draw_pieces(screen, board, selected_square):
    font = pygame.font.SysFont(None, 48)  # Font for piece letters
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            x = chess.square_file(square) * SQUARE_SIZE + SQUARE_SIZE // 2
            y = (7 - chess.square_rank(square)) * SQUARE_SIZE + SQUARE_SIZE // 2
            
            # Set color and shape based on piece type
            if piece.color == chess.WHITE:
                color = WHITE_PIECE_COLOR
                letter_color = BLACK_PIECE_COLOR  # Black letters for white pieces
            else:
                color = BLACK_PIECE_COLOR
                letter_color = WHITE_PIECE_COLOR  # White letters for black pieces
            
            # Draw shapes based on piece type and add letters
            if piece.piece_type == chess.PAWN:
                pygame.draw.circle(screen, color, (x, y), SQUARE_SIZE // 4)
                text = font.render('P', True, letter_color)
            elif piece.piece_type == chess.ROOK:
                pygame.draw.rect(screen, color, pygame.Rect(x - SQUARE_SIZE // 4, y - SQUARE_SIZE // 4, SQUARE_SIZE // 2, SQUARE_SIZE // 2))
                text = font.render('R', True, letter_color)
            elif piece.piece_type == chess.KNIGHT:
                pygame.draw.polygon(screen, color, [(x, y - SQUARE_SIZE // 4), (x - SQUARE_SIZE // 4, y), (x, y + SQUARE_SIZE // 4)])
                text = font.render('N', True, letter_color)
            elif piece.piece_type == chess.BISHOP:
                pygame.draw.polygon(screen, color, [(x, y - SQUARE_SIZE // 4), (x - SQUARE_SIZE // 4, y), (x + SQUARE_SIZE // 4, y)])
                text = font.render('B', True, letter_color)
            elif piece.piece_type == chess.QUEEN:
                pygame.draw.polygon(screen, color, [(x, y - SQUARE_SIZE // 4), (x - SQUARE_SIZE // 4, y + SQUARE_SIZE // 4), (x + SQUARE_SIZE // 4, y + SQUARE_SIZE // 4)])
                text = font.render('Q', True, letter_color)
            elif piece.piece_type == chess.KING:
                pygame.draw.rect(screen, color, pygame.Rect(x - SQUARE_SIZE // 8, y - SQUARE_SIZE // 4, SQUARE_SIZE // 4, SQUARE_SIZE // 2))
                text = font.render('K', True, letter_color)

            # Draw the letter on top of the piece
            text_rect = text.get_rect(center=(x, y))
            screen.blit(text, text_rect)

        # Highlight selected piece
        if selected_square is not None and selected_square == square:
            pygame.draw.rect(screen, SELECTED_COLOR, pygame.Rect(chess.square_file(square) * SQUARE_SIZE,
                                                                  (7 - chess.square_rank(square)) * SQUARE_SIZE,
                                                                  SQUARE_SIZE, SQUARE_SIZE), 5)

# Draw advantage
def draw_advantage(screen, advantage):
    screen.fill(DARK_GRAY, rect=pygame.Rect(0, HEIGHT - 100, WIDTH, 100))  # Draw a background for the advantage bar
    font = pygame.font.SysFont(None, 48)
    color = RED if advantage < 0 else BLUE
    text = font.render(f'Advantage: {"White" if advantage > 0 else "Black"} {abs(advantage):.2f}', True, color)
    screen.blit(text, text.get_rect(center=(WIDTH // 2, HEIGHT - 50)))  # Center text in the advantage area

# Handle user input and moves
def handle_click(pos):
    file = pos[0] // SQUARE_SIZE
    rank = 7 - pos[1] // SQUARE_SIZE  # Adjusted for header
    if 0 <= rank < 8:
        square = chess.square(file, rank)
        return square
    return None

# Initialize the chess board
board = chess.Board()

# Main game loop
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Chess Game with Advantage Prediction')

selected_square = None
advantage = predict_advantage(board)

running = True
while running:
    # Draw the chess board and pieces
    draw_board(screen)
    draw_pieces(screen, board, selected_square)
    
    # Draw the advantage display
    draw_advantage(screen, advantage)
    
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
                        advantage = predict_advantage(board)
                    selected_square = None

pygame.quit()
