import pygame
import chess
import joblib
import numpy as np
import os
from typing import Optional, Dict, Tuple, List
import sys
from engine.model import ChessEngine

# Constants
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
DIALOG_BG = (245, 245, 245)
DIALOG_BORDER = (100, 100, 100)

# Piece Images
PIECE_IMAGES = {
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

class ChessGame:
    def __init__(self):
        pygame.init()
        
        # Screen dimensions
        self.WIDTH = WINDOW_WIDTH
        self.HEIGHT = WINDOW_HEIGHT
        self.BOARD_SIZE = BOARD_SIZE
        self.SQUARE_SIZE = SQUARE_SIZE
        
        # Modern color scheme
        self.LIGHT_SQUARES = LIGHT_SQUARE
        self.DARK_SQUARES = DARK_SQUARE
        self.SELECTED_COLOR = SELECTED_COLOR
        self.HOVER_COLOR = HOVER_COLOR
        self.DARK_GRAY = BLACK
        self.RED = (231, 76, 60)
        self.BLUE = (52, 152, 219)
        self.WHITE = WHITE
        self.BLACK = BLACK
        self.GRAY = GRAY
        
        # Game state
        self.board = chess.Board()
        self.selected_square: Optional[int] = None
        self.hover_square: Optional[int] = None
        self.advantage = 0
        self.move_history: List[str] = []
        self.game_over = False
        self.current_player = "White"
        
        # Load resources
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption('Chess Game with Advantage Prediction')
        self.pieces = self.load_images()
        self.engine = self.load_engine()
        
        # Load fonts
        self.title_font = pygame.font.SysFont('Arial', 36, bold=True)
        self.text_font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 18)
        
        # FPS control
        self.clock = pygame.time.Clock()
        self.FPS = FPS

        # Promotion state
        self.awaiting_promotion = False
        self.promotion_move = None
        self.promotion_color = None

    def load_engine(self) -> ChessEngine:
        try:
            engine = ChessEngine(MODEL_PATH)
            engine.load_model()
            return engine
        except Exception as e:
            print(f"Error loading engine: {e}")
            sys.exit(1)

    def load_images(self) -> Dict[str, pygame.Surface]:
        pieces = {}
        try:
            for piece_name, image_path in PIECE_IMAGES.items():
                pieces[piece_name] = pygame.image.load(image_path).convert_alpha()
                pieces[piece_name] = pygame.transform.scale(
                    pieces[piece_name], 
                    (self.SQUARE_SIZE, self.SQUARE_SIZE)
                )
            return pieces
        except Exception as e:
            print(f"Error loading images: {e}")
            sys.exit(1)

    def predict_advantage(self) -> float:
        return self.engine.predict(self.board)

    def draw_board(self):
        # Calculate board position to center it
        board_x = (self.WIDTH - self.BOARD_SIZE - 300) // 2
        board_y = (self.HEIGHT - self.BOARD_SIZE) // 2
        
        # Draw board background
        pygame.draw.rect(self.screen, self.DARK_GRAY, 
                        (0, 0, self.WIDTH, self.HEIGHT))
        
        # Draw chess board
        for r in range(8):
            for c in range(8):
                color = self.LIGHT_SQUARES if (r + c) % 2 == 0 else self.DARK_SQUARES
                pygame.draw.rect(
                    self.screen, 
                    color, 
                    pygame.Rect(board_x + c * self.SQUARE_SIZE, 
                              board_y + r * self.SQUARE_SIZE, 
                              self.SQUARE_SIZE, self.SQUARE_SIZE)
                )
        
        # Draw board border
        pygame.draw.rect(self.screen, self.BLACK, 
                        (board_x, board_y, self.BOARD_SIZE, self.BOARD_SIZE), 2)
        
        # Draw coordinates
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['8', '7', '6', '5', '4', '3', '2', '1']
        
        for i, file in enumerate(files):
            text = self.small_font.render(file, True, self.BLACK)
            self.screen.blit(text, (board_x + i * self.SQUARE_SIZE + self.SQUARE_SIZE // 3, 
                                  board_y + self.BOARD_SIZE + 5))
        
        for i, rank in enumerate(ranks):
            text = self.small_font.render(rank, True, self.BLACK)
            self.screen.blit(text, (board_x - 20, board_y + i * self.SQUARE_SIZE + self.SQUARE_SIZE // 3))

    def draw_pieces(self):
        board_x = (self.WIDTH - self.BOARD_SIZE - 300) // 2
        board_y = (self.HEIGHT - self.BOARD_SIZE) // 2
        
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                x = board_x + chess.square_file(square) * self.SQUARE_SIZE
                y = board_y + (7 - chess.square_rank(square)) * self.SQUARE_SIZE
                piece_image = self.pieces[f"{'w' if piece.color == chess.WHITE else 'b'}{piece.symbol().upper()}"]
                self.screen.blit(piece_image, (x, y))

            # Draw hover effect
            if self.hover_square is not None and self.hover_square == square:
                s = pygame.Surface((self.SQUARE_SIZE, self.SQUARE_SIZE))
                s.set_alpha(128)
                s.fill(self.HOVER_COLOR)
                self.screen.blit(s, (board_x + chess.square_file(square) * self.SQUARE_SIZE,
                                   board_y + (7 - chess.square_rank(square)) * self.SQUARE_SIZE))

            # Draw selected piece highlight
            if self.selected_square is not None and self.selected_square == square:
                pygame.draw.rect(
                    self.screen, 
                    self.SELECTED_COLOR, 
                    pygame.Rect(board_x + chess.square_file(square) * self.SQUARE_SIZE,
                              board_y + (7 - chess.square_rank(square)) * self.SQUARE_SIZE,
                              self.SQUARE_SIZE, self.SQUARE_SIZE), 
                    3
                )

    def highlight_legal_moves(self):
        board_x = (self.WIDTH - self.BOARD_SIZE - 300) // 2
        board_y = (self.HEIGHT - self.BOARD_SIZE) // 2
        
        if self.selected_square is not None:
            for move in self.board.legal_moves:
                if move.from_square == self.selected_square:
                    to_square = move.to_square
                    file = chess.square_file(to_square)
                    rank = chess.square_rank(to_square)
                    # Draw circle for legal moves
                    pygame.draw.circle(
                        self.screen, 
                        self.SELECTED_COLOR, 
                        (board_x + file * self.SQUARE_SIZE + self.SQUARE_SIZE // 2, 
                         board_y + (7 - rank) * self.SQUARE_SIZE + self.SQUARE_SIZE // 2), 
                        self.SQUARE_SIZE // 6
                    )

    def draw_advantage(self):
        # Draw status panel background
        panel_width = 280
        panel_height = self.HEIGHT - 40
        panel_x = self.WIDTH - panel_width - 10
        panel_y = 20
        
        # Draw panel with shadow
        pygame.draw.rect(self.screen, self.GRAY, 
                        (panel_x + 2, panel_y + 2, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.WHITE, 
                        (panel_x, panel_y, panel_width, panel_height))
        pygame.draw.rect(self.screen, self.DARK_GRAY, 
                        (panel_x, panel_y, panel_width, panel_height), 2)
        
        # Draw title with underline
        title = self.title_font.render("Game Status", True, self.DARK_GRAY)
        self.screen.blit(title, (panel_x + 20, panel_y + 20))
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (panel_x + 20, panel_y + 60),
                        (panel_x + panel_width - 20, panel_y + 60), 2)
        
        # Draw current player with icon
        player_text = self.text_font.render(f"Current Player: {self.current_player}", 
                                          True, self.DARK_GRAY)
        self.screen.blit(player_text, (panel_x + 20, panel_y + 80))
        
        # Draw advantage bar with gradient
        bar_width = panel_width - 40
        bar_height = 20
        bar_x = panel_x + 20
        bar_y = panel_y + 120
        
        # Draw bar background with rounded corners
        pygame.draw.rect(self.screen, (200, 200, 200), 
                        (bar_x, bar_y, bar_width, bar_height))
        
        # Calculate bar fill
        max_advantage = 10
        normalized_advantage = max(min(self.advantage, max_advantage), -max_advantage)
        fill_width = int((normalized_advantage + max_advantage) * bar_width / (2 * max_advantage))
        
        # Draw filled portion with gradient
        color = self.BLUE if normalized_advantage >= 0 else self.RED
        pygame.draw.rect(self.screen, color, 
                        (bar_x, bar_y, fill_width, bar_height))
        
        # Draw advantage text with shadow
        shadow_text = self.text_font.render(
            f'Advantage: {"White" if self.advantage > 0 else "Black"} {abs(self.advantage):.2f}', 
            True, 
            self.GRAY
        )
        self.screen.blit(shadow_text, (bar_x + 1, bar_y + 31))
        
        text = self.text_font.render(
            f'Advantage: {"White" if self.advantage > 0 else "Black"} {abs(self.advantage):.2f}', 
            True, 
            color
        )
        self.screen.blit(text, (bar_x, bar_y + 30))

    def draw_move_history(self):
        panel_width = 280
        panel_x = self.WIDTH - panel_width - 10
        panel_y = 200
        
        # Draw move history title with underline
        title = self.text_font.render("Move History", True, self.DARK_GRAY)
        self.screen.blit(title, (panel_x + 20, panel_y))
        pygame.draw.line(self.screen, self.DARK_GRAY,
                        (panel_x + 20, panel_y + 30),
                        (panel_x + panel_width - 20, panel_y + 30), 2)
        
        # Draw moves with alternating background
        y = panel_y + 40
        for i, move in enumerate(self.move_history[-10:]):  # Show last 10 moves
            # Draw alternating background
            if i % 2 == 0:
                pygame.draw.rect(self.screen, (245, 245, 245),
                               (panel_x + 10, y - 5, panel_width - 20, 25))
            
            text = self.small_font.render(move, True, self.DARK_GRAY)
            self.screen.blit(text, (panel_x + 20, y))
            y += 25

    def handle_click(self, pos: Tuple[int, int]) -> Optional[int]:
        board_x = (self.WIDTH - self.BOARD_SIZE - 300) // 2
        board_y = (self.HEIGHT - self.BOARD_SIZE) // 2
        
        # Adjust click position relative to board
        adjusted_x = pos[0] - board_x
        adjusted_y = pos[1] - board_y
        
        if 0 <= adjusted_x < self.BOARD_SIZE and 0 <= adjusted_y < self.BOARD_SIZE:
            file = adjusted_x // self.SQUARE_SIZE
            rank = 7 - adjusted_y // self.SQUARE_SIZE
            if 0 <= rank < 8:
                square = chess.square(file, rank)
                return square
        return None

    def is_promotion_move(self, from_square: int, to_square: int) -> bool:
        """Check if a move would result in pawn promotion."""
        piece = self.board.piece_at(from_square)
        if piece is None or piece.piece_type != chess.PAWN:
            return False
        
        # Check if pawn reaches the opposite end
        rank = chess.square_rank(to_square)
        return (piece.color == chess.WHITE and rank == 7) or \
               (piece.color == chess.BLACK and rank == 0)

    def show_promotion_dialog(self, color: bool):
        """Show dialog for pawn promotion piece selection."""
        dialog_width = 200
        dialog_height = 280
        dialog_x = (self.WIDTH - dialog_width) // 2
        dialog_y = (self.HEIGHT - dialog_height) // 2

        # Draw dialog background with border
        pygame.draw.rect(self.screen, DIALOG_BG, 
                        (dialog_x, dialog_y, dialog_width, dialog_height))
        pygame.draw.rect(self.screen, DIALOG_BORDER, 
                        (dialog_x, dialog_y, dialog_width, dialog_height), 2)

        # Draw title
        title = self.text_font.render("Choose Promotion", True, BLACK)
        title_rect = title.get_rect(centerx=dialog_x + dialog_width//2, 
                                  y=dialog_y + 10)
        self.screen.blit(title, title_rect)

        # Draw piece options
        pieces = ['Q', 'R', 'B', 'N']
        piece_size = 60
        for i, piece in enumerate(pieces):
            piece_key = f"{'w' if color else 'b'}{piece}"
            piece_img = pygame.transform.scale(self.pieces[piece_key], 
                                            (piece_size, piece_size))
            x = dialog_x + (dialog_width - piece_size) // 2
            y = dialog_y + 50 + i * (piece_size + 10)
            
            # Draw piece background
            pygame.draw.rect(self.screen, WHITE, 
                           (x - 5, y - 5, piece_size + 10, piece_size + 10))
            pygame.draw.rect(self.screen, DIALOG_BORDER, 
                           (x - 5, y - 5, piece_size + 10, piece_size + 10), 1)
            
            # Draw piece
            self.screen.blit(piece_img, (x, y))

    def get_promotion_piece(self, pos: Tuple[int, int]) -> Optional[chess.PieceType]:
        """Get the selected promotion piece from click position."""
        dialog_width = 200
        dialog_height = 280
        dialog_x = (self.WIDTH - dialog_width) // 2
        dialog_y = (self.HEIGHT - dialog_height) // 2
        piece_size = 60

        # Check if click is within dialog
        if not (dialog_x <= pos[0] <= dialog_x + dialog_width and 
                dialog_y <= pos[1] <= dialog_y + dialog_height):
            return None

        # Calculate which piece was clicked
        pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
        for i, piece in enumerate(pieces):
            y = dialog_y + 50 + i * (piece_size + 10)
            if (dialog_x + (dialog_width - piece_size) // 2 - 5 <= pos[0] <= 
                dialog_x + (dialog_width + piece_size) // 2 + 5 and
                y - 5 <= pos[1] <= y + piece_size + 5):
                return piece
        return None

    def handle_move(self, from_square: int, to_square: int) -> bool:
        """Handle chess move including pawn promotion."""
        if self.is_promotion_move(from_square, to_square):
            self.awaiting_promotion = True
            self.promotion_move = (from_square, to_square)
            self.promotion_color = self.board.turn
            return False

        move = chess.Move(from_square, to_square)
        if move in self.board.legal_moves:
            self.board.push(move)
            self.advantage = self.predict_advantage()
            self.move_history.append(f"{len(self.move_history) + 1}. {move.uci()}")
            self.current_player = "Black" if self.current_player == "White" else "White"
            return True
        return False

    def run(self):
        running = True
        while running:
            self.clock.tick(self.FPS)
            
            # Update hover square if not awaiting promotion
            if not self.awaiting_promotion:
                mouse_pos = pygame.mouse.get_pos()
                self.hover_square = self.handle_click(mouse_pos)
            
            # Draw everything
            self.draw_board()
            self.draw_pieces()
            self.highlight_legal_moves()
            self.draw_advantage()
            self.draw_move_history()
            
            # Draw promotion dialog if needed
            if self.awaiting_promotion:
                self.show_promotion_dialog(self.promotion_color)
            
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if self.awaiting_promotion:
                        promotion_piece = self.get_promotion_piece(event.pos)
                        if promotion_piece:
                            # Create and execute promotion move
                            move = chess.Move(
                                self.promotion_move[0],
                                self.promotion_move[1],
                                promotion=promotion_piece
                            )
                            if move in self.board.legal_moves:
                                self.board.push(move)
                                self.advantage = self.predict_advantage()
                                self.move_history.append(f"{len(self.move_history) + 1}. {move.uci()}")
                                self.current_player = "Black" if self.current_player == "White" else "White"
                            
                            # Reset promotion state
                            self.awaiting_promotion = False
                            self.promotion_move = None
                            self.promotion_color = None
                            self.selected_square = None
                    else:
                        square = self.handle_click(event.pos)
                        if square is not None:
                            if self.selected_square is None:
                                if self.board.piece_at(square):
                                    self.selected_square = square
                            else:
                                if self.handle_move(self.selected_square, square):
                                    self.selected_square = None
                                else:
                                    self.selected_square = square if self.board.piece_at(square) else None
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_r:  # Reset game
                        self.board = chess.Board()
                        self.selected_square = None
                        self.advantage = 0
                        self.move_history = []
                        self.current_player = "White"
                        self.awaiting_promotion = False
                        self.promotion_move = None
                        self.promotion_color = None
                    elif event.key == pygame.K_ESCAPE:  # Cancel selection or promotion
                        if self.awaiting_promotion:
                            self.awaiting_promotion = False
                            self.promotion_move = None
                            self.promotion_color = None
                        self.selected_square = None

        pygame.quit()

if __name__ == "__main__":
    game = ChessGame()
    game.run()
