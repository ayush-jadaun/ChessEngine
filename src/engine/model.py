import chess
import chess.pgn
import chess.engine
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import joblib
import os
from typing import List, Tuple, Dict
import logging
import sklearn

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChessEngine:
    def __init__(self, model_path: str = None):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.feature_names = self._get_feature_names()
        
    def _get_feature_names(self) -> List[str]:
        """Generate feature names for the model."""
        features = []
        pieces = ['P', 'R', 'N', 'B', 'Q', 'K']
        colors = ['w', 'b']
        for square in range(64):
            for piece in pieces:
                for color in colors:
                    features.append(f"{square}_{color}{piece}")
        return features

    def extract_position_features(self, board: chess.Board) -> np.ndarray:
        """Extract position features in the old format."""
        piece_map = board.piece_map()
        position_features = [0] * 768  # 64 squares * 12 features per square
        for square, piece in piece_map.items():
            piece_type_index = (piece.piece_type - 1) * 2
            if piece.color == chess.BLACK:
                piece_type_index += 1
            position_features[square * 12 + piece_type_index] = 1
        return np.array(position_features)

    def train(self, games: List[chess.pgn.Game], engine_path: str, 
              model_type: str = 'gradient_boosting') -> None:
        """Train the model on a collection of games."""
        logger.info("Starting model training...")
        
        # Extract features and labels
        X, y = self._prepare_training_data(games, engine_path)
        
        # Scale features
        X = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Initialize and train model
        if model_type == 'gradient_boosting':
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        mse = mean_squared_error(y_test, self.model.predict(X_test))
        mae = mean_absolute_error(y_test, self.model.predict(X_test))
        
        logger.info(f"Training R² score: {train_score:.4f}")
        logger.info(f"Testing R² score: {test_score:.4f}")
        logger.info(f"Mean Squared Error: {mse:.4f}")
        logger.info(f"Mean Absolute Error: {mae:.4f}")
        
        # Save model and scaler
        if self.model_path:
            self.save_model()

    def _prepare_training_data(self, games: List[chess.pgn.Game], 
                             engine_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from games using Stockfish."""
        logger.info("Preparing training data...")
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Stockfish engine not found at {engine_path}")
        
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
        X, y = [], []
        
        for game in games:
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
                features = self.extract_position_features(board)
                info = engine.analyse(board, chess.engine.Limit(time=0.1))
                score = info["score"].relative.score(mate_score=10000) or 0
                X.append(features)
                y.append(score)
        
        engine.quit()
        logger.info(f"Extracted {len(X)} positions")
        return np.array(X), np.array(y)

    def predict(self, board: chess.Board) -> float:
        """Predict the advantage for a given position."""
        if self.model is None:
            raise ValueError("Model not trained or loaded")
        
        features = self.extract_position_features(board)
        try:
            features = self.scaler.transform(features.reshape(1, -1))
        except sklearn.exceptions.NotFittedError:
            # If scaler is not fitted, just use the raw features
            features = features.reshape(1, -1)
        return self.model.predict(features)[0]

    def save_model(self) -> None:
        """Save the model and scaler."""
        if self.model_path:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }
            joblib.dump(model_data, self.model_path)
            logger.info(f"Model saved to {self.model_path}")

    def load_model(self) -> None:
        """Load the model and scaler."""
        if self.model_path and os.path.exists(self.model_path):
            try:
                # Try loading new format
                model_data = joblib.load(self.model_path)
                if isinstance(model_data, dict):
                    self.model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.feature_names = model_data['feature_names']
                else:
                    # Old format - just the model
                    self.model = model_data
                    self.scaler = StandardScaler()
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                raise FileNotFoundError(f"Error loading model: {e}")
        else:
            raise FileNotFoundError(f"Model file not found at {self.model_path}")

def train_model(pgn_files: List[str], engine_path: str, 
                model_path: str, model_type: str = 'gradient_boosting') -> None:
    """Train a new model from PGN files."""
    engine = ChessEngine(model_path)
    
    # Load games
    games = []
    for file_path in pgn_files:
        if os.path.exists(file_path):
            logger.info(f"Loading games from {file_path}...")
            with open(file_path, encoding='utf-8') as pgn:
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    games.append(game)
            logger.info(f'Loaded {len(games)} games from {file_path}.')
        else:
            raise FileNotFoundError(f"PGN file not found at {file_path}")
    
    # Train model
    engine.train(games, engine_path, model_type)

if __name__ == "__main__":
    # Example usage
    pgn_files = [
        'data/pgn/Bobby Fischer - My 60 Memorable Games.pgn',
        'data/pgn/The Most Instructive Chess Games.pgn',
        'data/pgn/The Best Chess Studies.pgn'
    ]
    engine_path = 'assets/stockfish/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe'
    model_path = 'assets/models/chess_model.pkl'
    
    train_model(pgn_files, engine_path, model_path, model_type='gradient_boosting')
