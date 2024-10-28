import chess
import chess.pgn
import chess.engine
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

# Function to load and parse multiple PGN files
def load_games(file_paths):
    games = []
    for file_path in file_paths:
        if os.path.exists(file_path):
            print(f"Loading games from {file_path}...")
            with open(file_path, encoding='utf-8') as pgn:  # Specify UTF-8 encoding
                while True:
                    game = chess.pgn.read_game(pgn)
                    if game is None:
                        break
                    games.append(game)
            print(f'Loaded {len(games)} games from {file_path}.')
        else:
            raise FileNotFoundError(f"PGN file not found at {file_path}")
    return games

# Function to extract features and labels from games using Stockfish
def extract_features_and_labels(games, engine_path):
    if not os.path.exists(engine_path):
        raise FileNotFoundError(f"Stockfish engine not found at {engine_path}")
    
    print("Starting Stockfish engine...")
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    data = []
    labels = []
    for game in games:
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
            info = engine.analyse(board, chess.engine.Limit(time=0.1))
            score = info["score"].relative.score(mate_score=10000) or 0
            features = flatten_piece_map(board.piece_map())
            data.append(features)
            labels.append(score)
    engine.quit()
    print(f'Extracted {len(data)} features and labels using Stockfish.')
    return data, labels

# Flatten the piece_map dictionary into a feature vector for model compatibility
def flatten_piece_map(piece_map):
    flattened = [0] * 64 * 6 * 2  # 64 squares, 6 pieces, 2 colors
    for square, piece in piece_map.items():
        piece_type_index = (piece.piece_type - 1) * 2
        if piece.color == chess.BLACK:
            piece_type_index += 1
        flattened[square * 12 + piece_type_index] = 1
    return flattened

# Load the games from the specified PGN files
file_paths = [
    'C:\\Users\\Ayush\\Desktop\\Bobby Fischer - My 60 Memorable Games.pgn',
    'C:\\Users\\Ayush\\Desktop\\The Most Instructive Chess Games.pgn',
    'C:\\Users\\Ayush\\Desktop\\The Best Chess Studies.pgn'
]

games = load_games(file_paths)

# Extract features and labels using Stockfish
engine_path = 'C:\\Users\\Ayush\\Desktop\\Chess engime\\stockfish\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe'
data, labels = extract_features_and_labels(games, engine_path)

# Split the data
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

# Train the model
print("Training the model...")
model = RandomForestRegressor()
model.fit(X_train, y_train)
print("Model training completed.")

# Evaluate the model
print("Evaluating the model...")
y_pred = model.predict(X_test)
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')

# Save the model
print("Saving the model...")
joblib.dump(model, 'chess_engine_model.pkl')
print("Model saved.")

# Function to predict advantage
def predict_advantage(board):
    piece_map = board.piece_map()
    features = flatten_piece_map(piece_map)
    return model.predict([features])[0]

# Example usage
print("Making a prediction on a new board position...")
board = chess.Board()
advantage = predict_advantage(board)
print(f'Predicted Advantage: {advantage}')
