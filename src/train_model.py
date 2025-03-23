import os
from engine.model import ChessEngine
import chess.engine
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    # Paths
    pgn_dir = 'assets/pgn'
    model_path = 'assets/models/chess_model.pkl'
    stockfish_path = 'assets/stockfish/stockfish-windows-x86-64-avx2/stockfish/stockfish-windows-x86-64-avx2.exe'
    
    # Initialize engine
    engine = ChessEngine(model_path)
    
    # Get list of PGN files
    pgn_files = [f for f in os.listdir(pgn_dir) if f.endswith('.pgn')]
    
    if not pgn_files:
        logging.error("No PGN files found in %s", pgn_dir)
        return
    
    # Train model
    try:
        engine.train(
            pgn_files=[os.path.join(pgn_dir, f) for f in pgn_files],
            stockfish_path=stockfish_path,
            model_type='gradient_boosting'
        )
        logging.info("Model training completed successfully")
    except Exception as e:
        logging.error("Error during model training: %s", str(e))

if __name__ == '__main__':
    main() 