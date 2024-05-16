# Quantum Chess

Quantum Chess is a unique chess variant that introduces quantum mechanics concepts such as superposition, entanglement, and quantum tunneling to a traditional chess-like game. The game is played on a 4x4 board with each player having 4 pieces: 1 king (K), 1 queen (Q), and 2 pawns (P).

## Game Setup
- The board is a 4x4 grid with rows labeled 1-4 and columns labeled a-d.
- Each player starts with their pieces on the first row (1 for Player 1, 4 for Player 2).
- The initial positions are: King (a1/a4), Queen (b1/b4), Pawns (c1/c4, d1/d4).

## Piece Movements
- **King**: Moves one square in any direction (horizontal, vertical, or diagonal).
- **Queen**: Moves any number of squares in any direction (horizontal, vertical, or diagonal).
- **Pawn**: Moves one square forward.

## Quantum Moves
- **Quantum Split**: A piece can split into two quantum states, occupying two different positions at once. Each player has 3 quantum splits per game. To perform a quantum split, use the command "split [piece] [pos1] [pos2]".
- **Quantum Entanglement**: Two pieces of the same player can be entangled, connecting their fates. If one entangled piece is captured, the other is also removed. To entangle pieces, use the command "entangle [piece1] [piece2]".
- **Quantum Tunneling**: Once per game, each piece can perform a quantum tunneling move, allowing it to move to any empty square on the board. To perform quantum tunneling, use the command "tunnel [piece] [pos]".

## Game Play
1. Players take turns making moves.
2. On each turn, a player can make one of the following moves:
   - Regular Move: Move a piece to a new position using algebraic notation (e.g., "a2 a4").
   - Quantum Split: Split a piece into two quantum states using the "split" command (e.g., "split a2 a3 a4").
   - Quantum Entanglement: Entangle two pieces using the "entangle" command (e.g., "entangle a2 b2").
   - Quantum Tunneling: Move a piece to any empty square using the "tunnel" command (e.g., "tunnel a2 c3").
3. After each move, the game will update the board and check for any captures or checkmate.
4. The game ends when a player's king is in checkmate (i.e., the king is under attack and there is no legal move to escape).

## Additional Rules
- **Captures**: If a piece moves to a square occupied by an opponent's piece, the opponent's piece is captured and removed from the board.
- **Check**: If a player's king is under attack by an opponent's piece, the king is in check. The player must make a move to get out of check.
- **Checkmate**: If a player's king is in check and there is no legal move to get out of check, the king is in checkmate, and the game is over.

## Scripts

### qc.py
The main game script for Quantum Chess. This script contains the game logic, including piece movements, quantum mechanics moves, and game state checks.

### qc_prompt_instruct.txt
Instructions for players on how to play Quantum Chess, including game setup, piece movements, quantum moves, and additional rules.

## Future Expansion Plans
- Implement AI for single-player mode.
- Add more quantum mechanics features like quantum interference.
- Develop a graphical user interface (GUI) for the game.
- Create an online multiplayer version.
- Add more piece types with unique quantum abilities.

## Starting the Game
To start the game, run the `qc.py` script and follow the prompts to make your moves. Good luck and have fun playing Quantum Chess!
