class QuantumChessPiece:
    def __init__(self, piece_type, player, pos):
        self.piece_type = piece_type  # 'K', 'Q', 'P'
        self.player = player  # 1 or 2
        self.positions = [pos]  # List of positions for quantum states
        self.entangled_with = None  # Entangled piece, if any
        self.has_tunneling = True  # Available once per piece for quantum tunneling


class QuantumChess:
    def __init__(self):
        self.board = [[None for _ in range(4)] for _ in range(4)]
        self.pieces = []  # List of all pieces in play
        self.quantum_splits_remaining = {1: 3, 2: 3}  # Quantum splits per player
        self.current_player = 1  # Player 1 starts
        self.initialize_pieces()

    def initialize_pieces(self):
        # Initialize pieces on the board for each player
        for player in [1, 2]:
            row = 0 if player == 1 else 3
            self.pieces.append(QuantumChessPiece('K', player, (row, 0)))
            self.pieces.append(QuantumChessPiece('Q', player, (row, 1)))
            for col in [2, 3]:
                self.pieces.append(QuantumChessPiece('P', player, (row, col)))
        self.update_board()

    def update_board(self):
        # Clear the board and place pieces according to their current positions
        self.board = [[None for _ in range(4)] for _ in range(4)]
        for piece in self.pieces:
            for pos in piece.positions:
                self.board[pos[0]][pos[1]] = piece

    def move_piece(self, piece_pos, new_pos):
        # Move piece to new_pos; handle captures
        piece = self.get_piece_at_pos(piece_pos)
        if piece:
            captured_piece = self.get_piece_at_pos(new_pos)
            if captured_piece and captured_piece.player != piece.player:
                self.pieces.remove(captured_piece)
            piece.positions = [new_pos]  # Update positions for non-quantum move
            self.update_board()

    def quantum_split(self, piece_pos, pos1, pos2):
        # Perform a quantum split if available; update positions
        if self.quantum_splits_remaining[self.current_player] > 0:
            piece = self.get_piece_at_pos(piece_pos)
            if piece:
                piece.positions = [pos1, pos2]
                self.quantum_splits_remaining[self.current_player] -= 1
                self.update_board()

    def entangle_pieces(self, piece1_pos, piece2_pos):
        # Entangle two pieces
        piece1 = self.get_piece_at_pos(piece1_pos)
        piece2 = self.get_piece_at_pos(piece2_pos)
        if piece1 and piece2:
            piece1.entangled_with = piece2
            piece2.entangled_with = piece1

    def quantum_tunneling(self, piece_pos, new_pos):
        # Move a piece to a distant position if tunneling is available
        piece = self.get_piece_at_pos(piece_pos)
        if piece and piece.has_tunneling:
            piece.positions = [new_pos]
            piece.has_tunneling = False
            self.update_board()

    def get_piece_at_pos(self, pos):
        # Find and return the piece at the specified position
        for piece in self.pieces:
            if pos in piece.positions:
                return piece
        return None

    def switch_player(self):
        # Switch to the other player
        self.current_player = 2 if self.current_player == 1 else 1

    def is_in_check(self, player):
        # Check if the specified player is in check
        king_pos = None
        for piece in self.pieces:
            if piece.player == player and piece.piece_type == 'K':
                king_pos = piece.positions[0]
                break
        if king_pos:
            for piece in self.pieces:
                if piece.player != player:
                    if king_pos in self.get_possible_moves(piece):
                        return True
        return False

    def is_in_checkmate(self, player):
        # Check if the specified player is in checkmate
        if self.is_in_check(player):
            for piece in self.pieces:
                if piece.player == player:
                    for move in self.get_possible_moves(piece):
                        # Simulate the move
                        original_pos = piece.positions[0]
                        piece.positions = [move]
                        self.update_board()
                        if not self.is_in_check(player):
                            # Undo the move
                            piece.positions = [original_pos]
                            self.update_board()
                            return False
                        # Undo the move
                        piece.positions = [original_pos]
                        self.update_board()
            return True
        return False

    def get_possible_moves(self, piece):
        possible_moves = []
        row, col = piece.positions[0]

        if piece.piece_type == 'K':
            # King moves
            possible_moves = [
                (row - 1, col - 1), (row - 1, col), (row - 1, col + 1),
                (row, col - 1), (row, col + 1),
                (row + 1, col - 1), (row + 1, col), (row + 1, col + 1)
            ]
        elif piece.piece_type == 'Q':
            # Queen moves
            for i in range(4):
                possible_moves.extend([(row - i - 1, col), (row + i + 1, col), (row, col - i - 1), (row, col + i + 1)])
                possible_moves.extend(
                    [(row - i - 1, col - i - 1), (row - i - 1, col + i + 1), (row + i + 1, col - i - 1),
                     (row + i + 1, col + i + 1)])
        elif piece.piece_type == 'P':
            # Pawn moves
            direction = -1 if piece.player == 1 else 1
            possible_moves = [(row + direction, col)]
            if (row == 1 and piece.player == 1) or (row == 2 and piece.player == 2):
                possible_moves.append((row + 2 * direction, col))

        # Filter out moves that are out of bounds
        possible_moves = [(r, c) for r, c in possible_moves if 0 <= r < 4 and 0 <= c < 4]

        return possible_moves

    def is_valid_move(self, from_pos, to_pos):
        # Check if the move is valid
        piece = self.get_piece_at_pos(from_pos)
        if piece and piece.player == self.current_player:
            if to_pos in self.get_possible_moves(piece):
                return True
        return False

    def is_valid_split(self, piece_pos, split_pos1, split_pos2):
        # Check if the quantum split is valid
        piece = self.get_piece_at_pos(piece_pos)
        if piece and piece.player == self.current_player:
            if self.quantum_splits_remaining[self.current_player] > 0:
                if split_pos1 in self.get_possible_moves(piece) and split_pos2 in self.get_possible_moves(piece):
                    return True
        return False

    def is_valid_entanglement(self, piece1_pos, piece2_pos):
        # Check if the entanglement is valid
        piece1 = self.get_piece_at_pos(piece1_pos)
        piece2 = self.get_piece_at_pos(piece2_pos)
        if piece1 and piece2 and piece1.player == piece2.player:
            if piece1.piece_type != 'K' and piece2.piece_type != 'K':
                return True
        return False

    def is_valid_tunneling(self, piece_pos, tunnel_pos):
        # Check if the quantum tunneling is valid
        piece = self.get_piece_at_pos(piece_pos)
        if piece and piece.player == self.current_player:
            if piece.has_tunneling:
                if tunnel_pos in self.get_empty_positions():
                    return True
        return False

    def get_empty_positions(self):
        # Get all empty positions on the board
        empty_positions = []
        for row in range(4):
            for col in range(4):
                if not self.board[row][col]:
                    empty_positions.append((row, col))
        return empty_positions


def get_game_state(game, player):
    board_state = ""
    for row in game.board:
        for piece in row:
            if piece and piece.player == player:
                board_state += piece.piece_type + " "
            else:
                board_state += ". "
        board_state += "\n"

    state = f"""
Player {player}'s perspective:
Board:
{board_state}
Quantum splits remaining: {game.quantum_splits_remaining[player]}
"""

    return state


def play_quantum_chess():
    game = QuantumChess()
    print("Welcome to Quantum Chess!")
    print("Player 1: White")
    print("Player 2: Black")

    while True:
        # Prompt Player 1 for their move
        while True:
            player1_state = get_game_state(game, 1)
            player1_move = input(f"Player 1's move:\n{player1_state}\nEnter your move: ")

            # Process Player 1's move
            move_parts = player1_move.split()

            if len(move_parts) == 2:
                # Regular move
                from_pos = algebraic_to_coordinate(move_parts[0])
                to_pos = algebraic_to_coordinate(move_parts[1])
                if game.is_valid_move(from_pos, to_pos):
                    game.move_piece(from_pos, to_pos)
                    break
                else:
                    print("Invalid move. Please try again.")
            elif len(move_parts) == 4 and move_parts[0] == 'split':
                # Quantum split
                piece_pos = algebraic_to_coordinate(move_parts[1])
                split_pos1 = algebraic_to_coordinate(move_parts[2])
                split_pos2 = algebraic_to_coordinate(move_parts[3])
                if game.is_valid_split(piece_pos, split_pos1, split_pos2):
                    game.quantum_split(piece_pos, split_pos1, split_pos2)
                    break
                else:
                    print("Invalid quantum split. Please try again.")
            elif len(move_parts) == 3 and move_parts[0] == 'entangle':
                # Entangle pieces
                piece1_pos = algebraic_to_coordinate(move_parts[1])
                piece2_pos = algebraic_to_coordinate(move_parts[2])
                if game.is_valid_entanglement(piece1_pos, piece2_pos):
                    game.entangle_pieces(piece1_pos, piece2_pos)
                    break
                else:
                    print("Invalid entanglement. Please try again.")
            elif len(move_parts) == 3 and move_parts[0] == 'tunnel':
                # Quantum tunneling
                piece_pos = algebraic_to_coordinate(move_parts[1])
                tunnel_pos = algebraic_to_coordinate(move_parts[2])
                if game.is_valid_tunneling(piece_pos, tunnel_pos):
                    game.quantum_tunneling(piece_pos, tunnel_pos)
                    break
                else:
                    print("Invalid quantum tunneling. Please try again.")
            else:
                print("Invalid move format. Please try again.")

        # Check for checkmate
        if game.is_in_checkmate(2):
            print("\nPlayer 1 wins!")
            break

        # Prompt Player 2 for their move
        while True:
            player2_state = get_game_state(game, 2)
            player2_move = input(f"Player 2's move:\n{player2_state}\nEnter your move: ")

            # Process Player 2's move
            move_parts = player2_move.split()

            if len(move_parts) == 2:
                # Regular move
                from_pos = algebraic_to_coordinate(move_parts[0])
                to_pos = algebraic_to_coordinate(move_parts[1])
                if game.is_valid_move(from_pos, to_pos):
                    game.move_piece(from_pos, to_pos)
                    break
                else:
                    print("Invalid move. Please try again.")
            elif len(move_parts) == 4 and move_parts[0] == 'split':
                # Quantum split
                piece_pos = algebraic_to_coordinate(move_parts[1])
                split_pos1 = algebraic_to_coordinate(move_parts[2])
                split_pos2 = algebraic_to_coordinate(move_parts[3])
                if game.is_valid_split(piece_pos, split_pos1, split_pos2):
                    game.quantum_split(piece_pos, split_pos1, split_pos2)
                    break
                else:
                    print("Invalid quantum split. Please try again.")
            elif len(move_parts) == 3 and move_parts[0] == 'entangle':
                # Entangle pieces
                piece1_pos = algebraic_to_coordinate(move_parts[1])
                piece2_pos = algebraic_to_coordinate(move_parts[2])
                if game.is_valid_entanglement(piece1_pos, piece2_pos):
                    game.entangle_pieces(piece1_pos, piece2_pos)
                    break
                else:
                    print("Invalid entanglement. Please try again.")
            elif len(move_parts) == 3 and move_parts[0] == 'tunnel':
                # Quantum tunneling
                piece_pos = algebraic_to_coordinate(move_parts[1])
                tunnel_pos = algebraic_to_coordinate(move_parts[2])
                if game.is_valid_tunneling(piece_pos, tunnel_pos):
                    game.quantum_tunneling(piece_pos, tunnel_pos)
                    break
                else:
                    print("Invalid quantum tunneling. Please try again.")
            else:
                print("Invalid move format. Please try again.")

        # Check for checkmate
        if game.is_in_checkmate(1):
            print("\nPlayer 2 wins!")
            break


def print_board(board):
    # Print the current state of the board
    for row in board:
        print(' '.join(piece.piece_type if piece else '.' for piece in row))


def algebraic_to_coordinate(algebraic):
    # Convert algebraic notation (e.g., 'a2') to coordinate tuple (e.g., (1, 0))
    col = ord(algebraic[0]) - ord('a')
    row = int(algebraic[1]) - 1
    return (row, col)


# Start the game
play_quantum_chess()
