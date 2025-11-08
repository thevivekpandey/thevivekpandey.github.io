"""
Engine vs Engine Match
Plays multiple games between two Mini Leela engines and tracks results.
"""

import chess
import argparse
from typing import Optional
from leela_engine import Engine


def play_game(white_engine: Engine, black_engine: Engine,
              verbose: bool = True, max_moves: int = 200) -> tuple[Optional[str], list[str]]:
    """
    Play a single game between two engines.

    Args:
        white_engine: Engine playing as white
        black_engine: Engine playing as black
        verbose: If True, print moves as they are played
        max_moves: Maximum number of moves before declaring a draw

    Returns:
        Tuple of (result, moves) where:
            result is 'white', 'black', or 'draw'
            moves is list of moves in SAN format
    """
    board = chess.Board()
    moves_san = []
    move_count = 0

    while not board.is_game_over() and move_count < max_moves:
        # Get current engine
        current_engine = white_engine if board.turn == chess.WHITE else black_engine
        current_color = "White" if board.turn == chess.WHITE else "Black"

        try:
            # Get move from engine
            move = current_engine.get_move(board, temperature=0.0)

            # Convert to SAN before making the move
            san_move = board.san(move)
            moves_san.append(san_move)

            if verbose:
                if board.turn == chess.WHITE:
                   mv = len(moves_san) // 2 + 1
                   print(f"{mv}. {san_move}", end=" ")
                else:
                   print(f" {san_move}")

            # Make the move
            board.push(move)
            move_count += 1

        except Exception as e:
            print(f"\nError getting move for {current_color}: {e}")
            # If error occurs, opponent wins
            return 'black' if board.turn == chess.WHITE else 'white', moves_san

    if verbose:
        print()  # New line after moves

    # Determine result
    if board.is_checkmate():
        # The side whose turn it is has been checkmated
        winner = 'black' if board.turn == chess.WHITE else 'white'
    elif board.is_stalemate() or board.is_insufficient_material() or \
         board.can_claim_fifty_moves() or board.can_claim_threefold_repetition() or \
         move_count >= max_moves:
        winner = 'draw'
    else:
        winner = 'draw'

    return winner, moves_san


def play_match(white_engine: Engine, black_engine: Engine,
               num_games: int = 100, verbose: bool = True,
               switch_colors: bool = True) -> dict:
    """
    Play a match of multiple games between two engines.

    Args:
        white_engine: First engine
        black_engine: Second engine
        num_games: Number of games to play
        verbose: If True, print game moves
        switch_colors: If True, engines alternate colors each game

    Returns:
        Dictionary with match statistics
    """
    results = {
        'engine1_wins': 0,  # engine1 is white_engine
        'engine2_wins': 0,  # engine2 is black_engine
        'draws': 0,
        'games': []
    }

    print(f"\n{'='*70}")
    print(f"Starting match: {num_games} games")
    print(f"{'='*70}\n")

    for game_num in range(1, num_games + 1):
        # Determine which engine plays which color
        if switch_colors and game_num % 2 == 0:
            # Switch colors every other game
            white, black = black_engine, white_engine
            white_label, black_label = "Engine 2", "Engine 1"
        else:
            white, black = white_engine, black_engine
            white_label, black_label = "Engine 1", "Engine 2"

        print(f"\n{'='*70}")
        print(f"Game {game_num}/{num_games}")
        print(f"White: {white_label}, Black: {black_label}")
        print(f"{'='*70}")

        # Play the game
        result, moves = play_game(white, black, verbose=verbose)

        # Update results
        if result == 'white':
            if white_label == "Engine 1":
                results['engine1_wins'] += 1
            else:
                results['engine2_wins'] += 1
            print(f"\nResult: {white_label} (White) wins!")
        elif result == 'black':
            if black_label == "Engine 1":
                results['engine1_wins'] += 1
            else:
                results['engine2_wins'] += 1
            print(f"\nResult: {black_label} (Black) wins!")
        else:
            results['draws'] += 1
            print(f"\nResult: Draw")

        # Store game info
        results['games'].append({
            'game_num': game_num,
            'white': white_label,
            'black': black_label,
            'result': result,
            'moves': moves,
            'num_moves': len(moves)
        })

        # Print running score
        print(f"\nRunning Score:")
        print(f"  Engine 1: {results['engine1_wins']} wins")
        print(f"  Engine 2: {results['engine2_wins']} wins")
        print(f"  Draws: {results['draws']}")
        print(f"  Games played: {game_num}/{num_games}")

    return results


def print_match_summary(results: dict):
    """Print a summary of the match results."""
    total_games = len(results['games'])

    print(f"\n{'='*70}")
    print("MATCH SUMMARY")
    print(f"{'='*70}")
    print(f"\nTotal games played: {total_games}")
    print(f"\nFinal Score:")
    print(f"  Engine 1: {results['engine1_wins']} wins ({100*results['engine1_wins']/total_games:.1f}%)")
    print(f"  Engine 2: {results['engine2_wins']} wins ({100*results['engine2_wins']/total_games:.1f}%)")
    print(f"  Draws: {results['draws']} ({100*results['draws']/total_games:.1f}%)")

    # Calculate average game length
    avg_moves = sum(g['num_moves'] for g in results['games']) / total_games
    print(f"\nAverage game length: {avg_moves:.1f} moves")

    # Longest and shortest games
    longest_game = max(results['games'], key=lambda g: g['num_moves'])
    shortest_game = min(results['games'], key=lambda g: g['num_moves'])
    print(f"Longest game: Game {longest_game['game_num']} ({longest_game['num_moves']} moves)")
    print(f"Shortest game: Game {shortest_game['game_num']} ({shortest_game['num_moves']} moves)")

    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Play engine vs engine match')
    parser.add_argument('--model1', type=str, default='mini_leela_model.pth',
                        help='Path to model file for Engine 1')
    parser.add_argument('--model2', type=str, default=None,
                        help='Path to model file for Engine 2 (defaults to same as model1)')
    parser.add_argument('--games', type=int, default=100,
                        help='Number of games to play (default: 100)')
    parser.add_argument('--simulations', type=int, default=100,
                        help='Number of MCTS simulations per move (default: 100)')
    parser.add_argument('--simulations2', type=int, default=None,
                        help='Number of MCTS simulations for Engine 2 (defaults to same as simulations)')
    parser.add_argument('--quiet', action='store_true',
                        help='Suppress move-by-move output')
    parser.add_argument('--no-switch', action='store_true',
                        help='Do not switch colors between games')
    parser.add_argument('--max-moves', type=int, default=200,
                        help='Maximum moves per game before declaring draw (default: 200)')

    args = parser.parse_args()

    # Set model2 to model1 if not specified
    model2_path = args.model2 if args.model2 else args.model1
    simulations2 = args.simulations2 if args.simulations2 else args.simulations

    print(f"\n{'='*70}")
    print("MINI LEELA ENGINE VS ENGINE MATCH")
    print(f"{'='*70}")
    print(f"\nEngine 1:")
    print(f"  Model: {args.model1}")
    print(f"  Simulations: {args.simulations}")
    print(f"\nEngine 2:")
    print(f"  Model: {model2_path}")
    print(f"  Simulations: {simulations2}")
    print(f"\nMatch Settings:")
    print(f"  Number of games: {args.games}")
    print(f"  Switch colors: {not args.no_switch}")
    print(f"  Max moves per game: {args.max_moves}")

    # Initialize engines
    print(f"\n{'='*70}")
    print("Initializing engines...")
    print(f"{'='*70}\n")

    print("Loading Engine 1...")
    engine1 = Engine(args.model1, num_simulations=args.simulations)

    if model2_path == args.model1 and simulations2 == args.simulations:
        print("\nEngine 2 will use the same model and settings as Engine 1")
        engine2 = engine1  # Use same instance if models and settings are identical
    else:
        print("\nLoading Engine 2...")
        engine2 = Engine(model2_path, num_simulations=simulations2)

    # Play the match
    results = play_match(
        engine1, engine2,
        num_games=args.games,
        verbose=not args.quiet,
        switch_colors=not args.no_switch
    )

    # Print summary
    print_match_summary(results)

    # Save results to file
    output_file = f"match_results_{args.games}_games.txt"
    with open(output_file, 'w') as f:
        f.write("MINI LEELA ENGINE VS ENGINE MATCH RESULTS\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Engine 1: {args.model1} (simulations: {args.simulations})\n")
        f.write(f"Engine 2: {model2_path} (simulations: {simulations2})\n")
        f.write(f"Total games: {args.games}\n\n")
        f.write(f"Final Score:\n")
        f.write(f"  Engine 1: {results['engine1_wins']} wins\n")
        f.write(f"  Engine 2: {results['engine2_wins']} wins\n")
        f.write(f"  Draws: {results['draws']}\n\n")
        f.write("=" * 70 + "\n\n")

        for game in results['games']:
            f.write(f"Game {game['game_num']}: {game['white']} (W) vs {game['black']} (B)\n")
            f.write(f"Result: {game['result']}, Moves: {game['num_moves']}\n")
            f.write(f"Moves: {' '.join(game['moves'])}\n\n")

    print(f"Detailed results saved to: {output_file}")


if __name__ == "__main__":
    main()
