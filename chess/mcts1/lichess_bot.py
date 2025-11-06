#!/usr/bin/env python3

import requests
import json
import chess
import random
import time
import random_mover
import basic_mcts

# Configuration
#LICHESS_TOKEN = "<my token>"  # Replace with your actual token
BASE_URL = "https://lichess.org"

# Headers for API requests
headers = {
    "Authorization": f"Bearer {LICHESS_TOKEN}"
}

def get_legal_moves(board):
    """Get all legal moves for the current position."""
    return list(board.legal_moves)

def choose_random_move(board):
    """Choose a random legal move."""
    legal_moves = get_legal_moves(board)
    if legal_moves:
        return random.choice(legal_moves)
    return None

def make_move(game_id, move):
    """Make a move in a game."""
    url = f"{BASE_URL}/api/bot/game/{game_id}/move/{move.uci()}"
    response = requests.post(url, headers=headers)
    return response.status_code == 200

def handle_game_state(game_id, state):
    """Handle a game state update."""
    # Create board from moves
    board = chess.Board()
    moves_str = state.get("moves", "")
    
    if moves_str:
        moves = moves_str.split()
        for move_uci in moves:
            try:
                board.push_uci(move_uci)
            except Exception as e:
                print(f"Error applying move {move_uci}: {e}")
                return
    
    # Check if it's our turn
    print(f"\nCurrent position: {board.fen()}")
    print(f"Moves played: {moves_str}")
    
    # Only move if game is not over and it's our turn
    status = state.get("status")
    if status not in ["mate", "resign", "stalemate", "draw", "outoftime", "aborted"]:
        # Small delay to avoid making moves too quickly
        time.sleep(0.5)
        
        #engine = random_mover.Engine()
        engine = basic_mcts.Engine()
        move = engine.get_move(board)
        if move:
            print(f"Playing move: {board.san(move)}")
            mm = make_move(game_id, move)
            print('see :', mm)
            if mm:
                print(f"Move {board.san(move)} played successfully")
            else:
                print(f"Failed to play move {board.san(move)}")
        else:
            print("No legal moves available")
    else:
        print(f"Game ended with status: {status}")

def stream_game_state(game_id):
    """Stream game state and play moves."""
    url = f"{BASE_URL}/api/bot/game/stream/{game_id}"
    
    print(f"\nStarting game {game_id}")
    
    with requests.get(url, headers=headers, stream=True, timeout=60) as response:
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line.decode('utf-8'))
                    event_type = event.get("type")
                    
                    if event_type == "gameFull":
                        # Initial game state
                        print(f"Game started: White={event.get('white', {}).get('name')} vs Black={event.get('black', {}).get('name')}")
                        state = event.get("state", {})
                        handle_game_state(game_id, state)
                    
                    elif event_type == "gameState":
                        # Game state update (opponent moved)
                        handle_game_state(game_id, event)
                    
                    elif event_type == "chatLine":
                        # Chat message (ignore for now)
                        pass
                    
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                except Exception as e:
                    print(f"Error handling event: {e}")

def accept_challenge(challenge_id):
    """Accept a challenge."""
    url = f"{BASE_URL}/api/challenge/{challenge_id}/accept"
    response = requests.post(url, headers=headers)
    return response.status_code == 200

def decline_challenge(challenge_id):
    """Decline a challenge."""
    url = f"{BASE_URL}/api/challenge/{challenge_id}/decline"
    response = requests.post(url, headers=headers)
    return response.status_code == 200

def should_accept_challenge(challenge):
    """Decide whether to accept a challenge."""
    # Accept standard chess only
    variant = challenge.get("variant", {}).get("key")
    if variant != "standard":
        print(f"Declining non-standard variant: {variant}")
        return False
    
    # Accept all time controls for testing
    time_control = challenge.get("timeControl", {})
    print(f"Challenge received: {time_control}")
    
    return True

def stream_events():
    """Stream incoming events (challenges, games)."""
    url = f"{BASE_URL}/api/stream/event"
    
    print("Connecting to Lichess...")
    print("Waiting for challenges...\n")
    
    with requests.get(url, headers=headers, stream=True, timeout=None) as response:
        for line in response.iter_lines():
            if line:
                try:
                    event = json.loads(line.decode('utf-8'))
                    event_type = event.get("type")
                    
                    if event_type == "challenge":
                        challenge = event.get("challenge", {})
                        challenge_id = challenge.get("id")
                        challenger = challenge.get("challenger", {}).get("name", "Unknown")
                        
                        print(f"\nChallenge received from {challenger} (ID: {challenge_id})")
                        
                        if should_accept_challenge(challenge):
                            print(f"Accepting challenge {challenge_id}")
                            if accept_challenge(challenge_id):
                                print("Challenge accepted!")
                            else:
                                print("Failed to accept challenge")
                        else:
                            print(f"Declining challenge {challenge_id}")
                            decline_challenge(challenge_id)
                    
                    elif event_type == "gameStart":
                        game_id = event.get("game", {}).get("id")
                        print(f"\nGame starting: {game_id}")
                        time.sleep(5)
                        # Stream this game in the same connection
                        stream_game_state(game_id)
                    
                    elif event_type == "gameFinish":
                        game_id = event.get("game", {}).get("id")
                        print(f"\nGame finished: {game_id}")
                
                except json.JSONDecodeError as e:
                    print(f"JSON decode error: {e}")
                except Exception as e:
                    print(f"Error in event stream: {e}")

def challenge_bot(username, time_control={"minutes": 3, "increment": 2}, rated=False):
    """Challenge another bot to a game."""
    url = f"{BASE_URL}/api/challenge/{username}"
    
    data = {
        "rated": rated,
        "clock.limit": time_control["minutes"] * 60,
        "clock.increment": time_control["increment"],
        "color": "random"
    }
    
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        print(f"Challenge sent to {username}")
        return True
    else:
        print(f"Failed to challenge {username}: {response.status_code}")
        return False

def main():
    """Main bot loop."""
    print("=" * 60)
    print("Lichess Random Move Bot")
    print("=" * 60)
    
    if LICHESS_TOKEN == "YOUR_API_TOKEN_HERE":
        print("\nERROR: Please set your Lichess API token in the script!")
        print("Get your token from: https://lichess.org/account/oauth/token")
        return
    
    # Test connection
    response = requests.get(f"{BASE_URL}/api/account", headers=headers)
    if response.status_code == 200:
        account = response.json()
        print(f"\nConnected as: {account.get('username')}")
        print(f"Bot account: {account.get('title') == 'BOT'}")
        
        if account.get('title') != 'BOT':
            print("\nWARNING: This account is not a BOT account!")
            print("Please upgrade at: https://lichess.org/api/bot/account/upgrade")
            return
    else:
        print(f"\nERROR: Failed to connect to Lichess (Status: {response.status_code})")
        print("Please check your API token.")
        return
    
    print("\nStarting event stream...")
    print("The bot will accept challenges and play random moves.")
    print("Press Ctrl+C to stop.\n")
    
    #try:
    #    stream_events()
    #except KeyboardInterrupt:
    #    print("\n\nBot stopped by user.")
    #except Exception as e:
    #    print(f"\nError: {e}")

    bots_to_challenge = [
     #"maia1",  # ~1100 Elo
     #"sargon-1ply",  # ~1100 Elo
     #"pawnrobot",  # ~1100 Elo
     "StupidfishBOTBYDSCS",  # 800 Elo
     #"Lurch_0",  # random mover
    ]

    for bot_name in bots_to_challenge:
       ret = challenge_bot(bot_name, time_control={"minutes": 3, "increment": 2}, rated=False)

       try:
           stream_events()
       except KeyboardInterrupt:
           print("\n\nBot stopped by user.")
       except Exception as e:
           print(f"\nError: {e}")
       time.sleep(0.5)  # Wait between challenges


if __name__ == "__main__":
    main()
