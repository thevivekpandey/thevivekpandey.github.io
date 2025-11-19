#!/usr/bin/env python3
"""
Filter PGN file to only include games where:
1. Both White and Black Elo >= 2500
2. TimeControl is 300+ seconds
3. Remove clock annotations [%clk ...] from moves
4. Only keep Site, WhiteElo, BlackElo and TimeControl headers
"""

import re
import sys

def parse_time_control(time_control):
    """
    Parse TimeControl string and return base time in seconds.
    Format is typically "base+increment" e.g., "1800+0", "3600+30"
    """
    if not time_control or time_control == '-':
        return 0

    parts = time_control.split('+')
    if parts:
        try:
            return int(parts[0])
        except ValueError:
            return 0
    return 0

def process_pgn_file(input_file, output_file):
    """
    Process PGN file and write filtered games to output file.
    """
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:

        current_game = []
        white_elo = None
        black_elo = None
        time_control = None
        in_moves = False
        games_processed = 0
        games_written = 0

        for line in f_in:
            line = line.rstrip('\n')

            # Check if we're starting a new game
            if line.startswith('[Event '):
                # Process previous game if we have one
                if current_game:
                    games_processed += 1
                    if (white_elo and black_elo and time_control and
                        white_elo >= 2500 and black_elo >= 2500 and
                        time_control >= 300):
                        # Write the filtered game
                        f_out.write('\n'.join(current_game) + '\n\n')
                        games_written += 1

                    if games_processed % 100000 == 0:
                        print(f"Processed {games_processed} games, written {games_written} games...")

                # Reset for new game
                current_game = []
                white_elo = None
                black_elo = None
                time_control = None
                in_moves = False

            # Parse headers
            if line.startswith('[Site '):
                # Keep Site header
                current_game.append(line)
            elif line.startswith('[WhiteElo '):
                match = re.search(r'\[WhiteElo "(\d+)"\]', line)
                if match:
                    white_elo = int(match.group(1))
                    current_game.append(line)
            elif line.startswith('[BlackElo '):
                match = re.search(r'\[BlackElo "(\d+)"\]', line)
                if match:
                    black_elo = int(match.group(1))
                    current_game.append(line)
            elif line.startswith('[TimeControl '):
                match = re.search(r'\[TimeControl "([^"]+)"\]', line)
                if match:
                    time_control = parse_time_control(match.group(1))
                    current_game.append(line)
            elif line.startswith('['):
                # Skip all other headers
                pass
            elif line.strip() == '':
                # Empty line - might be between headers and moves or between games
                if in_moves:
                    current_game.append(line)
            else:
                # This is a move line
                in_moves = True
                # Remove all annotations in curly braces { ... }
                cleaned_line = re.sub(r'\s*\{[^}]*\}', '', line)
                # Convert from "1. e4 { ... } 1... e5 { ... }" to "1. e4 e5"
                # Remove the extra move numbers for black (like "1...")
                cleaned_line = re.sub(r'\s+\d+\.\.\.', '', cleaned_line)
                current_game.append(cleaned_line)

        # Don't forget the last game
        if current_game:
            games_processed += 1
            if (white_elo and black_elo and time_control and
                white_elo >= 2500 and black_elo >= 2500 and
                time_control >= 300):
                f_out.write('\n'.join(current_game) + '\n\n')
                games_written += 1

        print(f"\nDone! Processed {games_processed} games total.")
        print(f"Written {games_written} games that match criteria (Elo >= 2500, TimeControl >= 300s).")

if __name__ == '__main__':
    input_file = 'lichess_db_standard_rated_2024-11.pgn'
    output_file = 'trimmed_lichess_db_standard_rated_2024-11.pgn'

    print(f"Processing {input_file}...")
    print(f"Filtering for: WhiteElo >= 2500, BlackElo >= 2500, TimeControl >= 300s")
    print(f"Output will be written to: {output_file}")
    print()

    try:
        process_pgn_file(input_file, output_file)
    except FileNotFoundError:
        print(f"Error: Could not find input file '{input_file}'")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
