"""
Extract mate-in-1 puzzles from Lichess puzzle database
"""
import zstandard
import csv
import random

print("Decompressing and extracting mate-in-1 puzzles...")

mate_in_1_puzzles = []

# Decompress and read CSV
with open('lichess_puzzles.csv.zst', 'rb') as compressed:
    dctx = zstandard.ZstdDecompressor()
    with dctx.stream_reader(compressed) as reader:
        text_stream = reader.read().decode('utf-8').splitlines()
        csv_reader = csv.DictReader(text_stream)

        for i, row in enumerate(csv_reader):
            if i % 100000 == 0:
                print(f"Processed {i} puzzles, found {len(mate_in_1_puzzles)} mate-in-1 so far...")

            # Check if this is a mate-in-1 puzzle
            themes = row['Themes']
            if 'mateIn1' in themes:
                fen = row['FEN']
                moves = row['Moves'].split()
                solution_move = moves[0]  # First move is the solution

                puzzle_id = row['PuzzleId']
                rating = row['Rating']

                mate_in_1_puzzles.append({
                    'fen': fen,
                    'move': solution_move,
                    'id': puzzle_id,
                    'rating': int(rating)
                })

            # Stop after collecting enough
            if len(mate_in_1_puzzles) >= 15000:  # Get 15000 for diverse split
                break

print(f"\nFound {len(mate_in_1_puzzles)} mate-in-1 puzzles")

# Sort by rating to see the full difficulty range
mate_in_1_puzzles.sort(key=lambda x: x['rating'])
print(f"Rating range: {mate_in_1_puzzles[0]['rating']} to {mate_in_1_puzzles[-1]['rating']}")

# Target: 10K training, 4K test - sampled UNIFORMLY across all difficulty levels
num_training = min(10000, len(mate_in_1_puzzles) - 4000)
num_test = min(4000, len(mate_in_1_puzzles) - num_training)

training_puzzles = []
test_puzzles = []

# Strategy: Sample uniformly across the sorted rating range
# This ensures we get easy, medium, and hard puzzles
total_available = len(mate_in_1_puzzles)

# Take every Nth puzzle for training to spread across difficulties
training_step = total_available / (num_training + num_test) * (num_training / (num_training + num_test))
training_indices = set()
for i in range(num_training):
    idx = int(i * total_available / num_training)
    training_indices.add(idx)
    training_puzzles.append(mate_in_1_puzzles[idx])

# Take remaining puzzles for test, also spread across difficulties
remaining_indices = [i for i in range(total_available) if i not in training_indices]
test_step = max(1, len(remaining_indices) / num_test)
for i in range(num_test):
    idx = int(i * len(remaining_indices) / num_test)
    if idx < len(remaining_indices):
        test_puzzles.append(mate_in_1_puzzles[remaining_indices[idx]])

print(f"\nTraining: {len(training_puzzles)} puzzles")
print(f"  Rating range: {training_puzzles[0]['rating']} to {training_puzzles[-1]['rating']}")
print(f"  Avg rating: {sum(p['rating'] for p in training_puzzles) / len(training_puzzles):.0f}")

print(f"\nTest: {len(test_puzzles)} puzzles")
print(f"  Rating range: {test_puzzles[0]['rating']} to {test_puzzles[-1]['rating']}")
print(f"  Avg rating: {sum(p['rating'] for p in test_puzzles) / len(test_puzzles):.0f}")

# Write to Python file
with open('mate_in_1_positions.py', 'w') as f:
    f.write('"""\n')
    f.write('Mate-in-1 positions from Lichess puzzle database\n')
    f.write(f'{len(training_puzzles)} training + {len(test_puzzles)} test puzzles\n')
    f.write('Sampled uniformly across all difficulty ratings for challenging learning\n')
    f.write('"""\n\n')

    f.write(f"# {len(training_puzzles)} training positions\n")
    f.write("TRAINING_POSITIONS = [\n")
    for p in training_puzzles:
        desc = f"Lichess #{p['id']} (rating:{p['rating']})"
        f.write(f"    ('{p['fen']}', '{p['move']}', '{desc}'),\n")
    f.write("]\n\n")

    f.write(f"# {len(test_puzzles)} test positions\n")
    f.write("TEST_POSITIONS = [\n")
    for p in test_puzzles:
        desc = f"Lichess #{p['id']} (rating:{p['rating']})"
        f.write(f"    ('{p['fen']}', '{p['move']}', '{desc}'),\n")
    f.write("]\n\n")

    f.write("""
def get_training_fens():
    return [pos[0] for pos in TRAINING_POSITIONS]

def get_test_fens():
    return [pos[0] for pos in TEST_POSITIONS]

def get_all_fens():
    return get_training_fens() + get_test_fens()

def get_solution(fen):
    for pos_list in [TRAINING_POSITIONS, TEST_POSITIONS]:
        for pos_fen, solution, desc in pos_list:
            if pos_fen == fen:
                return solution, desc
    return None, None
""")

print("\n✓ Saved to mate_in_1_positions.py")
print("You can now use this for training!")
