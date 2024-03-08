import glob
import json
from datasets import load_from_disk

dataset = load_from_disk("dataset/train")

difficulties = ["EASY", "MEDIUM", "MEDIUM_HARD", "HARD", "VERY_HARD", "UNKNOWN_DIFFICULTY"]

total_solutions = 0
found_per_difficulty = {}
for difficulty in difficulties:
    found_per_difficulty[difficulty] = {
        "problems": 0,
        "solutions": 0
    }

generated_code_files = glob.glob("generated_code/all/*.json")

for f in generated_code_files:
    with open(f, "r") as json_file:
        content = json.load(json_file)
        # print(content[0]['task_id'])
        task_id = content[0]['task_id']
        # print(dataset[task_id]['difficulty'])
        difficulty = dataset[task_id]['difficulty']
        found_per_difficulty[difficulty]['problems'] += 1
        found_per_difficulty[difficulty]['solutions'] += len(content)
        total_solutions += len(content)

print("-------- SOLVED PROBLEMS ----------")

for k in found_per_difficulty:
    print(k, found_per_difficulty[k]['problems'], ' problems - ', found_per_difficulty[k]['solutions'], ' solutions')

print("Total problems: ", sum([found_per_difficulty[k]['problems'] for k in found_per_difficulty]))
print("Total solutions: ", total_solutions)

print("\n\n\n-------- UNSOLVED PROBLEMS ----------")

solved_problems = []
max_task_id = 0

for f in generated_code_files:
    with open(f, "r") as json_file:
        content = json.load(json_file)
    task_id = content[0]['task_id']
    solved_problems.append(task_id)
    max_task_id = max(max_task_id, task_id)


not_solved_problems = []
for i in range(0, max_task_id + 1):
    if i not in solved_problems:
        not_solved_problems.append(i)

print(len(not_solved_problems))


not_solved_problems_by_difficulty = {}
for diff in difficulties:
    not_solved_problems_by_difficulty[diff] = []

for prob in not_solved_problems:
    not_solved_problems_by_difficulty[dataset[prob]['difficulty']].append(prob)

for diff in not_solved_problems_by_difficulty:
    print(diff, len(not_solved_problems_by_difficulty[diff]))
