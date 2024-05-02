import json
import numpy as np
import re
import datetime
import glob
from datasets import load_from_disk
import argparse

from code_executor import execute_code_in_batch, CodeExecutionResponse

parser = argparse.ArgumentParser(description='Code Generation Parser')
parser.add_argument('--start_sample', dest='start_sample', type=int, help='Index of first sample', default=0)
parser.add_argument('--n_samples', dest='n_samples', type=int, help='Number of samples', default=10)
parser.add_argument("--output_path", dest='output_path', type=str, help='Path to save output file', default="generated_code")
parser.add_argument("--language_id", dest="language_id", type=int, help="Language ID", default=71) # default language is Python


def clean_code(code: str):
    return code[code.index('\n')+1:code.rindex('```')]


def clean_java_code(code: str):
    main_method_index = code.rindex("public static void main")

    truncated_code = code[:main_method_index]
    print(truncated_code)
    try:
        first_class_index = truncated_code.index("public class") + 13
    except ValueError:
        first_class_index = truncated_code.index("class") + 6

    truncated_code = truncated_code[first_class_index:]
    try:
        truncated_code = truncated_code[:truncated_code.index(' ')]
    except ValueError:
        truncated_code = truncated_code[:truncated_code.index('\n')]

    print(truncated_code)

    return code.replace(truncated_code, "Main", 1)


def testcode(task_id, problem, sols, output_path, language_id):
    filename = f'{task_id}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_file = f'{output_path}/{filename}'
    not_passed_output_file = f'{output_path}/../not_passed_solutions/{filename}'

    test_inputs: list = problem["private_tests"]["input"]
    test_inputs.extend(problem["generated_tests"]["input"])

    test_outputs: list = problem["private_tests"]["output"]
    test_outputs.extend(problem["generated_tests"]["output"])

    problem["input_output"] = json.dumps({
        "inputs": test_inputs,
        "outputs": test_outputs
    })
    problem["starter_code"] = ""
    passed_outputs = []
    not_passed_outputs = []
    accepted_solutions = []
    sol_count = 0

    for sol in sols:
        try:
            # before_preprocessed_sol = clean_code(sol)
            before_preprocessed_sol = sol
            preprocessed_sol = clean_java_code(before_preprocessed_sol)
        except ValueError:
            continue

        # preprocessed_sol = sol

        is_valid = True
        cleaned_solution = re.sub(r"[\n\t\s]*", "", preprocessed_sol)
        for accepted_solution in accepted_solutions:
            if cleaned_solution == accepted_solution:
                print(f"Task {task_id}: solution duplicated")
                is_valid = False
                break

        if not is_valid:
            continue

        execution_result = execute_code_in_batch(preprocessed_sol, problem, language_id)
        if execution_result.status.id != CodeExecutionResponse.ACCEPTED.id:
            print("Failed to execute code: ", execution_result.status.description)
            continue

        if not execution_result.results or len(execution_result.results) == 0:
            output = {
                'task_id': task_id,
                'solution': before_preprocessed_sol,
                'result': execution_result.results
            }
            not_passed_outputs.append(output)
            continue

        print(execution_result.results)
        result_np = np.array(execution_result.results)

        if np.any(result_np > 0):
            output = {
                'task_id': task_id,
                'solution': before_preprocessed_sol,
                'result': execution_result.results,
                'n_test_pass': int(np.sum(result_np > 0))
            }

            passed_outputs.append(output)
            accepted_solutions.append(cleaned_solution)
            sol_count += 1

            print(f'Task {task_id}, found {sol_count}')

        else:
            output = {
                'task_id': task_id,
                'solution': before_preprocessed_sol,
                'result': execution_result.results
            }
            not_passed_outputs.append(output)

    if len(passed_outputs) > 0:
        with open(output_file, 'w') as outfile:
            json.dump(passed_outputs, outfile, indent=4)

    if len(not_passed_outputs) > 0:
        with open(not_passed_output_file, "w") as outfile:
            json.dump(not_passed_outputs, outfile, indent=4)


def main(start, end, output_path, language_id):
    # dataset = load_from_disk('dataset/deepmind_codecontest/train')
    with open("dataset/java_train.json", "r") as f:
        dataset = json.load(f)

    # sol_files = glob.glob("generated_code/java/java_codes-20240426T121549Z-001/java_codes/*.json")
    sol_files = glob.glob("generated_code/java/gemini/solutions/*.json")
    sol_files = sorted(sol_files, key=lambda s: s)
    for sol_file in sol_files[start:end]:
        print(f"Start sample {sol_file}")
        with open(sol_file, "r") as f:
            content = json.load(f)

        # sol = content['output'][1]
        # with open("/home/hungpm/Work/test/Main.java", "w") as f:
        #     f.write(sol)

        # task_id = content['task_id']
        # problem = dataset[task_id]
        # sols = content['output']

        task_id = content[0]['task_id']
        problem = dataset[task_id]
        sols = [_['solution'] for _ in content]

        testcode(task_id, problem, sols, output_path, language_id)


if __name__ == "__main__":
    # Prepare parameter
    args = parser.parse_args()
    base_start = args.start_sample
    n_samples = args.n_samples
    output_path = args.output_path
    language_id = args.language_id

    main(base_start, base_start + n_samples, output_path, language_id)
