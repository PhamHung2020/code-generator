# API_KEY = 'AIzaSyAYHi5jnVDlZsFCtM3Ocm9xjBHKrSBQ7RE' 'AIzaSyAL_ngwshh23YzKNcSXp3JgVZAaGpSwKz0'
import os.path
import google.api_core.exceptions
import google.generativeai as genai

import json
import datetime
import numpy as np
import uuid
import time
from datasets import load_from_disk
import argparse
import multiprocessing
import re
from code_executor import execute_code_in_batch, CodeExecutionResponse

parser = argparse.ArgumentParser(description='Code Generation Parser')
parser.add_argument('--api_key', dest='api_key', type=str, help='Gemini API key', default='AIzaSyAYHi5jnVDlZsFCtM3Ocm9xjBHKrSBQ7RE')
parser.add_argument('--start_sample', dest='start_sample', type=int, help='Index of first sample', default=0)
parser.add_argument('--n_samples_per_thread', dest='n_samples_per_thread', type=int, help='Number of samples', default=10)
parser.add_argument('--n_threads', dest='n_threads', type=int, help='Number of threads', default=1)
parser.add_argument('--max_solutions', dest='max_solutions', type=int, help='Maximum number of solutions per sample', default=20)
parser.add_argument("--max_valid_solutions", dest='max_valid_solutions', type=int, help='Maximum number of valid solutions per sample', default=5)
parser.add_argument("--output_path", dest='output_path', type=str, help='Path to save output file', default="generated_code")
parser.add_argument("--language", dest="language", type=str, help="Language", default='Python') # default language is Python
parser.add_argument("--language_id", dest="language_id", type=int, help="Language ID", default=71) # default language is Python


FILTERED_PROBLEM_TASK_ID = []


def get_solving_code_prompt(taco_sample, language='Python'):
    prompt = f"""Solve the coding problem below in {language} programming language:

[{taco_sample['question']}]


"""

    starter_code = None if len(taco_sample["starter_code"]) == 0 else taco_sample["starter_code"]
    if starter_code:
        prompt += f"""Use starter code below as template:

[{starter_code}]

"""

    try:
        input_output = json.loads(taco_sample["input_output"])
        fn_name = (
            None if not input_output.get("fn_name") else input_output["fn_name"]
        )
    except ValueError:
        fn_name = None

    if not fn_name and not starter_code:
        prompt += "Use Standard Input format. "
    else:
        prompt += "Use Call-Based format. "

    # prompt += "ONLY return code, don't comment in code, don't explain anything, don't include test case or example usage, don't write unnecessary string to standard output."
    prompt += "ONLY return code, don't explain anything, don't include test case or example usage, don't write unnecessary string to standard output."

    if "Java" in language:
        prompt += " Source code must contain 'Main' class and 'main' function."

    return prompt


def clean_gemini_code(code: str):
    return code[code.index('\n')+1:code.rindex('```')]


def gen_code(task_id, sample, n_solutions, n_valid_solutions, output_path, language, language_id):
    if task_id not in FILTERED_PROBLEM_TASK_ID:
        return

    if sample['starter_code'] and len(sample['starter_code']) > 0:
        print("Sample is invalid: No input-output, Use starter code or picture in question, Input or Output is not a String or List")
        return

    model = genai.GenerativeModel('gemini-pro')

    # output_file = os.path.join(output_path, f'{task_id}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
    filename = f'{task_id}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    output_file = f'{output_path}/{filename}'
    not_passed_output_file = f'{output_path}/../not_passed_solutions/{filename}'

    outputs = []
    not_passed_outputs = []
    accepted_solutions = []

    prompt = get_solving_code_prompt(sample, language=language)
    valid_solutions = 0
    while n_solutions > 0:
    # for i in range(n_solutions):
        try:
            response = model.generate_content(prompt)
        except google.api_core.exceptions.InternalServerError:
            print("Google Internal Server Error")
            time.sleep(5)
            continue
        except google.api_core.exceptions.DeadlineExceeded:
            print("Deadline Exceed")
            time.sleep(10)
            continue

        try:
            # print(response.text)
            clean_code = clean_gemini_code(response.text)
        except Exception:
            print("Clean code failed")
            continue

        is_valid = True
        cleaned_solution = re.sub(r"[\n\t\s]*", "", clean_code)
        for accepted_solution in accepted_solutions:
            if cleaned_solution == accepted_solution:
                print(f"Task {task_id}: solution duplicated")
                is_valid = False
                break

        if not is_valid:
            continue

        execution_result = execute_code_in_batch(clean_code, sample, language_id)
        if execution_result.status.id != CodeExecutionResponse.ACCEPTED.id:
            print("Failed to execute code: ", execution_result.status.description)
            break

        n_solutions -= 1
        if not execution_result.results or len(execution_result.results) == 0:
            continue

        print(execution_result.results)
        result_np = np.array(execution_result.results)

        if np.any(result_np > 0):
            output = {
                'task_id': task_id,
                'solution_id': str(uuid.uuid4()),
                'solution': clean_code,
                'result': execution_result.results,
                'n_test_pass': int(np.sum(result_np > 0))
            }

            outputs.append(output)
            accepted_solutions.append(cleaned_solution)
            valid_solutions += 1

            print(f'Task {task_id}, found {valid_solutions}')

            if valid_solutions == n_valid_solutions:
                break
        else:
            output = {
                'task_id': task_id,
                'solution_id': str(uuid.uuid4()),
                'solution': clean_code,
                'result': execution_result.results
            }
            not_passed_outputs.append(output)

    if len(outputs) > 0:
        with open(output_file, 'w') as outfile:
            json.dump(outputs, outfile, indent=4)

    if len(not_passed_outputs) > 0:
        with open(not_passed_output_file, "w") as outfile:
            json.dump(not_passed_outputs, outfile, indent=4)


def main(dataset, start, end, max_solutions, max_valid_solutions, output_path, language, language_id):
    for i in range(start, end):
        print(f"Start sample {i}")
        gen_code(i, dataset[i - start], max_solutions, max_valid_solutions, output_path, language, language_id)


if __name__ == "__main__":
    args = parser.parse_args()
    n_threads = args.n_threads
    base_start = args.start_sample
    n_sample_per_thread = args.n_samples_per_thread
    output_path = args.output_path
    language = args.language
    language_id = args.language_id

    api_key = args.api_key
    genai.configure(api_key=api_key)
    threads = []

    with open("generated_code/taco_filtered/dataset_1.json", "r") as f:
        filtered_problems = json.load(f)

    FILTERED_PROBLEM_TASK_ID = [_['task_id'] for _ in filtered_problems]
    print(len(FILTERED_PROBLEM_TASK_ID))

    taco = load_from_disk("dataset/train")
    for thread_idx in range(n_threads):
        thread_start = base_start + thread_idx * n_sample_per_thread
        thread_end = thread_start + n_sample_per_thread
        thread_dataset = taco.select(range(thread_start, thread_end))
        threads.append(
            multiprocessing.Process(
                target=main,
                args=(thread_dataset, thread_start, thread_end, args.max_solutions, args.max_valid_solutions, output_path, language, language_id)
            )
        )

    for thread_idx, thread in enumerate(threads):
        thread.start()
        print(f"Start thread {thread_idx}")

    for thread in threads:
        thread.join()

    print("DONE")
