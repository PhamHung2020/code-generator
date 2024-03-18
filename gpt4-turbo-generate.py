import time

from datasets import load_from_disk
from openai import OpenAI
import openai
import json
import numpy as np
import multiprocessing
import datetime
import re
import uuid
import argparse

from helpers import clean_gpt_code, get_gpt_prompt, evaluate_generations

parser = argparse.ArgumentParser(description='Code Generation Parser')
parser.add_argument('--api_key', dest='api_key', type=str, help='OpenAI API key')
parser.add_argument('--start_sample', dest='start_sample', type=int, help='Index of first sample', default=0)
parser.add_argument('--n_samples_per_thread', dest='n_samples_per_thread', type=int, help='Number of samples', default=10)
parser.add_argument('--n_threads', dest='n_threads', type=int, help='Number of threads', default=1)
parser.add_argument('--max_solutions', dest='max_solutions', type=int, help='Maximum number of solutions per sample', default=10)
parser.add_argument("--max_valid_solutions", dest='max_valid_solutions', type=int, help='Maximum number of valid solutions per sample', default=1)


API_KEY = None
MODEL = 'gpt-4-turbo-preview'
TIMEOUT = 10
TEMPERATURE = 0.7
FILTERED_PROBLEM_TASK_ID = []


def gen_code(task_id, sample, n_solutions, n_valid_solutions):
    if task_id not in FILTERED_PROBLEM_TASK_ID:
        return

    client = OpenAI(api_key=API_KEY, max_retries=3)

    now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    passed_output_file = f'generated_code/python/gpt4-turbo/passed_solutions/{task_id}_{now}.json'
    not_passed_output_file = f'generated_code/python/gpt4-turbo/not_passed_solutions/{task_id}_{now}.json'

    not_passed_outputs = []
    passed_outputs = []
    valid_solutions = []

    prompt = get_gpt_prompt(sample)
    valid_solutions_count = 0
    for i in range(n_solutions):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": "You are a coding assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=TEMPERATURE
            )
        except openai.BadRequestError as e:
            # Handle error 400
            print(f"Error 400: {e}")
            return
        except openai.AuthenticationError as e:
            # Handle error 401
            print(f"Error 401: {e}")
            return
        except openai.UnprocessableEntityError as e:
            # Handle error 422
            print(f"Error 422: {e}")
            continue
        except openai.RateLimitError as e:
            # Handle error 429
            print(f"Error 429: {e}")
            time.sleep(30)
            continue
        except openai.InternalServerError as e:
            # Handle error >=500
            print(f"Error >=500: {e}")
            time.sleep(10)
            continue
        except openai.APIConnectionError as e:
            # Handle API connection error
            print(f"API connection error: {e}")
            time.sleep(10)
            continue

        try:
            clean_code = clean_gpt_code(response.choices[0].message.content)
        except Exception:
            print("Clean code failed")
            continue

        result = evaluate_generations(clean_code, sample, task_id, debug=False)
        result_np = np.array(result)

        print(result)

        is_valid = True
        cleaned_solution = re.sub(r"[\n\t\s]*", "", clean_code)
        for valid_solution in valid_solutions:
            if cleaned_solution == valid_solution:
                print(f"Task {task_id}: solution duplicated")
                is_valid = False
                break

        if not is_valid:
            continue

        valid_solutions.append(cleaned_solution)
        if np.any(result_np > 0):
            output = {
                'task_id': task_id,
                'solution_id': str(uuid.uuid4()),
                'solution': clean_code,
                'result': result,
                'n_test_pass': int(np.sum(result_np > 0)),
                'tokens': response.usage.completion_tokens
            }
            passed_outputs.append(output)
            valid_solutions_count += 1

            print(f'Task {task_id}, found {valid_solutions_count}')

            if valid_solutions_count == n_valid_solutions:
                break
        else:
            output = {
                'task_id': task_id,
                'solution_id': str(uuid.uuid4()),
                'solution': clean_code,
                'tokens': response.usage.completion_tokens
            }
            not_passed_outputs.append(output)

    if len(passed_outputs) > 0:
        with open(passed_output_file, 'w') as outfile:
            json.dump(passed_outputs, outfile, indent=4)

    if len(not_passed_outputs) > 0:
        with open(not_passed_output_file, 'w') as outfile:
            json.dump(not_passed_outputs, outfile, indent=4)


def main(dataset, start, end, max_solutions, max_valid_solutions):
    for i in range(start, end):
        print(f"Start sample {i}")
        gen_code(i, dataset[i - start], max_solutions, max_valid_solutions)


if __name__ == "__main__":
    with open("generated_code/taco_filtered/dataset_1.json", "r") as f:
        filtered_problems = json.load(f)

    FILTERED_PROBLEM_TASK_ID = [_['task_id'] for _ in filtered_problems]
    print(len(FILTERED_PROBLEM_TASK_ID))

    # taco = load_from_disk("dataset/train")
    # start = 1000
    # end = 10000
    # thread_dataset = taco.select(range(start, end))
    # main(thread_dataset, start, end, 10, 1)

    args = parser.parse_args()
    n_threads = args.n_threads
    base_start = args.start_sample
    n_sample_per_thread = args.n_samples_per_thread

    API_KEY = args.api_key
    threads = []

    taco = load_from_disk("dataset/train")
    for thread_idx in range(n_threads):
        thread_start = base_start + thread_idx * n_sample_per_thread
        thread_end = thread_start + n_sample_per_thread
        thread_dataset = taco.select(range(thread_start, thread_end))
        threads.append(
            multiprocessing.Process(
                target=main,
                args=(thread_dataset, thread_start, thread_end, args.max_solutions, args.max_valid_solutions)
            )
        )

    for thread_idx, thread in enumerate(threads):
        thread.start()
        print(f"Start thread {thread_idx}")

    for thread in threads:
        thread.join()

    print("DONE")
