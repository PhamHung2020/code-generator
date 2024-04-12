
import pathlib
import textwrap

import google.api_core.exceptions
import google.generativeai as genai

from IPython.display import display
from IPython.display import Markdown
import json
import datetime
import numpy as np
import uuid
import time
from datasets import load_dataset, load_from_disk
import argparse
import multiprocessing
import re

from metrics.testing_util import run_test

parser = argparse.ArgumentParser(description='Code Generation Parser')
parser.add_argument('--api_key', dest='api_key', type=str, help='Gemini API key')
parser.add_argument('--start_sample', dest='start_sample', type=int, help='Index of first sample', default=0)
parser.add_argument('--n_samples_per_thread', dest='n_samples_per_thread', type=int, help='Number of samples', default=10)
parser.add_argument('--n_threads', dest='n_threads', type=int, help='Number of threads', default=1)
parser.add_argument('--max_solutions', dest='max_solutions', type=int, help='Maximum number of solutions per sample', default=20)
parser.add_argument("--max_valid_solutions", dest='max_valid_solutions', type=int, help='Maximum number of valid solutions per sample', default=5)


TIMEOUT = 10


def to_markdown(text):
    text = text.replace('•', '  *')
    return Markdown(textwrap.indent(text, '> ', predicate=lambda _: True))


def list_models():
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)


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

    prompt += "Only return code, don't comment in code, don't explain anything, don't include test case or example usage."
    return prompt


def clean_gemini_code(code: str):
    return code[code.index('\n')+1:code.rindex('```')]


def check_correctness(sample, generated_code, timeout, debug=True):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""

    def _temp_run(sample, generation, debug, result):
        result.append(run_test(sample, test=generation, debug=debug))

    manager = multiprocessing.Manager()
    result = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(sample, generated_code, debug, result))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result:
        in_outs = json.loads(sample["input_output"])
        # consider that all tests failed
        result = [[-1 for i in range(len(in_outs["inputs"]))]]
        if debug:
            print(f"global timeout")
    return result[0]


def evaluate_generations(generated_code, sample, idx=None, debug=False):
    curr_res = [-2]
    try:
        curr_res = check_correctness(sample, generated_code, timeout=TIMEOUT, debug=debug)
        if debug:
            print(f"\nSuccessful compilation of task {idx}!")
        fixed = []
        for e in curr_res:
            if isinstance(e, np.ndarray):
                e = e.item(0)
            if isinstance(e, np.bool_):
                e = bool(e)
            fixed.append(e)
        curr_res = fixed
        if not np.all(curr_res):
            if debug:
                print(f"Results were not True for all test cases")
    except Exception as e:
        if debug:
            print(f"Compilation failed, test framework exception = {repr(e)}{e}\n")
    finally:
        assert isinstance(curr_res, list)

    return curr_res


def gen_code(task_id, sample, n_solutions, n_valid_solutions):
    model = genai.GenerativeModel('gemini-pro')

    output_file = f'generated_code/all/{task_id}_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    outputs = []
    cleaned_solutions = []

    prompt = get_solving_code_prompt(sample)
    valid_solutions = 0
    for i in range(n_solutions):
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
            clean_code = clean_gemini_code(response.text)
        except Exception:
            print("Clean code failed")
            continue

        result = evaluate_generations(clean_code, sample, task_id, debug=False)
        result_np = np.array(result)

        # print(prompt)
        # print(clean_code)
        print(result)
        if np.any(result_np > 0):
            is_valid = True
            cleaned_solution = re.sub(r"[\n\t\s]*", "", clean_code)
            for valid_solution in cleaned_solutions:
                if cleaned_solution == valid_solution:
                    print(f"Task {task_id}: solution duplicated")
                    is_valid = False
                    break

            if is_valid:
                cleaned_solutions.append(cleaned_solution)
                output = {'task_id': task_id, 'solution_id': str(uuid.uuid4()), 'solution': clean_code, 'result': result, 'n_test_pass': int(np.sum(result_np > 0))}
                outputs.append(output)
                valid_solutions += 1

                print(f'Task {task_id}, found {valid_solutions}')

                if valid_solutions == n_valid_solutions:
                    break

    if len(outputs) > 0:
        with open(output_file, 'w') as outfile:
            json.dump(outputs, outfile, indent=4)


def main(dataset, start, end, max_solutions, max_valid_solutions):
    for i in range(start, end):
        print(f"Start sample {i}")
        gen_code(i, dataset[i - start], max_solutions, max_valid_solutions)


if __name__ == "__main__":
    args = parser.parse_args()
    n_threads = args.n_threads
    base_start = args.start_sample
    n_sample_per_thread = args.n_samples_per_thread

    api_key = args.api_key
    genai.configure(api_key=api_key)
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
