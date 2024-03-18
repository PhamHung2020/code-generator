import json
import multiprocessing
import numpy as np
from metrics.testing_util import run_test

TIMEOUT = 10


def clean_gpt_code(code: str):
    return code[code.index('\n')+1:code.rindex('```')]


def get_gpt_prompt(taco_sample, language='Python'):
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
