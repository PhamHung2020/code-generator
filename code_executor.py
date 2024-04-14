import requests
import time
import json
import base64
import string
from http import HTTPStatus

base_api = "http://localhost:2358/"
create_submission_api = f"{base_api}/submissions"
status_submission_api = f"{base_api}/submissions/"
create_batch_submissions_api = f"{base_api}/submissions/batch"
status_batch_submissions_api = f"{base_api}/submissions/batch"


class CodeExecutionStatus:
    def __init__(self, status_id: int, description: str):
        self.id = status_id
        self.description = description


class CodeExecutionResult:
    def __init__(self, status: CodeExecutionStatus, results=None):
        if results is None:
            results = []

        self.status = status
        self.results = results


class CodeExecutionResponse:
    INVALID_SAMPLE = CodeExecutionStatus(
        -1,
        "Sample is invalid: No input-output, Use starter code or picture in question, Input or Output is not a String or List"
    )

    CONNECTION_ERROR = CodeExecutionStatus(-2, "Cannot connect to Judge0")

    IN_QUEUE = CodeExecutionStatus(1, "In Queue")

    PROCESSING = CodeExecutionStatus(2, "Processing")

    ACCEPTED = CodeExecutionStatus(3, "Accepted")

    WRONG_ANSWER = CodeExecutionStatus(4, "Wrong answer")

    COMPILATION_ERROR = CodeExecutionStatus(6, "Compilation error")

    INTERNAL_ERROR = CodeExecutionStatus(13, "Internal error")

    EXEC_FORMAT_ERROR = CodeExecutionStatus(14, "Exec format error")

    RUNTIME_ERROR_NZEC = CodeExecutionStatus(11, "Runtime error (NZEC)")


def get_base64_encoded(text: str):
    return base64.b64encode(text.encode('utf-8')).decode('utf-8')


def get_base64_decoded(text: str):
    return base64.b64decode(text.encode('utf-8')).decode('utf-8')


def preprocessing(sample) -> tuple[CodeExecutionStatus, list, list]:
    # check starter code
    if (sample['starter_code'] and len(sample['starter_code']) > 0) or not sample['input_output']:
        return CodeExecutionResponse.INVALID_SAMPLE, [], []

    # serialize input - output
    in_out = json.loads(sample['input_output'])
    if not in_out['inputs'] or not in_out['outputs'] or len(in_out['inputs']) != len(in_out['outputs']):
        return CodeExecutionResponse.INVALID_SAMPLE, [], []

    # standardize inputs and outputs
    inputs = []
    outputs = []

    for sample_input, sample_output in zip(in_out['inputs'], in_out['outputs']):
        if isinstance(sample_input, str):
            inputs.append(sample_input.strip())
        elif isinstance(sample_input, list):
            inputs.append('\n'.join(sample_input).strip())
        else:
            return CodeExecutionResponse.INVALID_SAMPLE, [], []

        if isinstance(sample_output, str):
            outputs.append(sample_output.strip())
        elif isinstance(sample_output, list):
            outputs.append('\n'.join(sample_output).strip())
        else:
            return CodeExecutionResponse.INVALID_SAMPLE, [], []

    # test connection to Judge0
    try:
        test_conn_response = requests.get(base_api + "statuses")
        if test_conn_response.status_code != HTTPStatus.OK.value:
            return CodeExecutionResponse.CONNECTION_ERROR, [], []

    except requests.exceptions.ConnectionError:
        return CodeExecutionResponse.CONNECTION_ERROR, [], []

    inputs = inputs[:300]
    outputs = outputs[:300]

    return CodeExecutionResponse.ACCEPTED, inputs, outputs


def execute_code(generated_code: str, sample, language_id: int) -> CodeExecutionResult:
    preprocessing_result, inputs, outputs = preprocessing(sample)
    if preprocessing_result.id != CodeExecutionResponse.ACCEPTED.id:
        return CodeExecutionResult(preprocessing_result)

    # iterate through each input-output pair and execute code with that pair
    execution_results = []
    n_test = 0
    for sample_input, sample_output in zip(inputs, outputs):
        n_test += 1
        # payload for API
        payload = {
            'language_id': language_id,
            # 'source_code': "for _ in range(int(input())):\n\tn=int(input())\n\tprint(n//2 if n%2==0 else n//2+1)",
            'source_code': get_base64_encoded(generated_code),
            'stdin': get_base64_encoded(sample_input),
            'expected_output': get_base64_encoded(sample_output),
        }

        # submit code to Judge0
        submission_creation_response = requests.post(
            url=create_submission_api + '?base64_encoded=true',
            json=payload
        )

        # check submission result and get token
        body = submission_creation_response.json()
        if 'token' not in body or 'error' in body:
            print("Error ", submission_creation_response)
            print(payload)
            return CodeExecutionResult(CodeExecutionResponse.INTERNAL_ERROR)

        token = body['token']
        # polling to get execution result
        while True:
            status_response = requests.get(f"{status_submission_api}{token}")
            status_response_body = status_response.json()
            if 'error' in status_response_body:
                print("Error ", submission_creation_response)
                print(payload)
                return CodeExecutionResult(CodeExecutionResponse.INTERNAL_ERROR)

            try:
                status_id = status_response_body['status']['id']
            except KeyError:
                print("Error ", status_response_body)
                status_id = CodeExecutionResponse.INTERNAL_ERROR.id
                break

            print("Test ", n_test, status_response_body['status']['description'])
            if status_id != CodeExecutionResponse.IN_QUEUE.id and status_id != CodeExecutionResponse.PROCESSING.id:
                break

            time.sleep(1.5)

        if status_id == CodeExecutionResponse.ACCEPTED.id:
            execution_results.append(True)
        elif status_id == CodeExecutionResponse.COMPILATION_ERROR.id:
            execution_results.append(-2)
            break
        elif status_id == CodeExecutionResponse.WRONG_ANSWER.id:
            execution_results.append(False)
        else:
            execution_results.append(-1)

    return CodeExecutionResult(CodeExecutionResponse.ACCEPTED, execution_results)


def execute_code_in_batch(generated_code: str, sample, language_id: int) -> CodeExecutionResult:
    preprocessing_result, inputs, outputs = preprocessing(sample)
    if preprocessing_result.id != CodeExecutionResponse.ACCEPTED.id:
        return CodeExecutionResult(preprocessing_result)

    n_in_out_per_loop = 20
    execution_results = []
    n_loops = int(len(inputs) / n_in_out_per_loop) + 1
    count = 0
    has_one_test_passed = False

    for i in range(n_loops):
        start_index = i * n_in_out_per_loop
        end_index = min(i * n_in_out_per_loop + n_in_out_per_loop, len(inputs))
        print(f"Start from {start_index} to {end_index}")
        submissions = []

        for j in range(start_index, end_index):
            submission = {
                "language_id": language_id,
                "source_code":  get_base64_encoded(generated_code),
                "stdin": get_base64_encoded(inputs[j]),
                "expected_output": get_base64_encoded(outputs[j]),
            }

            # if language_id == 54:
            #     submission['compiler_options'] = '-O3 --std=c++17 -Wall -Wextra -Wold-style-cast -Wuseless-cast -Wnull-dereference -Werror -Wfatal-errors -pedantic -pedantic-errors'

            submissions.append(submission)

        payload = {
            "submissions": submissions
        }

        # submit code to Judge0
        submission_creation_response = requests.post(
            url=create_batch_submissions_api + "?base64_encoded=true",
            json=payload,
            # headers={"Content-Type": "application/json; charset=utf-8"}
        )
        # print(payload)

        print("Submissions creation status: ", submission_creation_response.status_code)
        body = submission_creation_response.json()
        if submission_creation_response.status_code != 201:
            print(body)
            break

        n_token_per_request = 10
        tokens = [_['token'] for _ in body]
        while len(tokens) > 0:
            request_token = tokens[:n_token_per_request]
            tokens = tokens[n_token_per_request:]
            status_response = requests.get(
                url=status_batch_submissions_api + '?base64_encoded=true&tokens=' + ','.join(request_token)
            )

            if status_response.status_code != 200:
                print(status_response.status_code)
                print(status_response.text)
                # print(status_response.json())
                break

            status_response_body = status_response.json()
            submissions_results = status_response_body['submissions']
            for idx, submission_result in enumerate(submissions_results):
                status_id = submission_result['status']['id']
                if status_id == CodeExecutionResponse.IN_QUEUE.id or status_id == CodeExecutionResponse.PROCESSING.id:
                    print(f"Test {count}: ", submission_result['status']['description'])
                    request_token = request_token[idx:]
                    request_token.extend(tokens)
                    tokens = request_token
                    break
                if status_id == CodeExecutionResponse.ACCEPTED.id:
                    execution_results.append(True)
                    has_one_test_passed = True
                elif status_id == CodeExecutionResponse.COMPILATION_ERROR.id or status_id == CodeExecutionResponse.RUNTIME_ERROR_NZEC.id:
                    print("Compilation error/Runtime error NZEC")
                    if submission_result['compile_output']:
                        print(get_base64_decoded(submission_result['compile_output']))
                    return CodeExecutionResult(CodeExecutionResponse.ACCEPTED, execution_results)
                    # execution_results.append(-2)
                elif status_id == CodeExecutionResponse.WRONG_ANSWER.id:
                    execution_results.append(False)
                else:
                    execution_results.append(-1)

                print(f"Test {count}: ", submission_result['status']['description'])
                count += 1

            if has_one_test_passed:
                break

            time.sleep(2)

        if has_one_test_passed:
            break

    return CodeExecutionResult(CodeExecutionResponse.ACCEPTED, execution_results)


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


if __name__ == "__main__":
    from datasets import load_from_disk
    import google.generativeai as genai
    import numpy as np
    import uuid
    import datetime

    taco = load_from_disk("dataset/train")

    genai.configure(api_key="")
    model = genai.GenerativeModel('gemini-pro')

    output_file = f'generated_code/test_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    index = 11
    prompt = get_solving_code_prompt(taco[index], 'C++17')
    final_results = []

    # while len(outputs) == 0:
    response = model.generate_content(prompt)
    clean_code = clean_gemini_code(response.text)
    print("Got code from gemini")

    result = execute_code_in_batch(clean_code, taco[index], 54)
    print("Status: ", result.status.id)
    print("Status: ", result.status.description)
    print("Result: ", result.results, "\n\n")

    # result_np = np.array(result.results)
    # if np.any(result_np > 0):
    #     output = {'task_id': 123, 'solution_id': str(uuid.uuid4()), 'solution': clean_code, 'result': result.results, 'n_test_pass': int(np.sum(result_np > 0))}
    #     final_results.append(output)
    #
    # if len(final_results) > 0:
    #     with open(output_file, 'w') as outfile:
    #         json.dump(final_results, outfile, indent=4)
    pass
