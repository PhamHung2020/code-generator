import requests
import time
payload = {
    'language_id': 71,
    'source_code': "for _ in range(int(input())):\n\tn=int(input())\n\tprint(n//2 if n%2==0 else n//2+1)",
    'stdin': '3\n2\n3\n4',
    'expected_output': '1\n2\n2',
    'cpu_extra_time': '1',
    'cpu_time_limit': "5",
    'memory_limit': "128000"
}

base_api = "http://localhost:2358/"
create_submission_api = f"{base_api}/submissions"
status_submission_api = f"{base_api}/submissions/"
creation_response = requests.post(
    url=create_submission_api,
    json=payload
)

print(creation_response.status_code)
body = creation_response.json()
print(body)
token = body['token']
print(token)

while True:
    status_response = requests.get(f"{status_submission_api}{token}")
    status_response_body = status_response.json()
    print(status_response_body['status']['description'])
    if status_response_body['status']['id'] != 1 and status_response_body['status']['id'] != 2:
        break
    time.sleep(1.5)

print(status_response_body)
