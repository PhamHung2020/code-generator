from datasets import load_dataset, load_from_disk
import csv
import json


difficulties = ['EASY']

# Load train dataset from Hugging Face
# taco = load_dataset('BAAI/TACO', token='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', split='train', trust_remote_code=True)


# taco.save_to_disk("dataset/train")

# Load train dataset from disk
# taco_from_disk = load_from_disk("dataset/train")
# taco = taco_from_disk.filter(lambda entry: entry['difficulty'] in difficulties).select(range(5))
# print(taco[1]['input_output'])

# taco_10 = taco.select(range(10))
# print(taco_10.column_names)
# print(taco_10["input_output"][1])
# print(taco_10[1]["input_output"])
# input_output = json.loads(taco[2]["input_output"])
# print(input_output)

# for i in range(taco.num_rows):
#     try:
#         input_output = json.loads(taco[i]["input_output"])
#         fn_name = (
#             None if not input_output.get("fn_name") else input_output["fn_name"]
#         )
#     except ValueError:
#         fn_name = None
#
#     # if fn_name:
#     #     print(i)
#     #     print(fn_name)
#     #     break
#     starter_code = None if len(taco[i]["starter_code"]) == 0 else taco[i]["starter_code"]
#     # if starter_code:
#     #     print(i)
#     #     print(starter_code)
#     #     break
#
#     if fn_name and not starter_code:
#         print(i)

# taco_10.to_csv("dataset/train_easy_dataset.csv")
#
# print(taco.column_names)
# print(taco.description)
# print(taco.data.schema)
# with open("question.txt", "w") as of:
#     for i in range(10):
#         of.write(f"{i}\n")
#         of.write(bytes(taco[i]['question'], 'utf-8').decode('unicode_escape'))
#         of.write(f"\nEnd of {i}\n\n\n\n")


result = load_dataset("json", data_files="result.json").sort("task_id")
print(result['train'][5])
