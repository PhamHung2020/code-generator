from datasets import load_dataset, load_from_disk
import csv

difficulties = ['EASY']

# Load train dataset from Hugging Face
# taco = load_dataset('BAAI/TACO', token='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', split='train', trust_remote_code=True)
# taco.save_to_disk("dataset/train")

# Load train dataset from disk
taco_from_disk = load_from_disk("dataset/train")
taco = taco_from_disk.filter(lambda entry: entry['difficulty'] in difficulties)

taco_10 = taco.select(range(10))
print(taco_10.column_names)
print(taco_10["input_output"][1])
# print(taco_10['question'][0])


# taco_10.to_csv("dataset/train_easy_dataset.csv")
#
# print(taco.column_names)
# print(taco.description)
# print(taco.data.schema)
with open("question.txt", "w") as of:
    for i in range(10):
        of.write(f"{i}\n")
        of.write(bytes(taco[i]['question'], 'utf-8').decode('unicode_escape'))
        of.write(f"\nEnd of {i}\n\n\n\n")
