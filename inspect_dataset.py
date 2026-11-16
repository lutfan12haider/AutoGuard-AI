import os

dataset_path = "data/kwerimoto/prepared"

splits = ["train", "validation", "test", "real_test"]

total = 0

for split in splits:
    split_path = os.path.join(dataset_path, split)

    print("\n" + "=" * 40)
    print("SPLIT:", split)
    print("=" * 40)

    if not os.path.exists(split_path):
        print("Not found")
        continue

    split_total = 0

    for category in os.listdir(split_path):
        category_path = os.path.join(split_path, category)

        if os.path.isdir(category_path):
            files = [
                f for f in os.listdir(category_path)
                if f.lower().endswith(".wav")
            ]

            count = len(files)
            split_total += count

            print(category, ":", count)

    print("Split total:", split_total)
    total += split_total

print("\nTotal audio files:", total)