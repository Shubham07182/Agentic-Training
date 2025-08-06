import json

converted = []
with open("my_dataset.json", "r", encoding="utf-8") as f:
    for line in f:
        try:
            entry = json.loads(line)
            prompt = entry.get("text", "").strip()
            response = entry.get("code", "").strip()
            if prompt and response:
                converted.append({"prompt": prompt, "response": response})
        except json.JSONDecodeError:
            print("Skipping invalid line.")

# Save the new dataset as a valid JSONL
with open("my_dataset_converted.json", "w", encoding="utf-8") as f:
    for item in converted:
        f.write(json.dumps(item) + "\n")
