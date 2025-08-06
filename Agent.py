import os
import torch
from fastapi import FastAPI, Request
import ast
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM


def load_model(model_id="microsoft/phi-2"):
    print("Loading model and tokenizer...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    local_path = r"C:\Users\shubh\AppData\Local\Programs\Python\Python313\AI Agent\phi2-finetuned"

    if not os.path.isdir(local_path):
        raise FileNotFoundError(f" Folder '{local_path}' not found. Please train or place the fine-tuned model there.")

    tokenizer = AutoTokenizer.from_pretrained(local_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        local_path,
        local_files_only=True,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)

    tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    print(" Model loaded from local folder!\n")
    return tokenizer, model


def generate_response(prompt, tokenizer, model, max_new_tokens=100):
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if decoded_output.lower().startswith(prompt.lower()):
        decoded_output = decoded_output[len(prompt):].strip()

    return decoded_output.split("\n\n")[0].strip()


def check_syntax(code):
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f" SyntaxError on line {e.lineno}: {e.msg}"


def run_pyflakes(code):
    temp_file = "temp_debug_code.py"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write(code)
    result = subprocess.run(["pyflakes", temp_file], capture_output=True, text=True)
    os.remove(temp_file)
    return result.stdout.strip()


def main():
    tokenizer, model = load_model()
    print(" Mini Code Assistant is ready! (type 'exit' to quit)\n")

    while True:
        user_input = input(" Prompt: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print(" Goodbye!")
            break

        if user_input.lower() in {"hi", "hello", "hey"}:
            print(" Response:\nHello! Ask me to generate or debug code.\n")
            continue
        elif not user_input:
            print("️ Please type something!\n")
            continue

        if user_input.lower() == "debug":
            while True:
                print("️ Code upload mode activated.")
                combined_code = ""

                while True:
                    choice = input(" Paste your code or enter file path (p/f)? ").strip().lower()

                    if choice.startswith("def") or choice.startswith("import") or "\n" in choice:
                        combined_code += "\n" + choice
                    elif choice == "p":
                        code = input(" Paste your code now:\n")
                        combined_code += "\n" + code
                    elif choice == "f":
                        path = input(" Enter file path: ").strip()
                        try:
                            with open(path, "r", encoding="utf-8") as file:
                                combined_code += "\n" + file.read()
                        except Exception as e:
                            print(" File error:", str(e))
                            continue
                    else:
                        print(" Invalid input. Try again.")
                        continue

                    more = input("\n").strip().lower()
                    if more != "y":
                        break

                print("\n Running static checks...\n")
                syntax_msg = check_syntax(combined_code)
                pyflakes_msg = run_pyflakes(combined_code)

                if not syntax_msg and not pyflakes_msg:
                    print(" No syntax errors.")
                    print(" No static issues from pyflakes.")
                    print("\n Debugging complete.\n")
                else:
                    if syntax_msg:
                        print(pyflakes_msg)
                    if pyflakes_msg:
                        print(pyflakes_msg)

                    print("\nGenerating LLM-based fix suggestions...\n")
                    prompt = f"### Prompt:\nPlease debug this Python code:\n\n{combined_code}\n\n### Response:\n"
                    response = generate_response(prompt, tokenizer, model)
                    print(" Suggestions:\n", response, "\n")

                again = input("\n").strip().lower()
                if again != "y":
                    break
                user_input=""

            continue

        # Normal generation
       
        try:
            while not user_input:
                user_input = input(" Prompt: ").strip()

            prompt = f"### Prompt:\n{user_input}\n\n### Response:\n"
            response = generate_response(prompt, tokenizer, model)
            print(" Response:\n", response, "\n")
            user_input = ""  # Reset after printing
        except Exception as e:
            print(" Generation Error:", str(e))




if __name__ == "__main__":
    main()
