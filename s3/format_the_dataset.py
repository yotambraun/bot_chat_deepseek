from download_dataset import dataset

human_prompt = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{}

### Response:
{}"""

def format_prompts(examples):
    instructions = examples["prompt"]
    outputs = examples["chosen"]
    return {"text": [human_prompt.format(inst, out) for inst, out in zip(instructions, outputs)]}

dataset = dataset.map(format_prompts, batched=True)