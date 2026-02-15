from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import argparse
import os
from tqdm import tqdm
from peft import PeftModel

# Configuration
TEST_FILE = "/home/node/.openclaw/workspace/BhashaLLM/data/processed/test.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def compute_perplexity(model, tokenizer, text_path, max_length=1024, stride=512):
    print(f"Loading text from {text_path}...")
    with open(text_path, encoding='utf-8') as f:
        text = f.read()
    
    print("Tokenizing...")
    encodings = tokenizer(text, return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    print(f"Total tokens: {seq_len}")

    nlls = []
    prev_end_loc = 0
    
    print(f"Evaluating perplexity with context_length={max_length}, stride={stride}...")
    # Wrap in try-except for OOM safety
    try:
        for begin_loc in tqdm(range(0, seq_len, stride)):
            end_loc = min(begin_loc + max_length, seq_len)
            trg_len = end_loc - prev_end_loc
            
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss

            nlls.append(neg_log_likelihood)

            prev_end_loc = end_loc
            if end_loc == seq_len:
                break
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("WARNING: GPU Out of Memory! Try reducing --context_length or --stride.")
            torch.cuda.empty_cache()
            return float('inf')
        else:
            raise e

    if not nlls:
        return float('inf')

    ppl = torch.exp(torch.stack(nlls).mean())
    return ppl.item()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Base model name or path")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to LoRA adapter")
    parser.add_argument("--load_in_4bit", action="store_true", help="Load in 4-bit")
    parser.add_argument("--context_length", type=int, default=1024, help="Max context length")
    parser.add_argument("--stride", type=int, default=512, help="Stride length")
    args = parser.parse_args()

    print(f"Loading model: {args.model_name}")
    
    bnb_config = None
    if args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True
        )

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    
    # Auto-adjust device map if 4bit
    device_map = "auto" if args.load_in_4bit else None
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        low_cpu_mem_usage=True if args.load_in_4bit else False
    )
    
    if not args.load_in_4bit:
        model.to(DEVICE)

    if args.adapter_path:
        print(f"Loading adapter from {args.adapter_path}")
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()
    
    ppl = compute_perplexity(
        model, 
        tokenizer, 
        TEST_FILE, 
        max_length=args.context_length, 
        stride=args.stride
    )
    print(f"Perplexity: {ppl}")
    
    # Save results
    res_file = "perplexity_results.csv"
    mode = "a" if os.path.exists(res_file) else "w"
    with open(res_file, mode) as f:
        if mode == "w": f.write("model,adapter,perplexity,context_length\n")
        f.write(f"{args.model_name},{args.adapter_path},{ppl},{args.context_length}\n")

if __name__ == "__main__":
    main()
