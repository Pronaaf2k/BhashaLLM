import re
import json

def extract_metrics(log_file, output_file):
    with open(log_file, 'r', encoding='utf-8', errors='replace') as f:
        content = f.read()
    
    # HF logs typically look like {'loss': 1.1234, 'learning_rate': 0.0002, 'epoch': 0.01}
    # We will try to find lines that are valid dict representations.
    # To bypass \r characters from tqdm, we replace them with \n
    lines = content.replace('\r', '\n').split('\n')
    
    metrics = []
    for line in lines:
        if "'loss':" in line or "'eval_loss':" in line:
            # try to find the dict
            match = re.search(r"\{.*?\}", line)
            if match:
                dict_str = match.group(0).replace("'", '"')
                try:
                    data = json.loads(dict_str)
                    metrics.append(data)
                except:
                    pass

    # Save as CSV or markdown
    if metrics:
        import pandas as pd
        df = pd.DataFrame(metrics)
        df.to_csv(output_file, index=False)
        print(f"Extracted {len(metrics)} metric logs to {output_file}")
        print(df.tail())
    else:
        print("No metrics found yet. They might be waiting for the next logging step.")

if __name__ == "__main__":
    extract_metrics("/home/benaaf/Desktop/BhashaLLM/train_error.log", "/home/benaaf/Desktop/BhashaLLM/report/training_metrics.csv")
