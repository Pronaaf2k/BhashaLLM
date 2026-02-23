import json
import os
import shutil

def main():
    adapters_dir = "/home/benaaf/Desktop/BhashaLLM/models/instruct_adapters"
    final_dir = os.path.join(adapters_dir, "final_instruct_adapter")
    state_file = os.path.join(final_dir, "trainer_state.json")
    
    with open(state_file, "r") as f:
        state = json.load(f)
        
    log_history = state.get("log_history", [])
    
    # Extract eval_loss for each step
    step_to_loss = {}
    for entry in log_history:
        if "eval_loss" in entry and "step" in entry:
            step_to_loss[entry["step"]] = entry["eval_loss"]
            
    # List available checkpoints
    checkpoints = [d for d in os.listdir(adapters_dir) if d.startswith("checkpoint-")]
    
    ckpt_losses = []
    for ckpt in checkpoints:
        step = int(ckpt.split("-")[1])
        if step in step_to_loss:
            ckpt_losses.append((ckpt, step_to_loss[step]))
        else:
            # fallback to reading the trainer_state in the checkpoint itself if needed
            ckpt_state_file = os.path.join(adapters_dir, ckpt, "trainer_state.json")
            if os.path.exists(ckpt_state_file):
                with open(ckpt_state_file, "r") as f:
                    c_state = json.load(f)
                    c_history = c_state.get("log_history", [])
                    c_loss = float("inf")
                    for e in reversed(c_history):
                        if "eval_loss" in e:
                            c_loss = e["eval_loss"]
                            break
                    ckpt_losses.append((ckpt, c_loss))
            else:
                ckpt_losses.append((ckpt, float("inf")))
                
    # Sort by loss (ascending)
    ckpt_losses.sort(key=lambda x: x[1])
    
    print("Available Checkpoints Sorted by Eval Loss:")
    for ckpt, loss in ckpt_losses:
        print(f"  {ckpt}: {loss:.4f}")
        
    top_10 = ckpt_losses[:10]
    
    # Save the best 10 to TRAINED MODELS
    trained_models_dir = "/home/benaaf/Desktop/BhashaLLM/TRAINED MODELS"
    os.makedirs(trained_models_dir, exist_ok=True)
    
    print("\nCopying Top 10 Models to TRAINED MODELS folder...")
    for i, (ckpt, loss) in enumerate(top_10, 1):
        src_path = os.path.join(adapters_dir, ckpt)
        dst_path = os.path.join(trained_models_dir, f"rank_{i}_{ckpt}_loss_{loss:.4f}")
        print(f"Copying {src_path} -> {dst_path}")
        if not os.path.exists(dst_path):
            shutil.copytree(src_path, dst_path)
    
    print("Done picking best models!")

if __name__ == "__main__":
    main()
