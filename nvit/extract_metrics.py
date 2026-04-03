import os
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import pandas as pd

def extract_metrics(run_dir):
    print(f"Analyzing run: {run_dir}")
    tb_dir = os.path.join(run_dir, 'tensorboard')
    if not os.path.exists(tb_dir):
        # Try finding anywhere recursively if not in standard 'tensorboard' subdir
        event_files = []
        for root, dirs, files in os.walk(run_dir):
            for f in files:
                if 'events.out.tfevents' in f:
                    event_files.append(os.path.join(root, f))
    else:
        event_files = [os.path.join(tb_dir, f) for f in os.listdir(tb_dir) if 'events.out.tfevents' in f]
    
    if not event_files:
        return "No event files found."

    # Use the latest event file by mtime
    event_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    latest_event_file = event_files[0]
    print(f"Using event file: {latest_event_file}")

    # Load events
    # Size 0 means load all
    event_acc = EventAccumulator(latest_event_file, size_guidance={'scalars': 0})
    event_acc.Reload()

    tags = event_acc.Tags()['scalars']
    print(f"Found tags: {tags}")

    # Identify relevant tags
    target_tags = [t for t in tags if any(k in t.lower() for k in ['h36m', 'pa-mpjpe', 'mpjpe']) and 'val' in t.lower()]
    
    if not target_tags:
        # Fallback to broader search
        target_tags = [t for t in tags if 'pa-mpjpe' in t.lower() or 'mpjpe' in t.lower()]

    if not target_tags:
        return "No relevant PA-MPJPE tags found."

    results = {}
    for tag in target_tags:
        events = event_acc.Scalars(tag)
        data = [[e.step, e.value] for e in events]
        df = pd.DataFrame(data, columns=['step', 'value'])
        best_val = df['value'].min()
        best_step = df.loc[df['value'].idxmin(), 'step']
        results[tag] = {
            'best_val': best_val,
            'best_step': best_step,
            'history': df.tail(20).to_dict('records')
        }

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    args = parser.parse_args()
    
    res = extract_metrics(args.run_dir)
    print("\n--- RESULTS ---")
    if isinstance(res, str):
        print(res)
    else:
        for tag, data in res.items():
            print(f"Tag: {tag}")
            print(f"  Best Value: {data['best_val']:.2f}")
            print(f"  Best Step: {data['best_step']}")
            print("  Recent Values:")
            for h in data['history']:
                print(f"    Step {h['step']}: {h['value']:.2f}")
