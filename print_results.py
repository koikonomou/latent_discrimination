import json
import pandas as pd
from pathlib import Path
import numpy as np

def generate_full_breakdown(run_root="runs"):
    root = Path(run_root)
    all_data = []

    for json_path in root.rglob("params_baseline_*.json"):
        try:
            parts = json_path.parts
            
            main_run = next((p for p in parts if p.startswith("run_") or p.startswith("test_")), None)
            
            repeat = next((p for p in parts if p.startswith("repeat_")), None)
            
            if main_run is None:
                print(f"[Skipping] Found JSON outside of a run folder: {json_path}")
                continue
            if repeat is None:
                repeat = "Base"

            with open(json_path, 'r') as f:
                data = json.load(f)
                pct = float(data.get("coreset"))
                acc = data.get("best_test_acc") * 100
                
                all_data.append({
                    "Main_Run": main_run,
                    "Repeat": repeat,
                    "Percentage": pct,
                    "Accuracy": acc
                })
        except Exception as e:
            continue

    if not all_data:
        print(f"No valid data found in {root.absolute()}")
        return

    df = pd.DataFrame(all_data)
    main_runs = sorted(df["Main_Run"].unique())

    for run_id in main_runs:
        print("\n" + "="*100)
        print(f" REPEAT BREAKDOWN FOR: {run_id} ".center(100, "="))
        print("="*100)
        
        run_df = df[df["Main_Run"] == run_id]
        
        pivot_df = run_df.pivot_table(index="Repeat", columns="Percentage", values="Accuracy", aggfunc='mean')
        pivot_df = pivot_df.reindex(sorted(pivot_df.columns), axis=1)
        
        print(pivot_df.to_string(float_format=lambda x: f"{x:6.2f}%"))
        
        print("-" * 100)
        run_mean = pivot_df.mean()
        run_std = pivot_df.std().fillna(0.0)
        
        print(f"{'MEAN:':<10}" + "".join([f"{m:8.2f}%" for m in run_mean]))
        print(f"{'STD:':<10}" + "".join([f"{s:8.2f} " for s in run_std]))
        print("="*100)

    if len(main_runs) > 1:
        print("\n\n" + "#"*100)
        print(f" GLOBAL SUMMARY (MEAN OF ALL RUNS) ".center(100, "#"))
        print("#"*100)
        
        run_summaries = df.groupby(["Main_Run", "Percentage"])["Accuracy"].mean().unstack()
        run_summaries = run_summaries.reindex(sorted(run_summaries.columns), axis=1)
        
        print(run_summaries.to_string(float_format=lambda x: f"{x:6.2f}%"))
        print("-" * 100)
        
        final_mean = run_summaries.mean()
        final_std = run_summaries.std().fillna(0.0)
        
        print(f"{'TOTAL MEAN:':<12}" + "".join([f"{m:8.2f}%" for m in final_mean]))
        print(f"{'TOTAL STD:':<12}" + "".join([f" ±{s:5.2f} " for s in final_std]))
        print("#"*100)

if __name__ == "__main__":
    generate_full_breakdown()
