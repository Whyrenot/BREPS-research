import os
import argparse
import json
import ast
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

def find_critical_shift(csv_path, dist_thresh=10.0, iou_drop_thresh=0.75, max_cases=5):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return []
        
        # parse bbox
        bboxes = np.vstack(df['bbox'].apply(ast.literal_eval).values)
        ious = df['iou'].values
        
        # Find base predictions with IoU > 0.85
        base_mask = ious > 0.85
        if not base_mask.any():
            return []
            
        base_indices = np.where(base_mask)[0]
        # Check higher IoU cases first
        base_indices = base_indices[np.argsort(-ious[base_indices])]
        
        found_cases = []
        used_bad_indices = set()
        
        for idx in base_indices:
            base_iou = ious[idx]
            base_box = bboxes[idx]
            
            diffs = np.max(np.abs(bboxes - base_box), axis=1)
            close_mask = (diffs > 0) & (diffs <= dist_thresh)
            
            if not close_mask.any():
                continue
                
            close_indices = np.where(close_mask)[0]
            close_indices = [ci for ci in close_indices if ci not in used_bad_indices]
            if not close_indices:
                continue
                
            bad_idx = close_indices[np.argmin(ious[close_indices])]
            bad_iou = ious[bad_idx]
            bad_box = bboxes[bad_idx]
            
            iou_drop = base_iou - bad_iou
            if iou_drop >= iou_drop_thresh:
                used_bad_indices.add(bad_idx)
                csv_stem = Path(csv_path).stem
                if csv_stem.startswith('res_final_'):
                    csv_stem = csv_stem[len('res_final_'):]
                if csv_stem.endswith('_random'):
                    csv_stem = csv_stem[:-len('_random')]
                elif csv_stem.endswith('_'):
                    csv_stem = csv_stem[:-1]
                image_name = csv_stem
                found_cases.append({
                    "csv_path": str(csv_path),
                    "image_name": image_name,
                    "best_box": base_box.tolist(),
                    "bad_box": bad_box.tolist(),
                    "best_iou": float(base_iou),
                    "bad_iou": float(bad_iou),
                    "iou_drop": float(iou_drop)
                })
                
                if len(found_cases) >= max_cases:
                    break
        
        return found_cases
    except Exception as e:
        print(f"Error processing {csv_path}: {e}")
        return []

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to a single CSV file to test')
    parser.add_argument('--dir', type=str, help='Path to directory with CSV files')
    parser.add_argument('--out', type=str, default='critical_shifts.json', help='Output JSON file')
    parser.add_argument('--dist_thresh', type=float, default=10.0, help='Max coordinate difference in 1024x1024 space')
    parser.add_argument('--iou_drop_thresh', type=float, default=0.75, help='Minimum drop in IoU to be considered critical')
    parser.add_argument('--max_cases', type=int, default=5, help='Max cases per image')
    parser.add_argument('--continue_from', action='store_true', help='Skip CSVs that are already in the output JSON')
    args = parser.parse_args()

    results = []

    if args.csv:
        res_list = find_critical_shift(args.csv, args.dist_thresh, args.iou_drop_thresh, args.max_cases)
        for res in res_list:
            results.append(res)
            print(f"Found critical shift for {res['image_name']}: IoU drop {res['iou_drop']:.3f} (Best: {res['best_iou']:.3f} -> Bad: {res['bad_iou']:.3f})")
        if not res_list:
            print("No critical shift found in this CSV.")
    elif args.dir:
        existing_csvs = set()
        if args.continue_from and Path(args.out).exists():
            try:
                with open(args.out, 'r') as f:
                    old_results = json.load(f)
                    existing_csvs = {r['csv_path'] for r in old_results}
                    results.extend(old_results)
            except Exception as e:
                print(f"Warning: could not read {args.out} for continuing: {e}")

        csv_files = list(Path(args.dir).glob('**/*.csv'))
        for csv_path in tqdm(csv_files, desc="Processing CSVs"):
            if args.continue_from and str(csv_path) in existing_csvs:
                continue
            res_list = find_critical_shift(csv_path, args.dist_thresh, args.iou_drop_thresh, args.max_cases)
            results.extend(res_list)
        print(f"Found {len(results)} critical shifts across all files.")
        
        # Sort by biggest drop
        results = sorted(results, key=lambda x: x['iou_drop'], reverse=True)
    else:
        print("Please provide --csv or --dir")
        return

    with open(args.out, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Results saved to {args.out}")

if __name__ == '__main__':
    main()
