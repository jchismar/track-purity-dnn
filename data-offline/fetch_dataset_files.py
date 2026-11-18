#!/usr/bin/env python3
"""
Fetch all files from DAS for each dataset and create file lists for Condor jobs.
"""

import subprocess
import json
import os
from pathlib import Path

def get_files_for_dataset(dataset):
    """Query DAS to get all files for a dataset."""
    cmd = f'dasgoclient -query="file dataset={dataset}" -json'
    try:
        result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              universal_newlines=True, timeout=300)
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout)
            files = []
            for entry in data:
                if 'file' in entry:
                    for file_entry in entry['file']:
                        if 'name' in file_entry:
                            files.append(file_entry['name'])
            return files
        else:
            print(f"Error querying DAS for {dataset}")
            print(f"stderr: {result.stderr}")
            return []
    except subprocess.TimeoutExpired:
        print(f"Timeout querying DAS for {dataset}")
        return []
    except Exception as e:
        print(f"Exception querying DAS for {dataset}: {e}")
        return []

def main():
    datasets_file = Path(__file__).parent / "datasets.txt"
    output_dir = Path(__file__).parent / "file_lists"
    output_dir.mkdir(exist_ok=True)
    
    with open(datasets_file, 'r') as f:
        datasets = [line.strip() for line in f if line.strip() and not line.startswith('#')]
    
    print(f"Processing {len(datasets)} datasets...")
    
    all_jobs = []
    
    for dataset in datasets:
        print(f"\nProcessing: {dataset}")
        
        dataset_name = dataset.split('/')[1]
        
        files = get_files_for_dataset(dataset)
        
        if not files:
            print(f"  No files found for {dataset}")
            continue
        
        print(f"  Found {len(files)} files")
        
        if len(files) > 1000:
            files = files[:1000]
            print(f"  Limiting to {len(files)} files")
        
        file_list_path = output_dir / f"{dataset_name}_files.txt"
        with open(file_list_path, 'w') as f:
            for file_path in files:
                f.write(f"{file_path}\n")
        
        for idx, file_path in enumerate(files):
            all_jobs.append({
                'dataset': dataset,
                'dataset_name': dataset_name,
                'file': file_path,
                'file_idx': idx,
                'job_id': f"{dataset_name}_{idx}"
            })
        
        print(f"  Created {len(files)} job entries")
    
    job_list_path = output_dir / "all_jobs.txt"
    with open(job_list_path, 'w') as f:
        for job in all_jobs:
            f.write(f'{job["dataset"]},{job["dataset_name"]},{job["file"]},{job["file_idx"]}\n')
    
    print(f"\n{'='*60}")
    print(f"Total jobs to submit: {len(all_jobs)}")
    print(f"Job list saved to: {job_list_path}")
    print(f"File lists saved to: {output_dir}/")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
