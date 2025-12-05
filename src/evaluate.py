import json
import pandas as pd
import scipy.stats as stats
import numpy as np

LOG_FILE = "interaction_logs.jsonl"

def evaluate():
    logs = []
    with open(LOG_FILE, 'r') as f:
        for line in f:
            data = json.loads(line)
            if "experiment_group" in data:
                logs.append(data)
            
    if not logs:
        print("No experiment logs found.")
        return

    df = pd.DataFrame(logs)
    
    def has_click(actions):
        for aa in actions:
            if "Click" in aa:
                return 1
        return 0

    df['clicked'] = df['actions'].apply(has_click)
    
    results = df.groupby('experiment_group')['clicked'].agg(['count', 'sum', 'mean'])
    results.columns = ['Total Queries', 'Queries with Click', 'CTR']
    
    print("=== A/B Test Results ===")
    print(results)
    
    control_clicks = df[df['experiment_group'] == 'control']['clicked']
    treatment_clicks = df[df['experiment_group'] == 'treatment']['clicked']
    
    if len(control_clicks) > 0 and len(treatment_clicks) > 0:
        t_stat, p_val = stats.ttest_ind(control_clicks, treatment_clicks)
        print(f"\nStatistical Significance (t-test):")
        print(f"t-statistic: {t_stat:.4f}")
        print(f"p-value: {p_val:.4f}")
        
        if p_val < 0.05:
            print("Result is Statistically Significant (p < 0.05)")
        else:
            print("Result is NOT Statistically Significant (p >= 0.05)")
            

    dwell_times = []
    for _, row in df.iterrows():
        group = row['experiment_group']
        for action_list in row['actions']:
             for action in action_list:
                 if isinstance(action, dict) and "Dwell" in action:
                     secs = action["Dwell"]["secs"]
                     nanos = action["Dwell"]["nanos"]
                     total_secs = secs + nanos / 1e9
                     dwell_times.append({'group': group, 'dwell_time': total_secs})
    
    if dwell_times:
        dt_df = pd.DataFrame(dwell_times)
        dt_results = dt_df.groupby('group')['dwell_time'].mean()
        print("\n=== Average Dwell Time (seconds) ===")
        print(dt_results)

if __name__ == "__main__":
    evaluate()
