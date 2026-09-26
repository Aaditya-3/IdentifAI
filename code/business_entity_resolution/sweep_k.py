import time
from pathlib import Path
from business_entity_resolution.src.pipeline import validate
from business_entity_resolution.src.blocking import BlockingStore

def sweep():
    results = []
    top_ks = [20, 30]
    key_caps = [160]
    
    for key_cap in key_caps:
        for k in top_ks:
            print(f"Testing K={k}, key_cap={key_cap}")
            BlockingStore.KEY_CAP = key_cap
            BlockingStore.TOP_K = k
            
            t_start = time.time()
            score, threshold = validate(
                data_dir="student_resource/dataset/train",
                top_k=k,
                seed=42,
                scratch_dir="scratch"
            )
            runtime = time.time() - t_start
            print(f"Result: F0.5={score:.4f}, Threshold={threshold:.3f}, Time={runtime:.1f}s")
            
            results.append({
                "top_k": k,
                "key_cap": key_cap,
                "f05": score,
                "threshold": threshold,
                "runtime": runtime
            })
            
if __name__ == "__main__":
    sweep()
