import json
import time
from pathlib import Path

import numpy as np

from business_entity_resolution.src.blocking import BlockingStore

def monkey_patched_lsh_keys(self, text: str, prefix: str = "lsh") -> list[str]:
    import zlib
    import hashlib
    grams = self._grams(text)
    if len(grams) < 3:
        return []
    hashes = np.fromiter((zlib.crc32(gram.encode("utf-8")) for gram in grams), dtype=np.uint64, count=len(grams))
    values = np.min((hashes[:, None] * self._MINHASH_A + self._MINHASH_B) % self._MINHASH_PRIME, axis=0)
    band_size = getattr(self, "ROWS_PER_BAND", self.MINHASH_PERMUTATIONS // self.MINHASH_BANDS)
    return [
        f"{prefix}:{band}:{hashlib.blake2b(values[band * band_size:(band + 1) * band_size].tobytes(), digest_size=8).hexdigest()}"
        for band in range(self.MINHASH_BANDS)
    ]

BlockingStore._lsh_keys = monkey_patched_lsh_keys

def sweep():
    combinations = [
        (4, 3), (4, 4), (4, 5), (4, 6),
        (6, 3), (6, 4), (6, 5),
        (8, 3), (8, 4),
        (10, 3)
    ]
    
    results = []
    
    data_dir = Path("student_resource/dataset/train")
    truth = data_dir / "train_ground_truth.tsv"
    source = data_dir / "train_source1.tsv"
    targets = [data_dir / "train_source2.tsv", data_dir / "train_source3.tsv"]
    
    for bands, rows_per_band in combinations:
        if bands * rows_per_band > 32:
            continue
            
        print(f"Testing bands={bands}, rows_per_band={rows_per_band}")
        BlockingStore.MINHASH_BANDS = bands
        BlockingStore.ROWS_PER_BAND = rows_per_band
        
        db_path = Path(f"scratch/blocking_sweep.db")
        if db_path.exists():
            db_path.unlink()
            
        store = BlockingStore(db_path)
        store.reset()
        store.add_truth(truth)
        store.build_source_index(source)
        store.add_targets_and_retrieve(targets)
        store.finalize_candidates()
        
        diag = store.diagnostics
        
        results.append({
            "bands": bands,
            "rows_per_band": rows_per_band,
            "permutations": bands * rows_per_band,
            "raw_recall": diag.get("raw_blocking", {}).get("recall", 0),
            "final_recall": diag.get("final", {}).get("recall", 0),
            "avg_candidates": diag.get("final_stats", {}).get("avg_candidates", 0)
        })
        
        print(results[-1])
        store.close()
        db_path.unlink()
        
    with open("scratch/sweep_results.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    sweep()
