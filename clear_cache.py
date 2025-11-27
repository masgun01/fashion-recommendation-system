import os
import pickle

def clear_cache():
    """Clear semua cache files"""
    cache_files = ['shape_features_cache.pkl', 'shape_cache.pkl']
    
    for cache_file in cache_files:
        if os.path.exists(cache_file):
            os.remove(cache_file)
            print(f"🧹 Removed {cache_file}")
    
    print("✅ All cache cleared! System will rebuild on next startup.")

if __name__ == "__main__":
    clear_cache()