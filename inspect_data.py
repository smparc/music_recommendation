import pandas as pd
import sys

# --- Configuration ---
CSV_FILE_NAME = 'SpotifyCSV.csv' 
# ---------------------

def inspect_csv(file_path):
    """
    Loads a CSV file and prints its columns and the first 5 rows.
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(file_path, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: Data file not found at '{file_path}'")
        print("Please make sure the file is in the same directory.")
        return
    except Exception as e:
        print(f"An error occurred: {e}")
        return

    print(f"\n--- 🕵️ Inspecting: {file_path} ---\n")
    
    # 1. Print all column names
    print("Column Names:")
    print("--------------------")
    for col in df.columns:
        print(col)
        
    # 2. Print the first 5 rows
    print("\n\nSample Data (First 5 Rows):")
    print("--------------------")
    pd.set_option('display.max_columns', None)
    print(df.head())
    print("\n" + "-"*30)

if __name__ == "__main":
    # If a filename is passed as an argument, use it
    # Otherwise, use the default
    if len(sys.argv) > 1:
        file_to_inspect = sys.argv[1]
    else:
        file_to_inspect = CSV_FILE_NAME
        
    inspect_csv(file_to_inspect)
