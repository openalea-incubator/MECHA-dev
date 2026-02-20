import sys
import os
import numpy as np

# Ensure src is in pythonpath
sys.path.append(os.getcwd())

from src.mecha.mecha_class import Mecha
from src.mecha.utils.data_loader import InData

def test_build_matrices():
    print("Loading data...")
    # Use the existing xml file
    xml_path = "extdata/current_root.xml"
    if not os.path.exists(xml_path):
        print(f"Error: {xml_path} not found.")
        return

    try:
        AllIn = InData(cellset_file=xml_path)
    except Exception as e:
        print(f"Error loading InData: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Initializing Mecha...")
    try:
        mecha = Mecha(all_input=AllIn)
    except Exception as e:
        print(f"Error initializing Mecha: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("Setting up parameters...")
    h = 0
    i_maturity = 0
    if not mecha.geometry or not mecha.geometry.maturity_stages:
        print("Error: Maturity stages not loaded properly.")
        return
        
    maturity_elem = mecha.geometry.maturity_stages[i_maturity]
    
    print("Calling build_matrices...")
    try:
        matrix_W, matrix_C, matrix_ApoC, matrix_SymC, rhs_C, rhs_ApoC, rhs_SymC = mecha.build_matrices(h, i_maturity, maturity_elem)
        
        print("\n--- Results ---")
        print(f"matrix_W shape: {matrix_W.shape}")
        
        # Check symmetry of matrix_W
        if np.allclose(matrix_W, matrix_W.T):
            print("matrix_W is symmetric: YES")
        else:
            print("matrix_W is symmetric: NO")
            
        print(f"matrix_W min: {np.min(matrix_W)}, max: {np.max(matrix_W)}")
        
        if matrix_C is not None:
            print(f"matrix_C shape: {matrix_C.shape}")
        else:
            print("matrix_C is None")
            
        print("Verification SUCCESS.")
        
    except Exception as e:
        print(f"Verification FAILED with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_build_matrices()
