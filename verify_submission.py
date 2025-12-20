import os
import sys
import shutil

import r0927480

def test_run():
    
    print("Initializing solver...")
    solver = r0927480.r0927480()
    print("Running optimize...")
    solver.optimize("tour500.csv")
    print("Done.")

if __name__ == "__main__":
    test_run()
