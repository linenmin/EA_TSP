import os
import sys
import shutil

import r0927482

def test_run():
    
    print("Initializing solver...")
    solver = r0927482.r0927480()
    print("Running optimize...")
    solver.optimize("tour1000.csv")
    print("Done.")

if __name__ == "__main__":
    test_run()
