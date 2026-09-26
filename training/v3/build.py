"""Compile reproducible native experiment tools; run from the project root."""
import subprocess
from pathlib import Path
for source,name in [('train-late.cpp','train-late'),('train-goal.cpp','train-goal'),('train-joint.cpp','train-joint'),('benchmark.cpp','benchmark-fast'),('collect-late.cpp','collect'),('eval-late.cpp','eval-late'),('verify-targets.cpp','verify')]:
 subprocess.run(['g++','-O3','-march=native','-std=c++17',str(Path('training/v3')/source),'-o','/tmp/threes-v3-'+name],check=True)
subprocess.run(['/tmp/threes-v3-verify'],check=True)
