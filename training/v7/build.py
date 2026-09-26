from pathlib import Path
import subprocess
for name in ['collect','assess','train','benchmark','verify']:
 source=Path(f'training/v7/{name}.cpp')
 if source.exists():subprocess.run(['g++','-std=c++17','-O3','-march=native',str(source),'-o',f'/tmp/threes-v7-{name}'],check=True)
