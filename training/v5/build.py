import subprocess
for name in ['train','benchmark','verify']:
 subprocess.run(['g++','-O3','-march=native','-std=c++17','training/v5/'+name+'.cpp','-o','/tmp/threes-v5-'+name],check=True)
subprocess.run(['/tmp/threes-v5-verify'],check=True)
