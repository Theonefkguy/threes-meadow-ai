import subprocess
for name,source in [('train','train.cpp'),('benchmark','benchmark.cpp'),('collect','collect.cpp'),('verify','verify-model.cpp')]:
 subprocess.run(['g++','-O3','-march=native','-std=c++17','training/v4/'+source,'-o','/tmp/threes-v4-'+name],check=True)
subprocess.run(['/tmp/threes-v4-verify'],check=True)
