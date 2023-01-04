import subprocess

# print("Running debug with n=1")
# output = subprocess.check_output("python3 debug_mixed_assembly.py", shell=True)
# print(output.decode('utf-8'))


print("Running debug with n=2")

p = subprocess.Popen(
    "mpiexec -np 4 python3 debug_stubs.py",
    stdout=subprocess.PIPE,
    bufsize=1,
    shell=True,
)
for line in iter(p.stdout.readline, b""):
    print(line)
p.stdout.close()
p.wait()
# output_par = subprocess.check_output("mpiexec -np 4 python3 debug_mixed_assembly.py", stderr=sys.stderr, stdout=sys.stdout, shell=True)
# print(output_par.decode('utf-8'))
