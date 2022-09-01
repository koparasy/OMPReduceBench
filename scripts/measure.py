import sys
import subprocess
import glob
import re
import pandas as pd

exec_time_pattern = 'ELAPSED TIME:(.*)'
element_size= 'SIZE OF ELEMENT:(.*)'
reduction_type='REDUCTION TYPE:(.*)'

def check(val, cmd):
  if len(val) == 0:
    print('Could not match pattern:', cmd)
    print(val)
    sys.exit()
  return val[0]

def searchOut(stdout, cmd):
  time = re.findall(exec_time_pattern, stdout)
  time = float(check(time, cmd))
  eSize = re.findall(element_size, stdout)
  eSize = int(check(eSize, cmd))
  rType = re.findall(reduction_type, stdout)
  rType = str(check(rType, cmd))
  return [time, eSize, rType]

def execute(binary, mb):
  cmd = '%s %d' % (binary, mb)
  res = subprocess.run(cmd, shell=True, capture_output=True, text = True)
  return searchOut(res.stdout, cmd)

def main(args):
  reductionSizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
  repetitions = 1
  if len(args) != 3:
    print(f'CMD: {args[0]} pathToExecutables output-file')
    sys.exit()

  path_to_executables = args[1]
  print(path_to_executables)

  results = []
  for exe in glob.glob(f'{path_to_executables}/*.exe'):
    dTName = exe.split('/')[-1].split('.')[0].split('_')[1]
    for mb in reductionSizes:
      for t in range(repetitions):
        res = [dTName, mb, t] +  execute(exe, mb)
        print(*res)
        results.append(res)
  df = pd.DataFrame(results, columns=['Data Type', 'Total Reduction Size', '#Exp', 'Execution Time', 'Element Size', 'Reduction Type'])
  print(df)
  df.to_csv(args[2])

if __name__ == '__main__':
  main(sys.argv)