import pandas as pd
import sys
import seaborn as sns
import seaborn as sns
import matplotlib.pyplot as plt


def main(args):
  if len(args) != 2:
    print('Please provide csv file with results')
    sys.exit()
  fn = args[1]
  print(fn)
  df = pd.read_csv(fn)
  df = df.drop('Unnamed: 0', axis=1)
  print(df.columns)
  df['Data Type'] = df['Data Type'] + ' (' + df['Element Size'].map(str) + ')'
  df = df.drop('Element Size', axis =1)
  grp_id = list(df.columns)
  print(grp_id)
  grp_id.remove('#Exp')
  grp_id.remove('Execution Time')
  df = df.groupby(grp_id).mean().reset_index().drop('#Exp', axis=1)
  df['Implementation']  = 'OMP-Offload'
  print(df['Execution Time'].dtypes)
  g = sns.relplot(data=df, y='Execution Time', x='Total Reduction Size', row='Data Type', col='Reduction Type', hue='Implementation', facet_kws={'sharey': False, 'sharex': True})
  g.set(xscale="log")
  plt.savefig('teams_reduction.pdf')
  print(df)

if __name__ == '__main__':
  main(sys.argv)
