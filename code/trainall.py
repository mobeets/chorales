import pandas as pd
import itertools
from chorales.train import train

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

parts = ['soprano', 'alto', 'tenor', 'bass']
defargs = {
	'batch_size': 250,
	'optimizer': 'adam',
	'num_epochs': 200,
	'voice_num': None,
	'voices_to_zero': [],
	'latent_dim': 256,
	'seq_length': 2,
	'patience': 5,
	'log_dir': '../data/logs',
	'model_dir': '../data/models',
	'train_file': '../data/input/JSB Chorales_parts.pickle'}

def call(i, s):
	nm = parts[i]
	if s is not None:
		nm += '-' + ','.join([str(x) for x in s])
	args = dict((k,v) for k,v in defargs.iteritems())
	args['run_name'] = nm
	args['voice_num'] = i

	if s is not None:
		args['voices_to_zero'] = list(s)
	else:
		args['voices_to_zero'] = []
	args = AttrDict(args)
	print '----------------------'
	print nm
	_, info = train(args)
	print nm, info['val_acc']
	return nm, info['val_acc']

def main(outfile='../data/acc2.csv'):
	vals = []
	for i in xrange(4):
		inds = [j for j in xrange(4) if j != i]
		S = []
		for j in xrange(1,4):
			sets = set(itertools.combinations(inds,j))
			S.extend(sets)
		nm, acc = call(i, None)
		vals.append((nm, acc))
		pd.DataFrame(vals).to_csv(outfile)
		for s in S:
			nm, acc = call(i, s)
			vals.append((nm, acc))
			pd.DataFrame(vals).to_csv(outfile)

if __name__ == '__main__':
	main()

