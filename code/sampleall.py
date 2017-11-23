import os.path
from utils import load_chorales
from chorales.model import load_model
from chorales.sample import generate_sample, write_sample

# 29,"bass-1,2",0.445647059118
# 19,tenor-1,0.570588243358
# 8,alto,0.674823526074
# TOTAL: 0.564

def main():
    basedir = '../data/models/'
    models = ['bass-1,2', 'tenor-1', 'alto']
    sample_length = 64
    run_name = 'test'
    sample_dir = '../data/output/'

    start_ind = 112
    cur_nm = '0'
    voice_nums = [3,1,2]
    for i,fnm in enumerate(models):
        voice_num = voice_nums[i]
        cur_nm += str(voice_num)
        model_file = os.path.join(basedir, fnm + '.h5')
        model, margs = load_model(model_file)

        if i == 0:
            # load first voice, with the other voices zero'd
            P = load_chorales.load(margs['train_file'], voice_num, margs['seq_length'], margs['batch_size'], voices_to_zero=voice_nums[1:])
            X = P['x_valid'][start_ind:start_ind+sample_length,:,:]
            P0 = load_chorales.load(margs['train_file'], voice_num, margs['seq_length'], margs['batch_size'], voices_to_zero=[])
            X0 = P0['x_valid'][start_ind:start_ind+sample_length,:,:]
            y0 = P0['y_valid'][start_ind:start_ind+sample_length]
            write_sample(X0, y0, P0, run_name, sample_dir, margs, postfix='_v0312_true'.format(cur_nm))

        yh, Xnew = generate_sample(model, X, sample_length, margs['seq_length'], margs['y_dim'], P['ranges'], margs['voice_num'], arg_max=True)
        write_sample(X, yh, P, run_name, sample_dir, margs, postfix='_v{}_pred'.format(cur_nm))
        X = Xnew

if __name__ == '__main__':
    main()
