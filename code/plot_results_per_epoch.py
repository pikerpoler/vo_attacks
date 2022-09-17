import os
import torch
import matplotlib
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt


# TODO: make the following changes to utils.py and attacks.py:
'''
# in utils.py

# add this to parse_args():
parser.add_argument('--run_name', default='test-run', help='name of run for graphs. cannot have any "_" in it!!! (default: "test-run")')
# notice that run_name has to not include '_'. so 'rms-alpha=0.001-rot=dot' is fine but 'rms_alpha=0.001_rot=dot' is not, and will produce the name 'rot=dot' instead.

# add this to the end of compute_output_dir():
args.output_dir += '_' + str(args.run_name)

# it should look like this:
            args.output_dir += "/eps_" + str(args.eps).replace('.', '_') + \
                               "_attack_iter_" + str(args.attack_k) + \
                               "_alpha_" + str(args.alpha).replace('.', '_')
----->      args.output_dir += '_' + str(args.run_name)
            if not isdir(args.output_dir): 
                mkdir(args.output_dir)
'''

'''
# in run_attacks.py

# add these imports:
import os
from attacks import Const

# after this line:
best_pert, clean_loss_list, all_loss_list, all_best_loss_list = \
        attack.perturb(args.testDataloader, motions_target_list, eps=args.eps, device=args.device)

# insert:

torch.save(all_loss_list, os.path.join(args.output_dir, 'all_loss_list.pt'))
components_listname = args.output_dir.split('/')
listname = ''
for component in components_listname:
    listname += '_' + component
listname += '.pt'
listname = listname.split('opt_whole_trajectory')[-1]  # cutting down listname length this way is not elegant, but it works for now. alternatively you can save only run name, but this way custom filtration might be added in the future
dir = "results/loss_lists"
list_path = os.path.join(dir, listname)
if not isinstance(attack, Const):
    print(f'saving all_loss_list to {list_path}')
    if not isdir(dir):
        mkdir(dir)
    torch.save(all_loss_list, list_path)
    
'''


def cumul_sum_loss_from_list(loss_list):
    """
    :param loss_list: a list(epochs) of lists(trajectories) of lists(frames) of losses.
    :return: a list of the cumulative sum of the losses per epoch. this is what is printed in PGD.perturb during evaluation.
    """

    return [sum([sum(trajectory_losses) for trajectory_losses in epoch_losses]) for epoch_losses in loss_list]


def sum_loss_from_list(list3):
    """
    :param list3: a list(epochs) of lists(trajectories) of lists(frames) of losses.
    in the code:
    # list3 is trajectory losses per epoch
    # list2 is losses per trajectory
    # list1 is loss per frame of a trajectory
    :return: a list of the sum of the last frame losses per epoch. this is what I think we are going to be rated on.
    """

    last_frame_loss = [[list1[-1] for list1 in list2] for list2 in list3]
    sum_list = [sum(l) for l in last_frame_loss]
    return sum_list


def compare_results(loss_list_dir, partition='', agg=cumul_sum_loss_from_list, filters=(None,)):
    """
    :param loss_list_dir:  directory of loss_list.pt files
    :param agg: function to aggregate the loss lists. can be either cumul_sum_loss_from_list or sum_loss_from_list.
    """
    def extract_label(filename):
        components = filename.split('_')
        return components[-1]

    plt.figure(figsize=(16, 12))
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    for filename in os.listdir(os.path.join(loss_list_dir, partition)):

        if not any([f in filename for f in filters]): # and 'baseline' not in filename: # and 'second' not in filename:
            continue
        print(filename)
        f = os.path.join(loss_list_dir,partition, filename)
        # checking if it is a file
        if os.path.isfile(f) and filename.endswith('.pt'):
            loss_list = torch.load(f)
            loss_per_epoch = agg(loss_list)
            plt.plot(loss_per_epoch[:100], label=extract_label(filename), linewidth=2)
    plt.legend(loc='upper right', fontsize=16)
    plt.title(partition, fontsize=30)
    print("showing")
    # plt.show()
    plt.savefig(os.path.join(loss_list_dir, f'{partition} figure.png'))
    plt.close()


if __name__ == '__main__':
    print(os.environ['PYCHARM_DISPLAY_PORT'])
    print(f'matplotib backend: {matplotlib.get_backend()}')
    #['GTK3Agg', 'GTK3Cairo', 'GTK4Agg', 'GTK4Cairo', 'MacOSX', 'nbAgg', 'QtAgg', 'QtCairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg', 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']

    # to actually see the plots, this script should be run from pycharm(or any other ide?) with remote interpreter
    # see https://www.jetbrains.com/help/pycharm/configuring-remote-interpreters-via-ssh.html#ssh
    # alternatively you can copy the .pt files to your local machine and run this script from there.
    filters = ['exp9 ', 'exp1 pgd']
    # filters = ['exp7', 'exp1 pgd']
    # filters = ['baseline-conv', 'baseline-swapped-pgd', 'lr:1.0', 'lr:0.1']
    # filters = ['exp3 ', 'exp1 pgd']
    # filters = ['_exp3', 'exp1 pgd']
    compare_results("results/loss_lists", partition='oos', filters=filters)
    compare_results("results/loss_lists", partition='ood', filters=filters)
    compare_results("results/loss_lists", partition='real', filters=filters)
    compare_results("results/loss_lists", partition='cross_validation', filters=['exp0(cv) 50', 'pgd'])
