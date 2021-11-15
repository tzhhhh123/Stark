import _init_paths
import argparse
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8, 8]

from lib.test.analysis.plot_results import plot_results, print_results, print_per_sequence_results
from lib.test.evaluation import get_dataset, trackerlist

trackers = []

# baseline_lasot
parser = argparse.ArgumentParser()
parser.add_argument('parameter_name', type=str, default='lasot_words', help='parameter_name')
parser.add_argument('--dataset', type=str, default='lasot', help='parameter_name')
parser.add_argument('--ex', default=None, type=str, help='parameter_name')
args = parser.parse_args()
ex = args.ex
# trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
#                             run_ids=None, display_name='Ground'))
# trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
#                             run_ids=None, display_name='First'))
# trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
#                             run_ids=None, display_name='Interval-1'))
# trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
#                             run_ids=None, display_name='Interval-10'))
# trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
#                             run_ids=None, display_name='Interval-100'))
# trackers[1].name = '1'
# trackers[2].name = '2'
# trackers[3].name = '3'
# trackers[4].name = '4'
#
#
# trackers[0].results_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st/baseline_mlp_fuse_v2_nlp_grounding_tnl2k'
# trackers[1].results_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st/nl_test_0'
# trackers[2].results_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st/nl_test_1'
# trackers[3].results_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st/nl_test_10'
# trackers[4].results_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st/nl_test_100'


###lasot
trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
                            run_ids=None, display_name='TransNLT'))
trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
                            run_ids=None, display_name='TransNLT-ori'))
trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
                            run_ids=None, display_name='NL-TransNLT'))
trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
                            run_ids=None, display_name='NL-TransNLT-ori'))
# trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
#                             run_ids=None, display_name='SNLT'))

# trackers.extend(trackerlist(name='stark_st', parameter_name=args.parameter_name, dataset_name=args.dataset,
#                             run_ids=None, display_name='SNLT'))
trackers[1].name = '1'
trackers[2].name = '2'
trackers[3].name = '3'
# trackers[4].name = 'snlt'

trackers[0].results_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st/baseline_mlp_fuse_v2_fuse_box'
trackers[1].results_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st/baseline_mlp_fuse_v2_all/baseline_mlp_fuse_v2_fuse_box_860'
trackers[2].results_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st/baseline_mlp_fuse_v2_nlp_grounding_tnl2k'
trackers[3].results_dir = '/mnt/data1/tzh/Stark/test/tracking_results/stark_st/baseline_mlp_fuse_v2_all/baseline_mlp_fuse_v2_nlp_grounding/'
# trackers[4].results_dir = '/mnt/data1/tzh/Stark/test/tracking_results/lasot_fig/snlt_lasot/'

dataset = get_dataset(args.dataset)
plot_results(trackers, dataset, args.dataset, merge_results=True, plot_types=('success', 'norm_prec', 'prec'),
             skip_missing_seq=True, force_evaluation=True, plot_bin_gap=0.05, ex=ex)
print_results(trackers, dataset, args.dataset, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
# print_per_sequence_results(trackers, dataset, report_name="debug")
