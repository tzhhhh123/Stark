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
args = parser.parse_args()

trackers.extend(trackerlist(name='stark_s', parameter_name=args.parameter_name, dataset_name='lasot',
                            run_ids=None, display_name='STARK-S'))

dataset = get_dataset('lasot')
plot_results(trackers, dataset, 'lasot', merge_results=True, plot_types=('success', 'norm_prec'),
             skip_missing_seq=False, force_evaluation=True, plot_bin_gap=0.05)
print_results(trackers, dataset, 'lasot', merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
# print_per_sequence_results(trackers, dataset, report_name="debug")
