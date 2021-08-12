from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.antiuav_path = '/home/dsz/datasets/anti-uav'
    settings.davis_dir = ''
    settings.got10k_lmdb_path = '/mnt/data1/tzh/Stark/data/got10k_lmdb'
    settings.got10k_path = '/mnt/data1/tzh/Stark/data/got10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = ''
    settings.lasot_lmdb_path = '/mnt/data1/tzh/data/LaSOT/LaSOTBenchmark_lmdb'
    settings.lasot_path = '/mnt/data1/tzh/data/LaSOT/LaSOTBenchmark'
    settings.tnl2k_path = '/mnt/data1/tzh/data/TNL2K/'
    settings.network_path = '/mnt/data1/tzh/Stark/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '/mnt/data1/tzh/Stark'
    settings.result_plot_path = '/mnt/data1/tzh/Stark/test/result_plots'
    settings.results_path = '/mnt/data1/tzh/Stark/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/mnt/data1/tzh/Stark'
    settings.segmentation_path = '/mnt/data1/tzh/Stark/test/segmentation_results'
    settings.tn_packed_results_path = ''
    settings.tpl_path = ''
    settings.trackingnet_path = '/mnt/data1/tzh/Stark/data/trackingNet'
    settings.uav_path = ''
    settings.vot_path = '/mnt/data1/tzh/Stark/data/VOT2019'
    settings.youtubevos_dir = ''


    return settings

