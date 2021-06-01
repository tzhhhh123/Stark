class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/mnt/data3/tzh/Stark'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/mnt/data3/tzh/Stark/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/mnt/data3/tzh/Stark/pretrained_networks'
        self.lasot_dir = '/mnt/data3/tzh/Stark/data/lasot'
        self.got10k_dir = '/mnt/data3/tzh/Stark/data/got10k'
        self.lasot_lmdb_dir = '/mnt/data3/tzh/Stark/data/lasot_lmdb'
        self.got10k_lmdb_dir = '/mnt/data3/tzh/Stark/data/got10k_lmdb'
        self.antiuav_dir = '/home/dsz/datasets/anti-uav'
        self.trackingnet_dir = '/mnt/data3/tzh/Stark/data/trackingnet'
        self.trackingnet_lmdb_dir = '/mnt/data3/tzh/Stark/data/trackingnet_lmdb'
        self.coco_dir = '/mnt/data3/tzh/Stark/data/coco'
        self.coco_lmdb_dir = '/mnt/data3/tzh/Stark/data/coco_lmdb'
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = '/mnt/data3/tzh/Stark/data/vid'
        self.imagenet_lmdb_dir = '/mnt/data3/tzh/Stark/data/vid_lmdb'
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
