from prepfold import pfd
import os
import pickle
import numpy as np
from utils import normalize, downsample, shiftphase
import yaml

def load_config(config_path):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config

global_config = load_config('./config/global_cfg.yaml')

use_multiprocessing = global_config['MULTIPROCESSING']['ENABLED']

if use_multiprocessing:
    num_workers = global_config['MULTIPROCESSING']['NUM_WORKERS']

class pfdreader(pfd):
    def __init__(self, pfdfile):
        if os.access(pfdfile, os.R_OK) and os.path.splitext(pfdfile)[1] == '.pfd':
            super().__init__(pfdfile)
            self.pfdfile = pfdfile
            self.dedisperse(DM=self.bestdm, doppler=1)
            self.adjust_period()

    def getdata(self, *args, **kwargs):
        data = []
        for feature in args:
            for key, value in feature.items():
                if isinstance(value, list):
                    data.extend(self.extract(**{key:value}))
                else:
                    data.append(self.extract(**{key:value}))
        for key, value in kwargs.items():
            if isinstance(value, list):
                data.extend(self.extract(**{key:value}))
            else:
                data.append(self.extract(**{key:value}))
        return data
    
    def extract(self, phasebins=0, intervals=0, subbands=0, DMbins=0, centre=True, align=True):
        
        def get_sumprofs(bins):
            sumprofs = self.plot_sumprof()
            if type(bins) == list:
                data = []
                for bin in bins:
                    data.append(normalize(downsample(sumprofs, bin, align=self.align).ravel()))
                del sumprofs
                return data
            else:   
                sumprofs = normalize(downsample(sumprofs, bins, align=self.align).ravel())
                return sumprofs
            
        def get_intervals(bins):
            intervals = self.plot_intervals()
            intervals = intervals[np.any(intervals != 0, axis=1)]
            if type(bins) == list:
                data = []
                for bin in bins:
                    data.append(normalize(downsample(intervals, bin, align=self.align)))
                del intervals
                return data
            else:
                intervals = normalize(downsample(intervals, bins, align=self.align))
                return intervals
            
        def get_subbands(bins):
            subbands = self.plot_subbands()
            subbands = subbands[np.any(subbands != 0, axis=1)]
            if type(bins) == list:
                data = []
                for bin in bins:
                    data.append(normalize(downsample(subbands, bin, align=self.align)))
                del subbands
                return data
            else:
                subbands = normalize(downsample(subbands, bins, align=self.align))
                return subbands
            
        def get_DMcurve(bins):
            DMcurve = self.DM_curve()
            if type(bins) == list:
                data = []
                for bin in bins:
                    data.append(normalize(downsample(DMcurve, bin, align=self.align).ravel()))
                del DMcurve
                return data
            else:
                DMcurve = normalize(downsample(DMcurve, bins, align=self.align).ravel())
                return DMcurve
            
        if align:
            self.align = self.profs.sum(0).sum(0).argmax()

        if phasebins != 0:
            if type(phasebins) == list:
                if centre:
                    return list(map(shiftphase, get_sumprofs(phasebins)))
                else:
                    return get_sumprofs(phasebins)
            else:
                if centre:
                    return shiftphase(get_sumprofs(phasebins))
                else:
                    return get_sumprofs(phasebins)
        elif intervals != 0:
            if type(intervals) == list:
                if centre:
                    return list(map(shiftphase, get_intervals(intervals)))
                else:
                    return get_intervals(intervals)
            else:
                if centre:
                    return shiftphase(get_intervals(intervals))
                else:
                    return get_intervals(intervals)
        elif subbands != 0:
            if type(subbands) == list:
                if centre:
                    return list(map(shiftphase, get_subbands(subbands)))
                else:
                    return get_subbands(subbands)
            else:
                if centre:
                    return shiftphase(get_subbands(subbands))
                else:
                    return get_subbands(subbands)
        elif DMbins != 0:
            if type(DMbins) == list:
                if centre:
                    return list(map(shiftphase, get_DMcurve(DMbins)))
                else:
                    return get_DMcurve(DMbins)
            if centre:
                return shiftphase(get_DMcurve(DMbins))
            else:
                return get_DMcurve(DMbins)

from multiprocessing import Pool, cpu_count  

def parallel_pfdreader(filename):
    return pfdreader(filename)      

from tqdm import tqdm

class dataloader(object):
    """
    Load .pfd files using a text file that records data path and labels.
    """

    def __init__(self, filename, classmap=None):
        """
        args: 
        filename: the name of the pickle file
        classmap: mapping for different classes
        """
        self.trainclassifiers = {}
        if filename.endswith('.pkl'):
            with open(filename, 'rb') as fileobj:
                originaldata = pickle.load(fileobj)
                self.pfds = originaldata['pfds']
                if type(originaldata['target']) in [list] or originaldata['target'].ndim == 1:
                    self.orig_target = originaldata['target']
                    if classmap is None:
                        self.classmap = {0:[4,5], 1:[6,7]}
                    else:
                        self.classmap = classmap
                    self.target = self.orig_target[:]
                    for k, v in self.classmap.items():
                        for val in v:
                            self.target[self.orig_target == val] = k 
                else:
                    self.target = originaldata['target']
        elif filename.endswith('.txt'):
            with open(filename, 'r') as f:
                first_line = f.readline()
                second_line = f.readline().strip()
                num_columns = len(second_line.split())

            if num_columns == 6:
                dtype = [('fname', '|S200'), ('Overall', int), ('Profile', int), ('Interval', int), ('Subband', int), ('DMCurve', int)]
            elif num_columns == 2:
                dtype = [('fname', '|S200'), ('Overall', int)]
            else:
                raise ValueError(f"Unexpected number of columns: {num_columns}")
            
            data = np.loadtxt(filename, dtype=dtype, comments='#')

            with Pool(processes=num_workers) as pool:
                self.pfds = list(tqdm(pool.imap(parallel_pfdreader, data['fname'].astype(str)),
                                    total=len(data['fname']),
                                    desc='PFDs Reading',
                                    mininterval=1.0))
                  
            if num_columns == 6:
                self.target = np.vstack((data['Overall'], data['Profile'], data['DMCurve'], data['Interval'], data['Subband'])).T
                # self.target = data['Overall']
            elif num_columns == 2:
                self.target = data['Overall']
        else:
            print("Don't recognize the file surfix.")
            raise Exception("Unrecognized file suffix")
        self.extracted_feature = []
