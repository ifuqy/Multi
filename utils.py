import numpy as np
from scipy.interpolate import RectBivariateSpline as interp2d
from scipy import ndimage
import pickle

def save_to_pickle(data, filename):
    """
    Save data to a pickle file.
    
    Args:
        data: The data to be saved.
        filename: The name of the pickle file to save to.
        
    Returns:
        None. Prints success message or error details.
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        print(f"save to {filename}")
    except Exception as e:
        print(f"save error: {e}")
        
def load_from_pickle(filename):
    """
    Load data from a pickle file.
    
    Args:
        filename: The name of the pickle file.
        
    Returns:
        The loaded data if successful, None otherwise.
    """
    try:
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        print(f"Data successfully loaded from {filename}")
        return data
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def normalize(data):
    """
    Normalize a 1-3 dimensional array based on the median and standard deviation.
    (Keep the original dimensional structure of the data without modifying the input data)

    """
    if isinstance(data, list):
        return [normalize(sub_data) for sub_data in data]
    
    arr = np.asarray(data).astype(float)
    original_shape = arr.shape
    
    if arr.size == 0:
        return arr
    
    if arr.ndim > 1:
        arr_2d = arr.reshape(-1, original_shape[-1])
        
        medians = np.median(arr_2d, axis=1, keepdims=True)
        stds = np.std(arr_2d, axis=1, keepdims=True)
        
        arr_norm = arr_2d - medians
        np.divide(arr_norm, stds, out=arr_norm, where=(stds != 0))
        
        return arr_norm.reshape(original_shape)
    
    else:
        med = np.median(arr)
        std = np.std(arr)
        arr_norm = arr - med
        if std > 0:
            arr_norm /= std
        return arr_norm

import numpy as np
from scipy import ndimage

def downsample(a, n, align=0):
    '''a: input array of 1-3 dimensions
       n: downsample to n bins per dimension
       optional:
       align : if non-zero, downsample grid (coords) 
               will have a bin at same location as 'align'
               ( typically max(sum profile) )
               useful for plots vs. phase
    '''
    if isinstance(a, list):
        return [downsample(sub_a, n, align) for sub_a in a]
    
    a = np.asarray(a)
    D = a.ndim
    if D not in {1, 2, 3}:
        raise ValueError(f"Unsupported number of dimensions: {D}")
    
    if D == 1:
        m = len(a)
        x_phase = np.linspace(0, 1 - 1/m, m)
        if align:
            target_phase = np.linspace(0, 1 - 1/n, n) + x_phase[align]
            target_phase %= 1
            target_phase.sort()
        else:
            target_phase = np.linspace(0, 1 - 1/n, n)
        return np.interp(target_phase, x_phase, a)
    
    elif D == 2:
        d1, d2 = a.shape
        if align:
            x2_phase = np.linspace(0, 1 - 1/d2, d2)
            target_y_phase = np.linspace(0, 1 - 1/n, n) + x2_phase[align]
            target_y_phase %= 1
            target_y_phase.sort()
            
            coords_y = target_y_phase * (d2 - 1)
            coords_x = np.linspace(0, d1-1, n)
            
            grid_x, grid_y = np.meshgrid(coords_x, coords_y, indexing='ij')
            coords = np.stack([grid_x.ravel(), grid_y.ravel()], axis=0)
            result = ndimage.map_coordinates(
                a, coords, order=1, mode='reflect', prefilter=True
            ).reshape(n, n)
            return result
        else:
            zoom_factor = (n/d1, n/d2)
            return ndimage.zoom(a, zoom_factor, order=1, prefilter=True)
    
    elif D == 3:
        d1, d2, d3 = a.shape
        if align:
            x2_phase = np.linspace(0, 1 - 1/d2, d2)
            target_y_phase = np.linspace(0, 1 - 1/n, n) + x2_phase[align]
            target_y_phase %= 1
            target_y_phase.sort()
            
            coords_y = target_y_phase * (d2 - 1)
            coords_x = np.linspace(0, d1-1, n)
            coords_z = np.linspace(0, d3-1, n)
            
            grid_x, grid_y, grid_z = np.meshgrid(
                coords_x, coords_y, coords_z, indexing='ij'
            )
            coords = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=0)
            result = ndimage.map_coordinates(
                a, coords, order=1, mode='reflect', prefilter=True
            ).reshape(n, n, n)
            return result
        else:
            zoom_factor = (n/d1, n/d2, n/d3)
            return ndimage.zoom(a, zoom_factor, order=1, prefilter=True)
        
def shiftphase(x):
    """
    Shift the phase of a 1D or 2D array so that the maximum value is centered.
    
    """
    try:
        array = x.copy()
        dim = np.ndim(array)
        if dim == 1:
            num_col = len(array)
            sum_col = array
        elif dim == 2:
            _, num_col = array.shape
            sum_col = array.sum(axis=0)
        else:
            raise ValueError("Input array must be 1D or 2D.")
        max_index = np.argmax(sum_col)
        shift = num_col // 2 - max_index
        array = np.roll(array, int(shift), axis=-1)
        return array
    except ValueError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unknown error occurred: {e}")