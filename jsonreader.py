import json
import base64
import numpy as np

from utils import normalize, downsample, shiftphase


def _decode_numpy_array(array_dict):
    """
    Decode a NumPy array that is stored inside a JSON file.

    The array is saved using:
    - base64 encoded data
    - dtype (e.g. float32)
    - shape
    """

    # Decode base64 string back into raw bytes
    raw_bytes = base64.b64decode(array_dict["data"])

    # Reconstruct NumPy data type
    dtype = np.dtype(array_dict["dtype"])

    # Get array shape
    shape = tuple(array_dict["shape"])

    # Convert bytes to NumPy array and reshape
    array = np.frombuffer(raw_bytes, dtype=dtype)
    array = array.reshape(shape)

    return array


def extract_from_json(j):
    """
    Extract the main 2D time–phase array from a JSON candidate.

    The array is stored under "subints".
    It may be stored under "peaks" -> "values".
    """

    # Try to read from "subints"
    if "subints" in j and isinstance(j["subints"], dict) and "data" in j["subints"]:
        return _decode_numpy_array(j["subints"])

    # Fallback: try "peaks.values"
    if "peaks" in j and isinstance(j["peaks"], dict):
        if (
            "values" in j["peaks"]
            and isinstance(j["peaks"]["values"], dict)
            and "data" in j["peaks"]["values"]
        ):
            arr = _decode_numpy_array(j["peaks"]["values"])

            # Ensure the array is 2D
            if arr.ndim != 2:
                raise ValueError("Extracted array is not 2D")

            return arr

    # If no valid 2D array is found
    raise KeyError("Could not find a 2D array in the JSON file")


class jsonreader:
    """
    Reader class for pulsar candidate JSON files.
    """

    def __init__(self, jsonfile):
        # Store file path
        self.jsonfile = jsonfile

        # Load JSON file
        with open(jsonfile, "r") as f:
            j = json.load(f)

        # Handle case where JSON is wrapped in a list
        if isinstance(j, list):
            j = j[0]

        # Store full JSON content
        self.data = j

        # Extract 2D time–phase panel
        self.panel = extract_from_json(j).astype(np.float32)



        # Create 1D pulse profile by summing over time
        self.profile = self.panel.sum(axis=0)

        # Find index of maximum pulse intensity
        self.align = int(np.argmax(self.profile))

    

        

    def extract(self, phasebins=0, intervals=0, subbands=0, DMbins=0,
                centre=True, align=True):
        """
        Extract features from the JSON candidate.

        Supported features:
        - phasebins: 1D pulse profile
        - intervals: 2D time–phase panel
        """

        # Recompute alignment if requested
        if align:
            self.align = int(np.argmax(self.profile))

        # 1D pulse profile
        if phasebins != 0:

            def make_profile(b):
                # Downsample profile to required number of phase bins
                x = downsample(self.profile, b, align=self.align).ravel()

                # Normalize values
                x = normalize(x)

                # Center pulse if requested
                if centre:
                    x = shiftphase(x)

                return x

            # Multiple resolutions
            if isinstance(phasebins, list):
                out = []
                for b in phasebins:
                    out.append(make_profile(b))
                return out

            # Single resolution
            return make_profile(phasebins)

        # 2D time–phase panel 
        if intervals != 0:

            def make_interval(b):
                x = self.panel

                # Remove rows that contain only zeros
                x = x[np.any(x != 0, axis=1)]

                # Downsample along phase axis
                x = downsample(x, b, align=self.align)

                # Normalize values
                x = normalize(x)

                # Center pulse if requested
                if centre:
                    x = shiftphase(x)

                return x

            # Multiple resolutions
            if isinstance(intervals, list):
                out = []
                for b in intervals:
                    out.append(make_interval(b))
                return out

            # Single resolution
            return make_interval(intervals)

        # Unsupported feature request
        raise NotImplementedError(
            "Only phasebins (profile) and intervals (panel) are supported"
        )
    def get_path(self):
        return self.jsonfile