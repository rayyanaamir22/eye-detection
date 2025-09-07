"""
Test install of a lib (for tf since im on mac)
"""

if __name__ == '__main__':

    import numpy as np
    import tensorflow as tf

    print("TF:", tf.__version__)
    print("NumPy:", np.__version__)
    print("GPUs:", tf.config.list_physical_devices("GPU"))
