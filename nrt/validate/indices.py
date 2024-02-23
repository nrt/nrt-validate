class NDVI(object):
    def __init__(self, red='B04_20', nir='B8A'):
        self.red = red
        self.nir = nir

    def __call__(self, ds):
        da = (ds[self.nir].astype(np.float32) - ds[self.red].astype(np.float32)) / (ds[self.nir] + ds[self.red] + 0.0000001)
        return da
