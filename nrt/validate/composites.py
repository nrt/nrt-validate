from abc import ABC, abstractmethod
import numpy as np


class BaseComposite(ABC):
    """Abstract base class for all color compositors
    """
    @staticmethod
    def stretch(arr, blim=[20,2000], glim=[50,2000], rlim=[20,2000]):
        """
        Apply color stretching and [0,1] clipping to a 3 bands image.

        Args:
            arr (np.ndarray): 3D array; bands as last dimension in RGB order.
            blim, glim, rlim (list): min and max values between which to stretch the individual bands.

        Returns:
            np.ndarray: Stretched and clipped array.
        """
        bottom = np.array([[[rlim[0], glim[0], blim[0]]]])
        top = np.array([[[rlim[1], glim[1], blim[1]]]])
        arr_stretched = (arr - bottom) / (top - bottom)
        return np.clip(arr_stretched, 0.0, 1.0)

    @abstractmethod
    def __call__(self, ds):
        """Transform a given xarray Dataset into a color composite.

        Subclasses must implement this method to create a color composite.

        Args:
            ds (xarray.Dataset): Dataset from which to create the composite.

        Returns:
            The color composite as specified by the subclass.
        """
        pass


class SimpleComposite(BaseComposite):
    def __init__(self, b='B02', g='B03', r='B04',
                 blim=[20, 2000], glim=[50, 2000], rlim=[20, 2000]):
        """Initialize the SimpleComposite with specific bands and limits.

        Args:
            b, g, r (str): Band names for blue, green, and red components.
            blim, glim, rlim (list): Stretching limits for blue, green, and red bands.
        """
        self.b = b
        self.g = g
        self.r = r
        self.blim = blim
        self.glim = glim
        self.rlim = rlim

    def __call__(self, ds):
        """Create a stretched color composite from a multivariate xarray Dataset.

        Works only for a Dataset with a single temporal slice (e.g. only x and y
        coordinate valiables)

        Args:
            ds (xarray.Dataset): Dataset from which to create the composite.

        Returns:
            np.ndarray: A 3D numpy array representing the color composite.
        """
        # Extract the specified bands as numpy arrays
        rgb = np.stack([ds[self.r].values, ds[self.g].values, ds[self.b].values],
                       axis=-1)
        # Apply stretching
        rgb_stretched = self.stretch(rgb, self.blim, self.glim, self.rlim)
        return rgb_stretched


class S2CIR(SimpleComposite):
    def __init__(self, b='B03', g='B04', r='B08',
                 blim=[250, 1300], glim=[150, 1700], rlim=[1500, 4000]):
        """Initialize the S2CIR composite with specific bands for CIR (Color Infrared).

        Convenience class to quickly instantiate a Sentinel 2 Color Infrared
        compositor with sensible defaults (for L2A surface reflectance values x
        10000)
        """
        super().__init__(b=b, g=g, r=r, blim=blim, glim=glim, rlim=rlim)


class S2SWIR(SimpleComposite):
    def __init__(self, b='B04', g='B8A', r='B11',
                 blim=[150, 1200], glim=[1800, 3500], rlim=[800, 3000]):
        """Initialize the S2SWIR composite with specific bands for SWIR.

        Convenience class to quickly instantiate a Sentinel 2 SWIR color
        compositor with sensible defaults (for L2A surface reflectance values x
        10000)
        """
        super().__init__(b=b, g=g, r=r, blim=blim, glim=glim, rlim=rlim)


class S2SWIRx2(SimpleComposite):
    def __init__(self, b='B12', g='B11', r='B8A',
                 blim=[250, 1800], glim=[600, 3000], rlim=[1200, 4500]):
        """Initialize the color composite using both SWIR bands

        Convenience class to quickly instantiate a Sentinel 2 SWIRx2 color
        compositor with sensible defaults (for L2A surface reflectance values x
        10000)
        """
        super().__init__(b=b, g=g, r=r, blim=blim, glim=glim, rlim=rlim)


class S2TasseledCapComposite(BaseComposite):
    def __init__(self, blue='B02', green='B03', red='B04',
                 nir='B8A', swir1='B11', swir2='B12',
                 rlim=[300,1500], glim=[1200,4500], blim=[600,2900]):
        """Compute greenness, brightness and wetness from S2 L2A data and
        assemble the three components into a color composite

        Args:
            blue, green, red, nir, swir1, swir2 (str): Variable names in the input
                xarray Dataset
            blim, glim, rlim (list): Stretching limits for blue, green, and red bands.
        """
        self.blue = blue
        self.green = green
        self.red = red
        self.nir = nir
        self.swir1 = swir1
        self.swir2 = swir2
        self.blim = blim
        self.glim = glim
        self.rlim = rlim

    def greenness(self, ds):
        da = (-0.2848 * ds[self.blue]) - (0.2435 * ds[self.green]) \
             - (0.5436 * ds[self.red]) + (0.7243 * ds[self.nir]) \
             + (0.0840 * ds[self.swir1]) - (0.1800 * ds[self.swir2])
        return da

    def brightness(self, ds):
        da = (0.3037 * ds[self.blue]) + (0.2793 * ds[self.green]) \
             + (0.4743 * ds[self.red]) + (0.5585 * ds[self.nir]) \
             + (0.5082 * ds[self.swir1] + 0.1863 * ds[self.swir2])
        return da

    def wetness(self, ds):
        da = (0.1509 * ds[self.blue] + 0.1973 * ds[self.green]) \
             + (0.3279 * ds[self.red] + 0.3406 * ds[self.nir]) \
             - (0.7112 * ds[self.swir1] - 0.4572 * ds[self.swir2])
        return da

    def __call__(self, ds):
        greenness = self.greenness(ds)
        brightness = self.brightness(ds)
        wetness = self.wetness(ds)
        # Extract the specified bands as numpy arrays
        rgb = np.stack([brightness.values, greenness.values, wetness.values],
                       axis=-1)
        # Apply stretching
        rgb_stretched = self.stretch(rgb, self.blim, self.glim, self.rlim)
        return rgb_stretched
