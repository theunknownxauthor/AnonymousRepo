"""
============================================================
KpopVAT Raster Manager
============================================================
"""

import os
os.environ["GDAL_CACHEMAX"] = "256MB"

import rasterio
from config import *
from dataset_index import build_dataset_index
import numpy as np


class RasterManager:
    """
    Keeps raster datasets open during training.

    The manager never loads an entire raster into memory.
    It only stores raster handles.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.handles = {}
        self.open_all()

    def open_all(self):
        print("Opening raster datasets...")
        for split in ["train", "validation", "test"]:
            for roi in self.dataset[split]:
                country = roi["country"]
                roi_name = roi["roi"]
                for layer, path in roi["layers"].items():
                    key = (split, country, roi_name, layer)
                    self.handles[key] = rasterio.open(path)

                for task, path in roi["targets"].items():
                    key = (split, country, roi_name, task)
                    self.handles[key] = rasterio.open(path)

        print(f"Opened {len(self.handles)} raster datasets.")

    def get_dataset(self, split, country, roi, layer):
        key = (split, country, roi, layer)
        return self.handles[key]

    def close(self):
        print("Closing raster datasets...")
        for dataset in self.handles.values():
            dataset.close()

        self.handles.clear()
        print("All raster datasets closed.")

    def extract_pixel( self, split, country, roi, layer, i, j):
        """

        Returns
        -------
        numpy.ndarray

            Shape:

                (bands,)
        """
        dataset = self.get_dataset(split, country, roi, layer)
        pixel = dataset.read(        window=(  (i, i + 1),(j, j + 1)  )        )
        return pixel[:, 0, 0]

    def extract_patch( self, split, country, roi, layer, i, j, patch_size):
        dataset = self.get_dataset(split, country, roi, layer)
        half = patch_size // 2
        row0 = max(i - half, 0)
        row1 = min(i + half + 1, dataset.height)
        col0 = max(j - half, 0)
        col1 = min(j + half + 1, dataset.width)

        patch = dataset.read( window=((row0, row1), (col0, col1)) )
        patch = np.transpose(patch, (1, 2, 0))
        pad_top = max(0, half - i)
        pad_bottom = max( 0, (i + half + 1) - dataset.height )
        pad_left = max(0, half - j)
        pad_right = max( 0, (j + half + 1) - dataset.width )

        if ( pad_top or pad_bottom or pad_left or pad_right ):
            patch = np.pad( patch, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode="reflect" )
        return patch.astype(np.float32)
    
    def normalize(self, layer, data):
        data = data.astype(np.float32)
        if layer == RGB:
            rgb_keys = ["rgb_red", "rgb_green", "rgb_blue"]
            for band, key in enumerate(rgb_keys):
                stats = self.dataset["normalization"][key]
                minimum = stats["min"]
                maximum = stats["max"]
                if maximum != minimum:
                    data[..., band] = (data[..., band] - minimum) / (maximum - minimum)             
            return np.clip(data, 0.0, 1.0)
            
        stats = self.dataset["normalization"][layer]
        minimum = stats["min"]
        maximum = stats["max"]
        if maximum != minimum:
            data = (data - minimum) / (maximum - minimum)
        return np.clip(data, 0.0, 1.0)
        
    def build_xp(self, split, country, roi, i, j, selected_modalities):
        """
        Build the Xp tensor.

        Shape
        -----
        (1, 1, RGB + ancillary)
        """
        channels = []
        rgb = self.extract_pixel( split, country, roi, RGB, i, j)
        rgb = self.normalize(RGB, rgb)
        channels.extend(rgb.tolist())
        for layer in selected_modalities:
            value = self.extract_pixel(split, country, roi, layer, i, j)
            value = self.normalize(layer, value)
            channels.append(float(value[0]))
        xp = np.array( channels, dtype=np.float32 )
        xp = xp.reshape( 1, 1, len(channels))
        
        return xp
        
    def build_xc( self, split, country, roi, i, j, selected_modalities, patch_size ):
        patches = []
        rgb = self.extract_patch( split, country, roi, RGB, i, j, patch_size )
        rgb = self.normalize(RGB, rgb)
        patches.append(rgb)
        for layer in selected_modalities:
            patch = self.extract_patch( split, country, roi, layer, i, j, patch_size )
            patch = self.normalize(layer, patch)
            patches.append(patch)

        return np.concatenate( patches, axis=-1 )
        
    def build_xg( self, split, country, roi, i, j, selected_modalities, patch_size_global):
        patches = []
        rgb = self.extract_patch( split, country, roi, RGB, i, j, patch_size_global )
        rgb = self.normalize(RGB, rgb)
        patches.append(rgb)
        for layer in selected_modalities:
            patch = self.extract_patch( split, country, roi, layer, i, j, patch_size_global )
            patch = self.normalize(layer, patch)
            patches.append(patch)

        return np.concatenate( patches, axis=-1 )
    
    def build_target( self, split, country, roi, task, i, j):
        target = self.extract_pixel( split, country, roi, task, i, j)
        return float(target[0])
        
def main():

    dataset = build_dataset_index()

    manager = RasterManager(dataset)

    print()

    print("=" * 60)

    print("Raster Manager Test")

    print("=" * 60)

    print()

    print("Number of opened datasets :")

    print(len(manager.handles))

    print()
    print("\nFirst 10 keys:")

    for key in list(manager.handles.keys())[:10]:
        print(key)
    
    key = (
        "train",
        "u",
        "roi_1",
        "rgb"
    )

    raster = manager.handles[key]

    print("Example dataset")

    print("-----------------------")

    print("Key :", key)

    print("Width :", raster.width)

    print("Height :", raster.height)

    print("Bands :", raster.count)

    print("CRS :", raster.crs)

    selected_modalities = [
        NTL,
        BUILDING,
        LAND_COVER
    ]

    xp = manager.build_xp(
        "train",
        "u",
        "roi_1",
        2500,
        2500,
        selected_modalities
    )

    xc = manager.build_xc(
        "train",
        "u",
        "roi_1",
        2500,
        2500,
        selected_modalities,
        patch_size=11
    )

    xg = manager.build_xg(
        "train",
        "u",
        "roi_1",
        2500,
        2500,
        selected_modalities,
        patch_size_global=31
    )

    target = manager.build_target(
        "train",
        "u",
        "roi_1",
        TASK_POPULATION,
        2500,
        2500
    )

    print("\n========== KpopVAT Sample ==========")
    print("Xp :", xp.shape)
    print("Xc :", xc.shape)
    print("Xg :", xg.shape)
    print("Target :", target)



if __name__ == "__main__":

    main()