import csv
import numpy as np

from config import *
from raster_manager import RasterManager
from dataset_index import build_dataset_index

import numpy as np


def encode_gps(latitude, longitude):
    latitude = np.deg2rad(latitude)
    longitude = np.deg2rad(longitude)
    return np.asarray( [ np.sin(latitude), np.cos(latitude), np.sin(longitude), np.cos(longitude)], dtype=np.float32 )

class SampleGenerator:
    def __init__( self, raster_manager, patch_size,patch_size_global):
        self.raster_manager = raster_manager
        self.patch_size = patch_size
        self.patch_size_global = patch_size_global

    def parse_modalities(self, modalities_string):
        """
        Convert ntl-building-lc into ["ntl", "building", "lc"]
        """
        return modalities_string.split("-")

    def parse_mask(self, mask_string):
        """
        Convert 0101101 into float32 array
        """
        return np.array( [float(x) for x in mask_string], dtype=np.float32 )

    def build_sample(self, row):
        task = row["task"]
        country = row["country"]
        roi = row["roi"]
        split = row["split"]
        i = int(row["pixel_i"])
        j = int(row["pixel_j"])
        selected_modalities = self.parse_modalities( row["modalities"] )
        mask = self.parse_mask( row["mask"] )
        xp = self.raster_manager.build_xp( split, country, roi, i, j, selected_modalities )
        xc = self.raster_manager.build_xc( split, country, roi, i, j, selected_modalities,self.patch_size )
        xg = self.raster_manager.build_xg( split, country, roi, i, j, selected_modalities,self.patch_size_global )
        regression_target = self.raster_manager.build_target( split, country, roi, task, i, j)
        task_to_label = { TASK_POPULATION: 0, TASK_BIOMASS: 1, TASK_BUILDING: 2 }
        task_label = task_to_label[task]
        latitude, longitude = ROI_GPS[country][roi]
        gps = encode_gps(latitude, longitude)
        return ( xp, xc, xg, gps, mask, task_label, regression_target )


def main():

    dataset = build_dataset_index()

    manager = RasterManager(dataset)

    generator = SampleGenerator(
        manager,
        patch_size=11,
        patch_size_global=31
    )

    with open("train_index.csv", "r", newline="") as f:

        reader = csv.DictReader(f)

        row = next(reader)

    xp, xc, xg, mask, task_label, regression_target = generator.build_sample(row)

    print()

    print("=" * 60)
    print("Sample Generator Test")
    print("=" * 60)

    print()

    print("Task")
    print(row["task"])

    print()

    print("Task Label")
    print(task_label)

    print()

    print("Regression Target")
    print(regression_target)

    print()

    print("Selected Modalities")
    print(row["modalities"])

    print()

    print("Mask")
    print(mask)

    print()

    print("Xp")
    print("Shape :", xp.shape)
    print("Dtype :", xp.dtype)

    print()

    print("Xc")
    print("Shape :", xc.shape)
    print("Dtype :", xc.dtype)

    print()

    print("Xg")
    print("Shape :", xg.shape)
    print("Dtype :", xg.dtype)

    print()

    print("Expected Input Shapes")
    print("---------------------")
    print("Xp :", (1, 1, 6))
    print("Xc :", (11, 11, 6))
    print("Xg :", (31, 31, 6))
    print("Mask :", (7,))

    manager.close()
    

if __name__ == "__main__":

    main()