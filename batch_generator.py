"""
============================================================
KpopVAT Batch Generator
============================================================

Reads the master sample index, shuffles it once,
and builds mini-batches for KpopVAT.
"""

import csv
import random
import numpy as np
from sample_generator import SampleGenerator
from dataset_index import build_dataset_index
from raster_manager import RasterManager

class BatchGenerator:
    def __init__( self, csv_file, sample_generator, batch_size, shuffle=True, seed=42):
        self.csv_file = csv_file
        self.sample_generator = sample_generator
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.rows = self.load_rows()
        self.num_samples = len(self.rows)
        self.num_batches = int( np.ceil( self.num_samples / self.batch_size ) )
        
    def load_rows(self):
        with open(self.csv_file, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if self.shuffle:
            rng = random.Random(self.seed)
            rng.shuffle(rows)

        return rows

    def __len__(self):
        return self.num_batches

    # ======================================================
    # Build One Batch
    # ======================================================

    def __getitem__(self, index):

        start = index * self.batch_size
        stop = min( start + self.batch_size, self.num_samples )
        batch_rows = self.rows[start:stop]
        xp_batch = []
        xc_batch = []
        xg_batch = []
        mask_batch = []
        task_label_batch = []
        regression_target_batch = []
        for row in batch_rows:
            xp, xc, xg, mask, task_label, regression_target = self.sample_generator.build_sample(row)
            xp_batch.append(xp)
            xc_batch.append(xc)
            xg_batch.append(xg)
            mask_batch.append(mask)
            task_label_batch.append(task_label)
            regression_target_batch.append(regression_target)

        xp_batch = np.asarray(xp_batch, dtype=np.float32)
        xc_batch = np.asarray(xc_batch, dtype=np.float32)
        try:
            xg_batch = np.asarray(xg_batch, dtype=np.float32)
        except Exception:
            print("\n===== BAD BATCH =====\n")
            for k, x in enumerate(xg_batch):
                print("--------------------------------")
                print("Sample :", k)
                print("Type   :", type(x))
                if isinstance(x, np.ndarray):
                    print("Shape  :", x.shape)
                    print("dtype  :", x.dtype)
                else:
                    print(x)
            raise
        mask_batch = np.asarray(mask_batch, dtype=np.float32)
        task_label_batch = np.asarray(task_label_batch, dtype=np.int32)
        regression_target_batch = np.asarray(regression_target_batch, dtype=np.float32)
        return ( [xp_batch, xc_batch, xg_batch, mask_batch], [ task_label_batch, xc_batch ] )
        
def main():

    dataset = build_dataset_index()
    manager = RasterManager(dataset)
    sample_generator = SampleGenerator( manager, patch_size=11, patch_size_global=31 )
    generator = BatchGenerator( csv_file="train_index.csv", sample_generator=sample_generator, batch_size=8, shuffle=True, seed=42 )
    print()
    print("=" * 60)
    print("Batch Generator Test")
    print("=" * 60)
    print()
    print("Samples :", generator.num_samples)
    print("Batches :", len(generator))
    print()
    inputs, outputs, _= generator[0]

    xp, xc, xg, mask = inputs

    task_labels, reconstruction_target = outputs
    print("Xp :", xp.shape)

    print("Xc :", xc.shape)

    print("Xg :", xg.shape)

    print("Mask :", mask.shape)

    print("Task Labels :", task_labels.shape)

    print("Reconstruction Target :", reconstruction_target.shape)

    manager.close()


if __name__ == "__main__":

    main()