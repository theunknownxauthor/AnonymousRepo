import os
import csv
import random

from config import *

def load_normalization_statistics():
    statistics = {}
    with open(NORMALIZATION_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            statistics[row["layer"]] = {    "min": float(row["min"]),
                                            "max": float(row["max"])     }
    return statistics

def verify_dataset():
    print("Verifying dataset...")
    for layer in ANCILLARY_LAYERS + [RGB] + TASKS:
        folder = os.path.join(DATASET_ROOT, layer)
        if not os.path.isdir(folder):
            raise Exception(f"Missing folder: {folder}")
            
    if not os.path.isfile(NORMALIZATION_CSV):
        raise Exception("Normalization CSV not found.")
    print("Dataset verification completed.")

def build_roi(split, country, roi):
    roi_name = f"roi_{roi}.tif"
    layers = {}
    layers[RGB] = os.path.join( DATASET_ROOT, RGB, f"{RGB}_{country}", roi_name)

    for layer in ANCILLARY_LAYERS:
        layers[layer] = os.path.join( DATASET_ROOT, layer, f"{layer}_{country}", roi_name )

    targets = {}

    for task in TASKS:
        targets[task] = os.path.join(DATASET_ROOT, task, f"{task}_{country}", roi_name)

    return {    "split": split,
                "country": country,
                "roi": f"roi_{roi}",
                "country_name": COUNTRIES[country],
                "layers": layers,
                "targets": targets  }
                
def build_dataset_index():

    dataset = {

        "train": [],

        "validation": [],

        "test": [],

        "normalization": load_normalization_statistics()

    }
    
    for country in TRAIN_COUNTRIES:

        for roi in [1, 2]:

            dataset["train"].append(
                build_roi("train", country, roi)
            )

            dataset["train"].append(build_roi("train", country, roi))
    for country in VALIDATION_COUNTRIES:

        for roi in [1, 2]:

            dataset["validation"].append(
                build_roi("validation", country, roi)
            )

    for country in TEST_COUNTRIES:

        for roi in [1, 2]:

            dataset["test"].append(
                build_roi("test", country, roi)
            )

    return dataset

def get_rois(dataset, split):

    return dataset[split]

def get_candidate_modalities(task):

    return TASK_MODALITIES[task]

def sample_modalities(task):

    candidates = TASK_MODALITIES[task]

    return random.sample(candidates, SELECTED_MODALITIES)

def build_mask(selected_modalities):

    mask = []

    for layer in ANCILLARY_LAYERS:

        if layer in selected_modalities:
            mask.append(1.0)
        else:
            mask.append(0.0)

    return mask

def print_dataset_summary(dataset):

    print("----------------------------------")

    print("========== DATASET SUMMARY ==========")

    print("----------------------------------")

    print("Training ROIs :", len(dataset["train"]))

    print("Validation ROIs :", len(dataset["validation"]))

    print("Testing ROIs :", len(dataset["test"]))

    print("----------------------------------")

    print("Normalization layers :")

    for layer, stats in dataset["normalization"].items():

        print(f"    {layer:12s}  min = {stats['min']:.6f}   max = {stats['max']:.6f}")
    
    print("----------------------------------")


def main():
    verify_dataset()
    dataset = build_dataset_index()
    print_dataset_summary(dataset)
    task = TASK_POPULATION
    selected = sample_modalities(task)
    print("Task")
    print(task)
    print("----------------------------------")
    print("Selected modalities")
    print(selected)
    print("----------------------------------")
    mask = build_mask(selected)
    print("Mask")
    print(mask)
    print("----------------------------------")
    roi = dataset["train"][0]
    print("Example ROI")
    print("----------------------------------")
    print("Country :", roi["country_name"])
    print("ROI :", roi["roi"])
    print("----------------------------------")
    print("RGB")
    print(roi["layers"]["rgb"])
    print("----------------------------------")
    print("Population target")
    print(roi["targets"]["population"])
if __name__ == "__main__":

    main()