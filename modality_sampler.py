from itertools import combinations

from config import *


import random
from config import RASTER_WIDTH, RASTER_HEIGHT
import csv


def enumerate_combinations():
    """
    Enumerate all valid ancillary modality combinations.
    """
    task_combinations = {}
    for task in TASKS:
        candidates = TASK_MODALITIES[task]
        task_combinations[task] = list( combinations( candidates, SELECTED_MODALITIES ) )
    return task_combinations
   
def compute_combination_quota(roi_quota, combinations_list):
    """
    Returns
    -------
    dict

        {
            combination : quota
        }
    """
    
    n = len(combinations_list)
    base = roi_quota // n
    remainder = roi_quota % n
    quotas = {}
    for index, combination in enumerate(combinations_list):
        quota = base
        if index < remainder:
            quota += 1
        quotas[combination] = quota

    return quotas

def compute_roi_quota(country_quota):
    """
    Returns
    -------
    dict

        {
            "roi_1": quota,
            "roi_2": quota
        }    
        
        
    """

    n = len(ROI_NAMES)
    base = country_quota // n
    remainder = country_quota % n

    quotas = {}

    for index, roi in enumerate(ROI_NAMES):

        quota = base

        if index < remainder:
            quota += 1

        quotas[roi] = quota

    return quotas

def compute_task_quota(country_quota):
    """
    Compute a balanced quota for every task within a country.

    Parameters
    ----------
    country_quota : int
        Number of samples assigned to one country.

    Returns
    -------
    dict

        {
            "population": quota,
            "biomass": quota,
            "building": quota
        }
    dict { "population": quota, "biomass": quota, "building": quota }
    """

    n = len(TASKS)

    base = country_quota // n

    remainder = country_quota % n

    quotas = {}

    for index, task in enumerate(TASKS):

        quota = base

        if index < remainder:
            quota += 1

        quotas[task] = quota

    return quotas

def compute_country_quota(split, split_quota):
    """
    Compute a balanced quota for every country within a dataset split.

    Parameters
    ----------
    split : str
        "train", "validation", or "test"

    split_quota : int
        Number of samples assigned to the split.

    Returns
    -------
    dict

        {
            country_code : quota
        }
    
    """

    if split == "train":
        countries = TRAIN_COUNTRIES

    elif split == "validation":
        countries = VALIDATION_COUNTRIES

    elif split == "test":
        countries = TEST_COUNTRIES

    else:
        raise ValueError(f"Unknown split: {split}")

    n = len(countries)

    base = split_quota // n

    remainder = split_quota % n

    quotas = {}

    for index, country in enumerate(countries):

        quota = base

        if index < remainder:
            quota += 1

        quotas[country] = quota

    return quotas
    
def compute_split_quota(total_samples):
    """
    Returns
    -------
    dict

        {
            "train": quota,
            "validation": quota,
            "test": quota
        }
    
    """
    splits = { "train": len(TRAIN_COUNTRIES), "validation": len(VALIDATION_COUNTRIES),"test": len(TEST_COUNTRIES) }
    total_countries = sum(splits.values())
    quotas = {}
    allocated = 0
    split_names = list(splits.keys())
    for split in split_names[:-1]:
        quota = int(total_samples * splits[split] / total_countries)
        quotas[split] = quota
        allocated += quota
    quotas[split_names[-1]] = total_samples - allocated

    return quotas

def generate_unique_pixels(n_samples, used_pixels=None, seed=None):
    """
    Returns
    -------
    list of tuples
        [(i, j), (i, j), ...]
    """

    if used_pixels is None:
        used_pixels = set()

    if seed is not None:
        random.seed(seed)

    pixels = []
    max_attempts = n_samples * 50  # safety guard
    attempts = 0
    while len(pixels) < n_samples:
        if attempts > max_attempts:
            raise RuntimeError( "Too many attempts generating unique pixels. Check dataset size or quotas.")
        i = random.randint(0, RASTER_HEIGHT - 1)
        j = random.randint(0, RASTER_WIDTH - 1)
        attempts += 1
        if (i, j) in used_pixels:
            continue
        used_pixels.add((i, j))
        pixels.append((i, j))

    return pixels
    
def build_sample_rows(split, country, country_quota):
    """
    Returns
    -------
    list

        [
            {
                ...
            },
            ...
        ]
    """

    rows = []
    combinations_dict = enumerate_combinations()
    task_quotas = compute_task_quota(country_quota)
    for task, task_quota in task_quotas.items():
        roi_quotas = compute_roi_quota(task_quota)
        for roi, roi_quota in roi_quotas.items():
            combination_quotas = compute_combination_quota( roi_quota, combinations_dict[task])
            for combination, combination_quota in combination_quotas.items():
                used_pixels = set()
                pixels = generate_unique_pixels( combination_quota, used_pixels )
                mask = build_mask(list(combination))
                modality_string = "-".join(combination)
                for i, j in pixels:
                    rows.append({
                        "split": split,
                        "task": task,
                        "country": country,
                        "roi": roi,
                        "pixel_i": i,
                        "pixel_j": j,
                        "modalities": modality_string,
                        "mask": "".join(map(str, map(int, mask)))
                    })

    return rows

def build_sample_task_rows(split, country, country_quota, task):
    if task not in TASKS:
        raise ValueError(f"Unknown task: {task}")

    rows = []
    combinations_dict = enumerate_combinations()
    # ------------------------------------------------------
    # Entire quota goes to this task
    # ------------------------------------------------------
    roi_quotas = compute_roi_quota(country_quota)
    for roi, roi_quota in roi_quotas.items():
        combination_quotas = compute_combination_quota( roi_quota, combinations_dict[task] )
        for combination, combination_quota in combination_quotas.items():
            used_pixels = set()
            pixels = generate_unique_pixels( combination_quota, used_pixels )
            modality_string = "-".join(combination)
            mask = build_mask(list(combination))
            for i, j in pixels:
                rows.append({   "split": split,
                                "task": task,
                                "country": country,
                                "roi": roi,
                                "pixel_i": i,
                                "pixel_j": j,
                                "modalities": modality_string,
                                "mask": "".join(  map(str, map(int, mask)) ) })
    return rows

def build_mask(selected_modalities):
    """
    Returns
    -------
    list

        Example

        [0, 1, 0, 1, 1, 0, 0]
    """

    mask = []
    for layer in ANCILLARY_LAYERS:
        if layer in selected_modalities:
            mask.append(1)
        else:
            mask.append(0)

    return mask

def write_index_csv(filename, rows):
    if len(rows) == 0:
        print(f"No rows to write for {filename}")
        return
     
    fieldnames = [  "split",
                    "task",
                    "country",
                    "roi",
                    "pixel_i",
                    "pixel_j",
                    "modalities",
                    "mask"]

    with open(filename, "w", newline="") as csvfile:
        writer = csv.DictWriter( csvfile, fieldnames=fieldnames )
        writer.writeheader()
        writer.writerows(rows)

    print(f"{filename} written successfully.")
    print(f"Number of samples : {len(rows)}")
    
def main():

    TRAIN_SAMPLES = 300
    VALIDATION_SAMPLES = 90
    TEST_SAMPLES = 120
    splits = {  "train": TRAIN_SAMPLES,
                "validation": VALIDATION_SAMPLES,
                "test": TEST_SAMPLES }

    for split, split_quota in splits.items():
        country_quotas = compute_country_quota( split, split_quota )
        rows = []
        for country, country_quota in country_quotas.items():

            print(
                f"Processing {COUNTRIES[country]} "
                f"({country_quota} samples)"
            )

            rows.extend(

                build_sample_rows(

                    split,

                    country,

                    country_quota

                )

            )

        filename = f"{split}_index.csv"

        write_index_csv(
            filename,
            rows
        )

        print(
            f"{split} completed "
            f"({len(rows)} samples)"
        )



if __name__ == "__main__":

    main()
