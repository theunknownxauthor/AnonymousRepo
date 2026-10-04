"""
modality_sampler.py

Version 3

L1:
- Balanced ROI sampling.
- Balanced modality combinations inside every ROI.
"""

import random
from itertools import combinations

from config import TASKS
from config import TASK_MODALITIES
from config import ANCILLARY_LAYERS
from config import SELECTED_MODALITIES
from config import ROI_NAMES


class ModalitySampler:

    def __init__(self):

        self.task_modalities = TASK_MODALITIES
        self.all_modalities = ANCILLARY_LAYERS
        self.selected_modalities = SELECTED_MODALITIES

        self.total_samples = 0

        ####################################################
        # Global modality statistics
        ####################################################

        self.modality_counter = {
            m: 0 for m in self.all_modalities
        }

        ####################################################
        # ROI balancing
        ####################################################

        self.roi_counter = {}

        ####################################################
        # Combination balancing
        ####################################################

        self.combinations = {}

        self.combination_counter = {}

        for task in TASKS:

            combs = list(
                combinations(
                    self.task_modalities[task],
                    self.selected_modalities
                )
            )

            self.combinations[task] = combs

            self.roi_counter[task] = {}

            self.combination_counter[task] = {}

            for roi in ROI_NAMES:

                self.roi_counter[task][roi] = 0

                self.combination_counter[task][roi] = {
                    c: 0 for c in combs
                }

    ########################################################

    def sample(self, task, country=None):

        if task not in TASKS:
            raise ValueError(f"Unknown task: {task}")

        ####################################################
        # -------- L1 : ROI balancing --------
        ####################################################

        roi_counts = self.roi_counter[task]

        minimum = min(roi_counts.values())

        candidate_rois = []

        for roi, count in roi_counts.items():

            if count == minimum:

                candidate_rois.append(roi)

        roi = random.choice(candidate_rois)

        self.roi_counter[task][roi] += 1

        ####################################################
        # -------- L0 : Combination balancing --------
        ####################################################

        counters = self.combination_counter[task][roi]

        minimum = min(counters.values())

        candidate_combinations = []

        for comb, count in counters.items():

            if count == minimum:

                candidate_combinations.append(comb)

        selected = random.choice(candidate_combinations)

        counters[selected] += 1

        ####################################################
        # Statistics
        ####################################################

        self.total_samples += 1

        for modality in selected:

            self.modality_counter[modality] += 1

        mask = self.create_mask(selected)

        return roi, list(selected), mask

    ########################################################

    def create_mask(self, selected_modalities):

        mask = []

        for modality in self.all_modalities:

            if modality in selected_modalities:

                mask.append(1)

            else:

                mask.append(0)

        return mask

    ########################################################

    def print_statistics(self):

        print("\n==============================")
        print("Global Statistics")
        print("==============================")

        print("Total samples:", self.total_samples)

        print("\nModality usage")

        for modality in self.all_modalities:

            print(f"{modality:12s}: {self.modality_counter[modality]}")

        print("\n==============================")
        print("ROI Usage")
        print("==============================")

        for task in TASKS:

            print(f"\n{task}")

            for roi in ROI_NAMES:

                print(
                    f"{roi:8s}: {self.roi_counter[task][roi]}"
                )

        print("\n==============================")
        print("Combination Usage")
        print("==============================")

        for task in TASKS:

            print(f"\nTask : {task}")

            for roi in ROI_NAMES:

                print(f"\n   {roi}")

                counters = self.combination_counter[task][roi]

                for comb in sorted(counters.keys()):

                    print(
                        f"      {comb} : {counters[comb]}"
                    )


####################################################################

def main():

    sampler = ModalitySampler()

    print("\n==============================")
    print("Balanced Sampling")
    print("==============================")

    for task in TASKS:

        print("\n--------------------------------")
        print("Task:", task)
        print("--------------------------------")

        for _ in range(24):

            roi, selected, mask = sampler.sample(task)

            print(
                f"{roi:6s}",
                selected,
                mask
            )

    sampler.print_statistics()


if __name__ == "__main__":

    main()