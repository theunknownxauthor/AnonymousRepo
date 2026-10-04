import os
os.environ["GDAL_CACHEMAX"] = "256MB"
from itertools import product
import hyperparameters as hp
import GeoKRL
from config import *
from hyperparameters import *


from modality_sampler import ( compute_country_quota, build_sample_rows, write_index_csv )
from dataset_index import build_dataset_index
from raster_manager import RasterManager
from sample_generator import SampleGenerator
from batch_generator import BatchGenerator
import tensorflow as tf
import numpy as np
import time
import csv
import random
import gc



def generate_hyperparameter_grid():
    """
    Returns-------list of dict
    """

    search_space = {}

    # Automatically discover all search-space variables
    for name in dir(hp):
        if not name.isupper():
            continue
        value = getattr(hp, name)
        if isinstance(value, list):
            search_space[name] = value

    parameter_names = sorted(search_space.keys())
    parameter_values = [ search_space[name] for name in parameter_names]
    grid = []

    for combination in product(*parameter_values):
        configuration = dict( zip(parameter_names, combination) )
        grid.append(configuration)

    return grid

def build_experiment_name(config):
    """
    Parameters
    ----------
    config : dict

    Returns
    -------
    str
    """
    name = ( 
        f"P{config['PATCH_SIZE']}"
        f"_PG{config['PATCH_SIZE_GLOBAL']}"
        f"_Z{config['LATENT_DIM']}"
        f"_B{config['BATCH_SIZE']}"
        f"_LR{config['LEARNING_RATE']}"
        f"_RW{config['RECONSTRUCTION_LOSS_WEIGHT']}"
        f"_CW{config['CLASSIFICATION_LOSS_WEIGHT']}"
        f"_M{config['MODALITIES_PER_SAMPLE']}"
        f"_TB{config['TRAINING_BUDGET_FACTOR']}"
        f"_E{config['EPOCHS']}"
        f"_KLW{config['KL_LOSS_WEIGHT']}"   
        )


    return name.replace(".", "_")
    
def compute_training_budget(model, factor):
    """
    Returns
    -------
    dict

        {
            "train": ...,
            "validation": ...,
            "test": ...
        }
    """

    n_parameters = model.count_params()
    N = factor * n_parameters
    total_countries =  len(TRAIN_COUNTRIES) + len(VALIDATION_COUNTRIES) + len(TEST_COUNTRIES)

    validation_ratio = len(VALIDATION_COUNTRIES) / total_countries
    test_ratio = len(TEST_COUNTRIES) / total_countries
    train_ratio = len(TRAIN_COUNTRIES) / total_countries

    n_train = int(round(N * train_ratio))
    n_validation = int(round(N * validation_ratio))
    n_test = int(round(N * test_ratio))

    return { "parameters": n_parameters, "train": n_train, "validation": n_validation, "test": n_test}

def create_sample_indices( experiment_name, n_train, n_validation, n_test ):
    """
    Returns
    -------
    tuple
        (train_csv, validation_csv, test_csv)
    """

    split_quotas = { "train": n_train, "validation": n_validation, "test": n_test }
    generated_files = {}
    for split, split_quota in split_quotas.items():
        filename = os.path.join( CSV_INDEX_DIR, f"{split}_{experiment_name}.csv" )
        generated_files[split] = filename
        if os.path.isfile(filename):
            continue
        print()
        print("=" * 70)
        print(f"Generating {filename}")
        print("=" * 70)
        rows = []
        country_quotas = compute_country_quota( split, split_quota )
        for country, country_quota in country_quotas.items():
            rows.extend( build_sample_rows( split=split,
                                            country=country,
                                            country_quota=country_quota) )

        write_index_csv( filename, rows )
        print(f"{filename} generated.")

    return ( generated_files["train"], generated_files["validation"], generated_files["test"] )
    
def build_generators( train_csv, validation_csv, test_csv, config ):
    """
    Returns
    -------
    tuple
        (
            raster_manager,
            train_generator,
            validation_generator,
            test_generator
        )
    """
    print()
    print("=" * 80)
    print("Building Dataset Pipeline")
    print("=" * 80)
    
    dataset = build_dataset_index()
    raster_manager = RasterManager(dataset)
    sample_generator = SampleGenerator( raster_manager=raster_manager, patch_size=config["PATCH_SIZE"],
        patch_size_global=config["PATCH_SIZE_GLOBAL"] )
    train_generator = BatchGenerator( csv_file=train_csv, sample_generator=sample_generator, batch_size=config["BATCH_SIZE"],
        shuffle=True, seed=RANDOM_SEED)
    validation_generator = BatchGenerator( csv_file=validation_csv, sample_generator=sample_generator, 
        batch_size=config["BATCH_SIZE"], shuffle=False )
    test_generator = BatchGenerator( csv_file=test_csv, sample_generator=sample_generator, batch_size=config["BATCH_SIZE"],
        shuffle=False )
        
    print()
    print("Train samples      :", train_generator.num_samples)
    print("Validation samples :", validation_generator.num_samples)
    print("Test samples       :", test_generator.num_samples)

    print()
    print("Train batches      :", len(train_generator))
    print("Validation batches :", len(validation_generator))
    print("Test batches       :", len(test_generator))

    return ( raster_manager, train_generator, validation_generator, test_generator )

def build_loss_functions(config):
    classification_loss = tf.keras.losses.SparseCategoricalCrossentropy()
    reconstruction_loss = tf.keras.losses.MeanSquaredError()
    def kl_loss(z_mean, z_log_var):
        return -0.5 * tf.reduce_mean( 1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var) )

    return { "classification": classification_loss, "reconstruction": reconstruction_loss, "kl": kl_loss }
    
def build_optimizer(config):
    learning_rate = config["LEARNING_RATE"]
    optimizer_name = config.get("OPTIMIZER", "adam").lower()
    if optimizer_name == "adam":
        optimizer = tf.keras.optimizers.Adam( learning_rate=learning_rate )
    elif optimizer_name == "adamw":
        optimizer = tf.keras.optimizers.AdamW( learning_rate=learning_rate )
    elif optimizer_name == "sgd":

        optimizer = tf.keras.optimizers.SGD( learning_rate=learning_rate, momentum=0.9 )
    else:
        raise ValueError( f"Unknown optimizer: {optimizer_name}" )

    return optimizer
    
def train_one_epoch(model, train_generator, optimizer, losses, config):
    """
    Train the model for one epoch.

    Returns
    -------
    dict
        {
            "total_loss": ...,
            "classification_loss": ...,
            "reconstruction_loss": ...,
            "kl_loss": ...
        }
    """
    total_loss_sum = 0.0
    classification_sum = 0.0
    reconstruction_sum = 0.0
    kl_sum = 0.0
    n_batches = len(train_generator)
    epoch_start = time.time()
    for batch_index in range(n_batches):
        if batch_index % 100 == 0:
            gc.collect()
    
        batch_start = time.time()
        (inputs, targets) = train_generator[batch_index]
        xp, xc, xg, mask = inputs 
        task_true, reconstruction_true = targets
        with tf.GradientTape() as tape:
            # ( task_pred, reconstruction_pred, z_mean, z_log_var ) = model( [xp, xc, xg, mask], training=True )
            ( task_pred, reconstruction_pred, z_mean, z_log_var ) = model( [xp, xc, xg], training=True )

            classification_loss = losses["classification"]( task_true, task_pred )
            reconstruction_loss = losses["reconstruction"]( reconstruction_true, reconstruction_pred )
            kl_loss = losses["kl"]( z_mean, z_log_var )
            total_loss = (  config["CLASSIFICATION_LOSS_WEIGHT"] * classification_loss
                            + config["RECONSTRUCTION_LOSS_WEIGHT"]* reconstruction_loss
                            + config["KL_LOSS_WEIGHT"]* kl_loss )

        gradients = tape.gradient( total_loss, model.trainable_variables)
        optimizer.apply_gradients( zip( gradients, model.trainable_variables) )
        total_loss_sum += float(total_loss)
        classification_sum += float(classification_loss)
        reconstruction_sum += float(reconstruction_loss)
        kl_sum += float(kl_loss)
        elapsed = time.time() - epoch_start
        avg_total = total_loss_sum / (batch_index + 1)
        avg_cls = classification_sum / (batch_index + 1)
        avg_rec = reconstruction_sum / (batch_index + 1)
        avg_kl = kl_sum / (batch_index + 1)
        print(  f"\rTraining Batch {batch_index+1:5d}/{n_batches} | "
                f"Loss {avg_total} | "
                f"Cls {avg_cls} | "
                f"Rec {avg_rec} | "
                f"KL {avg_kl} | "
                f"Elapsed {elapsed:8.1f} s",
                end="",
                flush=True            )
    print()
    history = {     "total_loss": total_loss_sum / n_batches,
                    "classification_loss": classification_sum / n_batches,
                    "reconstruction_loss": reconstruction_sum / n_batches,
                    "kl_loss": kl_sum / n_batches    }      
    return history
   
def validate_one_epoch( model, validation_generator, losses, config, best_loss):
    """
    Returns
    -------
    history : dict

    improved : bool

    best_loss : float
    """

    total_loss_sum = 0.0
    classification_sum = 0.0
    reconstruction_sum = 0.0
    kl_sum = 0.0
    n_batches = len(validation_generator)
    epoch_start = time.time()
    for batch_index in range(n_batches):
        inputs, targets = validation_generator[batch_index]
        xp, xc, xg, mask = inputs
        task_true, reconstruction_true = targets
        ( task_pred, reconstruction_pred, z_mean, z_log_var ) = model( [xp, xc, xg], training=False )
        # ( task_pred, reconstruction_pred, z_mean, z_log_var ) = model( [xp, xc, xg, mask], training=False )
        classification_loss = losses["classification"]( task_true, task_pred )
        reconstruction_loss = losses["reconstruction"]( reconstruction_true, reconstruction_pred )
        kl_loss = losses["kl"]( z_mean, z_log_var )
        total_loss = (  config["CLASSIFICATION_LOSS_WEIGHT"] * classification_loss +
                        config["RECONSTRUCTION_LOSS_WEIGHT"] * reconstruction_loss +
                        config["KL_LOSS_WEIGHT"] * kl_loss )

        total_loss_sum += float(total_loss)
        classification_sum += float(classification_loss)
        reconstruction_sum += float(reconstruction_loss)
        kl_sum += float(kl_loss)

        elapsed = time.time() - epoch_start
        
        print(
            f"\rValidation Batch {batch_index+1:5d}/{n_batches}"
            f" | Time {elapsed:7.1f}s"
            f" | Cls {classification_sum/(batch_index+1)}"
            f" | Rec {reconstruction_sum/(batch_index+1)}"
            f" | KL {kl_sum/(batch_index+1)}"
            f" | Total Loss {total_loss_sum/(batch_index+1)}",
            end="" ,
            flush=True )

    avg_total = total_loss_sum / n_batches
    avg_classification = classification_sum / n_batches
    avg_reconstruction = reconstruction_sum / n_batches
    avg_kl = kl_sum / n_batches
    epoch_time = time.time() - epoch_start
        
    previous_best = best_loss
    improved = avg_total < best_loss
    if improved:
        best_loss = avg_total

    if improved: print( f"=> Validation completed"
                        f" | Improved : {previous_best:.6f} -> {best_loss:.6f}" )
    else:
        print(  f"=> Validation completed"
                f" | No improvement : still {best_loss:.6f}"  )
        


    history = {
                "total_loss": avg_total,
                "classification_loss": avg_classification,
                "reconstruction_loss": avg_reconstruction,
                "kl_loss": avg_kl }
    return history, improved, best_loss
    
def save_best_model(model, experiment_name):
    filename = os.path.join( MODEL_DIR, f"model_{experiment_name}.weights.h5")
    model.save_weights(filename)
    print(f"Best model saved -> {filename}")   

def evaluate_test(model, test_generator, losses, config):
    total_loss_sum = 0.0
    classification_sum = 0.0
    reconstruction_sum = 0.0
    kl_sum = 0.0
    n_batches = len(test_generator)
    start_time = time.time()
    print()
    print("=" * 80)
    print("Testing")
    print("=" * 80)
    for batch_index in range(n_batches):
        (inputs, targets) = test_generator[batch_index]
        xp, xc, xg, mask = inputs
        task_true, reconstruction_true = targets
        # (task_pred, reconstruction_pred, z_mean, z_log_var ) = model( [xp, xc, xg, mask], training=False )
        (task_pred, reconstruction_pred, z_mean, z_log_var ) = model( [xp, xc, xg], training=False )
        classification_loss = losses["classification"]( task_true, task_pred )
        reconstruction_loss = losses["reconstruction"]( reconstruction_true, reconstruction_pred )
        kl_loss = losses["kl"]( z_mean, z_log_var )
        total_loss = (  config["CLASSIFICATION_LOSS_WEIGHT"] * classification_loss+
                        config["RECONSTRUCTION_LOSS_WEIGHT"] * reconstruction_loss+ config["KL_LOSS_WEIGHT"] * kl_loss )
        total_loss_sum += float(total_loss)
        classification_sum += float(classification_loss)
        reconstruction_sum += float(reconstruction_loss)
        kl_sum += float(kl_loss)
        elapsed = time.time() - start_time

        print(  f"\rTesting Batch {batch_index+1:5d}/{n_batches}"
                f" | Time {elapsed:7.1f}s"
                f" | Cls {classification_sum/(batch_index+1):.5f}"
                f" | Rec {reconstruction_sum/(batch_index+1):.5f}"
                f" | KL {kl_sum/(batch_index+1):.5f}"
                f" | Total Loss {total_loss_sum/(batch_index+1):.5f}",
                end="",
                flush=True )
    history = { "total_loss": total_loss_sum / n_batches,
                "classification_loss": classification_sum / n_batches,
                "reconstruction_loss": reconstruction_sum / n_batches,
                "kl_loss": kl_sum / n_batches }
    return history

def save_training_history( experiment_name, model, train_generator, validation_generator, test_generator, config, train_history,
                            validation_history, test_history):
    filename = os.path.join( RESULT_DIR, f"result_{experiment_name}.txt" )
    with open(filename, "w") as f:

        f.write("=" * 120 + "\n")
        f.write("Experiment\n")
        f.write("=" * 120 + "\n")
        f.write(f"Name : {experiment_name}\n\n")
        f.write("=" * 120 + "\n")
        f.write("Hyperparameters\n")
        f.write("=" * 120 + "\n")
        for key in sorted(config.keys()):
            f.write(f"{key:35s}: {config[key]}\n")
            
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("Model\n")
        f.write("=" * 120 + "\n")
        f.write(f"Parameters : {model.count_params():,}\n")
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("Dataset\n")
        f.write("=" * 120 + "\n")
        f.write(f"Train samples      : {train_generator.num_samples:,}\n")
        f.write(f"Validation samples : {validation_generator.num_samples:,}\n")
        f.write(f"Test samples       : {test_generator.num_samples:,}\n")
        f.write("\n")
        f.write(f"Train batches      : {len(train_generator):,}\n")
        f.write(f"Validation batches : {len(validation_generator):,}\n")
        f.write(f"Test batches       : {len(test_generator):,}\n")

        # ==================================================
        # Epoch history
        # ==================================================

        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("Training History\n")
        f.write("=" * 120 + "\n\n")

        f.write(
            "Epoch\t"
            "Train_Total\t"
            "Train_Cls\t"
            "Train_Rec\t"
            "Train_KL\t"
            "Valid_Total\t"
            "Valid_Cls\t"
            "Valid_Rec\t"
            "Valid_KL\n"
        )

        for epoch in range(len(train_history)):

            tr = train_history[epoch]
            va = validation_history[epoch]

            f.write(    f"{epoch+1}\t"
                        f"{tr['total_loss']}\t"
                        f"{tr['classification_loss']}\t"
                        f"{tr['reconstruction_loss']}\t"
                        f"{tr['kl_loss']}\t"

                        f"{va['total_loss']}\t"
                        f"{va['classification_loss']}\t"
                        f"{va['reconstruction_loss']}\t"
                        f"{va['kl_loss']}\n" )
                        
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("Final Test\n")
        f.write("=" * 120 + "\n")
        f.write(f"Total Loss          : {test_history['total_loss']}\n")
        f.write(f"Classification Loss : {test_history['classification_loss']}\n")
        f.write(f"Reconstruction Loss : {test_history['reconstruction_loss']}\n")
        f.write(f"KL Loss             : {test_history['kl_loss']}\n")
    print()
    print(f"Training history saved -> {filename}")

def main():

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    summary_file = os.path.join(LOG_DIR, "training_summary.csv")

    # Create summary file once
    if not os.path.isfile(summary_file):

        with open(summary_file, "w", newline="") as csvfile:

            writer = csv.writer(csvfile, delimiter=";")

            writer.writerow([

                "experiment_name",

                "Train_Total",
                "Train_Cls",
                "Train_Rec",
                "Train_KL",

                "Valid_Total",
                "Valid_Cls",
                "Valid_Rec",
                "Valid_KL",

                "Test_Total",
                "Test_Cls",
                "Test_Rec",
                "Test_KL"

            ])

    grid = generate_hyperparameter_grid()

    print()
    print("=" * 80)
    print(f"{len(grid):,} configurations to evaluate")
    print("=" * 80)

    # ======================================================
    # Iterate over all configurations
    # ======================================================
    # ======================================================
    # Read already evaluated configurations
    # ======================================================
    completed_experiments = set()
    if os.path.isfile(summary_file):
        with open(summary_file, "r", newline="") as csvfile:
            reader = csv.DictReader(csvfile, delimiter=";")
            for row in reader:
                completed_experiments.add(row["experiment_name"])
                
    print()
    print(f"{len(completed_experiments):,} completed experiments found.")
    
    
    for configuration_id, config in enumerate(grid, start=1):

        print()
        print("=" * 80)
        print(f"Configuration {configuration_id}/{len(grid)}")
        print("=" * 80)
        experiment_name = build_experiment_name(config)
        print(experiment_name)
        # --------------------------------------------------
        # Skip already evaluated configuration
        # --------------------------------------------------
        if experiment_name in completed_experiments:
            print(f"Skipping {experiment_name} (already completed)")
            continue

        best_loss = float("inf")

        epochs = 10

        # --------------------------------------------------
        # Build model
        # --------------------------------------------------

        encoder, decoder, backbone, pretraining_model = GeoKRL.create_GeoKRL(

            patch_size=config["PATCH_SIZE"],
            patch_size_global=config["PATCH_SIZE_GLOBAL"],
            latent_dim=config["LATENT_DIM"],
            bands_context=CONTEXT_BANDS,
            bands=PIXEL_BANDS,
            mask_dim=MASK_DIM

        )

        # --------------------------------------------------
        # Training budget
        # --------------------------------------------------

        budget = compute_training_budget(

            pretraining_model,
            config["TRAINING_BUDGET_FACTOR"]

        )

        # --------------------------------------------------
        # Create CSV indices
        # --------------------------------------------------

        train_csv, validation_csv, test_csv = create_sample_indices(

            experiment_name,
            budget["train"],
            budget["validation"],
            budget["test"]

        )

        # --------------------------------------------------
        # Generators
        # --------------------------------------------------

        raster_manager, train_generator, validation_generator, test_generator = build_generators(

            train_csv,
            validation_csv,
            test_csv,
            config

        )

        # --------------------------------------------------
        # Optimizer & losses
        # --------------------------------------------------

        optimizer = build_optimizer(config)

        losses = build_loss_functions(config)

        train_history_all = []

        validation_history_all = []

        # --------------------------------------------------
        # Training loop
        # --------------------------------------------------
        for epoch in range(epochs):
            print()
            print(f"Epoch {epoch+1}/{epochs}")

            train_history = train_one_epoch(

                model=pretraining_model,
                train_generator=train_generator,
                optimizer=optimizer,
                losses=losses,
                config=config

            )

            validation_history, improved, best_loss = validate_one_epoch(

                model=pretraining_model,
                validation_generator=validation_generator,
                losses=losses,
                config=config,
                best_loss=best_loss

            )

            if improved:

                save_best_model(

                    pretraining_model,
                    experiment_name

                )
                

            train_history_all.append(train_history)

            validation_history_all.append(validation_history)

        # --------------------------------------------------
        # Final test
        # --------------------------------------------------

        test_history = evaluate_test(

            model=pretraining_model,
            test_generator=test_generator,
            losses=losses,
            config=config

        )

        # --------------------------------------------------
        # Save history
        # --------------------------------------------------

        save_training_history(

            experiment_name=experiment_name,
            model=pretraining_model,
            train_generator=train_generator,
            validation_generator=validation_generator,
            test_generator=test_generator,
            config=config,
            train_history=train_history_all,
            validation_history=validation_history_all,
            test_history=test_history

        )

        # --------------------------------------------------
        # Update summary CSV
        # --------------------------------------------------

        train_last = train_history_all[-1]

        validation_last = validation_history_all[-1]

        with open(summary_file, "a", newline="") as csvfile:

            writer = csv.writer(csvfile, delimiter=";")

            writer.writerow([

                experiment_name,

                train_last["total_loss"],
                train_last["classification_loss"],
                train_last["reconstruction_loss"],
                train_last["kl_loss"],

                validation_last["total_loss"],
                validation_last["classification_loss"],
                validation_last["reconstruction_loss"],
                validation_last["kl_loss"],

                test_history["total_loss"],
                test_history["classification_loss"],
                test_history["reconstruction_loss"],
                test_history["kl_loss"]

            ])

        print(f"Summary updated -> {summary_file}")

        # --------------------------------------------------
        # Release memory
        # --------------------------------------------------

        raster_manager.close()

        del encoder
        del decoder
        del backbone
        del pretraining_model

        del train_generator
        del validation_generator
        del test_generator

        tf.keras.backend.clear_session()

    print()
    print("=" * 80)
    print("All configurations completed.")
    print("=" * 80)
    
if __name__ == "__main__":

    main()