import os
os.environ["GDAL_CACHEMAX"] = "256MB"
from itertools import product
import GeoKRL
from config import *
from hyperparameters import *
from modality_sampler import ( compute_country_quota, build_sample_task_rows, write_index_csv )
from dataset_index import build_dataset_index
from raster_manager import RasterManager
from sample_generator_task import *
from batch_generator_task import BatchGenerator
import tensorflow as tf
import numpy as np
import time
import csv
import random
import gc
import task_hyperparameters as hp
import best_pretraining_config as best
import argparse
import train


def generate_hyperparameter_grid():
    """
    Returns list of dict
    """
    search_space = {}
    
    # Automatically discover search-space variables
    for name in dir(hp):
        if not name.isupper():
            continue
        value = getattr(hp, name)
        if isinstance(value, list):
            search_space[name] = value
           
    # Cartesian product
    parameter_names = sorted(search_space.keys())
    parameter_values = [ search_space[name] for name in parameter_names ]
    grid = []
    
    for combination in product(*parameter_values):
        configuration = dict( zip( parameter_names, combination ))
        grid.append(configuration)

    return grid
    
def build_experiment_name(config, task):
    name = (    f"{task}"f"_B{config['BATCH_SIZE']}"
                f"_LR{config['LEARNING_RATE']}"
                f"_WD{config['WEIGHT_DECAY']}"
                f"_PB{config['PREDICTION_HEAD_BLOCKS']}"
                f"_BN{config['PREDICTION_HEAD_BOTTLENECK']}"
                f"_DO{config['PREDICTION_HEAD_DROPOUT']}"
                f"_FB{int(config['FREEZE_BACKBONE'])}"
                f"_TB{config['TRAINING_BUDGET_FACTOR']}"
                f"_E{config['EPOCHS']}"
                f"_L{config['LOSS']}" )
    return name.replace(".", "_")
    
def compute_training_budget(model, factor):
    """
    Returns
    -------
    dict

        {
            "parameters": ...,
            "train": ...,
            "validation": ...,
            "test": ...
        }
    """
    n_parameters = model.count_params()
    total_samples = factor * n_parameters
    total_countries = ( len(TRAIN_COUNTRIES) + len(VALIDATION_COUNTRIES) + len(TEST_COUNTRIES) )
    train_ratio = len(TRAIN_COUNTRIES) / total_countries
    validation_ratio = len(VALIDATION_COUNTRIES) / total_countries
    test_ratio = len(TEST_COUNTRIES) / total_countries
    n_train = int(round(total_samples * train_ratio))
    n_validation = int(round(total_samples * validation_ratio))
    n_test = int(round(total_samples * test_ratio))
    return { "parameters": n_parameters, "train": n_train, "validation": n_validation, "test": n_test }
  
def create_sample_indices( experiment_name, task, n_train, n_validation, n_test ):
    """
    Returns
    -------
    tuple

        (
            train_csv,
            validation_csv,
            test_csv
        )
    """

    split_quotas = { "train": n_train, "validation": n_validation, "test": n_test }
    generated_files = {}
    os.makedirs( TASK_CSV_INDEX_DIR, exist_ok=True )

    for split, split_quota in split_quotas.items():
        filename = os.path.join( TASK_CSV_INDEX_DIR, f"{split}_{experiment_name}.csv")
        generated_files[split] = filename
        if os.path.isfile(filename):
            continue
        print()
        print("=" * 80)
        print(f"Generating {filename}")
        print("=" * 80)
        rows = []
        country_quotas = compute_country_quota( split, split_quota )
        for country, country_quota in country_quotas.items():
            rows.extend(   build_sample_task_rows(split=split, country=country, country_quota=country_quota, task=task)   )

        write_index_csv( filename, rows)
        print(f"{filename} generated.")

    return ( generated_files["train"], generated_files["validation"], generated_files["test"] )
    
def build_generators(train_csv, validation_csv, test_csv, config):
    print("=" * 80)
    print("Building Dataset Pipeline")
    print("=" * 80)
    # ======================================================
    # Dataset
    # ======================================================
    dataset = build_dataset_index()
    # ======================================================
    # Raster manager
    # ======================================================
    raster_manager = RasterManager(dataset)
    # ======================================================
    # Sample generator
    # ======================================================
    sample_generator = SampleGenerator( raster_manager=raster_manager, patch_size=config["PATCH_SIZE"],
                                        patch_size_global=config["PATCH_SIZE_GLOBAL"] )
    # ======================================================
    # Batch generators
    # ======================================================
    train_generator = BatchGenerator( csv_file=train_csv, sample_generator=sample_generator, batch_size=config["BATCH_SIZE"],
                                      shuffle=True, seed=RANDOM_SEED )

    validation_generator = BatchGenerator( csv_file=validation_csv, sample_generator=sample_generator,
                                            batch_size=config["BATCH_SIZE"], shuffle=False )
    test_generator = BatchGenerator( csv_file=test_csv, sample_generator=sample_generator, batch_size=config["BATCH_SIZE"],
                                        shuffle=False )
    print()
    print(f"Train samples      : {train_generator.num_samples:,}")
    print(f"Validation samples : {validation_generator.num_samples:,}")
    print(f"Test samples       : {test_generator.num_samples:,}")
    print()
    print(f"Train batches      : {len(train_generator):,}")
    print(f"Validation batches : {len(validation_generator):,}")
    print(f"Test batches       : {len(test_generator):,}")
    return ( raster_manager, train_generator, validation_generator, test_generator )
    
def build_loss_functions(task):
    if task == TASK_POPULATION:
        nodata = -99999.0
        def population_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            mask = tf.not_equal(y_true, nodata)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.math.log1p(tf.maximum(y_true, 0.0))
            return tf.reduce_mean( tf.square(y_true - tf.squeeze(y_pred)) )

    elif task == TASK_BIOMASS:
        nodata = -1.0
        def biomass_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            mask = tf.not_equal(y_true, nodata)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.math.log1p(tf.maximum(y_true, 0.0))
            return tf.reduce_mean( tf.square(y_true - tf.squeeze(y_pred)) )

    elif task == TASK_BUILDING:
        nodata = 256.0
        def building_loss(y_true, y_pred):
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            mask = tf.not_equal(y_true, nodata)
            y_true = tf.boolean_mask(y_true, mask)
            y_pred = tf.boolean_mask(y_pred, mask)
            y_true = tf.where( y_true >= 255.0, 1.0, 0.0 )
            return tf.reduce_mean( tf.square(y_true - tf.squeeze(y_pred)) )
    else:
        raise ValueError(f"Unknown task: {task}")
    if task == TASK_POPULATION:
        loss = population_loss
    elif task == TASK_BIOMASS:
        loss = biomass_loss
    else:
        loss = building_loss
    return { "regression": loss }
    
def build_optimizer(config):
    optimizer = tf.keras.optimizers.AdamW( learning_rate=config["LEARNING_RATE"], weight_decay=config["WEIGHT_DECAY"] )
    return optimizer
    
def train_one_epoch(model, train_generator, optimizer, losses, config):
    loss_sum = 0.0
    n_batches = len(train_generator)
    epoch_start = time.time()
    for batch_index in range(n_batches):
        if batch_index % 100 == 0:
            gc.collect()
        inputs, targets = train_generator[batch_index]
        xp, xc, xg, gps, mask = inputs
        with tf.GradientTape() as tape:
            prediction = model( [xp, xc, xg, gps], training=True )
            loss = losses["regression"](targets, prediction)
        gradients = tape.gradient( loss, model.trainable_variables )
        optimizer.apply_gradients(     zip( gradients, model.trainable_variables )     )
        loss_sum += float(loss)
        elapsed = time.time() - epoch_start
        avg_loss = loss_sum / (batch_index + 1)
        print(  f"\rTraining Batch {batch_index+1:5d}/{n_batches}"
                f" | Loss {avg_loss:.6f}"
                f" | Elapsed {elapsed:8.1f} s",
                end="", flush=True )
    print()
    history = { "total_loss": loss_sum / n_batches}
    return history
    
def validate_one_epoch( model, validation_generator, losses, config, best_loss):
    total_loss_sum = 0.0
    n_batches = len(validation_generator)
    epoch_start = time.time()
    for batch_index in range(n_batches):
        inputs, targets = validation_generator[batch_index]
        xp, xc, xg, gps, mask = inputs
        prediction = model( [xp, xc, xg, gps], training=False )
        loss = losses["regression"](targets, prediction)
        total_loss_sum += float(loss)
        elapsed = time.time() - epoch_start
        print(
            f"\rValidation Batch {batch_index+1:5d}/{n_batches}"
            f" | Time {elapsed:7.1f}s"
            f" | Total Loss {total_loss_sum/(batch_index+1)}",
            end="" ,
            flush=True )

    avg_total = total_loss_sum / n_batches
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
    history = { "total_loss": avg_total }
    return history, improved, best_loss
 
def save_best_model(model, experiment_name):
    filename = os.path.join( TASK_MODEL_DIR, f"model_{experiment_name}.weights.h5")
    model.save_weights(filename)
    print(f"Best model saved -> {filename}")   

def evaluate_test(model, test_generator, losses, config):
    total_loss_sum = 0.0
    n_batches = len(test_generator)
    start_time = time.time()
    print()
    print("=" * 80)
    print("Testing")
    print("=" * 80)
    for batch_index in range(n_batches):
   
        (inputs, targets) = test_generator[batch_index]
        xp, xc, xg, gps, mask = inputs
        prediction = model( [xp, xc, xg, gps], training=False )
        loss = losses["regression"](targets, prediction)
        total_loss_sum += float(loss)
        elapsed = time.time() - start_time
        print(  f"\rTesting Batch {batch_index+1:5d}/{n_batches}"
                f" | Time {elapsed:7.1f}s"
                f" | Total Loss {total_loss_sum/(batch_index+1):.5f}",
                end="",
                flush=True )
    history = { "total_loss": total_loss_sum / n_batches }
    return history

def save_training_history( experiment_name, model, train_generator, validation_generator, test_generator, config, train_history,
                            validation_history, test_history):
    filename = os.path.join( TASK_RESULT_DIR, f"result_{experiment_name}.txt" )
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
            "Valid_Total\t \n"
        )

        for epoch in range(len(train_history)):

            tr = train_history[epoch]
            va = validation_history[epoch]

            f.write(    f"{epoch+1}\t"
                        f"{tr['total_loss']}\t"
                        f"{va['total_loss']}\n" )
                        
        f.write("\n")
        f.write("=" * 120 + "\n")
        f.write("Final Test\n")
        f.write("=" * 120 + "\n")
        f.write(f"Total Loss          : {test_history['total_loss']}\n")
    print()
    print(f"Training history saved -> {filename}")

def main():
    # ======================================================
    # Reproducibility
    # ======================================================
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    # ======================================================
    # Parse command line
    # ======================================================
    parser = argparse.ArgumentParser()
    parser.add_argument( "--task", required=True, choices=[ TASK_POPULATION, TASK_BIOMASS, TASK_BUILDING ] )
    args = parser.parse_args()
    task = args.task
    summary_file = os.path.join( TASK_LOG_DIR, f"{task}_training_summary.csv" )
    # ======================================================
    # Build pretraining configuration
    # (same configuration that produced the best backbone)
    # ======================================================
    pretraining_config = {      "PATCH_SIZE": best.PATCH_SIZE,
                                "PATCH_SIZE_GLOBAL": best.PATCH_SIZE_GLOBAL,
                                "LATENT_DIM": best.LATENT_DIM,
                                "BATCH_SIZE": best.BATCH_SIZE,
                                "LEARNING_RATE": best.LEARNING_RATE,
                                "RECONSTRUCTION_LOSS_WEIGHT": best.RECONSTRUCTION_LOSS_WEIGHT,
                                "CLASSIFICATION_LOSS_WEIGHT": best.CLASSIFICATION_LOSS_WEIGHT,
                                "MODALITIES_PER_SAMPLE": best.MODALITIES_PER_SAMPLE,
                                "TRAINING_BUDGET_FACTOR": best.TRAINING_BUDGET_FACTOR,
                                "EPOCHS": best.EPOCHS,
                                "KL_LOSS_WEIGHT": best.KL_LOSS_WEIGHT   }
    # ======================================================
    # Recover pretrained experiment name
    # ======================================================
    pretraining_experiment = train.build_experiment_name( pretraining_config)
    pretrained_weights = os.path.join( MODEL_DIR, "model_"+pretraining_experiment + ".weights.h5" )
    print()
    print("Loading Pretrained Backbone")
    print(pretrained_weights)
    # ======================================================
    # Build backbone
    # ======================================================
    backbone = GeoKRL.create_backbone( patch_size=pretraining_config["PATCH_SIZE"], patch_size_global=pretraining_config["PATCH_SIZE_GLOBAL"],
                                        latent_dim=pretraining_config["LATENT_DIM"], bands_context=CONTEXT_BANDS,
                                        bands=PIXEL_BANDS, mask_dim=MASK_DIM )
    # ======================================================
    # Load pretrained weights
    # ======================================================
    backbone.load_weights( pretrained_weights, skip_mismatch=True )
    print("Backbone weights loaded successfully.")
    # ======================================================
    # Task hyperparameters
    # ======================================================
    grid = generate_hyperparameter_grid()

    print()
    print(f"{len(grid)} task configurations")
    # ======================================================
    # Train every configuration
    # ======================================================
    for config in grid:
        print()
        print("=" * 80)
        print("Task Configuration")
        print("=" * 80)

        experiment_name = build_experiment_name( config, task )
        best_loss = float("inf")
        print(experiment_name)
        task_model = GeoKRL.create_task_model( backbone=backbone, task=task, freeze_backbone=config["FREEZE_BACKBONE"],
                                                bottleneck_ratio=config["PREDICTION_HEAD_BOTTLENECK"],
                                                n_blocks=config["PREDICTION_HEAD_BLOCKS"],
                                                dropout_rate=config["PREDICTION_HEAD_DROPOUT"] )

        budget = compute_training_budget(task_model, config["TRAINING_BUDGET_FACTOR"])

        print("parameters=",budget["parameters"])
        print("train=",budget["train"])
        print("validation=",budget["validation"])
        print("test=",budget["test"])
        #
        train_csv, validation_csv, test_csv = create_sample_indices( experiment_name=experiment_name, task=task, 
                                                                     n_train=budget["train"], n_validation=budget["validation"],
                                                                     n_test=budget["test"] )

        ( raster_manager, train_generator, validation_generator, test_generator) = build_generators(
                                                          train_csv=train_csv, validation_csv=validation_csv,
                                                          test_csv=test_csv, config=pretraining_config )
        optimizer = build_optimizer(config)
        losses = build_loss_functions(task)
        epochs = config["EPOCHS"]
        train_history_all = []
        validation_history_all = []
        # ======================================================
        # Training
        # ======================================================
        for epoch in range(epochs):
            print()
            print(f"Epoch {epoch+1}/{epochs}")
            train_history = train_one_epoch(    model=task_model, train_generator=train_generator, optimizer=optimizer,
                                                losses=losses, config=config )
            validation_history, improved, best_loss = validate_one_epoch( model=task_model, validation_generator=validation_generator,
                                                            losses=losses, config=config, best_loss=best_loss )
            if improved:
                save_best_model( task_model, experiment_name )
            train_history_all.append(train_history)
            validation_history_all.append(validation_history)
        # ======================================================
        # Test
        # ======================================================
        test_history = evaluate_test( model=task_model, test_generator=test_generator, losses=losses, config=config )
        # ======================================================
        # Save experiment
        # ======================================================
        save_training_history( experiment_name=experiment_name, model=task_model, train_generator=train_generator,
                                validation_generator=validation_generator, test_generator=test_generator, config=config,
                                train_history=train_history_all, validation_history=validation_history_all,
                                test_history=test_history )
        train_last = train_history_all[-1]
        validation_last = validation_history_all[-1]
        file_exists = os.path.isfile(summary_file)
        with open(summary_file, "a", newline="") as csvfile:
            writer = csv.writer(csvfile, delimiter=";")
            if not file_exists:
                writer.writerow([ "experiment_name", "Train_Loss", "Validation_Loss", "Test_Loss" ])
            writer.writerow([ experiment_name, train_last["total_loss"], validation_last["total_loss"], test_history["total_loss"] ])
        print(f"Summary updated -> {summary_file}")

    print()
    print("=" * 80)
    print("Finished")
    print("=" * 80)
    
    
if __name__ == "__main__":

    main()