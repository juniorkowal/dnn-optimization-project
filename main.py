import os.path

from constants import DATASET_DIR, MODELS_DIR, SET_SEED
from prepare_dataset import preprocess_all
from training.train_baseline import baseline_training
from training.train_remapped import remapped_training
from optimization import sparsificate, compression
import glob

if __name__ == "__main__":
    SET_SEED(0)

    if not os.path.exists(os.path.join(DATASET_DIR, 'coins_cropped_aug')):
        if not os.path.exists(os.path.join(DATASET_DIR, 'coins_cropped_categorized_by_currency')):
            preprocess_all() # or download all

    if not os.path.exists(os.path.join(MODELS_DIR, 'ResNet50_BASELINE')):
        baseline_training(epochs=100)

    if not os.path.exists(os.path.join(MODELS_DIR, 'ResNet50_REMAPPED_CLASSES')):
        best_model_files = glob.glob(os.path.join(MODELS_DIR, 'ResNet50_BASELINE', '*BEST*'))[0]
        remapped_training(best_model_files, noise_percentage=0.1, epochs=50) # training with remapped classes and noisy labels

    # sparsificate the model
    if not os.path.exists(os.path.join(MODELS_DIR, 'ResNet50_SPARSIFIED')):
        model_files = glob.glob(os.path.join(MODELS_DIR, 'ResNet50_REMAPPED_CLASSES', '*BEST*'))[0]
        sparsificate.prune_model(model_files, amount=0.15)

    if not os.path.exists(os.path.join(MODELS_DIR, 'ResNet50_COMPRESS')):
        model_files = os.path.join(MODELS_DIR, 'ResNet50_SPARSIFIED', 'SPARSE_ResNet50_SPARSIFIED_epochs_50.pth')
        compression.model_compression(model_files)

    # quantize / compress the sparsed models
    # optimize with 'things' e.g. augmented data
    #
    # evaluate all of the models
