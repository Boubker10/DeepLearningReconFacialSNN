from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import traceback
from datetime import datetime
from project.datamodules.fer_dvs import FerDVS
from project.fer_module import FerModule
from project.utils.transforms import DVSTransform
import math
import numpy as np

batch_size = 32
learning_rate = 5e-3
timesteps = 6
epochs = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_dir = "data"
global X


def main():
    dataset, fold_number, mode, trans = get_args()
    transform = DVSTransform(
        sensor_size=FerDVS.sensor_size,
        timesteps=timesteps,
        transforms_list=trans,
        concat_time_channels="cnn" in mode,
    )

    train_set = FerDVS(
        save_to=data_dir,
        dataset=dataset,
        train=True,
        fold=fold_number,
        transform=transform,
    )
    train_workers = 8
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_workers,
        persistent_workers=True
    )
    for batch in train_loader:
        inputs, labels = batch
        initial_input = (inputs.cpu().numpy(), labels.cpu().numpy())
        break  # Sortie après le premier lot
    global X
    X=initial_input[0]
    np.save('X.npy', X)

    # Affichage des données d'entrée et des étiquettes
    print("Exemple d'inputs:", initial_input[0])  # inputs est le premier élément de initial_input
    print("Exemple d'étiquettes:", initial_input[1])  # labels est le deuxième élément de initial_input

    val_set = FerDVS(
        save_to=data_dir,
        dataset=dataset,
        train=False,
        fold=fold_number,
        transform=DVSTransform(
            FerDVS.sensor_size,
            timesteps=timesteps,
            transforms_list=[],
            concat_time_channels="cnn" in mode,
        ),
    )
    val_workers = 8
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=val_workers,persistent_workers=True
    )

    print(f"\n\nEXPERIENCE FOR DATASET={dataset} FOLD={fold_number}")
    print(f"|TRAIN SET|={len(train_set)}")
    print(f"|VAL SET|={len(val_set)}")

    acc = train(train_loader, val_loader, fold_number, dataset, trans, mode=mode, initial_input=initial_input)

    print(f"accuracy obtained for {mode} on {dataset} fold={fold_number}: {acc}")


def get_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="CKPlusDVS", choices=FerDVS.available_datasets
    )
    parser.add_argument("--fold_number", type=int, required=True)
    parser.add_argument(
        "--edas",
        type=str,
        default="flip,background_activity,crop,reverse,mirror,event_drop",
        help="List of employed event data augmentations. They must be separated by commas. Example: 'transform1,transform2,...,transformN'.",
    )
    parser.add_argument("--mode", type=str, choices=["snn", "cnn"], default="snn")
    args = parser.parse_args()

    dataset = args.dataset
    fold_number = args.fold_number
    mode = args.mode

    allowed_transforms = (
        "background_activity",
        "flip_polarity",
        "crop",
        "event_drop",
        "reverse",
        "mirror",
        "flip",
    )

    edas = args.edas.split(",")
    for eda in edas:
        if eda not in allowed_transforms:
            raise ValueError(
                f"edas arguments must contain only transforms in the following list: {allowed_transforms}. Got: {eda}."
            )

    trans = edas

    return dataset, fold_number, mode, trans


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_number: int,
    dataset: str,
    trans: list,
    mode="snn",
    initial_input=None
):
    print("Début de la fonction train")  # Pour vérifier que cette partie est atteinte

    if initial_input is not None:
        inputs, labels = initial_input
        print(f"Premier lot d'inputs: {inputs.shape}, Premières étiquettes: {labels.shape}")


    
    module = FerModule(
        learning_rate=learning_rate,
        timesteps=timesteps,
        n_classes=6,
        epochs=epochs,
        mode=mode,
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",  # TODO: select the logged metric to monitor the checkpoint saving
        filename=str(fold_number) + "_{epoch:03d}_{val_acc:.4f}",
        save_top_k=1,
        mode="max",
    )

    # create trainer
    trainer = pl.Trainer(
        max_epochs=epochs,
        gpus=1 if torch.cuda.is_available() else None,  # Indique à PyTorch Lightning d'utiliser les GPUs
        callbacks=[
            checkpoint_callback, 
        ],
        logger=pl.loggers.TensorBoardLogger(
            "experiments/", name=f"{dataset}_{fold_number}"
        ),
        log_every_n_steps=5,
        default_root_dir=f"experiments/{dataset}",
        # precision=16,
    )

    try:
        trainer.fit(module, train_loader, val_loader)
    except:
        mess = traceback.format_exc()
        report = open("errors.txt", "a")
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        report.write(f"{dt_string} ===> {mess}\n=========\n\n")
        report.flush()
        report.close()
        return -1

    report = open(f"report_{mode}_{dataset}.txt", "a")
    now = datetime.now()
    dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
    report.write(
        f"{dt_string} MODE={mode} DATASET={dataset} FOLD={fold_number} ACC={checkpoint_callback.best_model_score} TRANS={trans}\n"
    )
    report.flush()
    report.close()
    return checkpoint_callback.best_model_score


if __name__ == "__main__":
    main()
