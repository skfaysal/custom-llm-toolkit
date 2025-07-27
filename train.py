import json
import torch
import time
import re
from functools import partial
from torch.utils.data import DataLoader
from configs.config import DATA_DIR, BATCH_SIZE, NUM_WORKER, EPOCHS, CHOOSE_MODEL, CHECKPOINT_PATH
from src.models.gpt.download_model_weights import download_and_load_gpt2
from src.models.gpt.config import BASE_CONFIG, model_configs
from src.models.gpt.gpt_model import GPTModel
from src.models.gpt.load_weights_into_model import load_weights_into_gpt
from src.training.trainer import train_model_simple
from src.data.splits import list_data_splitter
from src.data.dataset import InstructionDataset
from src.tokenizer.tokenizer import tokenizer
from src.data.collator import custom_collate_fn
from src.device import set_device
from src.data.preprocessing import format_input
from src.models.gpt.generation import (
    generate,
    text_to_token_ids,
    token_ids_to_text
)
from src.visualizations.visualize import plot_losses

if __name__ == "__main__":

    ###------ Load Dataset --------##
    with open(DATA_DIR / "instruction-data.json", "r", encoding="utf-8") as file:
        data = json.load(file)
    print("Number of entries:", len(data))

    ####------ Split Dataset --------##
    train_data, test_data, val_data = list_data_splitter(data)
    print("Number of training entries:", len(train_data))
    print("Number of testing entries:", len(test_data))
    print("Number of validation entries:", len(val_data))

    ###------- Setup Pytorch Datasets --------##
    DEVICE = set_device()
    customized_collate_fn = partial(
        custom_collate_fn,
        device=DEVICE,
        allowed_max_length=1024
    )

    torch.manual_seed(123)

    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=NUM_WORKER
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKER
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKER
    )

    print("Data loaders created successfully.")
    print("Train loader batch size:", len(train_loader.dataset))
    for inputs, targets in train_loader:
        print(inputs.shape, targets.shape)
        break

    ### ------ Download model weights and initialize model -------- ##

    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(
        model_size=model_size,
        models_dir= CHECKPOINT_PATH / "gpt2"
    )

    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)

    ### ------- Train the model --------- ##
    start_time = time.time()

    torch.manual_seed(123)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, DEVICE,
        num_epochs=EPOCHS, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ### ------- Visualization --------- ##
    epochs_tensor = torch.linspace(0, EPOCHS, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    ### ------- Save the model --------- ##
    file_name = CHECKPOINT_PATH / f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")

    
