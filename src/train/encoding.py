import torch
from supertriplets.utils import move_tensors_to_device
from tqdm import tqdm


def get_triplet_embeddings(dataloader, model, device):
    model.eval()
    embeddings = {"anchors": [], "positives": [], "negatives": []}
    with torch.no_grad():
        for batch in tqdm(dataloader, total=len(dataloader)):
            for input_type in ["anchors", "positives", "negatives"]:
                inputs = {k: v for k, v in batch[input_type].items() if k != "label"}
                inputs = move_tensors_to_device(obj=inputs, device=device)
                batch_embeddings = model(**inputs).cpu()
                embeddings[input_type].append(batch_embeddings)
    embeddings = {k: torch.cat(v, dim=0).numpy() for k, v in embeddings.items()}
    return embeddings
