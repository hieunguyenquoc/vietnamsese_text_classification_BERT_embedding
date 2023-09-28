from sentence_transformers import SentenceTransformer
import torch
import os

def load_pretrained_embedding():
  if torch.cuda.is_available():
      device = torch.device("cuda")
  else:
      device = torch.device("cpu")
  model_path = os.path.abspath("sup-SimCSE-VietNamese-phobert-base")
  model = SentenceTransformer(model_path, device=device)

  return model
