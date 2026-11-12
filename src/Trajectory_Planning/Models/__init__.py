"""
Liste aller Modelle. Ein neues Modell hier eintragen, dann ist es über
"model": {"type": "<name>"} in der Config auswählbar.
"""

from .Base_Model import BaseModel
from .MLP import MLP

MODELS = {
    "mlp": MLP,
}
