import schnetpack.properties as structure
from schnetpack.transform.base import Transform
from typing import Dict, Optional

import torch

class ExclusionTransformer(Transform):
    is_preprocessor: bool = True
    is_postprocessor: bool = True   

    def __init__(self, excluded_atoms:list=None):
        super().__init__()
        self.excluded_atoms = excluded_atoms

    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for excluded_atom in self.excluded_atoms:
            if excluded_atom:
                if excluded_atom in inputs[structure.Z]:
                    # remove the molecule
                    inputs[structure.Z][inputs[structure.Z] == excluded_atom] = 0  

        return inputs