"""Molecule handling utilities for the drug discovery platform."""

from typing import List, Dict, Any, Optional, Tuple
import io
import os
import numpy as np
import random
import replicate
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors


def get_ligands_for_protein(
    protein_file: str, 
    n_results: int = 10
) -> List[Chem.Mol]:
    """
    Retrieve potential ligands for a protein using the Replicate API.
    
    Uses a 3D molecule generation model to generate molecules that fit the protein binding site.
    
    Args:
        protein_file: Path to the protein PDB file
        n_results: Maximum number of results to return
        
    Returns:
        List of RDKit molecule objects with properties set
    """
    try:
        # Check if API token is already set in environment
        if not os.environ.get('REPLICATE_API_TOKEN'):
            # Load from environment variable or raise an error in production
            raise EnvironmentError(
                "REPLICATE_API_TOKEN environment variable is not set. "
                "Please set this variable before running in production."
            )
        
        # Call the molecule generation model
        outputs = replicate.run(
            "tsa87/3d-synth-molgen:cecf1ffd1aef0a8d4aee4e47a0a2c9981d3c731c61598ae316b77e72cdbb17f7",
            input={
                "pocket_pdb": open(protein_file, "rb"),
                "seed": random.randint(1, 1000),
                "num_samples": n_results,  # Ensure we get enough samples
            }
        )
        
        # Process the output
        mols = []
        trajs = []
        for i, output in enumerate(outputs):   
            smiles = output["smiles"]
            trajs.append(output['sample_trajectory'])
                    
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # Generate 3D coordinates for the molecule
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol, randomSeed=42)
                AllChem.MMFFOptimizeMolecule(mol)
                
                # Add properties
                mol.SetProp("_Name", f"Compound_{i+1}")
                mol.SetProp("SMILES", smiles)
                mol.SetProp("QED", f"{Descriptors.qed(mol):.2f}")
                mol.SetProp("MW", f"{Descriptors.MolWt(mol):.2f}")
                mol.SetProp("LogP", f"{Descriptors.MolLogP(mol):.2f}")
                            
                mols.append(mol)
        
        return mols, trajs
    except Exception as e:
        raise RuntimeError(f"Error retrieving ligands via Replicate API: {str(e)}")


def download_ligands_as_sdf(ligands: List[Chem.Mol]) -> str:
    """
    Convert a list of molecules to SDF format for download.
    
    Args:
        ligands: List of RDKit molecule objects
        
    Returns:
        String in SDF format containing all molecules
    """
    try:
        sdf_file = io.StringIO()
        writer = Chem.SDWriter(sdf_file)
        for mol in ligands:
            writer.write(mol)
        writer.close()
        return sdf_file.getvalue()
    except Exception as e:
        raise RuntimeError(f"Error converting ligands to SDF: {str(e)}")


def calculate_drug_likeness(mol: Chem.Mol) -> Tuple[Dict[str, float], int]:
    """
    Calculate Lipinski's Rule of Five parameters for a molecule.
    
    Args:
        mol: RDKit molecule object
        
    Returns:
        Tuple of (properties_dict, violation_count)
    """
    try:
        properties = {
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "H_Donors": Descriptors.NumHDonors(mol),
            "H_Acceptors": Descriptors.NumHAcceptors(mol)
        }
        
        violations = 0
        if properties["MW"] > 500: violations += 1
        if properties["LogP"] > 5: violations += 1
        if properties["H_Donors"] > 5: violations += 1
        if properties["H_Acceptors"] > 10: violations += 1
        
        return properties, violations
    except Exception as e:
        raise ValueError(f"Error calculating drug-likeness: {str(e)}") 