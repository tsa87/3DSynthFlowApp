"""Visualization utilities for rendering proteins and molecules."""

from typing import Dict, Any, Optional
import py3Dmol
from stmol import showmol
import streamlit as st
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
import tempfile
import io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def render_protein(
    pdb_file: str, 
    width: int = 700, 
    height: int = 500, 
    spin: bool = True
) -> py3Dmol.view:
    """
    Render a protein structure from a PDB file using py3Dmol.
    
    Args:
        pdb_file: Path to the PDB file
        width: Viewer width in pixels
        height: Viewer height in pixels
        spin: Whether to enable rotation animation
        
    Returns:
        A py3Dmol view object of the rendered protein
    """
    try:
        with open(pdb_file, 'r') as f:
            pdb_data = f.read()
        
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_data, 'pdb')
        view.setStyle({'cartoon': {'color': 'spectrum'}})
        view.zoomTo()
        view.spin(spin)
        return view
    except Exception as e:
        raise ValueError(f"Error rendering protein: {str(e)}")

def display_mol(
    mol: Chem.Mol, 
    width: int = 300, 
    height: int = 200
) -> str:
    """
    Generate an SVG representation of a molecule.
    
    Args:
        mol: RDKit molecule object
        width: Image width in pixels
        height: Image height in pixels
        
    Returns:
        SVG string representation of the molecule
    """
    try:
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        # remove conformers
        mol.RemoveAllConformers()
        mol = Chem.RemoveHs(mol)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg
    except Exception as e:
        raise ValueError(f"Error displaying molecule: {str(e)}")

def mol_to_3dmol(mol, width=400, height=300, style="stick", surface=False, spin=False):
    """
    Convert an RDKit molecule to a py3Dmol view.
    
    Args:
        mol: RDKit molecule
        width: Viewer width
        height: Viewer height
        style: Visualization style ('stick', 'line', 'sphere', 'cartoon')
        surface: Whether to show molecular surface
        spin: Whether to enable spinning
        
    Returns:
        py3Dmol view object
    """
    if mol is None:
        return None
    
    # Ensure 3D coordinates
    if mol.GetNumConformers() == 0:
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol)
        AllChem.MMFFOptimizeMolecule(mol)
    
    # Convert to PDB block
    pdb_block = Chem.MolToPDBBlock(mol)
    
    # Create viewer
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_block, 'pdb')
    
    # Set style
    if style == "stick":
        view.setStyle({'stick': {'radius': 0.2, 'color': 'spectrum'}})
    elif style == "line":
        view.setStyle({'line': {'color': 'spectrum'}})
    elif style == "sphere":
        view.setStyle({'sphere': {'radius': 0.5, 'color': 'spectrum'}})
    elif style == "cartoon":
        view.setStyle({'cartoon': {'color': 'spectrum'}})
    else:
        # Default to stick
        view.setStyle({'stick': {'radius': 0.2, 'color': 'spectrum'}})
    
    # Add surface if requested
    if surface:
        view.addSurface(py3Dmol.VDW, {'opacity': 0.5, 'color': 'white'})
    
    # Center and zoom
    view.zoomTo()
    
    # Enable spinning if requested
    if spin:
        view.spin(True)
    
    return view

def display_mol_3d(mol, width=400, height=300, **kwargs):
    """Display a molecule in 3D in Streamlit."""
    view = mol_to_3dmol(mol, width, height, **kwargs)
    if view:
        return showmol(view, height=height, width=width)
    return None
