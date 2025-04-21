"""Protein structure handling utilities."""

import os
import math
from typing import List, Tuple, Optional


def extract_region_from_pdb(
    pdb_file_path: str,
    x: float,
    y: float,
    z: float,
    radius: float,
    output_file_path: Optional[str] = None
) -> str:
    """
    Extract atoms from a PDB file that are within a specified radius of a 3D point.
    
    Args:
        pdb_file_path: Path to the input PDB file
        x: X-coordinate of the center point
        y: Y-coordinate of the center point
        z: Z-coordinate of the center point
        radius: Radius (in Angstroms) around the center point to include atoms
        output_file_path: Path for the output PDB file. If None, generates a path based on the input
    
    Returns:
        Path to the output PDB file containing only atoms within the specified radius
    """
    if not os.path.exists(pdb_file_path):
        raise FileNotFoundError(f"PDB file not found: {pdb_file_path}")
    
    # Generate output file path if not provided
    if output_file_path is None:
        base_name = os.path.splitext(os.path.basename(pdb_file_path))[0]
        dir_name = os.path.dirname(pdb_file_path)
        output_file_path = os.path.join(dir_name, f"{base_name}_region_r{radius:.1f}.pdb")
    
    # Parse PDB file and filter atoms
    filtered_lines = []
    atom_count = 0
    
    with open(pdb_file_path, 'r') as pdb_file:
        for line in pdb_file:
            # Keep header lines and other non-ATOM/HETATM lines
            if not (line.startswith('ATOM') or line.startswith('HETATM')):
                filtered_lines.append(line)
                continue
            
            # Parse atom coordinates
            try:
                atom_x = float(line[30:38].strip())
                atom_y = float(line[38:46].strip())
                atom_z = float(line[46:54].strip())
            except ValueError:
                # Skip lines with invalid coordinates
                continue
            
            # Calculate distance from center point
            distance = math.sqrt(
                (atom_x - x)**2 + 
                (atom_y - y)**2 + 
                (atom_z - z)**2
            )
            
            # Include atom if within radius
            if distance <= radius:
                atom_count += 1
                # Keep the original atom numbering to maintain connectivity
                filtered_lines.append(line)
    
    # Write filtered atoms to output file
    with open(output_file_path, 'w') as output_file:
        output_file.writelines(filtered_lines)
    
    print(f"Extracted {atom_count} atoms within {radius}Å of point ({x}, {y}, {z})")
    return output_file_path
