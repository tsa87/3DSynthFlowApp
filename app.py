"""
Molecule Generation Platform

A Streamlit application for drug discovery through molecule generation.
For research purposes only - not for clinical use.
"""


import streamlit as st
import os
import logging
import sys
import tempfile
import requests
from rdkit import Chem
import numpy as np  # For centroid calculation


from utils.molecule_utils import get_ligands_for_protein
from utils.protein_utils import extract_region_from_pdb
from utils.consts import ADRB_CACHED_RESPONSE, ADRB_CENTER, ADRB_POCKET_PDB_FILE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)




def initialize_app():
    """Configure the Streamlit application."""
    st.set_page_config(
        page_title="3D Synthesis-based Molecule Generation", 
        layout="wide", 
        initial_sidebar_state="collapsed",
        menu_items={
            'About': "Structure-based drug design Platform | For research purposes only"
        }
    )
    
    # Initialize session state variables if they don't exist
    if 'rotate_molecules' not in st.session_state:
        st.session_state.rotate_molecules = True


def generate_molecules(center_x, center_y, center_z, num_molecules, protein_file=None):
    """Generate molecules based on the given parameters.
    
    Args:
        center_x: X coordinate of binding site center
        center_y: Y coordinate of binding site center
        center_z: Z coordinate of binding site center
        num_molecules: Number of molecules to generate
        protein_file: Optional path to protein PDB file
    
    Returns:
        List of generated molecules
    """
    if protein_file is None:
        st.error("Please upload a protein file first.")
        return []
    
    try:
        # In a real app, this would call a molecule generation model
        # For now, we'll use the existing function that returns mock data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as temp_file:
            extract_region_from_pdb(protein_file, center_x, center_y, center_z, 20, temp_file.name)
            
            molecules, trajectories = get_ligands_for_protein(
                protein_file=temp_file.name,
                n_results=num_molecules
            )
        return molecules, trajectories
    except Exception as e:
        logger.error(f"Molecule generation error: {e}", exc_info=True)
        st.error(f"Failed to generate molecules: {str(e)}")
        return []


def display_molecules(molecules, trajectories):
    """Display the generated molecules.
    
    Args:
        molecules: List of molecule objects
    """
    import pandas as pd
    from utils.visualization_utils import display_mol, display_mol_3d
    
    if not molecules:
        st.warning("No molecules generated.")
        return
        
    st.header(f"Generated {len(molecules)} Molecules")
    
    # Create a DataFrame for easier display
    molecule_data = []
    for mol in molecules:
        try:
            molecule_data.append({
                "Name": mol.GetProp("_Name"),
                "Structure": mol,
                "SMILES": mol.GetProp("SMILES"),
                "QED": float(mol.GetProp("QED")),
                "MW": float(mol.GetProp("MW")),
                "LogP": float(mol.GetProp("LogP"))
            })
        except Exception as e:
            st.warning(f"Error processing molecule: {str(e)}")
    
    df = pd.DataFrame(molecule_data)
    
    # Arrange molecules in two columns
    # Create pairs of molecules (handle odd number if needed)
    num_molecules = len(molecule_data)
    pairs = []
    for i in range(0, num_molecules, 2):
        if i + 1 < num_molecules:
            pairs.append((i, i + 1))  # Pair of two molecules
        else:
            pairs.append((i, None))   # Last unpaired molecule
    
    # Display molecules in pairs
    for pair in pairs:
        left_idx, right_idx = pair
        left_col, right_col = st.columns(2)
        
        # Process left molecule
        with left_col:
            if left_idx is not None:
                _display_single_molecule(df.iloc[left_idx], trajectories[left_idx])
        
        # Process right molecule
        with right_col:
            if right_idx is not None:
                _display_single_molecule(df.iloc[right_idx], trajectories[right_idx])
    
    # Add download button for all molecules
    if molecules:
        st.divider()
        try:
            from utils.molecule_utils import download_ligands_as_sdf
            sdf_data = download_ligands_as_sdf(molecules)
            st.download_button(
                label="Download All Molecules (SDF)",
                data=sdf_data,
                file_name="generated_molecules.sdf",
                mime="chemical/x-mdl-sdfile"
            )
        except Exception as e:
            st.error(f"Error creating download: {str(e)}")


def _display_single_molecule(row, traj):
    """Helper function to display a single molecule with its trajectory.
    
    Args:
        row: DataFrame row with molecule data
        traj: Trajectory data for the molecule
    """
    from utils.visualization_utils import display_mol, display_mol_3d
    from rdkit import Chem
    
    st.divider()
    
    # Create a 2-column layout for 2D and 3D structures
    structure_col1, structure_col2 = st.columns(2)
    
    # 2D Structure
    with structure_col1:
        st.markdown("##### 2D Structure")
        try:
            svg = display_mol(row["Structure"], width=200, height=200)
            st.markdown(f"<div>{svg}</div>", unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Error displaying 2D structure: {str(e)}")
    
    # 3D Structure using ETKG pose
    with structure_col2:
        st.markdown("##### 3D Structure")
        try:
            mol = row["Structure"]
            # Ensure molecule has 3D coordinates
            if mol.GetNumConformers() == 0:
                from rdkit.Chem import AllChem
                mol = Chem.AddHs(mol)
                AllChem.EmbedMolecule(mol)
                AllChem.MMFFOptimizeMolecule(mol)
            # Display 3D structure, using rotation preference from session state
            display_mol_3d(mol, width=200, height=200, style="stick", surface=False, 
                          spin=st.session_state.rotate_molecules)
        except Exception as e:
            st.warning(f"Error displaying 3D structure: {str(e)}")
    
    # Molecule name and SMILES always visible
    st.write(f"**{row['Name']}**")
    
    # Molecule properties in a foldable section (expanded by default)
    with st.expander("Molecule Properties", expanded=True):
        st.write(f"SMILES: `{row['SMILES']}`")
        
        # Display properties in a cleaner multi-column layout
        prop_col1, prop_col2, prop_col3 = st.columns(3)
        with prop_col1:
            st.write(f"**QED:** {row['QED']:.2f}")
        with prop_col2:
            st.write(f"**MW:** {row['MW']:.2f}")
        with prop_col3:
            st.write(f"**LogP:** {row['LogP']:.2f}")
    
    # Display the trajectory in a foldable section (collapsed by default)
    with st.expander("Generation Sequence", expanded=False):
        # Extract the building blocks and their names from the trajectory
        building_blocks = []
        reaction_types = []
        
        for i in range(len(traj)):
            if i == 0:
                # First element is typically empty
                continue
            else:
                # Get the SMILES and building block name
                smiles = traj[i][0]
                try:
                    block_info = traj[i][1]  # Format is typically ['block_name', 'id', 'smiles']
                    block_name = block_info[0]
                    
                    # Get reaction type if available (for all but first block)
                    if i > 1:
                        # The reaction type is in the current block's info
                        reaction_types.append(block_name)
                    
                    building_blocks.append({"smiles": smiles, "name": block_name})
                except (IndexError, TypeError) as e:
                    building_blocks.append({"smiles": smiles, "name": f"Step {i}"})
                    if i > 1:
                        reaction_types.append("rxn")
        
        # Add final molecule
        building_blocks.append({"smiles": row['SMILES'], "name": "Final Molecule"})
        
        # Create a layout with molecules and arrows alternating
        total_items = len(building_blocks) * 2 - 1  # molecules + arrows
        cols = st.columns(total_items)
        
        # Display each molecule with arrows between them
        for i, block in enumerate(building_blocks):
            # Calculate column index (every other column, since arrows are in between)
            col_idx = i * 2
            
            with cols[col_idx]:
                try:
                    # Display molecule structure
                    if block["smiles"]:
                        mol = Chem.MolFromSmiles(block["smiles"])
                        if mol:
                            svg = display_mol(mol, width=60, height=60)
                            st.markdown(f"<div style='text-align: center;'>{svg}</div>", unsafe_allow_html=True)
                            
                            # Display building block name
                            # st.markdown(f"<div style='text-align: center; font-size: 12px;'><b>{block['smiles']}</b></div>", 
                            #           unsafe_allow_html=True)
                except Exception as e:
                    st.warning(f"Error displaying molecule {i}: {str(e)}")
            
            # Add arrow between molecules (not after the last one)
            if i < len(building_blocks) - 1:
                with cols[col_idx + 1]:
                    # Add reaction type to the arrow
                    reaction_label = reaction_types[i-1] if i-1 < len(reaction_types) else ""
                    arrow_html = f"""
                    <div style='text-align: center; margin-top: 20px;'>
                        <div style='font-size: 10px; color: #555; font-weight: bold; margin-bottom: 5px;'>{reaction_label}</div>
                        <div style='display: flex; align-items: center; justify-content: center;'>
                            <div style='height: 2px; background-color: #000; width: 60%; margin-right: 1px;'></div>
                            <div style='width: 0; height: 0; border-top: 5px solid transparent; border-bottom: 5px solid transparent; border-left: 8px solid #000;'></div>
                        </div>
                    </div>
                    """
                    st.markdown(arrow_html, unsafe_allow_html=True)


def upload_protein_section():
    """Protein upload section without visualization."""
    st.subheader("Protein Upload")
    
    use_example = st.checkbox("Use example protein (Beta2 adrenergic receptor)")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a PDB file", type=["pdb"])
    

    protein_file = None
    
    if use_example:
        try:
            # Download example PDB file
            protein_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
            protein_file.write(ADRB_POCKET_PDB_FILE.encode('utf-8'))
            protein_file.close()
            st.success("Example protein loaded: Beta2 adrenergic receptor (PDB ID: ADRB2)")
            
            # Store that we're using the example protein
            st.session_state.is_example_protein = True
        except Exception as e:
            st.error(f"Error loading example protein: {str(e)}")
            return None
    elif uploaded_file is not None:
        try:
            protein_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdb")
            protein_file.write(uploaded_file.getvalue())
            protein_file.close()
            st.success(f"Uploaded: {uploaded_file.name}")
            
            # Not using the example protein
            st.session_state.is_example_protein = False
        except Exception as e:
            st.error(f"Error saving uploaded protein: {str(e)}")
            return None
    
    # Store in session state if file exists
    if protein_file:
        st.session_state.protein_file = protein_file
        
    return protein_file


def calculate_protein_centroid(protein_file):
    """Calculate the centroid of a protein structure.
    
    Args:
        protein_file: Path to protein PDB file
    
    Returns:
        Tuple of (x, y, z) coordinates of the centroid
    """
    try:
        atom_positions = []
        with open(protein_file, 'r') as f:
            for line in f:
                if line.startswith('ATOM') or line.startswith('HETATM'):
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    atom_positions.append((x, y, z))
        
        if not atom_positions:
            return (0.0, 0.0, 0.0)
        
        # Calculate centroid
        x_sum = sum(pos[0] for pos in atom_positions)
        y_sum = sum(pos[1] for pos in atom_positions)
        z_sum = sum(pos[2] for pos in atom_positions)
        
        count = len(atom_positions)
        return (x_sum/count, y_sum/count, z_sum/count)
    except Exception as e:
        logger.error(f"Error calculating protein centroid: {e}")
        return (0.0, 0.0, 0.0)


def main():
    """Run the main application."""
    try:
        # Configure the application
        initialize_app()
        
        # Header
        st.title("Molecule Generation Platform")
        
        # Protein upload section (without visualization)
        protein_file = upload_protein_section()
        
        st.markdown("---")
        
        # Molecule generation (full width)
        st.subheader("Generate Molecules")
        
        # Add display options in a small expander
        # with st.expander("Display Options"):
        #     st.session_state.rotate_molecules = st.checkbox("Rotate 3D molecules", value=st.session_state.rotate_molecules)
        
        # Binding site parameters
        st.markdown("#### Binding Site Parameters")
        
        # Option to use automatic centroid calculation
        use_auto_centroid = st.checkbox("Automatically use protein centroid", value=True)
        
        # Calculate centroid if requested
        centroid = (0.0, 0.0, 0.0)
        if use_auto_centroid and 'protein_file' in st.session_state:
            # If using example protein, use the pre-calculated centroid
            if st.session_state.get('is_example_protein', False):
                centroid = ADRB_CENTER
                st.info(f"Using pre-calculated protein centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
            else:
                centroid = calculate_protein_centroid(st.session_state.protein_file.name)
                st.info(f"Calculated protein centroid as the average of all atoms: ({centroid[0]:.2f}, {centroid[1]:.2f}, {centroid[2]:.2f})")
        
        # Create columns for x, y, z coordinates
        col1, col2, col3 = st.columns(3)
        with col1:
            center_x = st.number_input("X Coordinate", value=centroid[0], step=1.0, disabled=use_auto_centroid)
        with col2:
            center_y = st.number_input("Y Coordinate", value=centroid[1], step=1.0, disabled=use_auto_centroid)
        with col3:
            center_z = st.number_input("Z Coordinate", value=centroid[2], step=1.0, disabled=use_auto_centroid)
        
        # Number of molecules to generate
        num_molecules = st.slider("Number of Molecules to Generate", 1, 20, 5)
        
        # Generate button
        if st.button("Generate Molecules"):
            with st.spinner("Generating molecules... Due to server limitations, this may take up to 5 minutes..."):
                # If using example protein, use the cached results
                if st.session_state.get('is_example_protein', False):
                    # Use the cached results for the example protein
                    from rdkit import Chem
                    from rdkit.Chem import AllChem
                    cached_results = ADRB_CACHED_RESPONSE[:num_molecules]
                    
                    # Convert the cached results to RDKit molecules
                    molecules = []
                    trajectories = []
                    for i, result in enumerate(cached_results):
                        try:
                            mol = Chem.MolFromSmiles(result['smiles'])
                            trajectories.append(result['sample_trajectory'])
                            if mol:
                                # Add 3D coordinates
                                mol = Chem.AddHs(mol)
                                AllChem.EmbedMolecule(mol)
                                AllChem.MMFFOptimizeMolecule(mol)
                                
                                # Add properties
                                mol.SetProp("_Name", f"Molecule_{i+1}")
                                mol.SetProp("SMILES", result['smiles'])
                                mol.SetProp("QED", str(0.5 + 0.3 * np.random.random()))  # Mock QED value
                                mol.SetProp("MW", str(300 + 200 * np.random.random()))   # Mock molecular weight
                                mol.SetProp("LogP", str(2.0 + 3.0 * np.random.random())) # Mock LogP
                                
                                molecules.append(mol)
                        except Exception as e:
                            st.warning(f"Error processing cached molecule: {str(e)}")
                    
                    st.success(f"Generated {len(molecules)} molecules from cached results")
                elif 'protein_file' in st.session_state:
                    # Use either auto-calculated centroid or manual input
                    if use_auto_centroid:
                        x, y, z = centroid
                    else:
                        x, y, z = center_x, center_y, center_z
                        
                    molecules, trajectories = generate_molecules(
                        x, y, z, 
                        num_molecules, 
                        st.session_state.protein_file.name
                    )
          
                
                display_molecules(molecules, trajectories)
        
        # Footer
        st.divider()
        st.caption("Molecule Generation Platform | For research purposes only | Not for clinical use")
    
    except Exception as e:
        logger.error(f"Application error: {e}", exc_info=True)
        st.error(f"An unexpected error occurred: {str(e)}")


if __name__ == "__main__":
    main()