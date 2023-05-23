import inspect
from rdkit import Chem
from rdkit.Chem import Descriptors
from mordred import Calculator, descriptors, error

def calculate_all_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        return {"error": f"Invalid SMILES: {smiles}"}

    # Get all descriptor functions from RDKit Descriptors module
    rdkit_descriptor_functions = [
        descriptor for descriptor in dir(Descriptors)
        if callable(getattr(Descriptors, descriptor))
        and not descriptor.startswith('_')
        and len(inspect.signature(getattr(Descriptors, descriptor)).parameters) == 1
    ]

    # Calculate RDKit descriptor values for the given molecule
    rdkit_descriptor_values = [
        getattr(Descriptors, descriptor)(mol) for descriptor in rdkit_descriptor_functions
    ]
    print(len(rdkit_descriptor_values))
    # Create a descriptor calculator with all available Mordred descriptors
    mordred_calc = Calculator(descriptors, ignore_3D=True)

    try:
        # Calculate Mordred descriptor values for the given molecule
        mordred_descriptor_values = mordred_calc(mol)
    except ValueError as e:
        return {"error": f"{e}: {smiles}"}

    # Replace Mordred error objects with None
    mordred_descriptor_values = [value if not isinstance(value, error.Error) else None for value in mordred_descriptor_values]
    print(len(mordred_descriptor_values))
    # Combine RDKit and Mordred descriptor names and values
    descriptor_names = rdkit_descriptor_functions + [str(d) for d in mordred_calc.descriptors]
    descriptor_values = rdkit_descriptor_values + list(mordred_descriptor_values)

    # Return descriptor values with names as a dictionary
    return dict(zip(descriptor_names, descriptor_values))

calculate_all_descriptors("CC(=O)OC1=CC=CC=C1C(=O)O")