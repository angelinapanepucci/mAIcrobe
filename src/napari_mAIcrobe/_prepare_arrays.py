"""
Module responsible for preparing arrays for processing by the mAIcrobe plugin. This includes:
- Squeezing arrays to remove singleton dimensions.
- Converting xarray DataArrays to raw NumPy arrays if needed.
- Creating new layers in the viewer for the processed arrays without overwriting the originals.
"""

import numpy as np
from napari.layers import Image
from magicgui import magicgui


def squeeze_all_layers(viewer, suffix=" squeezed NumPy"):
    """
    Creates new squeezed NumPy versions of all Image layers in the viewer.
    Does not overwrite the original layers.
    """
    new_layers = []

    for layer in list(viewer.layers):
        data = layer.data
        

        # Convert xarray DataArray to raw NumPy values if needed
        if hasattr(data, "values"):
            data = data.values

        arr = np.asarray(data)
        arr_squeezed = np.squeeze(arr)

        new_name = layer.name + suffix

        # Add back as the same general layer type
        if isinstance(layer, Image):
            new_layer = viewer.add_image(
                arr_squeezed,
                name=new_name,
                colormap=layer.colormap,
                blending=layer.blending,
                opacity=layer.opacity,
                visible=layer.visible,
            )

            layer.visible = False  # Hide the original layer
            
        else:
            print(f"Skipping {layer.name}: not an Image layer")
            continue
        
        new_layers.append(new_layer)

    return new_layers

