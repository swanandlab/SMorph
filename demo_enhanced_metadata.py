"""
Enhanced Metadata Extraction Demo

This notebook demonstrates the improved metadata extraction capabilities
using the new MetadataExtractor class and enhanced TissueImage methods.
"""

# Import necessary modules
import sys
import os
from pathlib import Path

# Add the SMorph path to sys.path if needed
smorph_path = Path(__file__).parent / "smorph"
if str(smorph_path) not in sys.path:
    sys.path.append(str(smorph_path))

from smorph.util.autocrop.core import TissueImage
from smorph.util.autocrop.metadata_utils import MetadataExtractor, get_channel_summary

def demo_enhanced_metadata():
    """
    Demonstrate enhanced metadata extraction capabilities.
    """
    print("=== Enhanced Metadata Extraction Demo ===\n")
    
    # Example with a sample image (replace with actual path)
    sample_image_path = "path/to/your/sample/image.czi"  # Replace with actual path
    
    try:
        # Create TissueImage instance
        tissue_img = TissueImage(sample_image_path, channel=0)
        
        print(f"Image: {tissue_img.im_path}")
        print(f"Current channel: {tissue_img.channel_interest}")
        print(f"Image shape: {tissue_img.imoriginal.shape}")
        print(f"Physical pixel size: {tissue_img.SCALE}")
        print()
        
        # 1. Enhanced metadata for current channel
        print("=== Current Channel Enhanced Metadata ===")
        enhanced_meta = tissue_img.get_enhanced_metadata()
        if enhanced_meta:
            for key, value in enhanced_meta.items():
                if value:
                    print(f"  {key}: {value}")
        else:
            print("  No enhanced metadata available")
        print()
        
        # 2. All channels metadata summary
        print("=== All Channels Summary ===")
        all_channels = tissue_img.get_all_channels_metadata()
        if all_channels:
            for i, ch_meta in enumerate(all_channels):
                name = ch_meta.get('name', f'Channel_{i}')
                ex_wl = ch_meta.get('excitation_wavelength', 'N/A')
                em_wl = ch_meta.get('emission_wavelength', 'N/A')
                acq_mode = ch_meta.get('acquisition_mode', 'N/A')
                pinhole = ch_meta.get('pinhole_size', 'N/A')
                na = ch_meta.get('numerical_aperture', 'N/A')
                
                print(f"  Channel {i}: {name}")
                print(f"    Excitation: {ex_wl}nm, Emission: {em_wl}nm")
                print(f"    Acquisition: {acq_mode}, Pinhole: {pinhole}µm, NA: {na}")
                print()
        else:
            print("  No channels metadata available")
        
        # 3. Deconvolution parameters
        print("=== Deconvolution Parameters ===")
        deconv_params = tissue_img.get_deconvolution_params()
        if deconv_params:
            for key, value in deconv_params.items():
                print(f"  {key}: {value}")
        else:
            print("  No deconvolution parameters available")
        print()
        
        # 4. Raw metadata structure overview
        print("=== Raw Metadata Structure Overview ===")
        if hasattr(tissue_img, 'metadata') and tissue_img.metadata:
            print(f"  Metadata type: {type(tissue_img.metadata)}")
            if isinstance(tissue_img.metadata, dict):
                print(f"  Top-level keys: {list(tissue_img.metadata.keys())}")
                
                # Show some structure for common formats
                if 'ImageDocument' in tissue_img.metadata:
                    print("  Format: CZI")
                    img_doc = tissue_img.metadata['ImageDocument']
                    if 'Metadata' in img_doc:
                        metadata = img_doc['Metadata']
                        if 'Information' in metadata:
                            info = metadata['Information']
                            if 'Image' in info and 'Dimensions' in info['Image']:
                                dims = info['Image']['Dimensions']
                                if 'Channels' in dims:
                                    channels = dims['Channels'].get('Channel', [])
                                    if isinstance(channels, list):
                                        print(f"    Number of channels: {len(channels)}")
                                    else:
                                        print(f"    Number of channels: 1")
                                        
                elif 'images' in tissue_img.metadata:
                    print("  Format: OME")
                    images = tissue_img.metadata['images']
                    if images:
                        pixels = images[0].get('pixels', {})
                        channels = pixels.get('channels', [])
                        print(f"    Number of channels: {len(channels)}")
                        
                elif 'channels' in tissue_img.metadata:
                    print("  Format: ND2 or similar")
                    channels = tissue_img.metadata['channels']
                    if isinstance(channels, list):
                        print(f"    Number of channels: {len(channels)}")
                    else:
                        print(f"    Channels structure: {type(channels)}")
        else:
            print("  No metadata available")
        
    except Exception as e:
        print(f"Error loading image: {e}")
        print("Please provide a valid image path to test the enhanced metadata extraction.")
        return
        
    print("\n=== Enhanced vs Legacy Comparison ===")
    
    # Compare with legacy extraction
    try:
        from smorph.util.autocrop.gui.core import _auto_params_deconv
        
        legacy_params = _auto_params_deconv(tissue_img)
        enhanced_params = tissue_img.get_deconvolution_params()
        
        print("Legacy extraction:")
        if legacy_params:
            for key, value in legacy_params.items():
                print(f"  {key}: {value}")
        else:
            print("  No parameters extracted")
            
        print("\nEnhanced extraction:")
        if enhanced_params:
            for key, value in enhanced_params.items():
                print(f"  {key}: {value}")
        else:
            print("  No parameters extracted")
            
        print("\nComparison:")
        if legacy_params and enhanced_params:
            for key in set(list(legacy_params.keys()) + list(enhanced_params.keys())):
                legacy_val = legacy_params.get(key, 'N/A')
                enhanced_val = enhanced_params.get(key, 'N/A')
                match = "✓" if legacy_val == enhanced_val else "✗"
                print(f"  {key}: {legacy_val} -> {enhanced_val} {match}")
                
    except Exception as e:
        print(f"Error in comparison: {e}")


def demo_metadata_extractor_directly():
    """
    Demonstrate using MetadataExtractor directly on metadata structures.
    """
    print("\n=== Direct MetadataExtractor Demo ===")
    
    # Example metadata structures (simplified versions)
    
    # CZI-like structure
    czi_metadata = {
        'ImageDocument': {
            'Metadata': {
                'Information': {
                    'Image': {
                        'ObjectiveSettings': {'RefractiveIndex': 1.33},
                        'Dimensions': {
                            'Channels': {
                                'Channel': [
                                    {
                                        'Name': 'DAPI',
                                        'ExcitationWavelength': 358,
                                        'EmissionWavelength': 461,
                                        'ContrastMethod': 'Fluorescence'
                                    },
                                    {
                                        'Name': 'GFP',
                                        'ExcitationWavelength': 488,
                                        'EmissionWavelength': 507,
                                        'ContrastMethod': 'Fluorescence'
                                    }
                                ]
                            }
                        }
                    },
                    'Instrument': {
                        'Objectives': {
                            'Objective': {
                                'LensNA': 0.95,
                                'NominalMagnification': 40.0
                            }
                        }
                    }
                },
                'Experiment': {
                    'ExperimentBlocks': {
                        'AcquisitionBlock': {
                            'MultiTrackSetup': {
                                'TrackSetup': {
                                    'Detectors': {
                                        'Detector': [
                                            {
                                                'Name': 'PMT',
                                                'PinholeDiameter': 50.0,
                                                'Gain': 800
                                            }
                                        ]
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    
    print("Testing CZI-like metadata:")
    try:
        extractor = MetadataExtractor(czi_metadata, 'czi')
        
        # Test channel info extraction
        for i in range(2):  # Two channels in this example
            channel_info = extractor.get_channel_info(i)
            print(f"  Channel {i}: {channel_info.get('name', 'Unknown')}")
            print(f"    Excitation: {channel_info.get('excitation_wavelength', 'N/A')}nm")
            print(f"    Emission: {channel_info.get('emission_wavelength', 'N/A')}nm")
            print(f"    NA: {channel_info.get('numerical_aperture', 'N/A')}")
            print(f"    Pinhole: {channel_info.get('pinhole_size', 'N/A')}µm")
            
            # Test deconvolution parameters
            deconv_params = extractor.get_deconvolution_params(i)
            print(f"    Deconv params: {deconv_params}")
            print()
            
    except Exception as e:
        print(f"Error with CZI metadata: {e}")
    
    # OME-like structure
    ome_metadata = {
        'images': [{
            'pixels': {
                'channels': [
                    {
                        'name': 'Nuclei',
                        'excitation_wavelength': 405,
                        'emission_wavelength': 440,
                        'acquisition_mode': 'confocal'
                    },
                    {
                        'name': 'Membrane',
                        'excitation_wavelength': 561,
                        'emission_wavelength': 580,
                        'acquisition_mode': 'confocal'
                    }
                ]
            },
            'objectives': [{
                'numerical_aperture': 1.4,
                'nominal_magnification': 63.0,
                'refractive_index': 1.518
            }]
        }]
    }
    
    print("Testing OME-like metadata:")
    try:
        extractor = MetadataExtractor(ome_metadata, 'ome')
        
        all_channels = extractor.get_all_channels_info()
        for ch_info in all_channels:
            print(f"  Channel {ch_info.get('channel_index', '?')}: {ch_info.get('name', 'Unknown')}")
            print(f"    Excitation: {ch_info.get('excitation_wavelength', 'N/A')}nm")
            print(f"    Emission: {ch_info.get('emission_wavelength', 'N/A')}nm")
            print(f"    Mode: {ch_info.get('acquisition_mode', 'N/A')}")
            print(f"    NA: {ch_info.get('numerical_aperture', 'N/A')}")
            print()
            
    except Exception as e:
        print(f"Error with OME metadata: {e}")


if __name__ == "__main__":
    # Run the demos
    demo_enhanced_metadata()
    demo_metadata_extractor_directly()
    
    print("\n=== Summary ===")
    print("Enhanced metadata extraction provides:")
    print("1. Comprehensive channel information extraction")
    print("2. Support for multiple file formats (CZI, ND2, OME, etc.)")
    print("3. Robust error handling and fallback mechanisms")
    print("4. Standardized parameter extraction for deconvolution")
    print("5. Easy access through TissueImage methods")
    print("6. Backward compatibility with existing code")
