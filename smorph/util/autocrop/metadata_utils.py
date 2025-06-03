"""
Enhanced metadata utility functions for extracting comprehensive imaging parameters
from various microscopy file formats using rich image metadata structures.

This module provides standardized access to channel-specific metadata including
wavelengths, acquisition modes, pinhole sizes, numerical aperture, detector settings,
and other imaging parameters for automated parameter extraction in deconvolution
and image processing operations.
"""

import logging
from typing import Dict, Any, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """Enhanced metadata extractor that utilizes rich image metadata structures."""
    
    def __init__(self, metadata: Dict[str, Any], file_format: str = None):
        """
        Initialize metadata extractor.
        
        Parameters
        ----------
        metadata : dict
            Rich metadata structure from AICSImage or other sources
        file_format : str, optional
            File format hint (e.g., 'czi', 'nd2', 'lif', etc.)
        """
        self.metadata = metadata
        self.file_format = file_format or self._detect_format()
        
    def _detect_format(self) -> str:
        """Detect file format from metadata structure."""
        if 'ImageDocument' in self.metadata:
            return 'czi'
        elif 'channels' in self.metadata:
            # Check for common ND2 structure
            if any('emission_wavelength' in str(ch) for ch in str(self.metadata.get('channels', []))):
                return 'nd2'
        elif 'images' in self.metadata:
            return 'ome'
        return 'unknown'
    
    def get_channel_info(self, channel_idx: int = 0) -> Dict[str, Any]:
        """
        Extract comprehensive channel information.
        
        Parameters
        ----------
        channel_idx : int
            Channel index of interest
            
        Returns
        -------
        dict
            Dictionary containing channel metadata including:
            - name: Channel name
            - excitation_wavelength: Excitation wavelength in nm
            - emission_wavelength: Emission wavelength in nm
            - acquisition_mode: Acquisition mode (e.g., confocal, epifluorescence)
            - pinhole_size: Pinhole size in µm
            - numerical_aperture: Objective numerical aperture
            - detector: Detector information
            - magnification: Objective magnification
            - refractive_index: Medium refractive index
        """
        try:
            if self.file_format == 'czi':
                return self._extract_czi_channel_info(channel_idx)
            elif self.file_format == 'nd2':
                return self._extract_nd2_channel_info(channel_idx)
            elif self.file_format == 'ome':
                return self._extract_ome_channel_info(channel_idx)
            else:
                return self._extract_generic_channel_info(channel_idx)
        except Exception as e:
            logger.warning(f"Failed to extract channel info: {e}")
            return {}
    
    def _extract_czi_channel_info(self, channel_idx: int) -> Dict[str, Any]:
        """Extract channel info from CZI metadata."""
        channel_info = {}
        
        try:
            # Navigate CZI metadata structure
            if 'ImageDocument' in self.metadata:
                metadata = self.metadata['ImageDocument']['Metadata']
                
                # Extract objective information
                try:
                    objectives = metadata['Information']['Instrument']['Objectives']
                    if isinstance(objectives, dict) and 'Objective' in objectives:
                        obj = objectives['Objective']
                        if isinstance(obj, list):
                            obj = obj[0]  # Take first objective
                        
                        channel_info['numerical_aperture'] = float(obj.get('LensNA', 0))
                        channel_info['magnification'] = float(obj.get('NominalMagnification', 0))
                        channel_info['refractive_index'] = float(
                            metadata.get('Information', {})
                            .get('Image', {})
                            .get('ObjectiveSettings', {})
                            .get('RefractiveIndex', 1.33)
                        )
                except Exception as e:
                    logger.debug(f"Could not extract objective info: {e}")
                
                # Extract channel-specific information
                try:
                    im_meta = metadata['Information']['Image']
                    channels = im_meta.get('Dimensions', {}).get('Channels', {}).get('Channel', [])
                    
                    if isinstance(channels, dict):
                        channels = [channels]
                    
                    if channel_idx < len(channels):
                        ch = channels[channel_idx]
                        channel_info['name'] = ch.get('Name', f'Channel_{channel_idx}')
                        channel_info['excitation_wavelength'] = float(ch.get('ExcitationWavelength', 0))
                        channel_info['emission_wavelength'] = float(ch.get('EmissionWavelength', 0))
                        channel_info['acquisition_mode'] = ch.get('ContrastMethod', 'Unknown')
                        
                        # Extract fluor/dye information
                        if 'Fluor' in ch:
                            channel_info['fluor'] = ch['Fluor']
                except Exception as e:
                    logger.debug(f"Could not extract channel info: {e}")
                
                # Extract detector information
                try:
                    exp_blocks = metadata.get('Experiment', {}).get('ExperimentBlocks', {})
                    if 'AcquisitionBlock' in exp_blocks:
                        acq_block = exp_blocks['AcquisitionBlock']
                        if isinstance(acq_block, list):
                            acq_block = acq_block[0]
                        
                        # Navigate to detectors
                        track_setup = acq_block.get('MultiTrackSetup', {}).get('TrackSetup', {})
                        if isinstance(track_setup, list):
                            track_setup = track_setup[0]
                        
                        detectors = track_setup.get('Detectors', {}).get('Detector', [])
                        if isinstance(detectors, dict):
                            detectors = [detectors]
                        
                        if detectors:
                            detector = detectors[0] if len(detectors) > channel_idx else detectors[0]
                            channel_info['detector'] = detector.get('Name', 'Unknown')
                            
                            # Extract pinhole information
                            pinhole_diam = detector.get('PinholeDiameter', 0)
                            if pinhole_diam > 0:
                                channel_info['pinhole_size'] = float(pinhole_diam)
                                channel_info['pinhole_radius'] = float(pinhole_diam) / 2
                            
                            # Extract other detector parameters
                            channel_info['detector_gain'] = detector.get('Gain', 0)
                            channel_info['detector_offset'] = detector.get('DigitalOffset', 0)
                except Exception as e:
                    logger.debug(f"Could not extract detector info: {e}")
                    
        except Exception as e:
            logger.warning(f"Error extracting CZI metadata: {e}")
            
        return channel_info
    
    def _extract_nd2_channel_info(self, channel_idx: int) -> Dict[str, Any]:
        """Extract channel info from ND2 metadata."""
        channel_info = {}
        
        try:
            # Check if metadata is already in processed format (from previous extraction)
            if isinstance(self.metadata, dict) and 'excitation_wavelength' in str(self.metadata):
                # This appears to be pre-processed metadata
                if isinstance(self.metadata, list) and channel_idx < len(self.metadata):
                    ch_data = self.metadata[channel_idx]
                elif isinstance(self.metadata, dict):
                    ch_data = self.metadata
                else:
                    return channel_info
                
                channel_info['excitation_wavelength'] = ch_data.get('excitation_wavelength', 0)
                channel_info['emission_wavelength'] = ch_data.get('emission_wavelength', 0)
                channel_info['numerical_aperture'] = ch_data.get('num_aperture', 0)
                channel_info['refractive_index'] = ch_data.get('refr_index', 1.33)
                channel_info['pinhole_size'] = ch_data.get('pinhole_size', 0)
                if channel_info['pinhole_size'] > 0:
                    channel_info['pinhole_radius'] = channel_info['pinhole_size'] / 2
                
                return channel_info
            
            # Extract from raw ND2 metadata structure
            if 'channels' in self.metadata:
                channels = self.metadata['channels']
                if isinstance(channels, list) and channel_idx < len(channels):
                    ch = channels[channel_idx]
                    channel_info['name'] = ch.get('channel', {}).get('name', f'Channel_{channel_idx}')
                    
                    # Extract wavelength information
                    optical_configs = ch.get('loops', {}).get('NETimeLoop', {}).get('OpticalConfigurations', [])
                    if optical_configs:
                        opt_config = optical_configs[0]
                        channel_info['excitation_wavelength'] = opt_config.get('lasers', {}).get('wavelength', 0)
                        channel_info['emission_wavelength'] = opt_config.get('emissionWavelength', 0)
                        
                        # Extract acquisition mode
                        microscope_config = opt_config.get('microscopeConfig', {})
                        channel_info['acquisition_mode'] = microscope_config.get('method', 'Unknown')
            
            # Extract objective information
            if 'metadata' in self.metadata:
                meta = self.metadata['metadata']
                if 'contents' in meta:
                    # Look for objective information in contents
                    for content in meta['contents']:
                        if 'objectiveLens' in content:
                            obj = content['objectiveLens']
                            channel_info['numerical_aperture'] = float(obj.get('numericalAperture', 0))
                            channel_info['magnification'] = float(obj.get('magnification', 0))
                            channel_info['refractive_index'] = float(obj.get('refractiveIndex', 1.33))
                            break
                            
        except Exception as e:
            logger.warning(f"Error extracting ND2 metadata: {e}")
            
        return channel_info
    
    def _extract_ome_channel_info(self, channel_idx: int) -> Dict[str, Any]:
        """Extract channel info from OME metadata."""
        channel_info = {}
        
        try:
            if 'images' in self.metadata:
                images = self.metadata['images']
                if images and len(images) > 0:
                    image = images[0]
                    
                    # Extract pixels information
                    pixels = image.get('pixels', {})
                    channels = pixels.get('channels', [])
                    
                    if channel_idx < len(channels):
                        ch = channels[channel_idx]
                        channel_info['name'] = ch.get('name', f'Channel_{channel_idx}')
                        channel_info['excitation_wavelength'] = ch.get('excitation_wavelength', 0)
                        channel_info['emission_wavelength'] = ch.get('emission_wavelength', 0)
                        channel_info['acquisition_mode'] = ch.get('acquisition_mode', 'Unknown')
                        
                        # Extract light path information
                        if 'light_path' in ch:
                            light_path = ch['light_path']
                            if 'dichroics' in light_path:
                                channel_info['dichroic'] = light_path['dichroics']
                            if 'filters' in light_path:
                                channel_info['filters'] = light_path['filters']
                    
                    # Extract objective information
                    if 'objectives' in image:
                        objectives = image['objectives']
                        if objectives:
                            obj = objectives[0]  # Take first objective
                            channel_info['numerical_aperture'] = float(obj.get('numerical_aperture', 0))
                            channel_info['magnification'] = float(obj.get('nominal_magnification', 0))
                            channel_info['refractive_index'] = float(obj.get('refractive_index', 1.33))
                            
        except Exception as e:
            logger.warning(f"Error extracting OME metadata: {e}")
            
        return channel_info
    
    def _extract_generic_channel_info(self, channel_idx: int) -> Dict[str, Any]:
        """Extract channel info from generic metadata structure."""
        channel_info = {}
        
        # Try to find common metadata fields regardless of format
        try:
            # Look for channel-like structures
            for key in ['channels', 'channel', 'Channel']:
                if key in self.metadata:
                    channels = self.metadata[key]
                    if isinstance(channels, list) and channel_idx < len(channels):
                        ch = channels[channel_idx]
                        self._extract_common_fields(ch, channel_info)
                    elif isinstance(channels, dict):
                        self._extract_common_fields(channels, channel_info)
                    break
            
            # Look for objective information
            for obj_key in ['objective', 'objectives', 'Objective', 'Objectives']:
                if obj_key in self.metadata:
                    obj_data = self.metadata[obj_key]
                    if isinstance(obj_data, list):
                        obj_data = obj_data[0]
                    if isinstance(obj_data, dict):
                        channel_info['numerical_aperture'] = float(obj_data.get('numerical_aperture', 
                                                                                obj_data.get('LensNA', 0)))
                        channel_info['magnification'] = float(obj_data.get('magnification',
                                                                           obj_data.get('NominalMagnification', 0)))
                    break
                    
        except Exception as e:
            logger.warning(f"Error extracting generic metadata: {e}")
            
        return channel_info
    
    def _extract_common_fields(self, data: Dict[str, Any], channel_info: Dict[str, Any]):
        """Extract common fields from data dictionary."""
        # Common field mappings
        field_mappings = {
            'excitation_wavelength': ['excitation_wavelength', 'ExcitationWavelength', 'ex_wavelength'],
            'emission_wavelength': ['emission_wavelength', 'EmissionWavelength', 'em_wavelength'],
            'name': ['name', 'Name', 'channel_name'],
            'acquisition_mode': ['acquisition_mode', 'ContrastMethod', 'method'],
            'pinhole_size': ['pinhole_size', 'PinholeDiameter', 'pinhole'],
            'numerical_aperture': ['numerical_aperture', 'LensNA', 'NA'],
            'magnification': ['magnification', 'NominalMagnification', 'mag'],
            'refractive_index': ['refractive_index', 'RefractiveIndex', 'RI']
        }
        
        for field, possible_keys in field_mappings.items():
            for key in possible_keys:
                if key in data:
                    try:
                        if field in ['excitation_wavelength', 'emission_wavelength', 'pinhole_size', 
                                   'numerical_aperture', 'magnification', 'refractive_index']:
                            channel_info[field] = float(data[key])
                        else:
                            channel_info[field] = data[key]
                        break
                    except (ValueError, TypeError):
                        continue
    
    def get_deconvolution_params(self, channel_idx: int = 0) -> Dict[str, Any]:
        """
        Extract parameters specifically needed for deconvolution.
        
        Parameters
        ----------
        channel_idx : int
            Channel index of interest
            
        Returns
        -------
        dict
            Dictionary containing deconvolution parameters:
            - ex_wavelen: Excitation wavelength in nm
            - em_wavelen: Emission wavelength in nm
            - num_aperture: Numerical aperture
            - refr_index: Refractive index
            - pinhole_radius: Pinhole radius in µm
        """
        channel_info = self.get_channel_info(channel_idx)
        
        deconv_params = {}
        
        # Extract wavelengths
        if 'excitation_wavelength' in channel_info:
            deconv_params['ex_wavelen'] = channel_info['excitation_wavelength']
        if 'emission_wavelength' in channel_info:
            deconv_params['em_wavelen'] = channel_info['emission_wavelength']
            
        # Extract optical parameters
        if 'numerical_aperture' in channel_info:
            deconv_params['num_aperture'] = channel_info['numerical_aperture']
        if 'refractive_index' in channel_info:
            deconv_params['refr_index'] = channel_info['refractive_index']
            
        # Extract pinhole information
        if 'pinhole_radius' in channel_info:
            deconv_params['pinhole_radius'] = channel_info['pinhole_radius'] * 1e6  # Convert to micrometers
        elif 'pinhole_size' in channel_info:
            deconv_params['pinhole_radius'] = channel_info['pinhole_size'] / 2 * 1e6
            
        # Set defaults for missing parameters
        if 'refr_index' not in deconv_params:
            deconv_params['refr_index'] = 1.33  # Default for water
            
        return deconv_params
    
    def get_all_channels_info(self) -> List[Dict[str, Any]]:
        """
        Get information for all available channels.
        
        Returns
        -------
        list
            List of dictionaries containing channel information
        """
        channels_info = []
        
        # Try to determine number of channels
        num_channels = self._get_channel_count()
        
        for i in range(num_channels):
            channel_info = self.get_channel_info(i)
            if channel_info:  # Only add if we got some information
                channel_info['channel_index'] = i
                channels_info.append(channel_info)
                
        return channels_info
    
    def _get_channel_count(self) -> int:
        """Determine the number of channels from metadata."""
        try:
            if self.file_format == 'czi':
                metadata = self.metadata.get('ImageDocument', {}).get('Metadata', {})
                channels = (metadata.get('Information', {})
                           .get('Image', {})
                           .get('Dimensions', {})
                           .get('Channels', {})
                           .get('Channel', []))
                if isinstance(channels, dict):
                    return 1
                return len(channels) if isinstance(channels, list) else 0
                
            elif self.file_format == 'nd2':
                channels = self.metadata.get('channels', [])
                return len(channels) if isinstance(channels, list) else 1
                
            elif self.file_format == 'ome':
                images = self.metadata.get('images', [])
                if images:
                    pixels = images[0].get('pixels', {})
                    channels = pixels.get('channels', [])
                    return len(channels)
                    
        except Exception as e:
            logger.debug(f"Could not determine channel count: {e}")
            
        return 1  # Default to 1 channel


def extract_deconv_params(metadata: Dict[str, Any], channel_idx: int = 0, 
                         file_format: str = None) -> Optional[Dict[str, Any]]:
    """
    Convenience function to extract deconvolution parameters from metadata.
    
    Parameters
    ----------
    metadata : dict
        Rich metadata structure
    channel_idx : int
        Channel index of interest
    file_format : str, optional
        File format hint
        
    Returns
    -------
    dict or None
        Dictionary containing deconvolution parameters or None if extraction fails
    """
    try:
        extractor = MetadataExtractor(metadata, file_format)
        return extractor.get_deconvolution_params(channel_idx)
    except Exception as e:
        logger.error(f"Failed to extract deconvolution parameters: {e}")
        return None


def get_channel_summary(metadata: Dict[str, Any], file_format: str = None) -> List[Dict[str, Any]]:
    """
    Get a summary of all channels in the metadata.
    
    Parameters
    ----------
    metadata : dict
        Rich metadata structure
    file_format : str, optional
        File format hint
        
    Returns
    -------
    list
        List of dictionaries containing channel summaries
    """
    try:
        extractor = MetadataExtractor(metadata, file_format)
        return extractor.get_all_channels_info()
    except Exception as e:
        logger.error(f"Failed to get channel summary: {e}")
        return []
