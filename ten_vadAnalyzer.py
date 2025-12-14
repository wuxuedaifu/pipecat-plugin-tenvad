
"""TEN Voice Activity Detection (VAD) implementation for Pipecat.

This module provides a VAD analyzer based on the TEN VAD library,
which can detect voice activity in audio streams with high accuracy.
Supports various sample rates with configurable hop size and threshold.

License Follwing repo:
https://github.com/TEN-framework/ten-vad.git
"""

import time
from typing import Optional

import numpy as np
from loguru import logger

from pipecat.audio.vad.vad_analyzer import VADAnalyzer, VADParams

# How often should we reset internal model state
_MODEL_RESET_STATES_TIME = 5.0

try:
    from ten_vad import TenVad

except ModuleNotFoundError as e:
    logger.error(f"Exception: {e}")
    logger.error("In order to use TEN VAD, you need to install the ten_vad package.")
    raise Exception(f"Missing module(s): {e}")


class TenVADModel:
    """TEN VAD model wrapper.

    Provides voice activity detection using the TEN VAD library
    for efficient inference. Handles model state management
    and input validation for audio processing.
    """

    def __init__(self, hop_size: int = 256, threshold: float = 0.5):
        """Initialize the TEN VAD model.

        Args:
            hop_size: Number of audio samples per frame (typically 256 for 16kHz).
            threshold: Voice activity detection threshold (0.0 to 1.0).
        """
        self.hop_size = hop_size
        self.threshold = threshold
        self._model = TenVad(hop_size=hop_size, threshold=threshold)
        self._last_reset_time = 0

    def _validate_input(self, audio_data: np.ndarray, sample_rate: int):
        """Validate and preprocess input audio data."""
        if np.ndim(audio_data) != 1:
            raise ValueError(f"Audio data should be 1-dimensional, got {np.ndim(audio_data)} dimensions")

        expected_length = self.hop_size
        if len(audio_data) != expected_length:
            raise ValueError(
                f"Audio data length should be {expected_length} samples, got {len(audio_data)}"
            )

        # Ensure audio data is int16 as required by TEN VAD
        if audio_data.dtype != np.int16:
            # Convert float32 to int16 if needed
            if audio_data.dtype == np.float32:
                audio_data = (audio_data * 32767).astype(np.int16)
            else:
                audio_data = audio_data.astype(np.int16)

        return audio_data

    def reset_states(self):
        """Reset the internal model state."""
        # TEN VAD handles state management internally
        # We just update the reset time for periodic resets
        self._last_reset_time = time.time()

    def __call__(self, audio_data: np.ndarray, sample_rate: int):
        """Process audio input through the VAD model."""
        audio_data = self._validate_input(audio_data, sample_rate)
        
        try:
            # Process audio through TEN VAD
            probability, flag = self._model.process(audio_data)
            
            # We need to reset the model from time to time because it doesn't
            # really need all the data and memory will keep growing otherwise.
            curr_time = time.time()
            diff_time = curr_time - self._last_reset_time
            if diff_time >= _MODEL_RESET_STATES_TIME:
                self.reset_states()
                self._last_reset_time = curr_time

            return probability
        except Exception as e:
            logger.error(f"Error processing audio with TEN VAD: {e}")
            return 0.0


class TenVADAnalyzer(VADAnalyzer):
    """Voice Activity Detection analyzer using the TEN VAD library.

    Implements VAD analysis using the TEN VAD library for
    accurate voice activity detection. Supports various sample rates
    with configurable hop size and threshold parameters.
    """

    def __init__(
        self,
        *,
        sample_rate: Optional[int] = 16000,
        params: Optional[VADParams] = None,
        hop_size: int = 256,
        threshold: float = 0.5,
    ):
        """Initialize the TEN VAD analyzer.

        Args:
            sample_rate: Audio sample rate. If None, will be set later.
            params: VAD parameters for detection thresholds and timing.
            hop_size: Number of audio samples per frame (default: 256).
            threshold: Voice activity detection threshold (default: 0.5).
        """
        super().__init__(sample_rate=sample_rate, params=params)

        logger.debug("Loading TEN VAD model...")

        # Calculate hop size based on sample rate if not provided
        if sample_rate:
            # Default to 16ms frames (256 samples at 16kHz, 128 samples at 8kHz)
            if hop_size == 256:  # Default value
                self.hop_size = 256 if sample_rate == 16000 else 128
            else:
                self.hop_size = hop_size
        else:
            self.hop_size = hop_size

        self.threshold = threshold
        self._model = TenVADModel(hop_size=self.hop_size, threshold=self.threshold)

        logger.debug("Loaded TEN VAD")

    #
    # VADAnalyzer
    #

    def set_sample_rate(self, sample_rate: int):
        """Set the sample rate for audio processing.

        Args:
            sample_rate: Audio sample rate.

        Raises:
            ValueError: If sample rate is not supported.
        """
        # todo: ten-vad does not support 8000 hz sample rate
        # TEN VAD supports various sample rates, but we need to adjust hop size
        if sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {sample_rate}")

        # Adjust hop size based on sample rate (aim for ~16ms frames)
        if self.hop_size == 256:  # Default for 16kHz
            self.hop_size = 256 if sample_rate == 16000 else 128
        elif self.hop_size == 128:  # Default for 8kHz
            self.hop_size = 128 if sample_rate == 8000 else 256

        super().set_sample_rate(sample_rate)

        # Recreate model with new hop size
        self._model = TenVADModel(hop_size=self.hop_size, threshold=self.threshold)

    def num_frames_required(self) -> int:
        """Get the number of audio frames required for VAD analysis.

        Returns:
            Number of frames required (hop_size).
        """
        return self.hop_size

    def voice_confidence(self, buffer) -> float:
        """Calculate voice activity confidence for the given audio buffer.

        Args:
            buffer: Audio buffer to analyze.

        Returns:
            Voice confidence score between 0.0 and 1.0.
        """
        try:
            audio_int16 = np.frombuffer(buffer, np.int16)
            
            # Ensure we have the correct number of samples
            if len(audio_int16) != self.hop_size:
                logger.warning(
                    f"Audio buffer length {len(audio_int16)} doesn't match required hop size {self.hop_size}"
                )
                # Pad or truncate as needed
                if len(audio_int16) < self.hop_size:
                    audio_int16 = np.pad(audio_int16, (0, self.hop_size - len(audio_int16)))
                else:
                    audio_int16 = audio_int16[:self.hop_size]

            new_confidence = self._model(audio_int16, self.sample_rate)
            # logger.debug(f"New confidence: {new_confidence}, {self.sample_rate=}, {len(audio_int16)=}")
            return new_confidence
        except Exception as e:
            # This comes from an empty audio array or other processing errors
            logger.error(f"Error analyzing audio with TEN VAD: {e}")
            return 0.0


# Standalone script for testing TEN VAD
if __name__ == "__main__":
    import sys
    import scipy.io.wavfile as Wavfile

    if len(sys.argv) != 3:
        print("Usage: python ten-vad.py <input_wav_file> <output_file>")
        sys.exit(1)

    input_file, out_path = sys.argv[1], sys.argv[2]
    
    try:
        sr, data = Wavfile.read(input_file)
        
        # Use appropriate hop size based on sample rate
        hop_size = 256 if sr == 16000 else 128
        threshold = 0.5
        
        ten_vad_instance = TenVad(hop_size=hop_size, threshold=threshold)
        num_frames = data.shape[0] // hop_size
        
        # Streaming inference
        with open(out_path, "w") as f:
            for i in range(num_frames):
                audio_data = data[i * hop_size: (i + 1) * hop_size]
                out_probability, out_flag = ten_vad_instance.process(audio_data)
                print("[%d] %0.6f, %d" % (i, out_probability, out_flag))
                f.write("[%d] %0.6f, %d\n" % (i, out_probability, out_flag))
                
        print(f"VAD results written to {out_path}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        sys.exit(1)
