"""Audio implementation using sounddevice backend."""

import os
import threading
from collections import deque
from typing import Any, List, Optional

import numpy as np
import numpy.typing as npt
import scipy
import sounddevice as sd
import soundfile as sf

from reachy_mini.utils.constants import ASSETS_ROOT_PATH

from .audio_base import AudioBase


class SoundDeviceAudio(AudioBase):
    """Audio device implementation using sounddevice."""

    def __init__(
        self,
        frames_per_buffer: int = 256,
        log_level: str = "INFO",
    ) -> None:
        """Initialize the SoundDevice audio device."""
        super().__init__(log_level=log_level)
        self.frames_per_buffer = frames_per_buffer
        self.stream = None
        self._output_stream = None
        self._buffer: List[npt.NDArray[np.float32]] = []
        self._output_device_id = self._get_device_id(
            ["Reachy Mini Audio", "respeaker"], device_io_type="output"
        )
        self._input_device_id = self._get_device_id(
            ["Reachy Mini Audio", "respeaker"], device_io_type="input"
        )

        # Streaming output state
        self._streaming_active = False
        self._chunk_fifo: deque[npt.NDArray[np.float32]] = deque()
        self._queued_samples: int = 0
        self._tail: Optional[npt.NDArray[np.float32]] = None
        self._target_buffer_ms: int = 80
        self._max_queue_seconds: float = 3.0  # Reduced from 5.0 for better responsiveness
        self._underflows: int = 0
        self._lock = threading.Lock()

    def start_recording(self) -> None:
        """Open the audio input stream, using ReSpeaker card if available."""
        self.stream = sd.InputStream(
            blocksize=self.frames_per_buffer,
            device=self._input_device_id,
            callback=self._callback,
            samplerate=self.get_input_audio_samplerate(),
        )
        if self.stream is None:
            raise RuntimeError("Failed to open SoundDevice audio stream.")
        self._buffer.clear()
        self.stream.start()
        self.logger.info("SoundDevice audio stream opened.")

    def _callback(
        self,
        indata: npt.NDArray[np.float32],
        frames: int,
        time: int,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            self.logger.warning(f"SoundDevice status: {status}")

        self._buffer.append(indata.copy())

    def get_audio_sample(self) -> Optional[npt.NDArray[np.float32]]:
        """Read audio data from the buffer. Returns numpy array or None if empty."""
        if self._buffer and len(self._buffer) > 0:
            data: npt.NDArray[np.float32] = np.concatenate(self._buffer, axis=0)
            self._buffer.clear()
            return data
        self.logger.debug("No audio data available in buffer.")
        return None

    def get_input_audio_samplerate(self) -> int:
        """Get the input samplerate of the audio device."""
        return int(
            sd.query_devices(self._input_device_id, "input")["default_samplerate"]
        )

    def get_output_audio_samplerate(self) -> int:
        """Get the output samplerate of the audio device."""
        return int(
            sd.query_devices(self._output_device_id, "output")["default_samplerate"]
        )

    def stop_recording(self) -> None:
        """Close the audio stream and release resources."""
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None
            self.logger.info("SoundDevice audio stream closed.")

    def push_audio_sample(self, data: npt.NDArray[np.float32]) -> None:
        """Push audio samples to the output stream for playback."""
        if not self._streaming_active or self._output_stream is None:
            self.logger.warning("Output stream not active. Call start_playing() first.")
            return

        # Convert to mono if needed (match original behavior)
        if data.ndim > 1:
            data = np.mean(data, axis=1)
        data = np.asarray(data, dtype=np.float32, order="C")

        # Prevent unbounded queue growth - only drop if truly over capacity
        with self._lock:
            max_samples = int(self.SAMPLE_RATE * self._max_queue_seconds)
            
            # Only drop chunks if we're significantly over capacity
            # This prevents premature dropping that causes the overflow warnings
            if self._queued_samples > max_samples:
                dropped_count = 0
                target_after_drop = int(max_samples * 0.5)  # Drop to 50% capacity
                
                while self._queued_samples > target_after_drop and self._chunk_fifo:
                    dropped = self._chunk_fifo.popleft()
                    self._queued_samples -= dropped.shape[0]
                    dropped_count += dropped.shape[0]
                
                if dropped_count > 0:
                    self.logger.warning(
                        f"Audio queue overflow ({self._queued_samples + dropped_count} samples), "
                        f"dropped {dropped_count} samples to {self._queued_samples} samples"
                    )

            self._chunk_fifo.append(data)
            self._queued_samples += data.shape[0]

    def _target_buffer_samples(self) -> int:
        """Prebuffer watermark for smooth playback."""
        return int(self.SAMPLE_RATE * (self._target_buffer_ms / 1000.0))

    def _streaming_callback(
        self,
        outdata: npt.NDArray[np.float32],
        frames: int,
        time: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Fill audio output buffer from queued chunks."""
        if status and status.output_underflow:
            self._underflows += 1
            if self._underflows % 10 == 1:
                self.logger.debug(f"Audio underflow count: {self._underflows}")

        out = outdata[:, 0]
        out[:] = 0.0

        target = self._target_buffer_samples()
        written = 0

        # Drain carryover tail first (no lock needed, only accessed in callback)
        if self._tail is not None and self._tail.size:
            take = min(frames, self._tail.size)
            out[:take] = self._tail[:take]
            written += take
            if take < self._tail.size:
                self._tail = self._tail[take:]
            else:
                self._tail = None

        # Drain FIFO chunks - minimal lock time
        while written < frames:
            chunk = None
            
            with self._lock:
                # Quick check: do we have enough buffered data?
                if self._queued_samples < target:
                    break
                if not self._chunk_fifo:
                    break

                # Pop the chunk while holding the lock
                chunk = self._chunk_fifo.popleft()
                self._queued_samples -= chunk.shape[0]
            
            # Process chunk outside the lock
            if chunk is not None:
                need = frames - written
                if chunk.shape[0] <= need:
                    out[written : written + chunk.shape[0]] = chunk
                    written += chunk.shape[0]
                else:
                    out[written:frames] = chunk[:need]
                    self._tail = chunk[need:]
                    written = frames

    def start_playing(self) -> None:
        """Open the audio output stream."""
        if self._output_stream is not None:
            self.stop_playing()

        with self._lock:
            self._streaming_active = True
            self._chunk_fifo.clear()
            self._queued_samples = 0
            self._tail = None
            self._underflows = 0

        self._output_stream = sd.OutputStream(
            samplerate=self.get_output_audio_samplerate(),
            device=self._output_device_id,
            channels=1,
            dtype="float32",
            callback=self._streaming_callback,
        )
        if self._output_stream is None:
            raise RuntimeError("Failed to open SoundDevice audio output stream.")
        self._output_stream.start()

    def stop_playing(self) -> None:
        """Close the audio output stream."""
        if self._output_stream is not None:
            self._output_stream.stop()
            self._output_stream.close()
            self._output_stream = None

        with self._lock:
            self._streaming_active = False
            self._chunk_fifo.clear()
            self._queued_samples = 0
            self._tail = None

        self.logger.info("Audio output stream closed.")

    def play_sound(self, sound_file: str, autoclean: bool = False) -> None:
        """Play a sound file.

        Args:
            sound_file (str): Path to the sound file to play. May be given relative to the assets directory or as an absolute path.
            autoclean (bool): If True, the audio device will be released after the sound is played.

        """
        if not os.path.exists(sound_file):
            file_path = f"{ASSETS_ROOT_PATH}/{sound_file}"
            if not os.path.exists(file_path):
                raise FileNotFoundError(
                    f"Sound file {sound_file} not found in assets directory or given path."
                )
        else:
            file_path = sound_file

        data, samplerate_in = sf.read(file_path, dtype="float32")
        samplerate_out = self.get_output_audio_samplerate()

        if samplerate_in != samplerate_out:
            data = scipy.signal.resample(
                data, int(len(data) * (samplerate_out / samplerate_in))
            )
        if data.ndim > 1:  # convert to mono
            data = np.mean(data, axis=1)

        self.logger.debug(f"Playing sound '{file_path}' at {samplerate_in} Hz")

        self.stop_playing()
        start = 0  # current position in audio data
        length = len(data)

        def callback(
            outdata: npt.NDArray[np.float32],
            frames: int,
            time: Any,  # cdata 'struct PaStreamCallbackTimeInfo *
            status: sd.CallbackFlags,
        ) -> None:
            """Actual playback."""
            nonlocal start

            if status:
                self.logger.warning(f"SoundDevice output status: {status}")

            end = start + frames
            if end > length:
                # Fill the output buffer with the audio data, or zeros if finished
                outdata[: length - start, 0] = data[start:]
                outdata[length - start :, 0] = 0
                raise sd.CallbackStop()
            else:
                outdata[:, 0] = data[start:end]
            start = end

        stop_event = threading.Event()

        self._output_stream = sd.OutputStream(
            samplerate=samplerate_out,
            device=self._output_device_id,
            channels=1,
            callback=callback,
            finished_callback=stop_event.set,  # release the device when done
        )
        if self._output_stream is None:
            raise RuntimeError("Failed to open SoundDevice audio output stream.")
        self._output_stream.start()

        if autoclean:

            def _clean_up_thread() -> None:
                stop_event.wait()
                self.stop_playing()

            threading.Thread(
                target=_clean_up_thread,
                daemon=True,
            ).start()

    def _get_device_id(
        self, names_contains: List[str], device_io_type: str = "output"
    ) -> int:
        """Return the output device id whose name contains the given strings (case-insensitive).

        Args:
            names_contains (List[str]): List of strings that should be contained in the device name.
            device_io_type (str): 'input' or 'output' to specify device type.

        If not found, return the default output device id.

        """
        devices = sd.query_devices()

        for idx, dev in enumerate(devices):
            for name_contains in names_contains:
                if (
                    name_contains.lower() in dev["name"].lower()
                    and dev[f"max_{device_io_type}_channels"] > 0
                ):
                    return idx
        # Return default output device if not found
        self.logger.warning(
            f"No {device_io_type} device found containing '{names_contains}', using default."
        )
        return self._safe_query_device(device_io_type)

    def _safe_query_device(self, kind: str) -> int:
        try:
            return int(sd.query_devices(None, kind)["index"])
        except sd.PortAudioError:
            return (
                int(sd.default.device[1])
                if kind == "input"
                else int(sd.default.device[0])
            )
        except IndexError:
            return 0