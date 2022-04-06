from typing import Text

import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack
import torch

def torch_float32(x):
    """Ensure array/tensor is a float32 tf.Tensor."""
    if isinstance(x, torch.Tensor):
        return x.float()  # This is a no-op if x is float32.
    elif isinstance(x, np.ndarray):
       return torch.from_numpy(x).cuda()  # This is a no-op if x is float32.
    else:
        return torch.tensor(x, dtype=torch.float32).cuda()  # This is a no-op if x is float32.

def safe_log(x, eps=1e-7):
  """Avoid taking the log of a non-positive number."""
  safe_x = torch.where(x <= eps, eps, x.double())
  return torch.log(safe_x)

def stft(audio, frame_size=2048, overlap=0.75):
  """Differentiable stft in PyTorch, computed in batch."""
  assert frame_size * overlap % 2.0 == 0.0

  # Remove channel dim if present.
  audio = torch_float32(audio)
  if len(audio.shape) == 3:
    audio = torch.squeeze(audio, axis=-1)

  s = torch.stft(
      audio,
      n_fft=int(frame_size),
      hop_length=int(frame_size * (1.0 - overlap)),
      win_length=int(frame_size),
      window=torch.hann_window(int(frame_size)).to(audio),
      pad_mode='reflect',
      return_complex=True,
      )
  return s

def compute_mag(audio, size=2048, overlap=0.75):
  mag = torch.abs(stft(audio, frame_size=size, overlap=overlap))
  return torch_float32(mag)

def compute_logmag(audio, size=2048, overlap=0.75):
  return safe_log(compute_mag(audio, size, overlap))

def specplot(audio,
             vmin=-5,
             vmax=1,
             rotate=True,
             size=512 + 256,
             **matshow_kwargs):
  """Plot the log magnitude spectrogram of audio."""
  # If batched, take first element.
  if len(audio.shape) == 2:
    audio = audio[0]

  logmag = compute_logmag(torch_float32(audio), size=size)
  # logmag = spectral_ops.compute_logmel(core.tf_float32(audio), lo_hz=8.0, bins=80, fft_size=size)
  # logmag = spectral_ops.compute_mfcc(core.tf_float32(audio), mfcc_bins=40, fft_size=size)
  # if rotate:
  #   logmag = torch.rot90(logmag)
  logmag = torch.flip(logmag, [0])
  # Plotting.
  plt.matshow(logmag.detach().cpu(),
              vmin=vmin,
              vmax=vmax,
              cmap=plt.cm.magma,
              aspect='auto',
              **matshow_kwargs)
  plt.xticks([])
  plt.yticks([])
  plt.xlabel('Time')
  plt.ylabel('Frequency')

# Time-varying convolution -----------------------------------------------------
def get_fft_size(frame_size: int, ir_size: int, power_of_2: bool = True) -> int:
    """Calculate final size for efficient FFT.

    Args:
    frame_size: Size of the audio frame.
    ir_size: Size of the convolving impulse response.
    power_of_2: Constrain to be a power of 2. If False, allow other 5-smooth
      numbers. TPU requires power of 2, while GPU is more flexible.

    Returns:
    fft_size: Size for efficient FFT.
    """
    convolved_frame_size = ir_size + frame_size - 1
    if power_of_2:
        # Next power of 2.
        fft_size = int(2**np.ceil(np.log2(convolved_frame_size)))
    else:
        fft_size = int(fftpack.helper.next_fast_len(convolved_frame_size))
    return fft_size

def crop_and_compensate_delay(audio: torch.Tensor, audio_size: int, ir_size: int,
                              padding: Text,
                              delay_compensation: int) -> torch.Tensor:
    """Crop audio output from convolution to compensate for group delay.

    Args:
    audio: Audio after convolution. Tensor of shape [batch, time_steps].
    audio_size: Initial size of the audio before convolution.
    ir_size: Size of the convolving impulse response.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation < 0 it
      defaults to automatically calculating a constant group delay of the
      windowed linear phase filter from frequency_impulse_response().

    Returns:
    Tensor of cropped and shifted audio.

    Raises:
    ValueError: If padding is not either 'valid' or 'same'.
    """
    # Crop the output.
    if padding == 'valid':
        crop_size = ir_size + audio_size - 1
    elif padding == 'same':
        crop_size = audio_size
    else:
        raise ValueError('Padding must be \'valid\' or \'same\', instead '
                         'of {}.'.format(padding))

    # Compensate for the group delay of the filter by trimming the front.
    # For an impulse response produced by frequency_impulse_response(),
    # the group delay is constant because the filter is linear phase.
    total_size = int(audio.shape[-1])
    crop = total_size - crop_size
    start = ((ir_size - 1) // 2 -
           1 if delay_compensation < 0 else delay_compensation)
    end = crop - start
    return audio[:, start:-end]

def fft_convolve(audio: torch.Tensor,
                 impulse_response: torch.Tensor,
                 padding: Text = 'same',
                 delay_compensation: int = -1) -> torch.Tensor:
    """Filter audio with frames of time-varying impulse responses.

    Time-varying filter. Given audio [batch, n_samples], and a series of impulse
    responses [batch, n_frames, n_impulse_response], splits the audio into frames,
    applies filters, and then overlap-and-adds audio back together.
    Applies non-windowed non-overlapping STFT/ISTFT to efficiently compute
    convolution for large impulse response sizes.

    Args:
    audio: Input audio. Tensor of shape [batch, audio_timesteps].
    impulse_response: Finite impulse response to convolve. Can either be a 2-D
      Tensor of shape [batch, ir_size], or a 3-D Tensor of shape [batch,
      ir_frames, ir_size]. A 2-D tensor will apply a single linear
      time-invariant filter to the audio. A 3-D Tensor will apply a linear
      time-varying filter. Automatically chops the audio into equally shaped
      blocks to match ir_frames.
    padding: Either 'valid' or 'same'. For 'same' the final output to be the
      same size as the input audio (audio_timesteps). For 'valid' the audio is
      extended to include the tail of the impulse response (audio_timesteps +
      ir_timesteps - 1).
    delay_compensation: Samples to crop from start of output audio to compensate
      for group delay of the impulse response. If delay_compensation is less
      than 0 it defaults to automatically calculating a constant group delay of
      the windowed linear phase filter from frequency_impulse_response().

    Returns:
    audio_out: Convolved audio. Tensor of shape
        [batch, audio_timesteps + ir_timesteps - 1] ('valid' padding) or shape
        [batch, audio_timesteps] ('same' padding).

    Raises:
    ValueError: If audio and impulse response have different batch size.
    ValueError: If audio cannot be split into evenly spaced frames. (i.e. the
      number of impulse response frames is on the order of the audio size and
      not a multiple of the audio size.)
    """
    audio, impulse_response = torch_float32(audio), torch_float32(impulse_response)

    # Add a frame dimension to impulse response if it doesn't have one.
    ir_shape = list(impulse_response.shape)
    if len(ir_shape) == 2:
        impulse_response = torch.unsqueeze(impulse_response, axis = 2)
        ir_shape = list(impulse_response.shape)

    # Get shapes of audio and impulse response.
    batch_size_ir, n_ir_frames, ir_size = ir_shape
    batch_size, audio_size = list(audio.shape)

    # Validate that batch sizes match.
    if batch_size != batch_size_ir:
        raise ValueError('Batch size of audio ({}) and impulse response ({}) must '
                         'be the same.'.format(batch_size, batch_size_ir))

    # Cut audio into frames.
    frame_size = int(np.ceil(audio_size / n_ir_frames))
    hop_size = frame_size
    audio_frames = audio.unfold(1, frame_size, hop_size)

    # Check that number of frames match.
    n_audio_frames = int(audio_frames.shape[1])
    if n_audio_frames != n_ir_frames:
        raise ValueError(
            'Number of Audio frames ({}) and impulse response frames ({}) do not '
            'match. For small hop size = ceil(audio_size / n_ir_frames), '
            'number of impulse response frames must be a multiple of the audio '
            'size.'.format(n_audio_frames, n_ir_frames))

    # Pad and FFT the audio and impulse responses.
    fft_size = get_fft_size(frame_size, ir_size, power_of_2=True)
    audio_fft = torch.fft.rfft(audio_frames, fft_size)
    ir_fft = torch.fft.rfft(impulse_response, fft_size)

    # Multiply the FFTs (same as convolution in time).
    audio_ir_fft = torch.multiply(audio_fft, ir_fft)

    # Take the IFFT to resynthesize audio.
    audio_frames_out = torch.fft.irfft(audio_ir_fft)
    # audio_out = tf.signal.overlap_and_add(audio_frames_out, hop_size)
    audio_out = torch.squeeze(audio_frames_out, axis=1)

    # Crop and shift the output audio.
    return crop_and_compensate_delay(audio_out, audio_size, ir_size, padding,
                                   delay_compensation)

def get_modal_fir(gains, frequencies, dampings, n_samples=44100*2, sample_rate=44100):
    t = torch.reshape(torch.arange(n_samples)/sample_rate, (1, 1, -1)).cuda()
    g = torch.unsqueeze(gains, axis=2)
    f = torch.reshape(frequencies, (1, -1, 1))
    d = torch.reshape(dampings, (1, -1, 1))
    pure = torch.sin(2 * np.pi * f * t)
    damped = torch.exp(-1 * torch.abs(d) * t) * pure
    signal = torch.sum(g * damped, axis=1)
    return torch.cat((torch.zeros_like(signal), signal), axis=1)