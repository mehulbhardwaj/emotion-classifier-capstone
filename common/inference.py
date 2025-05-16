"""
Inference engine for emotion classification models.
"""

import os
import queue
import time # Used by timer, but EmotionInferenceEngine might use it directly later
import pandas as pd
import torch
import torchaudio
from tqdm import tqdm
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoProcessor # Added AutoProcessor

# from .config import BaseConfig # cfg is passed in, so direct config import might not be needed unless type hinting BaseConfig
# from .utils import ensure_dir # Potentially needed if engine saves files, for now cfg handles paths

class EmotionInferenceEngine:
    """
    Generic inference engine for emotion classification models.
    Supports inference from audio files, CSVs, or streaming audio.
    """
    
    def __init__(self, cfg, model, tokenizer, processor=None, asr_model_instance=None):
        """
        Initialize the inference engine.
        
        Args:
            cfg: Configuration object (expected to have attributes like sample_rate, device, id_to_emotion, asr_model_name, and optionally load_asr_model_on_init)
            model: Trained model
            tokenizer: Text tokenizer for the model's text encoder.
            processor: Optional ASR processor (e.g., WhisperProcessor for ASR task if separate from text encoder).
            asr_model_instance: Optional pre-loaded ASR model instance.
        """
        self.cfg = cfg
        self.model = model.to(self.cfg.device) # Ensure model is on the correct device
        self.tokenizer = tokenizer # Text tokenizer (e.g., from RoBERTa, Whisper's text part)
        self.asr_processor = processor # This is the ASR-specific processor (e.g. WhisperProcessor for ASR task)
        self.asr_model = None # Initialize as None
        
        # Initialize the audio processor for the model's audio encoder
        # This should match the processor used during training for the cfg.audio_encoder_model_name
        try:
            print(f"Loading audio processor for model's audio encoder: {cfg.audio_encoder_model_name}")
            self.model_audio_processor = AutoProcessor.from_pretrained(cfg.audio_encoder_model_name)
        except Exception as e:
            print(f"Warning: Could not load AutoProcessor for {cfg.audio_encoder_model_name}. Audio processing might fail. Error: {e}")
            self.model_audio_processor = None # Fallback or error

        if self.asr_processor and getattr(cfg, 'load_asr_model_on_init', False) and hasattr(cfg, 'asr_model_name') and cfg.asr_model_name:
            print(f"Initializing ASR model ({cfg.asr_model_name}) for EmotionInferenceEngine...")
            try:
                # asr_model_instance is passed if already loaded by main.py
                if asr_model_instance:
                    self.asr_model = asr_model_instance.to(self.cfg.device)
                    self.asr_model.eval()
                    print("Using pre-loaded ASR model for engine, set to eval mode.")
                else:
                    self.asr_model = WhisperForConditionalGeneration.from_pretrained(cfg.asr_model_name).to(self.cfg.device)
                    self.asr_model.eval()
                    print("ASR model loaded successfully and set to eval mode for engine.")
            except Exception as e:
                print(f"Warning: Could not load ASR model {cfg.asr_model_name} during engine init: {e}")
        elif asr_model_instance: # Handles case where ASR model is passed but load_asr_model_on_init might be false
             self.asr_model = asr_model_instance.to(self.cfg.device)
             self.asr_model.eval()
             print("Using pre-loaded ASR model for engine (passed directly), set to eval mode.")
        
        # Audio settings
        self.sample_rate = cfg.sample_rate # This should be the target SR for the model_audio_processor
        self.buffer_duration = getattr(cfg, 'buffer_duration', 5.0)  # Default buffer duration in seconds
        self.buffer_size = int(self.sample_rate * self.buffer_duration)
        
        # Streaming variables (sounddevice import will be needed for actual streaming)
        self.audio_buffer = queue.Queue()
        self.stream = None # sounddevice.InputStream
        self.is_running = False
        
        # Results storage
        self.last_prediction = None
        self.last_confidence = None
        self.last_transcript = None

        # Create id_to_emotion mapping from cfg.label_encoder
        self.id_to_emotion_map = {v: k for k, v in cfg.label_encoder.items()} if cfg.label_encoder else {}
    
    def infer_from_file(self, audio_file_path, text=None, use_asr=False):
        """
        Infer emotion from a single audio file.
        
        Args:
            audio_file_path (str): Path to audio file.
            text (str, optional): Optional text transcription. If None and use_asr is False, error.
            use_asr (bool): Whether to use ASR to get text if `text` is None. Requires processor to be set.
            
        Returns:
            dict: Inference results including 'emotion', 'confidence', 'transcript', 'file', and 'error' (if any).
        """
        print(f"\nInferring from file: {audio_file_path}")
        if text:
            print(f"Using provided text: \"{text}\"")
        elif use_asr and self.asr_processor:
            print("Using ASR for text transcription...")
        elif use_asr and not self.asr_processor:
            print("Warning: use_asr is True, but no ASR processor is configured for the engine.")
            # Fallback to error or allow proceeding if model can handle no text? For now, error.
            return {
                'emotion': None, 'confidence': None, 'transcript': None,
                'file': audio_file_path, 'error': "ASR requested but no processor configured."
            }
        elif not text and not use_asr:
             return {
                'emotion': None, 'confidence': None, 'transcript': None,
                'file': audio_file_path, 'error': "No text provided and ASR not requested."
            }

        try:
            # Load audio
            waveform, sr = torchaudio.load(audio_file_path)
            waveform = waveform.to(self.cfg.device) # Move waveform to device early
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate).to(self.cfg.device)
                waveform = resampler(waveform)
            if waveform.ndim == 1:  # Ensure 2D [channels, time]
                waveform = waveform.unsqueeze(0)
            
            # Process for inference
            emotion, confidence, transcript = self._process_audio_and_text(waveform, text, use_asr)
            
            results = {
                'emotion': emotion,
                'confidence': confidence,
                'transcript': transcript,
                'file': audio_file_path
            }
            
            print(f"  -> Predicted Emotion: {emotion} (Confidence: {confidence:.2f})")
            if transcript:
                print(f"  -> Transcript: \"{transcript}\"")
            
            return results
        
        except Exception as e:
            print(f"Error processing file {audio_file_path}: {e}")
            return {
                'emotion': None, 'confidence': None, 'transcript': None,
                'file': audio_file_path, 'error': str(e)
            }
    
    def infer_from_csv(self, csv_path, audio_dir=None, text_column='Utterance', 
                       dialogue_id_col='Dialogue_ID', utterance_id_col='Utterance_ID',
                       audio_path_col='audio_path', emotion_col='Emotion',
                       num_examples=None, use_asr_if_text_missing=False):
        """
        Infer emotions for all items in a CSV file.
        
        Args:
            csv_path (str): Path to CSV file.
            audio_dir (str, optional): Base directory for audio files if paths are relative or need construction.
            text_column (str): Name of the column containing text utterances.
            dialogue_id_col (str): Column name for dialogue ID (used if constructing audio path).
            utterance_id_col (str): Column name for utterance ID (used if constructing audio path).
            audio_path_col (str): Column name for explicit audio file paths.
            emotion_col (str): Column name for ground truth emotion (for accuracy calculation).
            num_examples (int, optional): Maximum number of examples to process.
            use_asr_if_text_missing (bool): If text is missing for a row, attempt ASR.
            
        Returns:
            list: List of inference results (dictionaries).
        """
        print(f"Inferring from CSV: {csv_path}")
        
        try:
            df = pd.read_csv(csv_path)
            print(f"Loaded CSV with {len(df)} rows")
            
            if num_examples and num_examples > 0:
                df = df.head(num_examples)
                print(f"Processing first {num_examples} examples")
            
            results_list = []
            
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing CSV"):
                current_audio_path = None
                if audio_path_col in row and pd.notna(row[audio_path_col]):
                    current_audio_path = row[audio_path_col]
                    if audio_dir and not os.path.isabs(current_audio_path): # If path is relative, join with audio_dir
                        current_audio_path = os.path.join(audio_dir, current_audio_path)
                elif audio_dir and dialogue_id_col in row and utterance_id_col in row:
                    # Construct path (assuming WAV extension, adjust if needed)
                    current_audio_path = os.path.join(
                        audio_dir,
                        f"dia{row[dialogue_id_col]}_utt{row[utterance_id_col]}.wav" 
                    )
                
                if not current_audio_path or not os.path.exists(current_audio_path):
                    print(f"Warning: Audio file not found or path missing for row {idx}. Path: {current_audio_path}. Skipping.")
                    results_list.append({'row_idx': idx, 'error': 'Audio file not found or path missing'})
                    continue
                    
                text_utterance = row.get(text_column) if text_column in row and pd.notna(row[text_column]) else None
                
                # Determine if ASR should be used for this row
                asr_for_row = False
                if not text_utterance and use_asr_if_text_missing and self.asr_processor:
                    asr_for_row = True
                elif not text_utterance and not use_asr_if_text_missing:
                    print(f"Warning: Text missing for row {idx} and ASR not requested. Skipping.")
                    results_list.append({'row_idx': idx, 'error': 'Text missing and ASR not requested'})
                    continue
                
                result = self.infer_from_file(current_audio_path, text=text_utterance, use_asr=asr_for_row)
                
                gt_emotion = row.get(emotion_col) if emotion_col in row else None
                if gt_emotion:
                    result['ground_truth'] = gt_emotion
                result['row_idx'] = idx
                
                results_list.append(result)
            
            # Print summary if ground truth was available
            valid_results_for_accuracy = [r for r in results_list if r.get('ground_truth') and r.get('emotion')]
            if valid_results_for_accuracy:
                correct = sum(1 for r in valid_results_for_accuracy if r['emotion'] == r['ground_truth'])
                print(f"\nAccuracy on {len(valid_results_for_accuracy)} valid samples: {correct}/{len(valid_results_for_accuracy)} = {correct/len(valid_results_for_accuracy):.2%}")
            
            return results_list
        
        except Exception as e:
            print(f"Error processing CSV {csv_path}: {e}")
            return [{'error': str(e)}]

    def _process_audio_and_text(self, waveform, text=None, use_asr=False):
        """
        Core logic to process audio and text, and get model prediction.
        Waveform should be [1, num_samples] on target device, at self.sample_rate.
        
        Args:
            waveform (torch.Tensor): Audio waveform tensor [1, num_samples] on target device.
            text (str, optional): Text transcription.
            use_asr (bool): Whether to use ASR if text is None.
            
        Returns:
            tuple: (emotion_label, confidence_score, transcript_used)
        """
        with torch.no_grad():
            transcript_to_tokenize = None
            
            if text:
                transcript_to_tokenize = text
            elif use_asr and self.asr_processor and self.asr_model:
                # Ensure waveform is suitable for ASR processor (e.g. numpy on CPU for some)
                waveform_for_asr = waveform.squeeze().cpu().numpy()
                
                # Use self.asr_processor (e.g., WhisperProcessor for ASR task)
                inputs_asr = self.asr_processor(
                    waveform_for_asr,
                    sampling_rate=self.sample_rate, # Assuming ASR model also uses this SR
                    return_tensors="pt"
                ).input_features.to(self.cfg.device)
                
                predicted_ids = self.asr_model.generate(inputs_asr)
                transcript_to_tokenize = self.asr_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
                self.last_transcript = transcript_to_tokenize
            elif use_asr and (not self.asr_processor or not self.asr_model):
                print("Warning: ASR inference requested, but ASR components not available/configured.")
                return "Error", 0.0, "ASR components not configured"
            elif not text and not use_asr:
                print("Error: No text provided and ASR not requested.")
                return "Error", 0.0, "No text input"

            if transcript_to_tokenize is None:
                 print("Error: Failed to obtain transcript for tokenization.")
                 return "Error", 0.0, "Failed to obtain transcript"

            tokenized = self.tokenizer(
                transcript_to_tokenize,
                truncation=True,
                padding="max_length", 
                max_length=self.cfg.max_seq_length_text, # Use correct attribute
                return_tensors="pt"
            )
            input_ids = tokenized['input_ids'].to(self.cfg.device)
            text_attention_mask = tokenized['attention_mask'].to(self.cfg.device) # Renamed for clarity
            
            # Process audio using the model_audio_processor
            # Waveform is already on device and at correct sample_rate
            if not self.model_audio_processor:
                print("Error: Model audio processor not initialized in inference engine.")
                return "Error", 0.0, "Model audio processor missing"

            # model_audio_processor expects a list of raw audio signals (numpy arrays or lists of floats)
            # or a single raw audio signal. It handles batching if a list is given.
            # Waveform is [1, num_samples], processor might expect 1D array or list.
            processed_audio = self.model_audio_processor(
                waveform.squeeze(0).cpu().numpy(), # Pass as 1D numpy array on CPU
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            # Move processed audio inputs to device
            audio_input_values = processed_audio.input_values.to(self.cfg.device)
            audio_attention_mask = processed_audio.get('attention_mask') # Get attention mask if processor provides it
            if audio_attention_mask is not None:
                audio_attention_mask = audio_attention_mask.to(self.cfg.device)

            batch = {
                # Keys should match exactly what the model's forward method expects
                'audio_input_values': audio_input_values, 
                'text_input_ids': input_ids,                     
                'text_attention_mask': text_attention_mask      
            }
            if audio_attention_mask is not None:
                 batch['audio_attention_mask'] = audio_attention_mask
            
            logits = self.model(batch) 
            
            probs = torch.softmax(logits, dim=1)
            confidence, prediction_idx = torch.max(probs, dim=1)
            
            emotion_id = prediction_idx.item()
            confidence_score = confidence.item()
            
            # Use the new id_to_emotion_map
            emotion_label = self.id_to_emotion_map.get(emotion_id, f"UnknownID:{emotion_id}")
            
            self.last_prediction = emotion_label
            self.last_confidence = confidence_score
            if not text: # if ASR was used, last_transcript was already set
                self.last_transcript = transcript_to_tokenize

            return emotion_label, confidence_score, transcript_to_tokenize

    # --- Streaming Methods (Requires sounddevice and further implementation) ---
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for sounddevice InputStream. Puts audio data into buffer."""
        if status:
            print(f"Stream callback status: {status}", flush=True)
        self.audio_buffer.put(indata.copy())

    def start_audio_stream(self, buffer_duration_override=None):
        """
        Start the audio stream for real-time inference.
        Requires `sounddevice` to be installed.
        """
        try:
            import sounddevice as sd
        except ImportError:
            print("Error: `sounddevice` library is not installed. Cannot start audio stream.")
            print("Please install it: pip install sounddevice")
            return False

        if self.is_running:
            print("Audio stream is already running.")
            return True

        effective_buffer_duration = buffer_duration_override if buffer_duration_override else self.buffer_duration
        effective_buffer_size = int(self.sample_rate * effective_buffer_duration)
        
        print(f"Starting audio stream with {self.sample_rate} Hz, {effective_buffer_duration}s buffer...")
        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='float32', # Standard for Whisper
                callback=self._audio_callback,
                blocksize=effective_buffer_size # Read in chunks of buffer_duration
            )
            self.stream.start()
            self.is_running = True
            print("Audio stream started. Waiting for audio data...")
            return True
        except Exception as e:
            print(f"Error starting audio stream: {e}")
            self.stream = None
            self.is_running = False
            return False

    def process_realtime_audio(self, use_asr=True):
        """
        Process audio from the stream buffer for real-time prediction.
        This should be called repeatedly in a loop after start_audio_stream.

        Args:
            use_asr (bool): Whether to use ASR for transcription. If False, this method
                            is less useful unless text is fed from another source.
        
        Returns:
            tuple: (emotion_label, confidence_score, transcript) or (None, None, None) if not enough data.
        """
        if not self.is_running or self.audio_buffer.empty():
            return None, None, None

        # Get all available data from buffer to form a segment
        # This strategy processes chunks when available.
        # More sophisticated VAD (Voice Activity Detection) could be added.
        audio_segment = []
        while not self.audio_buffer.empty():
            audio_segment.extend(self.audio_buffer.get())
        
        if not audio_segment:
            return None, None, None
            
        waveform_np = np.array(audio_segment).flatten()
        
        # Ensure enough data for a meaningful prediction, e.g., at least 1 second
        min_samples_for_prediction = self.sample_rate * 1 # Example: 1 second
        if len(waveform_np) < min_samples_for_prediction:
            # Not enough audio yet, put it back? Or wait for more? For now, just return None.
            # print(f"Not enough audio for prediction: {len(waveform_np)} samples. Need {min_samples_for_prediction}")
            return None, None, None

        # Keep only the most recent `buffer_size` if too much audio accumulated (optional)
        # if len(waveform_np) > self.buffer_size:
        #     waveform_np = waveform_np[-self.buffer_size:]

        waveform_tensor = torch.from_numpy(waveform_np).float().unsqueeze(0).to(self.cfg.device)
        
        # Here, text is None because it's real-time. ASR must be used.
        return self._process_audio_and_text(waveform_tensor, text=None, use_asr=use_asr)

    def stop_audio_stream(self):
        """Stop the audio stream."""
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                print("Audio stream stopped and closed.")
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None
                self.is_running = False
        else:
            print("Audio stream was not running.")
        # Clear buffer
        while not self.audio_buffer.empty():
            self.audio_buffer.get() 