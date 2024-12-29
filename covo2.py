from pathlib import Path
from enum import Enum

import torch
import torchaudio

from cosyvoice.cli.cosyvoice import CosyVoice2


__all__ = ["Covo2Wrap"]


def depress_warnings():
    # depress warnings
    import logging
    logging.getLogger().setLevel(logging.ERROR)
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    import onnxruntime
    onnxruntime.set_default_logger_severity(3)


depress_warnings()


class InferMode(Enum):
    ZERO_SHOT = 0
    INSTRUCT = 1
    CROSS_LINGUAL = 2


class Covo2Wrap(object):

    # CosyVoice2 model wrapper.

    _config = {
        "model_directory": "models",
        "output_directory": "output",
        "speaker_directory": "speakers",
        "model_name": "CosyVoice2-0.5B",
        "target_sample_rate": 16_000,
    }

    def __init__(self, **kwargs):
        for key, value in self._config.items():
            setattr(self, f"_{key}", kwargs.get(key) or value)
        self._model_name = kwargs.get("model_name") or self._config["model_name"]
        self._target_sample_rate = kwargs.get("target_sample_rate") or self._config["target_sample_rate"]
        self._text = None
        self._prompt_speech = None
        self._infer_mode = None
        self._prompt_speech_text = None  # zero shot
        self._instruct=None  # with instruct
        self._infer_with_cross_lingual = False  # with cross lingual
        self._save_path = None
        self._speech_text_path = None
        self._speech_path = None
        self._model = self._load_model()

    def _load_model(self):
        model_path = Path(self._model_directory) / self._model_name
        if not model_path.exists():
            raise Exception("Model not found."
                            "You may use Covo2Wrap.download_model() for downloading.")
        return CosyVoice2(str(model_path),
                          load_jit=True,
                          load_onnx=False,
                          load_trt=False)

    def download_model(self):
        model_path = Path(self._model_directory) / self._model_name
        if not model_path.exists():
            from modelscope import snapshot_download
            snapshot_download('iic/CosyVoice2-0.5B', local_dir=model_path)
        else:
            print("Model already existed."
                  "Or, manually remove the folder before downloading.")

    def _load_speech_text(self, speech_path, speech_text_path=None):
        self._speech_path = Path(speech_path)
        if speech_text_path is None:
            filename = str(self._speech_path.stem) + ".txt"
            self._speech_text_path = speech_path.parent / filename
        if Path(self._speech_text_path).exists():
            with open(self._speech_text_path, 'r', encoding='utf-8') as file:
                self._prompt_speech_text = file.read()

    def clone_voice(self, speech_path, speech_text_path=None):
        """ Load speech and the corresponding text.
        :param speech_path: speech file path.
        :param speech_text_path: the corresponding text of the speech.
        Note: if speech text takes the same name with speech audio,
        it loads the text file (with .txt extension) automatically (if exists).
        :return: self
        """
        # load speech_text
        self._load_speech_text(speech_path, speech_text_path)
        if self._prompt_speech_text is not None:
            self._infer_mode = InferMode.ZERO_SHOT
        # load speech
        speech, sample_rate = torchaudio.load(speech_path)
        # merge multichannel into mono-channel, while keeping "array shape".
        speech = speech.mean(dim=0, keepdim=True)
        if sample_rate < self._target_sample_rate:
            raise ValueError(f"speech sample rate too low! "
                             f"Expected = {self._target_sample_rate}, got = {sample_rate}")
        if sample_rate > self._target_sample_rate:
            speech = torchaudio.transforms.Resample(orig_freq=sample_rate,
                                                    new_freq=self._target_sample_rate)(speech)
        self._prompt_speech = speech

        return self

    def load_speaker(self, speaker_name):
        speech_path = Path(self._speaker_directory) / f"{speaker_name}.wav"
        self.clone_voice(speech_path)
        return self

    def load_text(self, filepath):
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(filepath)
        with open(filepath, 'r', encoding='utf-8') as file:
            self._text = file.read()
        return self

    def _check_loader(self, text):
        if text is None and self._text is None:
            raise ValueError("Use load_text or input text")
        elif self._text and text:
            print("Note: text input ignored, as text file has been loaded.")
        elif self._text is None and text:
            self._text = text

    @staticmethod
    def _get_speech(infer_result):
        """ Join the pieces of infer_result into one.
        :param infer_result: generator
        :return: wav
        """
        # Format result
        combined_speech = None
        for item in infer_result:
            speech_piece = item["tts_speech"]
            if combined_speech is None:
                combined_speech = speech_piece
            else:
                combined_speech = torch.cat((combined_speech, speech_piece), dim=1)
        return combined_speech

    def _infer(self, **kwargs):
        if self._infer_mode == InferMode.ZERO_SHOT:
            return self._model.inference_zero_shot(self._text,
                                                   self._prompt_speech_text,
                                                   self._prompt_speech,
                                                   **kwargs)
        if self._infer_mode == InferMode.INSTRUCT:
            return self._model.inference_instruct2(self._text,
                                                   self._instruct,
                                                   self._prompt_speech,
                                                   **kwargs)
        if self._infer_mode == InferMode.CROSS_LINGUAL:
            return self._model.inference_cross_lingual(self._text,
                                                       self._prompt_speech,
                                                       **kwargs)
        if self._infer_mode is None:
            raise Exception("Infer mode is not known."
                            "Please give a instruct, using Covo2Wrap.with_instruct(instruct)")

    def _save(self, speech, save_to=None):
        if save_to is None:
            index = 0
            while True:
                filename = f"covo2_{self._infer_mode.name.lower()}_{index}.wav"
                save_to = Path(self._output_directory) / filename
                if not save_to.exists():
                    break
                index += 1
        self._save_path = save_to
        torchaudio.save(self._save_path, speech, self._model.sample_rate)

    def _print_info(self):
        print(f"\n[Processing] >> ...")
        print(f"|-- infer mode = {self._infer_mode.name}")
        print(f"|-- speaker = {self._speech_path.stem}")
        print(f"    |-- speech path = {self._speech_path}")
        print(f"    |-- speech text path = {self._speech_text_path}")
        show_text_max_size = 64
        dots = " ..." if len(self._text) > show_text_max_size else ""
        print(f"|-- text = {self._text[0:show_text_max_size]}{dots}")
        if self._instruct:
            print(f"|-- instruct = {self._instruct}")

    def to_speech(self, text=None, save_to=None, **kwargs):
        """ Speech generation.
        :param text: text to speech
        :param save_to: file save path

        For fine-grained control of text generation, predefined labels can be used.
        E.g. [laughter], [breath], <strong> </strong>, ...
        For more labels, please check cosyvoice/tokenizer/tokenizer.py#L248
        """
        self._check_loader(text)
        self._print_info()
        infer_result = self._infer(**kwargs)
        speech = self._get_speech(infer_result)
        # Save result
        self._save(speech, save_to)
        print(f"[Done] >> saved to '{self._save_path}'")
        self._reset()

    def _reset(self):
        self._text = None
        self._instruct = None
        self._infer_with_cross_lingual = False
        self._save_path = None
        self._speech_path = None
        self._speech_text_path = None

    def with_instruct(self, instruct):
        self._instruct = instruct
        self._infer_mode = InferMode.INSTRUCT
        return self

    def with_cross_lingual(self):
        self._infer_mode = InferMode.CROSS_LINGUAL
        return self