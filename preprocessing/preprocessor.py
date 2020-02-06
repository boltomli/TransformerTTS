import numpy as np

from preprocessing.tokenizer import CharTokenizer


class Preprocessor:
    
    def __init__(self,
                 mel_channels: int,
                 start_vec_val: float,
                 end_vec_val: float,
                 tokenizer: CharTokenizer,
                 lowercase=True,
                 clip_val=1e-5):
        self.start_vec = np.ones((1, mel_channels)) * start_vec_val
        self.end_vec = np.ones((1, mel_channels)) * end_vec_val
        self.tokenizer = tokenizer
        self.lowercase = lowercase
        self.clip_val = clip_val
    
    def __call__(self, sample, divisible_by=1):
        text, mel_path = sample[0], sample[1]
        mel = np.load(mel_path)
        return self.encode(text, mel, divisible_by=divisible_by)
    
    def encode(self, text, mel, divisible_by=1):
        if self.lowercase:
            text = text.lower()
        encoded_text = self.tokenizer.encode(text)
        extra_end = (divisible_by - ((len(encoded_text) + 2) % divisible_by)) % divisible_by
        encoded_text = [self.tokenizer.start_token_index] + encoded_text + [self.tokenizer.end_token_index] + [
            0] * extra_end
        norm_mel = np.log(mel.clip(1e-5))
        norm_mel_len = norm_mel.shape[-2]
        extra_end = (divisible_by - ((norm_mel.shape[-2] + 2) % divisible_by)) % divisible_by
        divisibility_pads = np.zeros(self.end_vec.shape)
        norm_mel = np.concatenate([self.start_vec, norm_mel, self.end_vec, np.tile(divisibility_pads, (extra_end, 1))],
                                  axis=0)
        stop_probs = np.ones((norm_mel.shape[0]))
        stop_probs[norm_mel_len:] = 2
        return norm_mel, encoded_text, stop_probs