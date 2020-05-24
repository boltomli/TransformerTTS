import os
import re

prosody_pattern = re.compile(r'#\d')
with open(os.path.join('databaker', 'ProsodyLabeling', '000001-010000.txt'), encoding='utf-8') as f:
     with open(os.path.join('databaker', '000001-010000.txt'), 'w', encoding='utf-8') as w:
        for line in f:
            if not line.startswith('\t'):
                parts = line.strip().split('\t')
                wav_id = parts[0]
                text = ''.join(re.split(prosody_pattern, parts[1]))
                w.write('|'.join([wav_id, text, text]) + '\n')
