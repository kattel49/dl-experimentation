# import whisper

# model = whisper.load_model('base').to('cuda')

# print(model.transcribe('r1.flac'))

import whisper
model = whisper.load_model('base').to('cuda')

audio = whisper.load_audio('r3.flac')
audio = whisper.pad_or_trim(audio)

#make log-mel spectrogram
mel = whisper.log_mel_spectrogram(audio).to('cuda')

_, probs = model.detect_language(mel)
print("probability: ", probs)
# decode the audio
options = whisper.DecodingOptions(task='translate')
result = whisper.decode(model, mel, options)
print(result)