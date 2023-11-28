from flask import Flask, render_template, jsonify, request
from transformers import MarianMTModel, MarianTokenizer
#from google_trans_new import google_translator
from gtts import gTTS
from playsound import playsound
import os
import uuid

app = Flask(__name__, static_url_path='/static', static_folder='static')


#translator = google_translator()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/translate', methods=['POST'])
def translate():
    data = request.get_json()
    speech_text = data.get('speech_text', '')

    # Translate the speech text
    translated_text = translate_text(speech_text, source_lang='en', target_lang='fr')

    # Convert the translated text to speech
    voice = gTTS(translated_text, lang='fr')
    audio_filename = f"audio_{uuid.uuid4()}.mp3"
    voice.save(audio_filename)

    return jsonify({'translated_text': translated_text, 'audio_filename': audio_filename})

def translate_text(text, source_lang='en', target_lang='fr'):
    model_name = f'Helsinki-NLP/opus-mt-{source_lang}-{target_lang}'
    model = MarianMTModel.from_pretrained(model_name)
    tokenizer = MarianTokenizer.from_pretrained(model_name)

    inputs = tokenizer(text, return_tensors="pt", truncation=True)
    translation = model.generate(**inputs)
    translated_text = tokenizer.batch_decode(translation, skip_special_tokens=True)[0]

    return translated_text

if __name__ == '__main__':
    app.run(debug=True)
