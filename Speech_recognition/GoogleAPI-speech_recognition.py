import speech_recognition as sr
r = sr.Recognizer()
predict_text = []

for i in range(1, 1394):
    audio_file = '~/dl_project/data/'+str(i)+'.wav'
    with sr.AudioFile(audio_file) as source:
        audio = r.record(source)
    try:
        predict_text.append(r.recognize_google(audio))
    except sr.UnknownValueError:
        predict_text.append(" ")

try:
    print("Google Speech Recognition thinks you said " + r.recognize_google(audio))
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))
