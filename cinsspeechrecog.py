#!/usr/bin/env python3
# Requires PyAudio and PySpeech.
 
import speech_recognition as sr
import simpleaudio as sa
import playsound as ps 

# Record Audio
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Please wait. Calibrating microphone...")  
    # listen for 5 seconds and create the ambient noise energy level  
    r.adjust_for_ambient_noise(source, duration=5)   
    print("Say something!")
    audio = r.listen(source)
 
# Speech recognition using Google Speech Recognition
try:
    # for testing purposes, we're just using the default API key
    # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
    # instead of `r.recognize_google(audio)`
    print("You said: " + r.recognize_google(audio)) #(audio,key="None",language="en-US"))
    audwav = audio.get_wav_data()
    
except sr.UnknownValueError:
    print("Google Speech Recognition could not understand audio")
except sr.RequestError as e:
    print("Could not request results from Google Speech Recognition service; {0}".format(e))


# write audio to a WAV file
with open("cins_microphone.wav", "wb") as f:
    f.write(audio.get_wav_data())
    

## Microsoft Identification from Speech
#import cognitive_sr as mscog
#
#speech_identification =mscog.SpeechIdentification()
#
#result = speech_identification.identify_profile(profile_ids,wav_data)
#print('Identified wav as profile: ', result['identifiedProfileId'])
#print('Confidence is: ', result['confidence'])
 