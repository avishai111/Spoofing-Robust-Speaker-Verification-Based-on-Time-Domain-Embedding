import csv
import numpy as np
import pandas as pd


class speaker_class:
    def __init__(self, speaker_id, enrollment_files,gender):
        self.speaker_id = speaker_id
        self.enrollment_files = enrollment_files
        self.gender = gender
        self.enrollment_embeddings = None
        self.indexes_big_embeddings_list = None
        
    def get_enrollment_files(self):
        return self.enrollment_files
    
    def get_speaker_id(self):
        return self.speaker_id
    
    def get_gender(self):
        return self.gender
    
    def get_enrollment_embeddings(self):
        return self.enrollment_embeddings
    
    def get_indexes_big_embeddings_list(self):
        return self.indexes_big_embeddings_list
        
    def __str__(self):
        return f"Speaker ID: {self.speaker_id}, Gender: {self.gender}, \n Enrollment_Files: {self.enrollment_files}"


class enrollment_file:
    def __init__(self, speakers_list):
        self.speakers_list = speakers_list
        
    def get_speaker(self,speaker_id):
        for speaker in self.speakers_list:
            if speaker.speaker_id == speaker_id:
                return speaker
        return None
    
    def get_all_enrollment_files(self):
        return [speaker.get_enrollment_files() for speaker in self.speakers_list]
    
    def get_all_enrollment_embeddings(self):
        return [speaker.get_enrollment_embeddings() for speaker in self.speakers_list]
    
    def get_all_speakers_id(self):
        return [speaker.get_speaker_id() for speaker in self.speakers_list]
    
    def get_all_gender(self):
        return [speaker.get_gender() for speaker in self.speakers_list]
    
    def get_all_indexes_big_embeddings_list(self):
        return [speaker.get_indexes_big_embeddings_list() for speaker in self.speakers_list]

    def __str__(self):
        return '\n'.join(str(speaker) for speaker in self.speakers_list)

    def __add__(self, other):
        if isinstance(other, enrollment_file) and isinstance(self, enrollment_file):
            return enrollment_file(self.speakers_list + other.speakers_list)
        else:
            return None