import os
from sklearn.metrics.pairwise import pairwise_distances_argmin,cosine_similarity

from chatterbot import ChatBot
from chatterbot.trainers import ChatterBotCorpusTrainer
import pandas as pd
from utils import *


def rank_candidates(question, candidates, dim=50):
    """
        question: a string
        candidates: a list of strings (candidates) which we want to rank
        embeddings: some embeddings
        dim: dimension of the current embeddings
        
        result: a list of pairs (initial position in the list, question)
    """
    
    ######################################
    ######### YOUR CODE HERE #############
    ######################################
    
    output = []
    for idx,cand in enumerate(candidates):
      
      sim = cosine_similarity(cand.ravel().reshape((1,-1)),question.ravel().reshape((1,-1)))[0][0]
      temp = [idx,cand,sim]
      output.append(temp)
    

    output = sorted(output,key=lambda x: x[2],reverse=True)
    # output = [(i[0],i[1]) for i in output]
#     print(output)
    return output[0][0]

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings,_ = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.embeddings_dim = 50
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        """ Returns id of the most similar thread for the question.
            The search is performed across the threads with a given tag.
        """
        print("@@@@@@@@")
        print(question)
        print(tag_name)
        print(type(tag_name))
        print("@@@@@@@@")
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)

        # HINT: you have already implemented a similar routine in the 3rd assignment.
        
        # question_vec =question_to_vec(question, embeddings=self.word_embeddings, dim=50)
        # best_thread = rank_candidates(question_vec, thread_embeddings, dim=50)
        question = text_prepare(question)
        question_vec = question_to_vec(question, self.word_embeddings, 50).reshape(1,-1)
        print('---')
        print("vecs:")
        print(question_vec[:,:5])
        print('~~')
        print(thread_embeddings[:3,:5])
        print('---')

        best_thread = pairwise_distances_argmin(question_vec, thread_embeddings, metric='cosine')[0]
        print(best_thread)
        print(thread_ids[best_thread])
        return thread_ids[best_thread]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'I think its about %s\nThis thread might help you: https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

        self.create_chitchat_bot()
        
    def create_chitchat_bot(self):
        """Initializes self.chitchat_bot with some conversational model."""

        # Hint: you might want to create and train chatterbot.ChatBot here.
        # It could be done by creating ChatBot with the *trainer* parameter equals 
        # "chatterbot.trainers.ChatterBotCorpusTrainer"
        # and then calling *train* function with "chatterbot.corpus.english" param
        
        ########################
        #### YOUR CODE HERE ####
        ########################
        self.chatbot = ChatBot("Awesome_bot")
        trainer = ChatterBotCorpusTrainer(self.chatbot)
        trainer.train('chatterbot.corpus.english')
        ###



       
    def generate_answer(self, question):
        """Combines stackoverflow and chitchat parts using intent recognition."""

        # Recognize intent of the question using `intent_recognizer`.
        # Don't forget to prepare question and calculate features for the question.
        
        prepared_question = text_prepare(question)
        print(prepared_question)
        features = self.tfidf_vectorizer.transform(pd.Series(prepared_question))
        intent = self.intent_recognizer.predict(features)

        # Chit-chat part:   
        if intent == 'dialogue':
            # Pass question to chitchat_bot to generate a response.       
            response = self.chatbot.get_response(question)
            return response
        
        # Goal-oriented part:
        else:        
            # Pass features to tag_classifier to get predictions.
            tag = self.tag_classifier.predict(features)
            
            # Pass prepared_question to thread_ranker to get predictions.
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag[0])
           
            return self.ANSWER_TEMPLATE % (tag, thread_id)

