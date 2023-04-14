
from abc import ABC, abstractmethod
from basic_model import BasicModel
import os
import json
import openai

class OpenAIModel(BasicModel):
    def __init__(self, save_dir, model_id,  key_path, save_interval=50):
        super().__init__(save_dir, model_id)

        # load api key
        with open(key_path, 'r') as f:
            api_key = f.readline()
        # openai.organization = "academics uw"
        openai.api_key = api_key
        self.save_interval = save_interval
    
    @abstractmethod
    def call_API(self, message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7, **kwargs):
        pass
    
    def construct_qao_prompt(self, query, fact, level, qa_type):
        if level == "atom":
            instruction = "Generate the most relevant question-answer pair about the following fact.\n\n"
            instruction += "Query: Tell me a bio of James Taylor.\n"
            instruction += "Fact: James Taylor was born on March 12, 1948.\n"
            instruction += "Q: When was James Taylor born?\n"
            instruction += "A: March 12, 1948.\n"
            instruction += "\n"
            instruction += "Query: {}\n".format(query.strip())
            instruction += "Fact: {}\n".format(fact.strip())
        elif level == "sent":
            pass
        else:
            raise NotImplementedError()
        
        return instruction

    def extract_qao(output, level, qa_type):
        if level == "atom":
            lines = output.split("\n")
            question = None
            answer = None
            for line in lines:
                if line.startwith('Q:'):
                    assert not question
                    question = line[3:].strip()
                elif line.startwith('A:'):
                    assert not answer
                    answer = line[3:].strip()
            assert question and answer, options
            return {"question": question, "answer": answer, options: "options"}
        elif level == "sent":
            pass
        else:
            raise NotImplementedError()
    
    def construct_answering_prompt(self, question, context=None):
        if context:
            instruction = "Answer the question based on the given context.\n"
            instruction += "Context: {}\n\n".format(context)
        else:
            instruction = "Answer the question.\n"
        instruction += "Question: {}\n\n".format(question)
        instruction += "Answer: "
        return instruction


    @classmethod
    def is_answer_correct(cls, gold_answer, test_answer):
        return test_answer.strip().startwith(gold_answer.strip()) or gold_answer.strip() in test_answer.strip()