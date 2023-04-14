from abc import ABC, abstractmethod
import os
import json

class BasicModel(ABC):
    
    def __init__(self, save_dir, model_id):
        # intialize the save directories
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        self.save_dir = save_dir
        self.model_id = model_id
    
    @abstractmethod
    def generate_response_from_prompt(self, prompts, passage_index, source_id):
        pass
    
    @abstractmethod
    def generate_qao_from_fact(self, queries_and_facts, qa_type, level, source_id):
        pass

    @abstractmethod
    def answer_question(self, questions, source_id_question, source_id_passage, passages=None):
        pass
    

    def get_passages(self, passage_index, source_id):
        source_id = f"{source_id}_{passage_index}"
        return self.open_data("passages", source_id)
    

    def get_qaos(self, level, source_id):
        source_id = f"{source_id}_{level}"
        return self.open_data("qaos", source_id)

    def get_answers(self, source_id_question, source_id_retrieval):
        source_id = f"{source_id_question}_{source_id_retrieval}"
        return self.open_data("answers", source_id)


    def open_data(self, key, source_id):
        key_path = os.path.join(self.save_dir, key)
        if not os.path.exists(key_path):
            return {}
        file_path = os.path.join(key_path, f"{source_id}_{self.model_id}")
        if not os.path.exists(file_path):
            return {}
        with open(file_path, 'r') as f:
            return json.load(f)
    

    def save_data(self, data, key, source_id):
        key_path = os.path.join(self.save_dir, key)
        if not os.path.exists(key_path):
            os.mkdir(key_path)
        file_path = os.path.join(key_path, f"{source_id}_{self.model_id}")
        with open(file_path, 'w') as f:
            return json.dump(data, f)
        

    @classmethod
    def generate_fv_from_fact(cls, queries_and_facts, level, source_id, save_dir):
        if type(queries_and_facts) != list:
            queries_and_facts = [queries_and_facts]
        source_id = f"{source_id}_{level}"
        
        key_path = os.path.join(save_dir, "fvs")
        if not os.path.exists(key_path):
            os.mkdir(key_path)
            fvs = {}
        file_path = os.path.join(key_path, source_id)
        if not os.path.exists(file_path):
            fvs = {}
        else:
            with open(file_path, 'r') as f:
                fvs = json.load(f)
        
        ret_fvs = {}
        for i, queries_and_fact in enumerate(queries_and_facts):
            if queries_and_fact not in fvs:
                query = queries_and_fact[0]
                fact = queries_and_fact[1]
                # manually construct fact verification questions
                fvs[queries_and_fact] =  [{"question": "Is it true that {}?".format(fact), "answer": "Yes"}]
            ret_fvs[queries_and_fact] = queries_and_fact[queries_and_fact]

        with open(file_path, 'r') as f:
            fvs = json.load(f)
        return ret_fvs

    @classmethod
    def get_fvs(cls, level, source_id, save_dir):
        source_id = f"{source_id}_{level}"
        key_path = os.path.join(save_dir, "fvs")
        if not os.path.exists(key_path):
            return {}
        file_path = os.path.join(key_path, source_id)
        if not os.path.exists(file_path):
            return {}
        with open(file_path, 'r') as f:
            return json.load(f)