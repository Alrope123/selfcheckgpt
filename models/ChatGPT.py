from openai_model import OpenAIModel
import openai
import tqdm
import os
import time
import sys

class ChatGPT(OpenAIModel):
    def __init__(self, save_dir, key_path, save_interval=50):
        super().__init__(save_dir, "ChatGPT", key_path, save_interval)

    def generate_response_from_prompt(self, prompts, passage_index, source_id):
        # Modify the format of the input
        if type(prompts) != list:
            prompts = [prompts]
        source_id = f"{source_id}_{passage_index}"
        
        passages = self.open_data("passages", source_id)
        cache = self.open_data("cache", source_id)
        messages = [(prompt, [{"role": "system", "content": "You are a helpful and knowledgeable assistant that answers qaos truthfully."}, 
                {"role": "user", "content": prompt}]) for prompt in prompts]
        
        ret_passages = {}
        for i, (prompt, message) in tqdm(enumerate(messages)):
            if prompt not in passages:
                response = self.call_API(message, temp=1.0)
                passages[prompt] = response["choices"][0]["message"]["content"]
                cache.update(response)
                if i > 0 and i % self.save_interval == 0:
                    self.save_data(cache, "cache", source_id)
                    self.save_data(passages, "passages", source_id)
            ret_passages[prompt] = passages[prompt]

        self.save_data(cache, "cache", source_id)
        self.save_data(passages, "passages", source_id)
        return ret_passages


    def generate_qao_from_fact(self, queries_and_facts, qa_type, level, source_id):
        # Modify the format of the input
        if type(queries_and_facts) != list:
            queries_and_facts = [queries_and_facts]
        source_id = f"{source_id}_{level}_{qa_type}"

        # Get question cache
        qaos = self.open_data("qaos", source_id)
        cache = self.open_data("cache", source_id)

        ret_qaos = {}
        for i, queries_and_fact in tqdm(enumerate(queries_and_facts)):
            if queries_and_fact not in qaos:
                query = queries_and_fact[0]
                fact = queries_and_fact[1]

                # Construct the prompt send to ChatGPT
                message = [{"role": "user", "content": self.construct_qao_prompt(query, fact, qa_type)}]
                # Call API
                response = self.call_API(message)
                # Get the question answer pairs from the call
                output = response["choices"][0]["message"]["content"]
                # Update the outputfiles
                qaos[fact] = self.extract_qao(output, qa_type)
                cache.update(response)
                if i > 0 and i % self.save_interval == 0:
                    self.save_data(cache, "cache", source_id)
                    self.save_data(qaos, "qaos", source_id)

            ret_qaos[queries_and_fact] = queries_and_fact[queries_and_fact]
        
        self.save_data(cache, "cache", source_id)
        self.save_data(qaos, "qaos", source_id)
        return ret_qaos


    def answer_question(self, questions, source_id_question, source_id_passage, passages=None):
        if passages:
            assert type(questions) == type(passages) 
            if type(questions) != list:
                queries_and_facts = [queries_and_facts]
                passages = [passages]
            assert len(question) == len(passages)
            source_id = f"{source_id_question}_{source_id_passage}"
        else:
            if type(questions) != list:
                queries_and_facts = [queries_and_facts]
            source_id = f"{source_id_question}_cb"

        # get caches
        answers = self.open_data("answers", source_id)
        cache = self.open_data("cache", source_id)
        
        ret_answers = {}
        for i, (question, passage) in tqdm(enumerate(zip(questions, passages))):
            key = (question, source_id)
            if key not in answers:
                # Construct the prompt send to ChatGPT
                message = [{"role": "user", "content": self.construct_answering_prompt(question, context=passage)}]
                # Call API
                response = self.call_API(message)
                # Get the question answer pairs from the call
                answers[key] = response["choices"][0]["message"]["content"]
                cache.update(response)
                if i > 0 and i % self.save_interval == 0:
                    self.save_data(cache, "cache", source_id)
                    self.save_data(answers, "answers", source_id)
                
            ret_answers[key] = answers[key]

        self.save_data(cache, "cache", source_id)
        self.save_data(answers, "answers", source_id)
                
        return ret_answers


    def call_API(self, message, model_name="gpt-3.5-turbo", max_len=1024, temp=0.7):
        # call GPT-3 API until result is provided and then return it
        response = None
        received = False
        while not received:
            try:
                response = openai.ChatCompletion.create(model=model_name,
                                                    messages=message,
                                                    max_tokens=max_len,
                                                    temperature=temp)
                received = True
            except:
                error = sys.exc_info()[0]
                if error == openai.error.InvalidRequestError:
                    # something is wrong: e.g. prompt too long
                    print(f"InvalidRequestError\nPrompt passed in:\n\n{message}\n\n")
                    assert False
                print("API error:", error)
                time.sleep(1)
        return response