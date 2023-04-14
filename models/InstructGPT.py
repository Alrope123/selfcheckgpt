from openai_model import OpenAIModel
import openai
import tqdm
import os
import time
import sys

class InstructGPT(OpenAIModel):
    def __init__(self, save_dir, key_path, save_interval=50, batch_size=16):
        super().__init__(save_dir, "instructGPT", key_path, save_interval)
        self.batch_size = batch_size

    def generate_response_from_prompt(self, prompts, passage_index, source_id):
        # Modify the format of the input
        if type(prompts) != list:
            prompts = [prompts]
        source_id = f"{source_id}_{passage_index}"
        
        passages = self.open_data("passages", source_id)
        cache = self.open_data("cache", source_id)
        
        ret_passages = {}
        # if already cached, used the cached value
        for prompt in prompts:
            if prompt in passages:
                ret_passages[prompt] = passages[prompt]
        
        # call API for not cached prompt
        prompts = [prompt for prompt in prompts if prompt not in passages]
        for i in tqdm(range(0, len(prompts), self.batch_size)):
            response = self.call_API(prompts[i: i+self.batch_size], temp=1.0)
            for j in range(len(response["choices"])):
                passages[prompts[i+j]] = response["choices"][j]["text"]
                ret_passages[prompts[i+j]] = passages[prompts[i+j]]
            cache.update(response)
            if i > 0 and i % self.save_interval == 0:
                self.save_data(cache, "cache", source_id)
                self.save_data(passages, "passages", source_id)

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
        # if already cached, used the cached value
        for queries_and_fact in queries_and_facts:
            if queries_and_fact in qaos:
                qaos[queries_and_fact] = qaos[queries_and_fact]
        
        # call API for not cached prompt
        queries_and_facts = [queries_and_fact for queries_and_fact in queries_and_facts if queries_and_fact not in qaos]
        prompts = [self.construct_qao_prompt(queries_and_fact[0], queries_and_fact[1], qa_type) 
                   for queries_and_fact in queries_and_facts]
        for i in tqdm(range(0, len(prompts), self.batch_size)):
            response = self.call_API(prompts[i: i+self.batch_size], temp=1.0)
            for j in range(len(response["choices"])):
                qaos[queries_and_facts[i+j]] = response["choices"][j]["text"]
                ret_qaos[queries_and_facts[i+j]] = qaos[queries_and_facts[i+j]]
            cache.update(response)
            if i > 0 and i % self.save_interval == 0:
                self.save_data(cache, "cache", source_id)
                self.save_data(qaos, "qaos", source_id)
        
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
        
        prompts = []
        ret_answers = {}
        for question, passage in zip(questions, passages):
            key = (question, source_id)
            if key in answers:
                ret_answers[key] = answers[key]
            else:
                prompts.append(self.construct_answering_prompt(question, context=passage))
            messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
            for i, message in tqdm(enumerate(messages)):
                response = self.call_API(message)
                answers[key] = response["choices"][0]["message"]["content"]
                cache.update(response)
                if i > 0 and i % self.save_interval == 0:
                    self.save_data(cache, "cache", source_id)
                    self.save_data(answers, "answers", source_id)
            
            ret_answers[key] = answers[key]

        self.save_data(cache, "cache", source_id)
        self.save_data(answers, "answers", source_id)

        ret_answers = {}
        # if already cached, used the cached value
        for question, passage in zip(queries_and_facts):
            key = (question, source_id)
            if key in answers:
                ret_answers[key] = answers[key]
        
        # call API for not cached prompt
        questions_and_passages = [(question, passage) for question in questions if (question, source_id) not in answers]
        prompts = [self.construct_answering_prompt(questions_and_passage[0], context=questions_and_passage[1]) 
                   for questions_and_passage in questions_and_passages]
        for i in tqdm(range(0, len(prompts), self.batch_size)):
            response = self.call_API(prompts[i: i+self.batch_size], temp=1.0)
            for j in range(len(response["choices"])):
                answers[(questions_and_passages[i+j][0], source_id)] = response["choices"][j]["text"]
                ret_answers[(questions_and_passages[i+j][0], source_id)] = answers[(questions_and_passages[i+j][0], source_id)]
            cache.update(response)
            if i > 0 and i % self.save_interval == 0:
                self.save_data(cache, "cache", source_id)
                self.save_data(answers, "answers", source_id)
        
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