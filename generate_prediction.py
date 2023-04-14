import os
import json
import numpy as np
import argparse
from models.ChatGPT import ChatGPT
from models.InstructGPT import InstructGPT
from tqdm import tqdm
from nltk.tokenize import sent_tokenize

            
def predict_from_score(facts, scores):
    assert type(facts) == type(scores)
    is_list = type(facts) == list
    if not is_list:
        facts = [facts]
        scores = [scores]
    
    # Get predictions based on score
    predictions = []
    for fact, score in zip(fact, scores):
        if score > 0.75:
            predictions.append('IR')
        elif score > 0.25:
            predictions.append('NS')
        else:
            predictions.append('S')
    return predictions if is_list else predictions[0]


def initialize_model(name, save_dir, key_path=None, batch_size=16):
    if name == "ChatGPT":
        return ChatGPT(save_dir, key_path)
    elif name == "InstructGPT":
        return InstructGPT(save_dir, key_path, batch_size=batch_size)
    else:
        raise NotImplementedError()


def score_facts(data, qa_type, question_type, qao_model, level, answer_model, source_id, use_context, save_dir, 
                passage_model=None, passage_num=None, key_path=None, source_id_retrieval=None):
    ##################################### Setup ###################################################################
    is_list = type(data) == list
    if not is_list:
        data = [data]
    facts = [dp['fact'] for dp in data]
    queries = [dp['query'] for dp in data]
    retrieval_passages = [dp['retrieval_passages'] for dp in data]
    original_outputs = [dp['original_output'] for dp in data]
    
    ################################### Get question, answer, and option ############################################
        
    # if it's fact verfication, no need to use model
    if question_type == "fv":
        query_and_fact_to_qa = ChatGPT.generate_fv_from_fact(zip(queries, facts), level, source_id, save_dir)
        qaos = [query_and_fact_to_qa[query, fact] for query, fact in zip(queries, facts)]
        for dp in data:
            dp["qaos"] = query_and_fact_to_qa[dp['query'], dp['fact']]
    elif question_type == "qa": # need to use qa model to generate
        assert qao_model
        model = initialize_model(qao_model, save_dir, key_path)
        query_and_fact_to_qa = model.generate_qao_from_fact(zip(queries, facts), qa_type, level, source_id)
        qaos = [query_and_fact_to_qa[query, fact] for query, fact in zip(queries, facts)]
        for dp in data:
            dp["qaos"] = query_and_fact_to_qa[dp['query'], dp['fact']]

    ################################### Get context passages ############################################
    passages = None
    if use_context:
        passages = []
        # has to provide original outputs to compare
        assert passage_num and original_outputs
        # if we are provided with external passages, no need to generate ourselves
        if retrieval_passages:
            # each facts should have passage_num contexts
            assert source_id_retrieval and len(retrieval_passages[0]) == passage_num
            passages = retrieval_passages
        else: # have to generate new passages
            assert passage_model and queries
            # initalize the model
            model = initialize_model(passage_model, save_dir, key_path)
            # generate passage_num different passages
            for i in range(len(passage_num)):
                prompt_to_passages = model.generate_response_from_prompt(list(set(queries)), i, source_id)
                passages.append(prompt_to_passages[prompt] for prompt in queries)
        # for dp in data:
        #     if 'passages' not in dp:
        #         dp['passages'] = []
        #     dp['passages'].append(prompt_to_passages[dp['prompt']])


    ################################### Answer the questions ###############################################
    assert answer_model
    # initalize the model
    model = initialize_model(answer_model, save_dir, key_path)
    if not use_context: # if no context
        questions = [(question['questions'], None) for question_set in qaos for question in question_set]
        question_to_answer = model.answer_question(questions=questions, source_id_question=source_id, source_id_passage="none")
        for dp in data:
            for question_set in dp['qaos']:
                for question in question_set:
                    question['gold_answer'] = question['answer']
                    question['test_answers'] = [question_to_answer[question['question']]]
    else: # if use context
        # get answers from the original output
        questions_and_passages = [(question, original_outputs[i]) for i, question_set in enumerate(qaos) for question in question_set]
        question_to_answer = model.answer_question(questions=[questions_and_passage[0] for questions_and_passage in questions_and_passages], 
                                                        source_id_question=source_id,
                                                        passages=[questions_and_passage[1] for questions_and_passage in questions_and_passages],
                                                        source_id_passage='original')
        for dp in data:
            for question in dp['qaos']:
                question['gold_answer'] = question_to_answer[question['question']]
        # get answers from passages
        for i in range(len(passage_num)):
            questions_and_passages = [(question, passages[i][j]) for j, question_set in enumerate(qaos) for question in question_set]
            question_to_answer = model.answer_question(questions=[questions_and_passage[0] for questions_and_passage in questions_and_passages], 
                                                        source_id_question=source_id,
                                                        passages=[questions_and_passage[1] for questions_and_passage in questions_and_passages],
                                                        source_id_passage=source_id_retrieval if retrieval_passages else source_id)
            for dp in data:
                for question in dp['qaos']:
                    if not 'test_answers' in question:
                        question['test_answers'] = [] 
                    question['test_answers'].append(question_to_answer[question['question']])

    ################################### Score the facts ###############################################
    for dp in data:
        scores = []
        for question in dp['qaos']:
            scores.extend([model.is_answer_correct(question['gold'], test_answer) for test_answer in question['test_answers']])
        dp['score'] = 1 - sum(scores) / len(scores)
    
    return [dp['score'] for dp in data] if is_list else data[0]['score']


def main(args):
    # config data paths
    source_id = f"{args.type}_{args.model}_v0"
    data_filename = f"{source_id}_v0.jsonl"
    data_path = os.path.join(args.data_dir, data_filename)

    retrieval_passages = None
    if args.retrieval_passages_dir:
        source_id_retrieval = os.path.basename(args.retrieval_passages_dir).split('.')[0]
        with open(args.retrieval_passages_dir, 'r') as f:
            retrieval_passages = json.load(f)
    
    prediction_dict = {}
    with open(data_path, "r") as f:
        for i, line in enumerate(f):
            dp = json.loads(line)
            promptId = dp["promptId"]
            response = dp["response"]

            if response is not None:

                fact_data = response["fact_data"]
                predictions = []

                '''
                predictions is a list of dicts with two keys
                - `sent_prediction`: prediction for the sentence
                - `predictions`: prediction for each fact. `None` if the gold
                sent_prediction is IR
                '''

                for sent_idx, sent_dp in enumerate(fact_data):
                    # predict atomic facts
                    sent_prediction = None
                    atom_predictions = None
                    if args.method in ["atom", "sent_and_atom", "sent_from_atom"]:
                        data = [{
                            "fact": atom,
                            "query": response['prompt'],
                            "retrieval_passages": retrieval_passages[i],
                            "original_output": sent_dp["orig_text"],
                        } for atom in sent_dp["original_facts"]]
                        atoms_scores = score_facts(data, args.qa_type, args.question_type, args.qao_model, args.level, args.answer_model,
                                                source_id, args.use_context, args.save_dir, passage_model=args.passage_model,
                                                passage_num=args.passage_num, key_path=args.key_path, source_id_retrieval=source_id_retrieval)
                        atom_predictions = predict_from_score(sent_dp["original_facts"], atoms_scores)
                    # predict sentences
                    if args.method in ["sent", "sent_and_atom"]:
                        data = {
                            "fact": sent_dp["orig_sentence"],
                            "query": response['prompt'],
                            "retrieval_passages": retrieval_passages[i],
                            "original_output": sent_dp["orig_text"],
                        }
                        sent_score = score_facts(data, args.qa_type, args.question_type, args.qao_model, args.level, args.answer_model,
                                                source_id, args.use_context, args.save_dir, passage_model=args.passage_model,
                                                passage_num=args.passage_num, key_path=args.key_path, source_id_retrieval=source_id_retrieval)
                        sent_prediction = predict_from_score(sent_dp["orig_sentence"], sent_score)
                    # predict sent from atom
                    if args.method == "sent_from_atom":
                        score = sum(atoms_scores) / len(atoms_scores)
                        if score > 0.75:
                            sent_prediction = 'IR'
                        elif score > 0.25:
                            sent_prediction = 'NS'
                        else:
                            sent_prediction = 'S'

                    predictions.append({
                        "sent_prediction": sent_prediction,
                        "predictions": atom_predictions
                    })

                prediction_dict[promptId] = predictions


    with open(os.path.join(args.data_dir, f"{source_id}_{args.qa_type}_{args.question_type}_{args.qao_model}_\
                           {args.level}_{args.answer_model}_{args.use_context}_{args.passage_model}_{args.passage_num}_\
                           {source_id_retrieval}_predictions.json"), "w") as f:
        json.dump(prediction_dict, f)


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--save_dir", type=str, default="./")
    parser.add_argument("--type", type=str, default="bio")
    parser.add_argument("--method", type=str, default="sent", options=["sent", "atom", "sent_and_atom", "sent_from_atom"])
    parser.add_argument("--qa_type", type=str, default="ff", options=["ff", "mc"])
    parser.add_argument("--question_type", type=str, default="qa", options=["qa", ""])
    parser.add_argument("--qao_model", type=str, default="ChatGPT", options=["ChatGPT", "InstructGPT"])
    parser.add_argument("--answer_model", type=str, default="ChatGPT", options=["ChatGPT", "InstructGPT"])
    parser.add_argument("--use_context", default=False, action="store_true")
    parser.add_argument("--retrieval_passages_dir", type=str, default=None)
    parser.add_argument("--passage_model", type=str, default="ChatGPT", options=["ChatGPT", "InstructGPT"])
    parser.add_argument("--passage_num", type=str, default="5")
    
    args = parser.parse_args()

    main(args)