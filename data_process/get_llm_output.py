

import argparse
import os
import os.path as osp
import random
import time
from logging import getLogger
import openai
from utils import get_res_batch, load_json, intention_prompt, preference_prompt_1, preference_prompt_2, amazon18_dataset2fullname, write_json_file
import json



def get_intention_train(args, inters, item2feature, reviews, api_info):

    intention_train_output_file = os.path.join(args.root,"intention_train.json")


    # Suggest modifying the prompt based on different datasets
    prompt = intention_prompt
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    print(dataset_full_name)

    prompt_list = []

    inter_data = []

    for (user,item_list) in inters.items():
        user = int(user)
        item = int(item_list[-3])
        history = item_list[:-3]

        inter_data.append((user,item,history))

        review = reviews[str((user, item))]["review"]
        item_title = item2feature[str(item)]["title"]
        input_prompt = prompt.format(item_title=item_title,dataset_full_name=dataset_full_name,review=review)
        prompt_list.append(input_prompt)

    st = 0
    with open(intention_train_output_file, mode='a') as f:

        while st < len(prompt_list):
        # while st < 3:
            print(st)
            # if st < 25631:
            #     st += args.batchsize
            #     continue


            res = get_res_batch(args.model_name, prompt_list[st:st+args.batchsize], args.max_tokens, api_info)

            for i, answer in enumerate(res):
                user, item, history = inter_data[st+i]
                # print(answer)
                # print("=============")

                if answer == '':
                    print("answer null error")
                    answer = "I enjoy high-quality item."

                if answer.strip().count('\n') != 1:
                    if 'haracteristics:' in answer:
                        answer = answer.strip().split("The item's characteristics:")
                    else:
                        answer = answer.strip().split("The item's characteristic:")
                else:
                    answer = answer.strip().split('\n')

                if '' in answer:
                    answer.remove('')

                if len(answer) == 1:
                    print(answer)
                    user_preference = item_character = answer[0]
                elif len(answer) >= 3:
                    print(answer)
                    answer = answer[-1]
                    user_preference = item_character = answer
                else:
                    user_preference, item_character = answer

                if ':' in user_preference:
                    idx = user_preference.index(':')
                    user_preference = user_preference[idx+1:]
                user_preference = user_preference.strip().replace('}','')
                user_preference = user_preference.replace('\n','')

                if ':' in item_character:
                    idx = item_character.index(':')
                    item_character = item_character[idx+1:]
                item_character = item_character.strip().replace('}','')
                item_character = item_character.replace('\n','')


                dict = {"user":user, "item":item, "inters": history,
                        "user_related_intention":user_preference, "item_related_intention": item_character}

                json.dump(dict, f)
                f.write("\n")

            st += args.batchsize

    return intention_train_output_file


def get_intention_test(args, inters, item2feature, reviews, api_info):

    intention_test_output_file = os.path.join(args.root,"intention_test.json")

    # Suggest modifying the prompt based on different datasets
    prompt = intention_prompt
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    print(dataset_full_name)

    prompt_list = []

    inter_data = []

    for (user,item_list) in inters.items():
        user = int(user)
        item = int(item_list[-1])
        history = item_list[:-1]

        inter_data.append((user,item,history))

        review = reviews[str((user, item))]["review"]
        item_title = item2feature[str(item)]["title"]
        input_prompt = prompt.format(item_title=item_title,dataset_full_name=dataset_full_name,review=review)
        prompt_list.append(input_prompt)

    st = 0
    with open(intention_test_output_file, mode='a') as f:

        while st < len(prompt_list):
        # while st < 3:
            print(st)
            # if st < 4623:
            #     st += args.batchsize
            #     continue

            res = get_res_batch(args.model_name, prompt_list[st:st+args.batchsize], args.max_tokens, api_info)

            for i, answer in enumerate(res):
                user, item, history = inter_data[st+i]

                if answer == '':
                    print("answer null error")
                    answer = "I enjoy high-quality item."

                if answer.strip().count('\n') != 1:
                    if 'haracteristics:' in answer:
                        answer = answer.strip().split("The item's characteristics:")
                    else:
                        answer = answer.strip().split("The item's characteristic:")
                else:
                    answer = answer.strip().split('\n')

                if '' in answer:
                    answer.remove('')

                if len(answer) == 1:
                    print(answer)
                    user_preference = item_character = answer[0]
                elif len(answer) >= 3:
                    print(answer)
                    answer = answer[-1]
                    user_preference = item_character = answer
                else:
                    user_preference, item_character = answer

                if ':' in user_preference:
                    idx = user_preference.index(':')
                    user_preference = user_preference[idx+1:]
                user_preference = user_preference.strip().replace('}','')
                user_preference = user_preference.replace('\n','')

                if ':' in item_character:
                    idx = item_character.index(':')
                    item_character = item_character[idx+1:]
                item_character = item_character.strip().replace('}','')
                item_character = item_character.replace('\n','')


                dict = {"user":user, "item":item, "inters": history,
                        "user_related_intention":user_preference, "item_related_intention": item_character}

                json.dump(dict, f)
                f.write("\n")

            st += args.batchsize

    return intention_test_output_file




def get_user_preference(args, inters, item2feature, reviews, api_info):

    preference_output_file = os.path.join(args.root,"user_preference.json")


    # Suggest modifying the prompt based on different datasets
    prompt_1 = preference_prompt_1
    prompt_2 = preference_prompt_2


    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    dataset_full_name = dataset_full_name.replace("_", " ").lower()
    print(dataset_full_name)

    prompt_list_1 = []
    prompt_list_2 = []

    users = []

    for (user,item_list) in inters.items():
        users.append(user)
        history = item_list[:-3]
        item_titles = []
        for j, item in enumerate(history):
            item_titles.append(str(j+1) + '.' + item2feature[str(item)]["title"])
        if len(item_titles) > args.max_his_len:
            item_titles = item_titles[-args.max_his_len:]
        item_titles = ", ".join(item_titles)
        
        input_prompt_1 = prompt_1.format(dataset_full_name=dataset_full_name, item_titles=item_titles)
        input_prompt_2 = prompt_2.format(dataset_full_name=dataset_full_name, item_titles=item_titles)

        prompt_list_1.append(input_prompt_1)
        prompt_list_2.append(input_prompt_2)


    st = 0
    with open(preference_output_file, mode='a') as f:

        while st < len(prompt_list_1):
        # while st < 3:
            print(st)
            # if st < 22895:
            #     st += args.batchsize
            #     continue

            res_1 = get_res_batch(args.model_name, prompt_list_1[st:st + args.batchsize], args.max_tokens, api_info)
            res_2 = get_res_batch(args.model_name, prompt_list_2[st:st + args.batchsize], args.max_tokens, api_info)
            for i, answers in enumerate(zip(res_1, res_2)):

                user = users[st + i]

                answer_1, answer_2 = answers
                # print(answers)
                # print("=============")

                if answer_1 == '':
                    print("answer null error")
                    answer_1 = "I enjoy high-quality item."
                    
                if answer_2 == '':
                    print("answer null error")
                    answer_2 = "I enjoy high-quality item."

                if answer_2.strip().count('\n') != 1:
                    if 'references:' in answer_2:
                        answer_2 = answer_2.strip().split("Short-term preferences:")
                    else:
                        answer_2 = answer_2.strip().split("Short-term preference:")
                else:
                    answer_2 = answer_2.strip().split('\n')

                if '' in answer_2:
                    answer_2.remove('')

                if len(answer_2) == 1:
                    print(answer_2)
                    long_preference = short_preference = answer_2[0]
                elif len(answer_2) >= 3:
                    print(answer_2)
                    answer_2 = answer_2[-1]
                    long_preference = short_preference = answer_2
                else:
                    long_preference, short_preference = answer_2

                if ':' in long_preference:
                    idx = long_preference.index(':')
                    long_preference = long_preference[idx+1:]
                long_preference = long_preference.strip().replace('}','')
                long_preference = long_preference.replace('\n','')

                if ':' in short_preference:
                    idx = short_preference.index(':')
                    short_preference = short_preference[idx+1:]
                short_preference = short_preference.strip().replace('}','')
                short_preference = short_preference.replace('\n','')

                dict = {"user":user,"user_preference":[answer_1, long_preference, short_preference]}
                # print(dict)
                json.dump(dict, f)
                f.write("\n")

            st += args.batchsize

    return preference_output_file

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments', help='Instruments / Arts / Games')
    parser.add_argument('--root', type=str, default='')
    parser.add_argument('--api_info', type=str, default='./api_info.json')
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--max_tokens', type=int, default=512)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--max_his_len', type=int, default=20)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    args.root = os.path.join(args.root, args.dataset)

    api_info = load_json(args.api_info)
    openai.api_key = api_info["api_key_list"].pop()


    inter_path = os.path.join(args.root, f'{args.dataset}.inter.json')
    inters = load_json(inter_path)


    item2feature_path = os.path.join(args.root, f'{args.dataset}.item.json')
    item2feature = load_json(item2feature_path)

    reviews_path = os.path.join(args.root, f'{args.dataset}.review.json')
    reviews = load_json(reviews_path)

    intention_train_output_file = get_intention_train(args, inters, item2feature, reviews, api_info)
    intention_test_output_file = get_intention_test(args, inters, item2feature, reviews ,api_info)
    preference_output_file = get_user_preference(args, inters, item2feature, reviews, api_info)

    intention_train = {}
    intention_test = {}
    user_preference = {}

    with open(intention_train_output_file, "r") as f:
        for line in f:
            # print(line)
            content = json.loads(line)
            if content["user"] not in intention_train:
                intention_train[content["user"]] = {"item":content["item"],
                                                "inters":content["inters"],
                                                "querys":[ content["user_related_intention"], content["item_related_intention"] ]}


    with open(intention_test_output_file, "r") as f:
        for line in f:
            content = json.loads(line)
            if content["user"] not in intention_train:
                intention_test[content["user"]] = {"item":content["item"],
                                                "inters":content["inters"],
                                                "querys":[ content["user_related_intention"], content["item_related_intention"] ]}


    with open(preference_output_file, "r") as f:
        for line in f:
            content = json.loads(line)
            user_preference[content["user"]] = content["user_preference"]

    user_dict = {
        "user_explicit_preference": user_preference,
        "user_vague_intention": {"train": intention_train, "test": intention_test},
    }

    write_json_file(user_dict, os.path.join(args.root, f'{args.dataset}.user.json'))
