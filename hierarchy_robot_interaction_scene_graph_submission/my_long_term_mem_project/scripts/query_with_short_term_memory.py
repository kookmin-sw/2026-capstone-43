import json
import os
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

def load_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read().strip()

def extract_task(description):
    # : 위치 찾기
    colon_index = description.find(':')
    if colon_index != -1:
        # : 뒤부터 첫 번째 . 사이의 부분 추출
        task = description[colon_index + 1:].split('.')[0].strip()
        return task
    return None

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def save_to_file(file_path, content):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)


def cosine_scores_cpu(query_embedding, corpus_embeddings):
    """
    sentence-transformers 임베딩 device mismatch를 피하기 위해
    유사도 계산 전 두 텐서를 모두 CPU로 맞춘다.
    """
    q = query_embedding.detach().cpu()
    c = corpus_embeddings.detach().cpu()
    return util.pytorch_cos_sim(q, c)[0].cpu().numpy()

def update_memory_with_state(memory_file, analysis_file):
    # memory3.json 파일 읽기
    with open(memory_file, 'r', encoding="utf-8") as file:
        memory_data = json.load(file)

    # 분석 결과가 아직 없으면 업데이트를 건너뛴다.
    if (not os.path.exists(analysis_file)) or os.path.getsize(analysis_file) == 0:
        print(f"[query] skip state update: analysis file not found or empty -> {analysis_file}")
        return

    # analysis_results.json 파일 읽기
    with open(analysis_file, 'r', encoding="utf-8") as file:
        analysis_data = json.load(file)

    if not analysis_data:
        print(f"[query] skip state update: no analysis entries -> {analysis_file}")
        return

    # analysis_results.json의 마지막 데이터 읽어오기
    last_analysis_key = list(analysis_data.keys())[-1]
    state_data = {obj.lower(): state for obj, state in analysis_data[last_analysis_key].items()}

    # memory3.json 파일 내용 업데이트
    for item in memory_data:
        object_type = item.get('objectType', '').lower()
        if object_type in state_data:
            item['state'] = state_data[object_type]

    # 업데이트된 데이터를 memory3.json에 다시 쓰기
    with open(memory_file, 'w', encoding="utf-8") as file:
        json.dump(memory_data, file, ensure_ascii=False, indent=4)


# 경로 설정
analysis_file_path = '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/analysis_results.json'
short_term_memory_file_path = '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/prompts/short_term_memory.txt'

# memory3.json의 객체 데이터
memory_file_path = '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/memory/memory3.json'
example_file_path = '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/experience/experience.json'
examples_output_path = '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/prompts/examples.txt'

# memory3.json을 state 가지고 업데이트
update_memory_with_state(memory_file_path, analysis_file_path)

items = load_json(memory_file_path)

# planner 입력 파일은 항상 존재하게 보장
if not os.path.exists(short_term_memory_file_path):
    save_to_file(short_term_memory_file_path, "")

# instruction.txt 읽고 추출
instruction_file_path = '/home/yuchaehee/long_term_memory_project/my_long_term_mem_project/prompts/instruction.txt'
description = load_file(instruction_file_path)
extracted_task = extract_task(description)

# 추출한 instruction description 출력
if extracted_task:
    print(f"Extracted task: {extracted_task}")
else:
    print("No task found.")
    extracted_task = ""

# 추출한 instruction description similarity 계산
if extracted_task:
    item_texts = [item['objectType'] for item in items] # 아이템 이름 뽑아내기
    if not item_texts:
        print("[query] skip object similarity: no items in memory file.")
    else:
        # pre-trained transformer model
        model = SentenceTransformer('all-mpnet-base-v2')

        # item name embedding
        item_embeddings = model.encode(item_texts, convert_to_tensor=True)

        # instruction description embedding
        query = extracted_task
        query_embedding = model.encode(query, convert_to_tensor=True)

        # cosine similarity 계산으로 아이템 별 similarity (CPU 통일)
        cosine_scores = cosine_scores_cpu(query_embedding, item_embeddings)

        # 가장 유사한 항목 가져오기
        top_result_idx = np.argsort(cosine_scores)[::-1][0]
        top_result_item = items[top_result_idx]

        # 가장 유사한 항목 출력
        print("Top matching item:")
        print(f"Object Type: {top_result_item['objectType']}, Position: {top_result_item['position']}, Score: {cosine_scores[top_result_idx]:.4f}")

        # planner가 읽는 short_term_memory.txt 경로(prompts)에 저장
        object_type = top_result_item['objectType']
        position = top_result_item['position']
        formatted_content = f"{object_type} is at position ({position['x']:.2f}, {position['y']:.2f}, {position['z']:.2f})"
        save_to_file(short_term_memory_file_path, formatted_content)

        print(f"Top matching item has been saved to {short_term_memory_file_path}")


# experience
if extracted_task:
    example_data = load_json(example_file_path)
    tasks = [example['task'] for example in example_data]

    # pre-trained transformer model
    model = SentenceTransformer('all-mpnet-base-v2')

    # 예시 tasks 임베딩
    task_embeddings = model.encode(tasks, convert_to_tensor=True)

    # instruction description encoding
    query_embedding = model.encode(extracted_task, convert_to_tensor=True)

    # task별 similarity 구하기 (CPU 통일)
    cosine_scores = cosine_scores_cpu(query_embedding, task_embeddings)

    # 가장 유사한 3개 예시의 인덱스 뽑기
    top_k_indices = np.argsort(cosine_scores)[::-1][:3]

    # 유사한 예시의 decompositions 내역 가져오기
    top_decompositions = [example_data[idx]['decomposition'] for idx in top_k_indices]

    # 추출한 내용을 examples.txt에 저장
    with open(examples_output_path, 'w', encoding='utf-8') as file:
        for i, decomposition in enumerate(top_decompositions):
            file.write(f"Example {i+1} Decomposition:\n")
            file.write('\n'.join(decomposition))
            file.write('\n\n')

    print(f"Top 3 task decompositions have been saved to {examples_output_path}")
