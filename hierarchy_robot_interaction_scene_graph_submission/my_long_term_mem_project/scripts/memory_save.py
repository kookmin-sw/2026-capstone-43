import json
import math

def read_json_file(file_path):
    # JSON 파일을 읽어서 데이터를 반환하는 함수
    with open(file_path, 'r') as file:
        return json.load(file)
    
def calculate_distance(position1, position2):
    # 두 위치 간의 유클리드 거리를 계산하는 함수
    return math.sqrt((position1['x'] - position2['x']) ** 2 + 
                     (position1['y'] - position2['y']) ** 2 +
                     (position1['z'] - position2['z']) ** 2)

def compare_objects_location(objects_locations1, objects_locations2, output_file, threshold=0.3, max_objects=100):
    # 두 JSON 파일에서 물체의 위치가 유의미하게 변경되었는지 비교하고, 일치하지 않는 물체 정보를 새 파일로 출력하는 함수
    data1 = read_json_file(objects_locations1) # objects_locations1 파일에서 데이터 읽기
    data2 = read_json_file(objects_locations2) # objects_locations2 파일에서 데이터 읽기

    objects_data1 = {item['objectId']: item for item in data1} # objects_locations1 데이터(value)를 objectId를 키로 하는 딕셔너리로 변환
    objects_data2 = {item['objectId']: item for item in data2} # objects_locations2 데이터(value)를 objectId를 키로 하는 딕셔너리로 변환

    differences = [] # 위치가 변경된 물체 정보를 저장할 리스트

    # 모든 물체를 순회하면서 위치 정보를 비교
    for object_id in objects_data1:
        if object_id in objects_data2:
            # 위치 정보 추출
            position1 = objects_data1[object_id]['position'] 
            position2 = objects_data2[object_id]['position']
            # 위치 간의 거리가 threshold보다 큰 경우, 위치가 변경된 것으로 간주하여 differences 리스트에 추가
            if calculate_distance(position1, position2) > threshold:
                differences.append(objects_data2[object_id])

    # output_file에서 기존 데이터를 읽어서 merged_data에 differences와 합치기
    try:
        with open(output_file, 'r') as file:
            output_data = json.load(file)
    except FileNotFoundError:
        output_data = [] # output_file이 존재하지 않는 경우, 빈 리스트로 초기화
    merged_data = output_data + differences # 기존 데이터와 위치가 변경된 물체 정보를 합치기

    unique_objects = {} # objectId를 키로 하는 딕셔너리를 사용하여 중복된 물체 제거 (최신 정보 유지)

    for obj in merged_data:
        obj_id = obj['objectId']
        unique_objects[obj_id] = obj # 최신 정보로 덮어쓰기 (objectId가 동일한 경우)
    
    # unique_objects의 개수가 max_objects보다 큰 경우, timestamp를 기준으로 내림차순 정렬하여 최신 max_objects개만 유지
    # 근데 사실 timestamp 정보가 objects_location1, 2에 포함되어 있지 않음;;
    # 이건 다음에 수정하던가 해야 할 듯
    if len(unique_objects) > max_objects:
        sorted_objects = sorted(unique_objects.items(), key=lambda x: x[1]['timestamp'], reverse=True)
        unique_objects = dict(sorted_objects[:max_objects])
    
    with open(output_file, 'w') as file:
        json.dump(list(unique_objects.values()), file, indent=4)