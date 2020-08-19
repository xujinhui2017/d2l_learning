import sys
import time


def run(original_file: str, result_filename: str):
    stud_encode_dict = dict()
    content_encode_dict = dict()
    stats = dict()
    stud_current_encode, content_current_encode = [1, 1]
    with open(original_file, "r", encoding="utf-8") as txt_file:
        for line in txt_file:
            student_id, content_type, content_id, status, raw_time = line.strip().split("\t")
            if "Click" in status:
                rank = 1
            else:
                rank = 0
            if student_id not in stats:
                stats[student_id] = [0, 0]
            stats[student_id][rank] += 1
    
    with open(original_file, "r", encoding="utf-8") as txt_file, open(result_filename, "w", encoding="utf-8") as result_file:
        for line in txt_file:
            student_id, content_type, content_id, status, raw_time = line.strip().split("\t")
            if stats[student_id][0] < 20 or stats[student_id][1] < 10:
                continue
            if student_id not in stud_encode_dict:
                stud_encode_dict[student_id] = stud_current_encode
                stud_current_encode += 1
            if content_id not in content_encode_dict:
                content_encode_dict[content_id] = content_current_encode
                content_current_encode += 1
            if "Click" in status:
                rank = 1
            else:
                rank = 0
            
            time_array = time.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
            time_stamp = int(time.mktime(time_array))
            result_file.write(write_format(target_list=[stud_encode_dict[student_id], content_encode_dict[content_id],
                                                        rank, time_stamp]))
    return stud_encode_dict, content_encode_dict


def write_format(target_list: list):
    return "\t".join([str(i) for i in target_list]) + "\n"
    

def write_encode_file(filename: str, target_dict: dict):
    with open(filename, "w", encoding="utf-8") as txt_file:
        for local_id in target_dict:
            encode_value = target_dict[local_id]
            txt_file.write(write_format(target_list=[local_id, encode_value]))


def read_encode_file(filename: str):
    result = dict()
    with open(filename, "r", encoding="utf-8") as txt_file:
        for line in txt_file:
            target_id, encoding = line.strip().split("\t")
            result[target_id] = encoding
    return result


def test_run(original_file: str, result_filename: str, stud_encode_dict: dict, content_encode_dict: dict):
    with open(original_file, "r", encoding="utf-8") as txt_file, open(result_filename, "w", encoding="utf-8") as result_file:
        for line in txt_file:
            student_id, content_type, content_id, status, raw_time = line.strip().split("\t")
            if student_id not in stud_encode_dict or content_id not in content_encode_dict:
                continue
            else:
                stud_encoding = stud_encode_dict[student_id]
                content_encoding = content_encode_dict[content_id]
            if "Click" in status:
                rank = 1
            else:
                rank = 0
            time_array = time.strptime(raw_time, "%Y-%m-%d %H:%M:%S")
            time_stamp = int(time.mktime(time_array))
            result_file.write(write_format(target_list=[stud_encoding, content_encoding, rank, time_stamp]))
            
            
if __name__ == "__main__":
    original_filename, stud_encode_filename, content_encode_filename, model_input_filename, status_type = sys.argv[1:]
    if status_type == "train":
        stud_dict, content_dict = run(original_file=original_filename, result_filename=model_input_filename)
        write_encode_file(filename=stud_encode_filename, target_dict=stud_dict)
        write_encode_file(filename=content_encode_filename, target_dict=content_dict)
    else:
        stud_dict = read_encode_file(filename=stud_encode_filename)
        content_dict = read_encode_file(filename=content_encode_filename)
        test_run(original_file=original_filename, result_filename=model_input_filename,
                 stud_encode_dict=stud_dict, content_encode_dict=content_dict)
    pass
