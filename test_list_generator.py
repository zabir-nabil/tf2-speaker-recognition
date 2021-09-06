import glob
import random
root_path = "/Audio/voxceleb1/wav"
f = open("test/voxceleb1_veri_test.txt", "w")

data_dict = {}
for sub in glob.glob(f"{root_path}/*/*/*.wav"):
    file_path = sub.replace(root_path + "/", "")
    # print(file_path)
    sub_id = file_path.split("/")[0]
    # print(sub_id)
    try:
        data_dict[sub_id].append(file_path)
    except:
        data_dict[sub_id] = []
        data_dict[sub_id].append(file_path)
    
    #if len(data_dict) >= 20:
    #    break
print(len(data_dict))
print(sum(len(data_dict[k]) for k in data_dict))

tot_cnt = 0
# generate pairs
for k in data_dict:
    for m in data_dict:
        if k == m:
            for i in range(len(data_dict[k])):
                for j in range(len(data_dict[m])):
                    if data_dict[k][i] != data_dict[m][j]:
                        if random.randint(1,6) == 6:
                            f.write(f"1 {data_dict[k][i]} {data_dict[m][j]}\n")
                            tot_cnt += 1
                            if tot_cnt >= 1000:
                                break
                if tot_cnt >= 1000:
                    break

    
        else:
            for i in range(len(data_dict[k])):
                for j in range(len(data_dict[m])):
                    if random.randint(1,6) == 6:
                        f.write(f"0 {data_dict[k][i]} {data_dict[m][j]}\n")
                        tot_cnt += 1
                        if tot_cnt >= 2000:
                            break
                if tot_cnt >= 2000:
                    break
        
        if tot_cnt >= 2000:
                break

f.close()




