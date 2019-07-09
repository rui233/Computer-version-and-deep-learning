import surprise

### 使用NMF
from surprise import NMF, evaluate
from surprise import Dataset

file_path = os.path.expanduser('./popular_music_suprise_format.txt')
# 指定文件格式
reader = Reader(line_format='user item rating timestamp', sep=',')
# 从文件读取数据
music_data = Dataset.load_from_file(file_path, reader=reader)
# 构建数据集和建模
algo = NMF()
trainset = music_data.build_full_trainset()
algo.train(trainset)

user_inner_id = 4
user_rating = trainset.ur[user_inner_id]
items = map(lambda x:x[0], user_rating)
for song in items:
    print(algo.predict(algo.trainset.to_raw_uid(user_inner_id), algo.trainset.to_raw_iid(song), r_ui=1), song_id_name_dic[algo.trainset.to_raw_iid(song)])