import pandas as pd

# To download dialogueText just go this link https://www.kaggle.com/rtatman/ubuntu-dialogue-corpus

filename_list = ['dialogueText']  # ['dialogueText_196','dialogueText_301']


def add_record(files_dict, filename, record):
    files_dict[filename].append(
        f'{record["dialogueID"]},{record["date"]},{record["from"]},{record["to"]},\"{record["text"]}\"\n')


def segregate_data_by_custom_field(name_field):
    df = pd.read_csv(f'../{filename_list[0]}.csv')
    files_dict = dict()

    for index, row in df.iterrows():
        filename = row[name_field]
        if not filename in files_dict:
            files_dict[filename] = []
        add_record(files_dict, filename, row)

    headers = 'dialogueID,date,from,to,text\n'
    for key in files_dict:
        lines = files_dict[key]
        f = open(f'./output/{key}.csv', 'w', encoding='utf-8')
        f.write(headers)
        f.writelines(lines)
        f.close()


segregate_data_by_custom_field('from')
