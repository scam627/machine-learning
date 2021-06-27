import pandas as pd

filename_list = ['ActionParsnip', 'bazhang', 'Dr_Willis', 'jrib', 'Pici']


def transform_files():
    for i in range(5):
        df = pd.read_csv(f'../{filename_list[i]}.csv')
        f = open(f'./output/{filename_list[i]}.txt', 'w', encoding='utf-8')
        lines = []
        for line in df['text']:
            lines.append(line)
        f.writelines(lines)
        f.close()


transform_files()
