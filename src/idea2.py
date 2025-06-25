'''from datasets import load_dataset

train_ds = load_dataset("roneneldan/TinyStories", split = 'train')


text = ""
for i in range(100):
    text += train_ds[i]['text'] + "\n"
    
with open("data/tiny_stories.txt", "w") as f:
    f.write(text)'''
    
    
import re    

r_expression = r'\w+|\s|[-,:?;!.\'\"]'

#txt = "what is your name? my name is ajay. cars I have:toyato,rolls royce, cullinan!\"diwakar\" mani\'s car\nlatha val;"
with open("data/tiny_stories.txt", "r") as f:
    text = f.read()
    
x = re.findall(r_expression, text.lower())
#print(set(x))
print(x[:50])
print(len(x))
print(len(set(x)))