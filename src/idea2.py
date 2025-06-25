import re 

'''from datasets import load_dataset

train_ds = load_dataset("roneneldan/TinyStories", split = 'train')


text = ""
for i in range(100):
    text += train_ds[i]['text'] + "\n"
    
with open("data/tiny_stories.txt", "w") as f:
    f.write(text)'''
    


def remove_suffixes(word):
    if word in stop_words:
        return (word, )
    
    elif word.endswith('ing') and len(word) > 4:
        root = word[:-3]
        if len(root) >= 2 and root[-1] == root[-2] and root[-1] in 'bcdfghjklmnpqrstvwxyz':
            return (root[:-1], root[-1], word[-3:])
        return (root, word[-3:])
    
    elif word.endswith('ed') and len(word) > 3:
        root = word[:-2]
        if len(root) >= 2 and root[-1] == root[-2] and root[-1] in 'bcdfghjklmnpqrstvwxyz':
            return (root[:-1], root[-1], word[-2:])
        return (root, word[-2:])
    
    elif word.endswith('s') and len(word) > 2 and not word.endswith('ss'):
        return (word[:-1], word[-1])
    
    elif word.endswith("ly") and len(word) > 3:
        root = word[:-2]
        if len(root) >= 2 and root[-1] == root[-2] and root[-1] in 'bcdfghjklmnpqrstvwxyz':
            return (root[:-1], root[-1], word[-2:])
        return (root, word[-2:])
    
    return (word, )




r_expression = r'\w+|\s|[-,:?;!.\'\"]'

''''txt = "what is your name? my name is ajay. cars I have:toyato,rolls royce, cullinan!\"diwakar\" mani\'s car\nlatha val;she is running. I walked into a river"
x = re.findall(r_expression, txt.lower())
print(x)
removed_suffixes = []
for word in x:
    w = remove_suffixes(word)
    for t in w:
        removed_suffixes.append(t)
print(removed_suffixes)'''

stop_words = [
    "a", "about", "after", "again", "against", "all", "also", "am", "an", "and",
    "any", "are", "around", "as", "at", "be", "because", "been", "before", 
    "being", "below", "between", "both", "but", "by", "can", "could", "did", 
    "do", "does", "down", "each", "else", "ever", "every", "for", "from", 
    "further", "had", "have", "he", "her", "here", "him", "himself", "his", 
    "how", "i", "if", "in", "into", "is", "it", "its", "just", "let", "may", 
    "me", "might", "more", "most", "must", "my", "never", "no", "not", "now", 
    "of", "off", "on", "once", "only", "or", "other", "our", "out", "over", 
    "own", "re", "s", "same", "she", "should", "so", "some", "still", "such", 
    "t", "than", "that", "the", "their", "them", "themselves", "then", "there", 
    "these", "they", "this", "those", "through", "to", "too", "under", "until", 
    "up", "us", "very", "was", "we", "well", "were", "what", "when", "where", 
    "while", "who", "why", "will", "with", "would", "you", "your", "yours", "ed", "ing", "ly",
    "g", '!', 'a', '?', ',', '.', ':', ';', '-', '(', ')', '"', "'", '“', '”',
    '\n', '\t', ' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 
    'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]


with open("data/tiny_stories.txt", "r") as f:
    text = f.read()
    
words = re.findall(r_expression, text.lower())

x = []
for word in words:
    w = remove_suffixes(word)
    for t in w:
        x.append(t)
y = x.copy()

#x_ixs = [i for i in range(len(x))]

filtered_x = [w for w in x if w not in stop_words]
filtered_x_ixs = [i for i, w in enumerate(x) if w not in stop_words]

x = filtered_x
x_ixs = filtered_x_ixs



stop_set = [word for word in set(y) if word in stop_words]
print(len(x), len(y))
print(len(set(x)))
print(len(stop_set))
print(x[:50], y[:x_ixs[49]+1])
