import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')

# the below is a famous monologue from Shakespeare

paragraph = """Yet here, Laertes! aboard, aboard, for shame!
The wind sits in the shoulder of your sail,
And you are stay’d for. There; my blessing with thee!
And these few precepts in thy memory
See thou character. Give thy thoughts no tongue,
Nor any unproportioned thought his act.
Be thou familiar, but by no means vulgar.
Those friends thou hast, and their adoption tried,
Grapple them to thy soul with hoops of steel;
But do not dull thy palm with entertainment
Of each new-hatch’d, unfledged comrade. Beware
Of entrance to a quarrel, but being in,
Bear’t that the opposed may beware of thee.
Give every man thy ear, but few thy voice;
Take each man’s censure, but reserve thy judgment.
Costly thy habit as thy purse can buy,
But not express’d in fancy; rich, not gaudy;
For the apparel oft proclaims the man,
And they in France of the best rank and station
Are of a most select and generous chief in that.
Neither a borrower nor a lender be;
For loan oft loses both itself and friend,
And borrowing dulls the edge of husbandry.
This above all: to thine ownself be true,
And it must follow, as the night the day,
Thou canst not then be false to any man.
Farewell: my blessing season this in thee!"""
               
sentences = nltk.sent_tokenize(paragraph)
print(sentences)

stemmer = PorterStemmer()

for i in range(len(sentences)):
    words = nltk.word_tokenize(sentences[i])
    words = [stemmer.stem(w) for w in words if w not in set(stopwords.words('english'))]
    sentences[i] = ' '.join(words)
    
