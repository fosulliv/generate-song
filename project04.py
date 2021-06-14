
import os
import pandas as pd
import numpy as np
import requests
import time
import re

# ---------------------------------------------------------------------
# Question #1
# ---------------------------------------------------------------------

def get_book(url):
    """
    get_book that takes in the url of a 'Plain Text UTF-8' book and 
    returns a string containing the contents of the book.

    The function should satisfy the following conditions:
        - The contents of the book consist of everything between 
        Project Gutenberg's START and END comments.
        - The contents will include title/author/table of contents.
        - You should also transform any Windows new-lines (\r\n) with 
        standard new-lines (\n).
        - If the function is called twice in succession, it should not 
        violate the robots.txt policy.

    :Example: (note '\n' don't need to be escaped in notebooks!)
    >>> url = 'http://www.gutenberg.org/files/57988/57988-0.txt'
    >>> book_string = get_book(url)
    >>> book_string[:20] == '\\n\\n\\n\\n\\nProduced by Chu'
    True
    """
    response = requests.get(url, timeout=5)
    response.encoding = 'utf-8'

    book = response.text
    book = re.sub('\r\n', '\n', book)

    book = book[book.index(re.findall('\*\*\* [A-Za-z ]* \*\*\*', book)[0])+3:]
    book = book[book.index(re.findall('\*\*\*', book)[0])+3:]
    book = book[:book.index(re.findall('\*\*\*', book)[0])]


    return book
    
# ---------------------------------------------------------------------
# Question #2
# ---------------------------------------------------------------------


def tokenize(book_string):
    """
    tokenize takes in book_string and outputs a list of tokens 
    satisfying the following conditions:
        - The start of any paragraph should be represented in the 
        list with the single character \x02 (standing for START).
        - The end of any paragraph should be represented in the list 
        with the single character \x03 (standing for STOP).
        - Tokens in the sequence of words are split 
        apart at 'word boundaries' (see the regex lecture).
        - Tokens should include no whitespace.

    :Example:
    >>> test_fp = os.path.join('data', 'test.txt')
    >>> test = open(test_fp, encoding='utf-8').read()
    >>> tokens = tokenize(test)
    >>> tokens[0] == '\x02'
    True
    >>> tokens[9] == 'dead'
    True
    >>> sum([x == '\x03' for x in tokens]) == 4
    True
    >>> '(' in tokens
    True
    """
    s = re.sub('(\n){2,30}', ' \x03 \x02 ', book_string)
    s = re.sub('\n', ' ', s)
    tokens = re.findall('[\w]+|[^\s\w]', s)
    tokens.insert(0, '\x02')
    tokens.append('\x03')

    return tokens
    
# ---------------------------------------------------------------------
# Question #3
# ---------------------------------------------------------------------


class UniformLM(object):
    """
    Uniform Language Model class.
    """

    def __init__(self, tokens):
        """
        Initializes a Uniform languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
        
    def train(self, tokens):
        """
        Trains a uniform language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the (uniform) probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> isinstance(unif.mdl, pd.Series)
        True
        >>> set(unif.mdl.index) == set('one two three four'.split())
        True
        >>> (unif.mdl == 0.25).all()
        True
        """
        token_ser = pd.Series(tokens)
        unique_count = token_ser.nunique()
        ind = token_ser.value_counts().index

        return pd.Series(index=ind, data=1/unique_count)
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> unif.probability(('five',))
        0
        >>> unif.probability(('one', 'two')) == 0.0625
        True
        """
        train_ser = self.mdl
        prob = 1
        for i in words:
        	if i in train_ser:
        		prob = prob * train_ser[i]
        	else:
        		prob = 0

        return prob
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unif = UniformLM(tokens)
        >>> samp = unif.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True)
        >>> np.isclose(s, 0.25, atol=0.05).all()
        True
        """
        train_ser = self.mdl
        df = pd.DataFrame(train_ser)
        df.columns = ['probs']

        samp = df.sample(n=M, replace=True, weights='probs', axis=0)
        result_str = ' '.join(samp.index.to_list())

        return result_str

            
# ---------------------------------------------------------------------
# Question #4
# ---------------------------------------------------------------------


class UnigramLM(object):
    
    def __init__(self, tokens):
        """
        Initializes a Unigram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """
        self.mdl = self.train(tokens)
    
    def train(self, tokens):
        """
        Trains a unigram language model given a list of tokens.
        The output is a series indexed on distinct tokens, and
        values giving the probability of a token occuring
        in the language.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> isinstance(unig.mdl, pd.Series)
        True
        >>> set(unig.mdl.index) == set('one two three four'.split())
        True
        >>> unig.mdl.loc['one'] == 3 / 7
        True
        """
        num_words = len(tokens)

        return pd.Series(tokens).value_counts() / num_words
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> unig.probability(('five',))
        0
        >>> p = unig.probability(('one', 'two'))
        >>> np.isclose(p, 0.12244897959, atol=0.0001)
        True
        """
        train_ser = self.mdl
        prob = 1
        for i in words:
            if i in train_ser:
                prob = prob * train_ser[i]
            else:
                prob = 0

        return prob
        
    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        >>> tokens = tuple('one one two three one two four'.split())
        >>> unig = UnigramLM(tokens)
        >>> samp = unig.sample(1000)
        >>> isinstance(samp, str)
        True
        >>> len(samp.split()) == 1000
        True
        >>> s = pd.Series(samp.split()).value_counts(normalize=True).loc['one']
        >>> np.isclose(s, 0.41, atol=0.05).all()
        True
        """
        train_ser = self.mdl
        df = pd.DataFrame(train_ser)
        df.columns = ['probs']
        samp = df.sample(n=M, replace=True, weights='probs', axis=0)
        result_str = ' '.join(samp.index.to_list())

        return result_str
        
    
# ---------------------------------------------------------------------
# Question #5,6,7,8
# ---------------------------------------------------------------------

class NGramLM(object):
    
    def __init__(self, N, tokens):
        """
        Initializes a N-gram languange model using a
        list of tokens. It trains the language model
        using `train` and saves it to an attribute
        self.mdl.
        """

        self.N = N
        ngrams = self.create_ngrams(tokens)

        self.ngrams = ngrams
        self.mdl = self.train(ngrams)

        if N < 2:
            raise Exception('N must be greater than 1')
        elif N == 2:
            self.prev_mdl = UnigramLM(tokens)
        else:
            mdl = NGramLM(N-1, tokens)
            self.prev_mdl = mdl

    def create_ngrams(self, tokens):
        """
        create_ngrams takes in a list of tokens and returns a list of N-grams. 
        The START/STOP tokens in the N-grams should be handled as 
        explained in the notebook.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, [])
        >>> out = bigrams.create_ngrams(tokens)
        >>> isinstance(out[0], tuple)
        True
        >>> out[0]
        ('\\x02', 'one')
        >>> out[2]
        ('two', 'three')
        """
        N = self.N
        ngrams = []
        for i in range(len(tokens)):
            if (i+N) > len(tokens):
                break
            gram = tokens[i:i+N]
            ngrams.append(gram)
        
        return ngrams
        
    def train(self, ngrams):
        """
        Trains a n-gram language model given a list of tokens.
        The output is a dataframe with three columns (ngram, n1gram, prob).

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> set(bigrams.mdl.columns) == set('ngram n1gram prob'.split())
        True
        >>> bigrams.mdl.shape == (6, 3)
        True
        >>> bigrams.mdl['prob'].min() == 0.5
        True
        """
        result_df = pd.DataFrame(columns=['ngram', 'n1gram', 'prob'])
        # ngram counts C(w_1, ..., w_n)
        ng_counts = pd.Series(ngrams).value_counts()

        result_df['ngram'] = ng_counts.index
        # n-1 gram counts C(w_1, ..., w_(n-1))
        def helper_ngram(row):
            ng = row['ngram']
            l = len(ng)
            return ng[0:l-1]
        result_df['n1gram'] = result_df.apply(helper_ngram, axis=1)
        raw_n1g_counts = result_df['n1gram'].value_counts()

        def count_n1g(row):
            ng = row['ngram']
            n1g = row['n1gram']
            ng_count = ng_counts[ng]
            n1g_count = raw_n1g_counts[n1g]
            n1g_count = n1g_count + (ng_count - 1)
            return n1g_count

        n1g_counts = result_df.apply(count_n1g, axis=1)

        # Create the conditional probabilities
        def helper_ngram2(row):
            ng = row['ngram']
            n1g = row['n1gram']
            ng_count = ng_counts[ng]
            ind = result_df[result_df['n1gram'] == n1g].index.values[0]
            n1g_count = n1g_counts[ind]
            return ng_count / n1g_count
        
        result_df['prob'] = result_df.apply(helper_ngram2, axis=1)

        # Put it all together

        return result_df
    
    def probability(self, words):
        """
        probability gives the probabiliy a sequence of words
        appears under the language model.
        :param: words: a tuple of tokens
        :returns: the probability `words` appears under the language
        model.

        :Example:
        >>> tokens = tuple('\x02 one two one three one two \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> p = bigrams.probability('two one three'.split())
        >>> np.isclose(p, (1/4)*(1/2)*(1/3))
        True
        >>> bigrams.probability('one two five'.split()) == 0
        True
        """
        words = tuple(words)
        unigram = self.prev_mdl
        if words[0] in unigram.mdl:
            prob = unigram.mdl[words[0]]
        else:
            prob = 0

        ngrams = []
        for i in range(len(words)):
            if (i+self.N) > len(words):
                break
            ngrams.append(words[i:i+self.N])

        for i in ngrams:
            gram = self.mdl[self.mdl['ngram'] == i]
            if len(gram) == 0:
                prob = 0
            else:
                prob = prob * gram['prob'].values[0]

        return prob

    def sample(self, M):
        """
        sample selects tokens from the language model of length M, returning
        a string of tokens.

        :Example:
        >>> tokens = tuple('\x02 one two three one four \x03'.split())
        >>> bigrams = NGramLM(2, tokens)
        >>> samp = bigrams.sample(3)
        >>> len(samp.split()) == 4  # don't count the initial START token.
        True
        >>> samp[:2] == '\\x02 '
        True
        >>> set(samp.split()) <= {'\\x02', '\\x03', 'one', 'two', 'three', 'four'}
        True
        """
        N = self.N
        dic = {}
        first_mdl = self
        words = []


        for i in reversed(range(1, N+1)):
            dic['%smdl'%i] = first_mdl
            if isinstance(first_mdl.mdl, pd.Series):
                break
            first_mdl = first_mdl.prev_mdl

        first = dic['1mdl'].mdl
        first_df = pd.DataFrame(first)
        first_df.columns = ['probs']
        first_df['value'] = first_df.index
        samp = first_df.sample(n=1, weights='probs', axis=0)
        first_word = samp['value'][0]
        words.append(first_word)

        for i in reversed(dic.keys()):
            if i == '1mdl':
                continue
            else:
                N = int(i[0])
                w = words[-N-1:]
                tup = tuple(w)
                mdl = dic[i].mdl
                only_word = mdl[mdl['n1gram'] == tup]
                if len(only_word) == 0:
                    words.append('\x03')
                else:
                    samp = only_word.sample(n=1, weights='prob', axis=0)['ngram'].iloc[0]
                    words.append(samp[-1])

        remaining = M - len(words)
        mdl = self.mdl
        N = self.N - 1
        for i in range(remaining):
            w = words[-N:]
            tup = tuple(w)
            only_word = mdl[mdl['n1gram'] == tup]
            if len(only_word) == 0:
                words.append('\x03')
            else:
                samp = only_word.sample(n=1, weights='prob', axis=0)['ngram'].iloc[0]
                words.append(samp[-1])
        words.insert(0, '\x02')
        result_str = ' '.join(words)


        return result_str


# ---------------------------------------------------------------------
# DO NOT TOUCH BELOW THIS LINE
# IT'S FOR YOUR OWN BENEFIT!
# ---------------------------------------------------------------------


# Graded functions names! DO NOT CHANGE!
# This dictionary provides your doctests with
# a check that all of the questions being graded
# exist in your code!

GRADED_FUNCTIONS = {
    'q01': ['get_book'],
    'q02': ['tokenize'],
    'q03': ['UniformLM'],
    'q04': ['UnigramLM'],
    'q05': ['NGramLM']
}


def check_for_graded_elements():
    """
    >>> check_for_graded_elements()
    True
    """
    
    for q, elts in GRADED_FUNCTIONS.items():
        for elt in elts:
            if elt not in globals():
                stmt = "YOU CHANGED A QUESTION THAT SHOULDN'T CHANGE! \
                In %s, part %s is missing" %(q, elt)
                raise Exception(stmt)

    return True
