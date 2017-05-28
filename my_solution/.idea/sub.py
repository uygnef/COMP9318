## import modules here
import pandas as pd
import numpy as np
import string
from collections import deque, Counter, OrderedDict
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, scale, normalize
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, classification_report
import re
import nltk
import pickle

################# helper data ##############

# Vowel Phonemes
vowels = ('AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH'
          , 'IY', 'OW', 'OY', 'UH', 'UW')

# Consonants Phonemes
consonants = ('P', 'B', 'CH', 'D', 'DH', 'F', 'G', 'HH', 'JH', 'K', 'L', 'M', 'N',
              'NG', 'R', 'S', 'SH', 'T', 'TH', 'V', 'W', 'Y', 'Z', 'ZH')

strong_suffixes = set(('al', 'ance', 'ancy', 'ant', 'ard', 'ary', 'àte', 'auto', 'ence', 'ency', 'ent',
                       'ery', 'est', 'ial', 'ian', 'iana', 'en', 'ésce', 'ic', 'ify', 'ine', 'ion', 'tion',
                       'ity', 'ive', 'ory', 'ous', 'ual', 'ure', 'wide', 'y', 'se', 'ade', 'e', 'ee', 'een',
                       'eer', 'ese', 'esque', 'ette', 'eur', 'ier', 'oon', 'que'))

strong_prefixes = set(
    ('ad', 'co', 'con', 'counter', 'de', 'di', 'dis', 'e', 'en', 'ex', 'in', 'mid', 'ob', 'para', 'pre', 're', 'sub',
     'a', 'be', 'with', 'for'))

neutral_prefixes = set(
    ('down', 'fore', 'mis', 'over', 'out', 'un', 'under', 'up', 'anti', 'bi', 'non', 'pro', 'tri', 'contra', 'counta',
     'de', 'dis', 'extra', 'inter', 'intro', 'multi', 'non', 'post', 'retro', 'super', 'trans', 'ultra'))

neutral_suffixes = set(
    ('able', 'age', 'al', 'ate', 'ed', 'en', 'er', 'est', 'ful', 'hood', 'ible', 'ing', 'ile', 'ish', 'ism',
     'ist', 'ize', 'less', 'like', 'ly''man', 'ment', 'most', 'ness', 'old', 's', 'ship', 'some', 'th', 'ward', 'wise',
     'y'))

suffixes = (
    'inal', 'ain', 'tion', 'sion', 'osis', 'oon', 'sce', 'que', 'ette', 'eer', 'ee', 'aire', 'able', 'ible', 'acy',
    'cy', 'ade',
    'age', 'al', 'al', 'ial', 'ical', 'an', 'ance', 'ence',
    'ancy', 'ency', 'ant', 'ent', 'ant', 'ent', 'ient', 'ar', 'ary', 'ard', 'art', 'ate', 'ate', 'ate', 'ation', 'cade',
    'drome', 'ed', 'ed', 'en', 'en', 'ence', 'ency', 'er', 'ier',
    'er', 'or', 'er', 'or', 'ery', 'es', 'ese', 'ies', 'es', 'ies', 'ess', 'est', 'iest', 'fold', 'ful', 'ful', 'fy',
    'ia',
    'ian', 'iatry', 'ic', 'ic', 'ice', 'ify', 'ile',
    'ing', 'ion', 'ish', 'ism', 'ist', 'ite', 'ity', 'ive', 'ive', 'ative', 'itive', 'ize', 'less', 'ly', 'ment',
    'ness',
    'or', 'ory', 'ous', 'eous', 'ose', 'ious', 'ship', 'ster',
    'ure', 'ward', 'wise', 'ize', 'phy', 'ogy')

prefixes = (
    'ac', 'ad', 'af', 'ag', 'al', 'an', 'ap', 'as', 'at', 'an', 'ab', 'abs', 'acer', 'acid', 'acri', 'act', 'ag', 'acu',
    'aer', 'aero', 'ag', 'agi',
    'ig', 'act', 'agri', 'agro', 'alb', 'albo', 'ali', 'allo', 'alter', 'alt', 'am', 'ami', 'amor', 'ambi', 'ambul',
    'ana',
    'ano', 'andr', 'andro', 'ang',
    'anim', 'ann', 'annu', 'enni', 'ante', 'anthrop', 'anti', 'ant', 'anti', 'antico', 'apo', 'ap', 'aph', 'aqu',
    'arch',
    'aster', 'astr', 'auc', 'aug',
    'aut', 'aud', 'audi', 'aur', 'aus', 'aug', 'auc', 'aut', 'auto', 'bar', 'be', 'belli', 'bene', 'bi', 'bine', 'bibl',
    'bibli', 'biblio', 'bio', 'bi',
    'brev', 'cad', 'cap', 'cas', 'ceiv', 'cept', 'capt', 'cid', 'cip', 'cad', 'cas', 'calor', 'capit', 'capt', 'carn',
    'cat', 'cata', 'cath', 'caus', 'caut'
    , 'cause', 'cuse', 'cus', 'ceas', 'ced', 'cede', 'ceed', 'cess', 'cent', 'centr', 'centri', 'chrom', 'chron',
    'cide',
    'cis', 'cise', 'circum', 'cit',
    'civ', 'clam', 'claim', 'clin', 'clud', 'clus claus', 'co', 'cog', 'col', 'coll', 'con', 'com', 'cor', 'cogn',
    'gnos',
    'com', 'con', 'contr', 'contra',
    'counter', 'cord', 'cor', 'cardi', 'corp', 'cort', 'cosm', 'cour', 'cur', 'curr', 'curs', 'crat', 'cracy', 'cre',
    'cresc', 'cret', 'crease', 'crea',
    'cred', 'cresc', 'cret', 'crease', 'cru', 'crit', 'cur', 'curs', 'cura', 'cycl', 'cyclo', 'de', 'dec', 'deca',
    'dec',
    'dign', 'dei', 'div', 'dem', 'demo',
    'dent', 'dont', 'derm', 'di', 'dy', 'dia', 'dic', 'dict', 'dit', 'dis', 'dif', 'dit', 'doc', 'doct', 'domin', 'don',
    'dorm', 'dox', 'duc', 'duct', 'dura',
    'dynam', 'dys', 'ec', 'eco', 'ecto', 'en', 'em', 'end', 'epi', 'equi', 'erg', 'ev', 'et', 'ex', 'exter', 'extra',
    'extro', 'fa', 'fess', 'fac', 'fact',
    'fec', 'fect', 'fic', 'fas', 'fea', 'fall', 'fals', 'femto', 'fer', 'fic', 'feign', 'fain', 'fit', 'feat', 'fid',
    'fid',
    'fide', 'feder', 'fig', 'fila',
    'fili', 'fin', 'fix', 'flex', 'flect', 'flict', 'flu', 'fluc', 'fluv', 'flux', 'for', 'fore', 'forc', 'fort',
    'form',
    'fract', 'frag',
    'frai', 'fuge', 'fuse', 'gam', 'gastr', 'gastro', 'gen', 'gen', 'geo', 'germ', 'gest', 'giga', 'gin', 'gloss',
    'glot',
    'glu', 'glo', 'gor', 'grad', 'gress',
    'gree', 'graph', 'gram', 'graf', 'grat', 'grav', 'greg', 'hale', 'heal', 'helio', 'hema', 'hemo', 'her', 'here',
    'hes',
    'hetero', 'hex', 'ses', 'sex', 'homo',
    'hum', 'human', 'hydr', 'hydra', 'hydro', 'hyper', 'hypn', 'an', 'ics', 'ignis', 'in', 'im', 'in', 'im', 'il', 'ir',
    'infra', 'inter', 'intra', 'intro', 'ty',
    'jac', 'ject', 'join', 'junct', 'judice', 'jug', 'junct', 'just', 'juven', 'labor', 'lau', 'lav', 'lot', 'lut',
    'lect',
    'leg', 'lig', 'leg', 'levi', 'lex',
    'leag', 'leg', 'liber', 'liver', 'lide', 'liter', 'loc', 'loco', 'log', 'logo', 'ology', 'loqu', 'locut', 'luc',
    'lum',
    'lun', 'lus', 'lust', 'lude', 'macr',
    'macer', 'magn', 'main', 'mal', 'man', 'manu', 'mand', 'mania', 'mar', 'mari', 'mer', 'matri', 'medi', 'mega',
    'mem',
    'ment', 'meso', 'meta', 'meter', 'metr',
    'micro', 'migra', 'mill', 'kilo', 'milli', 'min', 'mis', 'mit', 'miss', 'mob', 'mov', 'mot', 'mon', 'mono', 'mor',
    'mort', 'morph', 'multi', 'nano', 'nasc',
    'nat', 'gnant', 'nai', 'nat', 'nasc', 'neo', 'neur', 'nom', 'nom', 'nym', 'nomen', 'nomin', 'non', 'non', 'nov',
    'nox',
    'noc', 'numer', 'numisma', 'ob', 'oc',
    'of', 'op', 'oct', 'oligo', 'omni', 'onym', 'oper', 'ortho', 'over', 'pac', 'pair', 'pare', 'paleo', 'pan', 'para',
    'pat', 'pass', 'path', 'pater', 'patr',
    'path', 'pathy', 'ped', 'pod', 'pedo', 'pel', 'puls', 'pend', 'pens', 'pond', 'per', 'peri', 'phage', 'phan',
    'phas',
    'phen', 'fan', 'phant', 'fant', 'phe',
    'phil', 'phlegma', 'phobia', 'phobos', 'phon', 'phot', 'photo', 'pico', 'pict', 'plac', 'plais', 'pli', 'ply',
    'plore',
    'plu', 'plur', 'plus', 'pneuma',
    'pneumon', 'pod', 'poli', 'poly', 'pon', 'pos', 'pound', 'pop', 'port', 'portion', 'post', 'pot', 'pre', 'pur',
    'prehendere', 'prin', 'prim', 'prime',
    'pro', 'proto', 'psych', 'punct', 'pute', 'quat', 'quad', 'quint', 'penta', 'quip', 'quir', 'quis', 'quest', 'quer',
    're', 'reg', 'recti', 'retro', 'ri', 'ridi',
    'risi', 'rog', 'roga', 'rupt', 'sacr', 'sanc', 'secr', 'salv', 'salu', 'sanct', 'sat', 'satis', 'sci', 'scio',
    'scientia', 'scope', 'scrib', 'script', 'se',
    'sect', 'sec', 'sed', 'sess', 'sid', 'semi', 'sen', 'scen', 'sent', 'sens', 'sept', 'sequ', 'secu', 'sue', 'serv',
    'sign', 'signi', 'simil', 'simul', 'sist', 'sta',
    'stit', 'soci', 'sol', 'solus', 'solv', 'solu', 'solut', 'somn', 'soph', 'spec', 'spect', 'spi', 'spic', 'sper',
    'sphere', 'spir', 'stand', 'stant', 'stab',
    'stat', 'stan', 'sti', 'sta', 'st', 'stead', 'strain', 'strict', 'string', 'stige', 'stru', 'struct', 'stroy',
    'stry',
    'sub', 'suc', 'suf', 'sup', 'sur', 'sus',
    'sume', 'sump', 'super', 'supra', 'syn', 'sym', 'tact', 'tang', 'tag', 'tig', 'ting', 'tain', 'ten', 'tent', 'tin',
    'tect', 'teg', 'tele', 'tem', 'tempo', 'ten',
    'tin', 'tain', 'tend', 'tent', 'tens', 'tera', 'term', 'terr', 'terra', 'test', 'the', 'theo', 'therm', 'thesis',
    'thet', 'tire', 'tom', 'tor', 'tors', 'tort'
    , 'tox', 'tract', 'tra', 'trai', 'treat', 'trans', 'tri', 'trib', 'tribute', 'turbo', 'typ', 'ultima', 'umber',
    'umbraticum', 'un', 'uni', 'vac', 'vade', 'vale',
    'vali', 'valu', 'veh', 'vect', 'ven', 'vent', 'ver', 'veri', 'verb', 'verv', 'vert', 'vers', 'vi', 'vic', 'vicis',
    'vict', 'vinc', 'vid', 'vis', 'viv', 'vita', 'vivi'
    , 'voc', 'voke', 'vol', 'volcan', 'volv', 'volt', 'vol', 'vor', 'with', 'zo')


# Upper Convert set ot upper
def upper(iterable):
    return {x.upper() for x in iterable}


neutral_prefixes = upper(neutral_prefixes)
neutral_suffixes = upper(neutral_suffixes)
strong_prefixes = upper(strong_prefixes)
strong_suffixes = upper(strong_suffixes)
full_suffixes_set = upper(suffixes)
full_prefixes_set = upper(prefixes)

# Classification Map
classifications = {'10': 0,
                   '100': 0,
                   '1000': 0,
                   '01': 3,
                   '001': 3,
                   '0001': 3,
                   '010': 1,
                   '0100': 1,
                   '0010': 2
                   }

vector_map = vowels + consonants


################# training #################

def train(data, classifier_file, multi=None, DEBUG=None):  # do not change the heading of the function
    words = word_data(data)
    classifier_type = LogisticRegression

    if multi:
        classifier_cls = multi_classifier(classifier_type, class_weight='balanced')
    else:
        classifier_cls = classifier(classifier_type, class_weight='balanced')

    features = ['str_pre', 'str_suf', 'neu_pre', 'neu_suf', 'prefix', 'suffix', 'phoneme_length',
                'vowel_count'] + words.type_tags
    classifier_cls.set_features(features)

    train_X = np.array(words.df[features])
    train_Y = words.df.classification

    classifier_cls.train(train_X, train_Y)
    save_Pickle((words, classifier_cls), classifier_file)

    if DEBUG:
        print("Finished Training")

    return


################# testing #################

def test(data, classifier_file, sample=None, DEBUG=None):  # do not change the heading of the function
    words, classifier_cls = get_Pickle(classifier_file)
    test_words = word_data(data, train_type_tags=words.type_tags)

    if sample:
        test_words.df = test_words.df.sample(sample)

    features = classifier_cls.get_features()
    feature_array = np.array(test_words.df[features])

    test_words.set_predicted_classes(classifier_cls.predict_classifications(feature_array))
    pred = test_words.df.predicted_primary_index.tolist()

    if DEBUG:
        print(classification_report(test_words.df.primary_stress_index, pred))
        print(f1_score(test_words.df.primary_stress_index, pred, average='macro'))
        return

    return pred


################# classes ##################

'''

word_data       = Class to hold word data and perform all requisite pre-processing

    Attributes
lines           = List of word and stressed phonemes
df              = dataframe to hold and process word data
pn_list         = list of phonemes
vowel_map       = 2-4 bit string depicting location of primary stress
classifications = Group Index of stressed vowel, 0 is 1st, 3 is last irrespective of vowel count/word length
                  1 and 2 are then 2nd and 3rd respecively.
ngrams          = All possible ngrams of pn_list
ngrams_counts   = Dict object of ngrams 


'''


class word_data(object):
    def __init__(self, data, train_type_tags=[]):
        self.lines = [line_split(line) for line in data]
        self.df = pd.DataFrame(data=self.lines, columns=('word', 'pronunciation'))
        self.df['pn_list'] = self.df.pronunciation.apply(str.split)
        self.df['phoneme_length'] = self.df.pn_list.str.len()
        self.df['destressed_pn_list'] = self.df.pronunciation.apply(filter_stress, args=('[012]',))
        self.df['vowel_map'] = self.df.destressed_pn_list.apply(phoneme_map, args=(vowels,))
        self.df['consonant_map'] = self.df.destressed_pn_list.apply(phoneme_map, args=(consonants,))
        self.df['vowel_count'] = self.df.vowel_map.apply(np.sum)
        self.df['consonant_count'] = self.df.consonant_map.apply(np.sum)
        self.df['vowel_map_string'] = self.df.vowel_map.apply(to_string)
        self.df['stress_map'] = self.df.pn_list.apply(get_stress_map)
        self.df['classification'] = self.df.stress_map.apply(get_classification)
        self.df['primary_stress_index'] = self.df.apply(get_classsification_index, args=('classification',), axis=1)
        self.df['ngrams'] = self.df.pn_list.apply(get_all_ngrams)
        self.df['ngram_counts'] = self.df.ngrams.apply(Counter)
        self.df['destressed_ngrams'] = self.df.destressed_pn_list.apply(get_all_ngrams)
        self.df['destressed_ngram_counts'] = self.df.destressed_ngrams.apply(Counter)
        self.df['prefix'] = self.df.word.apply(check_prefix, args=(full_prefixes_set,))
        self.df['suffix'] = self.df.word.apply(check_suffix, args=(full_suffixes_set,))
        self.df['str_pre'] = self.df.word.apply(check_prefix, args=(strong_prefixes,))
        self.df['str_suf'] = self.df.word.apply(check_suffix, args=(strong_suffixes,))
        self.df['neu_pre'] = self.df.word.apply(check_prefix, args=(neutral_prefixes,))
        self.df['neu_suf'] = self.df.word.apply(check_suffix, args=(neutral_suffixes,))
        self.df['type_tag'] = self.df.word.apply(get_pos_tag)

        self._encode_type_tag(train_type_tags)

    def _encode_type_tag(self, train_type_tags):
        if not train_type_tags:
            self.type_tags = self.df.type_tag.unique().tolist()
            type_tag_dummies = pd.get_dummies(self.df.type_tag)
        else:
            self.type_tags = train_type_tags
            # Type_tags for test_data
            test_type_tag_dummies = pd.get_dummies(self.df.type_tag)
            # Word Types that are in both test and training
            existing_columns = set(train_type_tags) & set(self.df.type_tag.unique())
            # Word Types in the Training data that need 0 values in test
            columns_to_be_added = set(train_type_tags) - existing_columns
            # Get existing columns from testing dummies and add those that don't exist with default
            type_tag_dummies = test_type_tag_dummies[list(existing_columns)]
            # Add columns that don't exist in test
            zeroes = np.zeros(shape=(len(self.df), len(columns_to_be_added)))
            non_existant_columns = pd.DataFrame(zeroes, columns=list(columns_to_be_added))
            # Concat and Sort like training data
            type_tag_dummies = pd.concat([type_tag_dummies, non_existant_columns], axis=1)[train_type_tags]

        self.df = pd.concat([self.df, type_tag_dummies], axis=1)

    def set_predicted_classes(self, classes):
        self.df['predicted_classes'] = classes
        self.df['predicted_primary_index'] = self.df.apply(get_classsification_index, args=('predicted_classes',),
                                                           axis=1)


'''
classifier      = Class to hold classifier and training/testing/prediction methods

    Attributes
clf             = Passed in Classifier
encoder         = LabelEncoder for classes
vectorizer      = DictVectorizer (Sparse Matrix) to hold features
train_X         = Vectorized training features
train_Y         = Label Encoded training classifications
test_X          = 


    Methods
train           = Encode Features and Classifications, Train Classifier
test

'''


class classifier(object):
    def __init__(self, classifier_type, *args, **kwargs):
        self.clf = classifier_type(**kwargs)
        self.encoder = LabelEncoder()
        self.scaler = scale
        self.vectorizer = DictVectorizer(dtype=int, sparse=True)

    def set_features(self, feature_list):
        self.features = feature_list

    def get_features(self):
        return self.features

    def train(self, X, Y):
        self.train_X = X
        self.normalized_train_x = self.scaler(X)
        self.train_Y = self.encoder.fit_transform(Y)
        self.clf.fit(self.normalized_train_x, self.train_Y)

    def _encode_test_features(self, X):
        return self.vectorizer.transform(X.tolist())

    def predict_classifications(self, X):
        predicted_Y = self.clf.predict(self.scaler(X))
        return predicted_Y

    def get_prob(self, X):
        return self.clf.predict_proba(self.scaler(X))


'''
multi_classifier      = Class to hold multiple classifier and training/testing/prediction methods

    Attributes
Zero            = Classifier for Zero
One             = Classifier for One
Two             = Classifier for Two
Three           = Classifier for Three

    Methods


'''


class multi_classifier(object):
    def __init__(self, classifier_type, **kwargs):
        self.Zero = classifier(classifier_type, **kwargs)
        self.One = classifier(classifier_type, **kwargs)
        self.Two = classifier(classifier_type, **kwargs)
        self.Three = classifier(classifier_type, **kwargs)
        self.clfs = OrderedDict({0: self.Zero,
                                 1: self.One,
                                 2: self.Two,
                                 3: self.Three
                                 })
        self.vectorizer = DictVectorizer(dtype=int, sparse=True)

    def train(self, X, Y):
        for idx, clf in self.clfs.items():
            Y_bin = [cls == idx for cls in Y]
            clf.train(X, Y_bin)

    def set_features(self, feature_list):
        self.features = feature_list

    def get_features(self):
        return self.features

    def _encode_training_features(self, X):
        self.vectorizer.fit_transform(X.tolist())

    def _encode_test_features(self, X):
        return self.vectorizer.fit([X])

    def predict_classifications(self, X):
        probs = pd.DataFrame()

        for idx, clf in self.clfs.items():
            class_probs = clf.get_prob(X)
            probs[idx] = class_probs.transpose()[1]

        probs['classification'] = probs.idxmax(axis=1)

        return probs.classification


################# helper functions #########

# Pickler
def save_Pickle(obj, file):
    with open(file, 'wb') as f:
        pickle.dump(obj, f)
    f.close()


def get_Pickle(file):
    with open(file, 'rb') as f:
        obj = pickle.load(f)
    f.close()
    return (obj_i for obj_i in obj)


# Return all ngrams of particular length
def get_ngram_possibilities(pronunciation_list, length):
    return tuple(zip(*(pronunciation_list[i:] for i in range(length))))


# Develop deque of all possible ngrams
def get_all_ngrams(pn_list, restrict_length=None):
    ngrams = set()
    if not restrict_length:
        restrict_length = len(pn_list)
    for i in range(2, restrict_length + 1):
        ngrams.update(get_ngram_possibilities(pn_list, i))
    return ngrams


# Convert list to tuple
def as_tuple(list_to_convert):
    return tuple(list_to_convert)


# Filter stress from string

def filter_stress(string_to_be_filtered, to_filter=None):
    if type(string_to_be_filtered) in [list, tuple]:
        string_to_be_filtered = ' '.join(string_to_be_filtered)
    return tuple(re.sub(to_filter, '', string_to_be_filtered).split())


# Filter non-important stresses
def filter_non_primary_stress(pronunciation):
    pronunciation = pronunciation.replace('0', '')
    return pronunciation.replace('2', '')


# Maps the location of the stress, 1 if stress at position
# 0 otherwise
def stress_map(pronunciation, stress='1'):
    return [1 if stress in num else 0 for num in pronunciation]


# Maps the the location of phenom, 1 in phenom_list
# 0 otherwise
def phoneme_map(pronunciation, phoneme_list):
    return [1 if phoneme in phoneme_list else 0 for phoneme in pronunciation]


# Map existence of one iterable in another
def iterable_map(list_to_map, iterable):
    return [1 if iter_item in list_to_map else 0 for iter_item in iterable]


# Get nltk pos_tag
def get_pos_tag(word):
    return nltk.pos_tag([word])[0][1]


# Returning string as a classification
def get_stress_position(stress_map_list, stress=1):
    return str(stress_map_list.index(stress) + 1)


# Check if prefix exists
def check_prefix(word, prefixes_set):
    for letter_idx in range(len(word) + 1):
        if word[:letter_idx] in prefixes_set:
            return 1
    return 0


# Check if suffix exists
def check_suffix(word, suffixes_set):
    word_length = len(word)
    for letter_idx in range(word_length + 1):
        if word[abs(letter_idx - word_length):] in suffixes_set:
            return 1
    return 0


# Get ascii index of first letter
def get_first_letter_idx(word):
    return string.ascii_lowercase.index(word[0].lower()) + 1


# Return the stressed vowel
def get_stressed_vowel(pn_list):
    for vowel in pn_list:
        if '1' in vowel:
            return filter_stress(vowel, to_filter='1')[0]


# Return all possible consecutive tuples length n from list
def sub_string(pronunciation_list, n):
    return tuple(zip(*(pronunciation_list[i:] for i in range(n))))


# Build a dict of all possible sequences of phonemes
def get_sequences(phoneme_series):
    sequences = {}
    max_length = max(phoneme_series.str.len())
    for i in range(2, max_length + 1):
        for pn_list in phoneme_series:
            # Next iteration if pn_list is shorter then the sequence length be built
            if len(pn_list) < i:
                continue
            word_sequences = sub_string(pn_list, i)
            for seq in word_sequences:
                sequences[seq] = sequences.get(seq, 0) + 1
    return sequences


def in_list(pn_list, sequence):
    if pn_list in sequence:
        return 1
    return 0


# Return 1 if sequence has a primary stress in it
def is_primary(sequence):
    for phoneme in sequence:
        if '1' in phoneme:
            return True
    return False


# Return classification for pn_list
def get_stress_map(pn_list):
    vowels = str()
    for pn in pn_list:
        if pn in consonants:
            continue
        elif '1' in pn:
            vowels += '1'
        elif '0' in pn or '2' in pn:
            vowels += '0'
    return vowels


def get_classification(vowel_map):
    return classifications[vowel_map]


# Return the index of the stressed vowel based on classification
def get_classsification_index(df, classification_column):
    vowel_idx = [idx.start() for idx in re.finditer('1', df.vowel_map_string)]
    if df[classification_column] > len(vowel_idx) - 1:
        return vowel_idx[-1]
    if df[classification_column] < 3:
        return vowel_idx[df[classification_column]]
    else:
        return vowel_idx[-1]


def to_string(list_to_convert):
    return ''.join([str(x) for x in list_to_convert])


def line_split(line):
    line = line.split(':')
    return line[0], line[1]


'''
Dataframe to hold list of words
word : Word
pronunciation: String of phonemes
pn_list: List of pronunciation phonemes
primary_stress_map: binary vector with position of primary stress
primary_stress_idx: Index of primary stress
secondary_stress_map: binary vector with position of secondary stress
vowel_map: binary vector with position of vowels
consonant_map: binary vector with position of consonants
vector_map: Binary vector for vowel and constant existence
type_tag: Pos_Tag for the word from nltk
first_letter_index: Alphabetic index of first letter
phenom_length: Number of phonemes
prefix: 1 if prefix exists 0 otherwise
suffix: 1 if suffix exists 0 otherwise
'''


def get_words(datafile):
    lines = [line_split(line) for line in datafile]

    words['pn_list'] = words.pronunciation.apply(str.split)
    words['destressed_pn_list'] = words.pronunciation.apply(filter_stress, args=('[012]',))
    words['primary_stress_map'] = words.pn_list.apply(stress_map)
    words['primary_stress_index'] = words.primary_stress_map.apply(list.index, args=(1,))
    words['secondary_stress_map'] = words.pn_list.apply(stress_map, stress='2')
    words['vowel_map'] = words.destressed_pn_list.apply(phoneme_map, args=(vowels,))
    words['vowel_map_string'] = words.vowel_map.apply(to_string)
    words['consonant_map'] = words.destressed_pn_list.apply(phoneme_map, args=(consonants,))
    words['vector_map'] = words.destressed_pn_list.apply(iterable_map, args=(vector_map,))
    words['vowel_count'] = words.vowel_map.apply(np.sum)
    words['classification'] = words.pn_list.apply(get_classification)
    words['consonant_count'] = words.consonant_map.apply(np.sum)
    words['primary_stress_index'] = words.primary_stress_map.apply(list.index, args=(1,))
    words['classification_index'] = words.apply(get_classsification_index, axis=1)
    words['secondary_stress_map'] = words.pn_list.apply(stress_map, stress='2')
    # words['type_tag'] = words.word.apply(get_pos_tag)
    words['1st_letter_idx'] = words.word.apply(get_first_letter_idx)
    words['phoneme_length'] = words.pn_list.str.len()
    # words['prefix'] = words.word.apply(check_prefix)
    # words['suffix'] = words.word.apply(check_suffix)
    # words['prefix_suffix_vector'] = words.
    # words['primary_stress_idx'] = words.primary_stress_map.apply(get_stress_position)
    words['stressed_vowel'] = words.pn_list.apply(get_stressed_vowel)
    words['ngrams'] = words.pn_list.apply(get_all_ngrams)
    words['ngram_counts'] = words.ngrams.apply(Counter)

    # Unpack vector map into single columns
    # unpacked_vector_map = pd.DataFrame.from_records(words.vector_map.tolist(),columns=vector_map)
    # words = pd.concat([words, unpacked_vector_map],axis=1)
    return words
