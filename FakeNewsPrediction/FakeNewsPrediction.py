import itertools
import json
import shutil
import requests

import joblib
from genetic_selection import GeneticSelectionCV
from newspaper import Article
from newspaper import fulltext
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, plot_roc_curve
from mlxtend.plotting import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from tabulate import tabulate

from .ModelUtils import *

# warnings.showwarning = warn_with_traceback
warnings.filterwarnings('ignore')
plt.rcParams.update({'figure.max_open_warning': 0})

__license__ = 'GPL'
__version__ = 'v0.0'
__status__ = 'Production'
__copyright__ = 'xxx'
__credits__ = ['Sara Suleiman <saratagelsir@gmail.com>']


class FakeNewsPrediction(object):
    """
    ***Fake News Prediction***
    To run the main function of the class FakeNewsPrediction:
        1- Importing the class:
           from FakeNewsPrediction import *
        2- Create an empty object from the class:
           mdl = FakeNewsPrediction()
        3- Pass an input json string to the main function, for example:
           caller_json = '{"modelrun":"covid19_news_training","news_contents":"bla-bla-bla"}'
           model_output = mdl.main(caller_json)

    or you can call the main-function with the following arrangements: python ModelTesting.py [Model_Run]
    """

    def __init__(self):
        # Initialize model parameters
        self.model_start = time.time()
        self.model_name = self.__class__.__name__
        self.user = get_username()
        self.modelrun = ''
        self.run_mode = 'training'
        self.news_contents = ''
        self.file_prefix = ''
        self.rundatetime = pd.to_datetime('today')
        self.randstate = 42

        self.with_features_selection = False
        self.ml_algorithm = ''
        self.HyperParams = []
        self.stopwords = []

        self.predictors = []
        self.key_features = []
        self.score_cutoff = 0.5

        self.modeldir = get_code_dir()
        self.data_file = ''
        self.model_output_file = ''
        self.trained_model_file = ''
        self.data = pd.DataFrame()
        self.model = type('', (), {})()

        log_writer('User %s started %s %s at %s' % (
            self.user, self.model_name, __version__, time.strftime('%Y-%m-%d %H:%M:%S')))
        log_writer('The model directory is: $MODELDIR=%s' % self.modeldir)
        log_writer('An empty object from the class %s is created' % self.model_name)

    def parse_model_info(self, caller_json):
        # Create necessary folders
        folder_names = ['data', 'def', 'output']
        folders = [os.path.join(t[0], t[1]) for t in itertools.product(*[[self.modeldir], folder_names])]
        for folder in folders:
            if not os.path.exists(folder):
                os.makedirs(folder)

        # Read the triggering json string caller
        input_str = json.loads(caller_json)

        info_string = 'Parsed information from caller: '
        for attribute, value in input_str.items():
            info_string += '%s: %s, ' % (attribute, value)
            setattr(self, attribute, value)

        log_writer(info_string[:-2])

        # Parse modelrun string
        self.file_prefix = '_'.join(str(s) for s in self.modelrun.split('_')[:-1])
        self.run_mode = self.modelrun.split('_')[-1]

        # Model directories
        data_folder = os.path.join(self.modeldir, 'data')
        def_folder = os.path.join(self.modeldir, 'def')

        # Read model def json file
        def_filename = os.path.join(def_folder, '%s_def.json' % self.file_prefix)
        json_file = open(def_filename, 'r')
        json_string = json_file.read()
        config_data = json.loads(json_string)

        # Model definition attributes
        for attribute, value in config_data.items():
            setattr(self, attribute, value)

        self.data_file = os.path.join(data_folder, '%s_data.csv' % self.file_prefix)
        self.model_output_file = os.path.join(data_folder, '%s_data.csv' % self.file_prefix)
        self.trained_model_file = os.path.join(data_folder, '%s_trained_model.pkl' % self.file_prefix)
        log_writer('Finished parsing model information')

    def read_raw_data(self):
        if self.run_mode == 'prediction':
            news = self.news_contents
            try:
                # Download and parse article
                # article = Article(self.news_contents)
                # article.download()
                # article.parse()
                # news = article.text
                html = requests.get(self.news_contents).text
                news = fulltext(html)
            except Exception:
                log_writer('News contents provided is text')

            data = pd.DataFrame({'text': [news]})
            data['text'] = data['text'].apply(lambda x: cleaning_data(x, self.stopwords))
        else:
            data = pd.read_csv(self.data_file)
            data = data.drop_duplicates()
            data = data.dropna()

            # Removing stopwords
            # stopwords = set(nltk.corpus.stopwords.words('english'))
            # stopwords.update(self.stopwords)
            # func = lambda x: ' '.join(word for word in x.split() if word not in stopwords)
            # data['text'] = data['text'].apply(func)

        self.data = data.copy()
        log_writer('Raw data is read into dataframe with %d rows and %d columns' % self.data.shape)

    def model_training(self):
        # Randamized data splitting based on 80% for training and 20% for testing
        x_train, x_test, y_train, y_test = train_test_split(self.data['text'], self.data['label'], test_size=0.2, random_state=self.randstate)

        name = '%s_%s' % (self.ml_algorithm, self.file_prefix)
        test_metrics = pd.DataFrame(columns=['Accuracy', 'F1 Score', 'Precision', 'Recall', 'Time [secs]'], index=[name])

        # Model pipeline and model training
        clfs = {'RF': RandomForestClassifier(random_state=self.randstate), 'DT': DecisionTreeClassifier(random_state=self.randstate),
                'GNB': GaussianNB(), 'MNB': MultinomialNB(), 'L_SVM': LinearSVC(random_state=self.randstate),
                'LR': LogisticRegression(random_state=self.randstate)}
        estimator = LogisticRegression(solver="liblinear", multi_class="ovr")
        genet = GeneticSelectionCV(estimator, cv=5, verbose=0, scoring="r2", max_features=10000, n_population=50, crossover_proba=0.5,
                                   mutation_proba=0.2, n_generations=50, crossover_independent_proba=0.5, mutation_independent_proba=0.05,
                                   tournament_size=3, n_gen_no_change=10, caching=True, n_jobs=-1)
        pipeline = [('vect', CountVectorizer()), ('tfidf', TfidfTransformer())]
        if self.with_features_selection:
            pipeline += [('genet', genet)]

        pipeline += [('clf', clfs[self.ml_algorithm])]
        self.model = Pipeline(pipeline)
        self.model.fit(x_train, y_train)
        test_pred = self.model.predict(x_test)
        get_feature_importance(self.model, name, self.modeldir)

        # Model performance metrics
        output_folder = os.path.join(self.modeldir, 'output')
        test_metrics.loc[name, 'Accuracy'] = accuracy_score(y_test, test_pred) * 100
        test_metrics.loc[name, 'F1 Score'] = f1_score(y_test, test_pred) * 100
        test_metrics.loc[name, 'Precision'] = precision_score(y_test, test_pred) * 100
        test_metrics.loc[name, 'Recall'] = recall_score(y_test, test_pred) * 100
        test_metrics.loc[name, 'Time [secs]'] = (time.time() - self.model_start)
        print(tabulate(test_metrics, headers='keys', tablefmt='psql'))
        test_metrics.to_csv(os.path.join(output_folder, '%s_models_performance_metrics.csv' % self.file_prefix))

        # AUC and ROC curve
        fig_auc = plot_roc_curve(self.model, x_test, y_test)
        fig_auc.figure_.suptitle('%s: ROC curve comparison' % name)
        fig_auc.figure_.savefig(os.path.join(output_folder, '%s_ROC_Curve.png' % name), bbox_inches='tight')
        fig_auc.figure_.clf()

        # Confusion matrix
        cm = confusion_matrix(y_test, test_pred, labels=[0, 1])
        fig, ax = plot_confusion_matrix(conf_mat=cm, show_absolute=True, show_normed=True, colorbar=True)
        plt.title('%s: Confusion Matrix' % name)
        plt.savefig(os.path.join(output_folder, '%s_cm_test_dataset' % name), bbox_inches='tight')
        fig.clf()

        # Saving the trained model in pkl-file
        if os.path.isfile(self.trained_model_file):
            shutil.move(self.trained_model_file, os.path.splitext(self.trained_model_file)[0] + '_old.pkl')

        joblib.dump(self.model, self.trained_model_file, compress=True)

    def model_prediction(self):
        # Load trained model and predict
        self.model = joblib.load(self.trained_model_file)

        model_output = self.model.predict(self.data['text'])
        return model_output

    def main(self, caller_json):
        status = 1
        err_short = ''
        description = ''
        model_output = ''
        np.random.seed(self.randstate)

        try:
            self.parse_model_info(caller_json)
            self.read_raw_data()

            run_mode_func = getattr(self, 'model_%s' % self.run_mode.lower())
            model_output = run_mode_func()
        except Exception:
            status = -1
            exc_type, exc_value, exc_traceback = sys.exc_info()
            err_list = traceback.format_exception(exc_type, exc_value, exc_traceback)
            err_log = clean_string(print_path(''.join(err_list)))
            description = 'Failure: ' + err_log
            err_short = clean_string(print_path(err_list[-1]))

        if status == 1:
            description = 'Model run finished successfully at %s (runtime %.2f secs)' % (
                time.strftime('%Y-%m-%d %H:%M:%S'), (time.time() - self.model_start))
        else:
            input_json = '"input_json": %s' % caller_json
            model_output = '{%s, "status": %d, "description": "%s", "full_description": "%s"}' % (
                input_json, status, err_short, description)

        log_writer(description)
        return model_output
