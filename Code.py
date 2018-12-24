import nltk
from sklearn.svm import LinearSVC
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import LogisticRegression
import xlwt


def main():

    posts = nltk.corpus.nps_chat.xml_posts()

    print(len(posts))
    print(sorted(nltk.FreqDist(p.attrib['class'] for p in posts).keys()))


    featuresets = []
    prev_ = None

    for post in posts:
        featuresets.append((dialogue_act_features(post.text,prev_), post.get('class')))

        if post.get('class') != 'Statement':
            prev_ = post.get('class')

    size = int(len(featuresets) * 0.01)
    train_set, test_set = featuresets[size:], featuresets[:size]

    # Linear Support vector classification
    classif = SklearnClassifier(LinearSVC())
    classif.train(train_set)
    
    # Logistic Regression method
    # classif = SklearnClassifier(LogisticRegression())
    # classif.train(train_set)

    dialog_Act_A = []
    print("Accuracy : ", nltk.classify.accuracy(classif, test_set) * 100)

    classAprev = None
    book = xlwt.Workbook()
    sh1 = book.add_sheet('Group A')
    index = 0
    openFile = open("output.txt","a", encoding='utf-8')

    with open('test-inputs.txt','r', encoding='utf-8') as groupA:
        for text in groupA:
            class_ = classif.classify(dialogue_act_features(text, classAprev))
            classAprev = class_
            if class_ != 'Statement':
                classAprev = class_
                if class_.find('Question') != -1:
                    class_ = "1"
                else:
                    class_ = "0"
            openFile.write(text.rstrip() + ", " + class_ + "\n")
            sh1.write(index, 0, text)
            sh1.write(index, 1, classAprev)
            index = index + 1
            dialog_Act_A.append(class_)

    groupA.close()
    book.save('QuestionAnalysis.xls')

def dialogue_act_features(post,prev_):
    features = {}

    for word in nltk.word_tokenize(post):
        features['contains({})'.format(word.lower())] = True

    features['prev']=prev_
    return features


if __name__ == '__main__':
    main()
