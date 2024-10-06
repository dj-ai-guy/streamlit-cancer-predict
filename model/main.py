# import libraries 
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


def create_model(data):
    # getting our features and labels 
    X = data.drop(columns='diagnosis', axis=1)
    y = data['diagnosis']

    # scaling the data 
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # split the data into training and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,y, test_size=0.2, random_state=42
    )

    # Train the model with train data 
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Test the model 
    # This will make predictions based on the test data 
    y_pred = model.predict(X_test)
    print('Accuracy of our model: ', accuracy_score(y_test, y_pred))
    print('Classification Report: \n', classification_report(y_test, y_pred))

    # returning the model, also returning the scaler because we'll need it 
    # to make predictions later
    return model, sc




def get_clean_data():
    data = pd.read_csv("data/data.csv")

    # dropping Unnamed 32 & id because it has no data and id is not needed
    data = data.drop(columns=['Unnamed: 32', 'id'], axis=1)
    
    # Encoding the diagnosis column. M = 1 B = 0 
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data

def main():
    data = get_clean_data()
    
    model, sc = create_model(data)

    # Save our model and sc (scaler) as binary files in our model folder 
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(sc, f)



    if __name__ == '__main__':
        main()