import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import numpy as np


def create_model(df, tuned=True, mean_plot=False, n_features=5):
    """
    The following function created a predictive model using the processed data obtained from the "comb_features"
    function, as outliers are removed, combined features are created, and data is normalized.

    :param pd.dataframe df: Dataframe of x + y (input + output)
    :param bool tuned: Whether the models would be created with tuned parameters or not (n_features and some subjects
    that are causing noise to the model due to outliers that were not entirely detected.
    :param bool mean_plot: Create mean plot.
    :param int n_features: Default number of features, this could be changed, but preferably < n_subjects.
    :return None: The created model is saved in a .pkl file, the file name depends on whether Empatica's features were
    included in the model or not.
    """

    empatica = False
    # x = df[df.Subject.isin([3, 4, 6, 9])]
    print(df.Subject.unique())
    print(Counter(df.Subject))

    # Counter({1: 331, 3: 269, 8: 85, 9: 27, 6: 5})

    # x = df[df.Subject.isin([3, 8, 9])]
    # x = x[~((x.Subject == 9) & (x.Scene == 4))]
    x = df
    x = df[~df.Subject.isin([1, 2, 9])]

    # print(len(x.Subject.unique()), x.Subject.unique())
    print(Counter(x.Subject))
    print(Counter(x.Scene))
    # print(Counter([1 if (35 > x >= 22) else 0 if x < 22 else 2 for x in
    #                [x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()]]))
    # print([x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()])

    # Fatigue score is popped from the x DataFrame, as these would be the prediction and thus the y.
    y = x.pop('Scene') - 1

    # RandomForestRegressor is used as a feature selection method, and thus the importance of each feature is sorted.
    s = pd.Series(index=x.drop(['Subject'], axis=1).columns,
                  data=RandomForestClassifier(random_state=50).fit(x.drop(['Subject'], axis=1),
                                                                   y).feature_importances_).sort_values(ascending=False)
    print(s.head(n_features))

    # The sorted list is used to obtain only the best "n_features" according to the integer selected.
    x = x.loc[:, list(s.index[:n_features]) + ['Subject']]
    classif = 'binary'
    y = [1 if (scene == 3) else 0 for scene in y] if classif == 'binary' else list(y)

    acc = []
    f1 = []
    for subject in x.Subject.unique():
        print(subject)
        x['Scene'] = y
        xtrain = x[x.Subject != subject]
        xtest = x[x.Subject == subject]
        x = x.drop('Scene', axis=1)
        ytrain = xtrain.pop('Scene')
        ytest = xtest.pop('Scene')
        model = RandomForestClassifier(random_state=1).fit(xtrain, ytrain)
        # print(xtest)
        # print(ytest)
        print(accuracy_score(model.predict(xtest), ytest))
        print(f1_score(model.predict(xtest), ytest))
        acc.append(accuracy_score(model.predict(xtest), ytest))
        f1.append(f1_score(model.predict(xtest), ytest))
        print(confusion_matrix(model.predict(xtest), ytest))
    print(acc)
    print(f1)
    print(np.mean(acc))
    print(np.mean(f1))
    exit()

    # The following for loop generates two 2D plots, these plots would depend on the number of classes:
    # either multi-classifier or binary fear/nofear classification
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    name_plot = 'EEG & Empatica' if empatica else 'EEG'
    name_file = 'EEG_Empatica' if empatica else 'EEG'
    scene_decoding = {'multi': ['Express', 'Name', 'Read', 'Fear'],
                      'binary': ['Non-Fear', 'Fear']}
    for classif in ['multi', 'binary']:
        fig = plt.figure()

        if mean_plot:
            # A scatter plot is generated using the top 2 features, according to the s series.
            x['Scene'] = y
            l_scene, x_data, y_data, fatigue_enc = [], [], [], []
            for current_subject in x.Subject.unique():
                l_scene.append(x[x.Subject == current_subject].reset_index(drop=True).loc[0, 'FAS'])
                x_data.append(
                    x[x.Subject == current_subject].reset_index(drop=True).loc[0, list(s.index[:n_features])[0]])
                y_data.append(
                    x[x.Subject == current_subject].reset_index(drop=True).loc[0, list(s.index[:n_features])[1]])
                if classif == 'multi':
                    fatigue_enc.append(1 if (l_fatigue[-1] >= 22) else 0)
                elif classif == 'binary':
                    fatigue_enc.append(1 if (35 > l_fatigue[-1] >= 22) else 0 if l_fatigue[-1] < 22 else 2)

            df = pd.DataFrame({'X': x_data, 'Y': y_data, 'FAS': l_fatigue, 'FAS_Enc': fatigue_enc})

            for clas in range(n_cat):
                df_sub = df[df.FAS_Enc == clas]
                plt.scatter(df_sub.X, df_sub.Y, label=cat_decoding[n_cat][clas], color=colors[clas])

            plt.legend()
            y = x.pop('FAS')

        else:
            scene_cat = [1 if (scene == 3) else 0 for scene in y] if classif == 'binary' else list(y)
            fig = plt.figure()

            # A scatter plot is generated usign the top 2 features, according to the s series.
            plt.scatter(x.loc[:, list(s.index[:n_features])[0]], x.loc[:, list(s.index[:n_features])[1]],
                        color = [colors[cat] for cat in scene_cat], alpha=0.5)
            # plt.scatter(x['P7_Theta'], x['P7_Beta'], color=[colors[cat] for cat in scene_cat])

            # A legend is manually generated using patches of colors, according to default matplotlib's colors.
            plt.legend(handles=[mpatches.Patch(color=c) for c in colors[:max(scene_cat)+1]], labels=scene_decoding[classif])

        # The axes object is used to set the title, x-label, and y-label.
        ax = fig.gca()
        ax.set_title('Best features visualization on {}-scene classification {}'.format(classif, name_plot))
        ax.set_xlabel(list(s.index[:n_features])[0])
        ax.set_ylabel(list(s.index[:n_features])[1])
        # ax.set_xlim(-1, 1)
        # ax.set_ylim(-1, 1)

        # Figure is saved in a .pdf file to mantain the best resolution when including in .tex files.
        fig.savefig('figs/{}_2D_plot_{}.pdf'.format(classif, name_file),
                    bbox_inches='tight')
        plt.close()

    exit()

    # Random Forest
    #  N      2          3       R
    # 20    0.643      0.429    0.06
    # 15    0.643      0.5      0.051
    # 14    0.643      0.5      0.051
    # 13    0.714      0.571    0.092
    # 12    0.714      0.571    0.126
    # 11    0.714      0.571    0.140
    # 10    0.714      0.571    0.17
    # 9     0.714      0.571    0.143
    # 8     0.714      0.571    0.104
    # 7     0.643      0.5      0.09
    # 6     0.643      0.5      0.11
    # 5     0.643      0.429    0.25

    # Linear Regression
    #  N      3          2       R
    # 10    0.786      0.5      -0.383

    # previous_features = set(list(x.drop(['Subject'], axis=1).columns))
    # scores_pearson = [np.abs(pearsonr(y, x[feature])[0]) for feature in x.columns[:N_FEATURES]]
    # p_value = [np.abs(pearsonr(y, x[feature])[1]) for feature in x.columns[:N_FEATURES]]
    # df_correlations = pd.DataFrame({'Feature': x.columns[:N_FEATURES], 'Correlation': scores_pearson, 'P': p_value})
    # df_correlations = (df_correlations[df_correlations.P < 0.05].sort_values('Correlation', ascending=False).round(6)
    #                   .reset_index(drop=True))
    #       Con                 Sin
    # 0.714 0.5 -0.031  0.643 0.429 0.06
    # filtered_features = set(list(df_correlations.Feature))
    # stad_reject_features = previous_features.difference(filtered_features)
    # print(df_correlations.head(df_correlations.shape[0]))
    # x = x.drop(list(previous_features.difference(filtered_features)), axis=1)
    # print('{} Features were rejected'.format(len(stad_reject_features)))
    # print('{} Features were accepted'.format(len(filtered_features)))

    print(x)

    # An empty DataFrame of results is generated, for continuous predictions, as well as 2-class, 3-class categorical.
    df_results = pd.DataFrame(columns=['Cat_Pred', 'Cat_True', 'Cat_Pred_Bin', 'Cat_True_Bin'])

    # The following for loop iterates over all subjects, in order to implement a Leave-One-Out (LOO) validation scheme,
    # using approximately 10 subjects, the LOO validation consist on using 90% of the data for training and 10% of the
    # data for testing, as the current subjects' data would be used to test the model, while the rest of the subjects'
    # data would be used for model training.
    for current_subject in x.Subject.unique():
        # Train index are the rows that does not correspond to the current subject's ID, while the test index are the
        # rows that contain the current subject's ID.
        train_index = x[x.Subject != current_subject].index
        test_index = x[x.Subject == current_subject].index

        # Indexes are used to create the set of training and testing data, removing subject as a feature.
        x_train, x_test = x.iloc[train_index, :].drop('Subject', axis=1), x.iloc[test_index, :].drop('Subject', axis=1)
        y_train, y_test = y[train_index], y[test_index]

        # Multiple linear regression is used, and training data is used to train the model.
        model = RandomForestRegressor().fit(x_train, y_train)

        # A raw prediction is done using the training model and the testing rows.
        raw_prediction = model.predict(x_test)

        # The raw prediction is transformed to a final number, because remember that FAS score is assigned to all
        # samples from a subject, and thus we are predicting the FAS score for each row assigned to the subject. So,
        # in order to compare both scores, a single score must be generated, in this case the mean of all predicted
        # scores was used, although, if the mean of score is not in a valid range (0 > x > 50), median is used.
        prediction = np.median(np.round(raw_prediction)) if (0 > mean(raw_prediction) > 50) else round(
            mean(raw_prediction))

        # 2-Class and 3-Class Categorical encoding is applied to the prediction.
        prediction_cat_2 = 1 if (prediction >= 22) else 0
        prediction_cat_3 = 1 if (35 > prediction >= 22) else 0 if prediction < 22 else 2

        # Subject's true FAS score is recovered from the "y_true" pandas series.
        y_true = y_test.head(1).item()

        # 2-Class and 3-Class Categorical encoding is applied to the true value.
        y_true_cat_2 = 1 if (y_true >= 22) else 0
        y_true_cat_3 = 1 if (35 > y_true >= 22) else 0 if y_true < 22 else 2

        # Results are appended into the "df_results" using a dictionary and the zip function.
        df_results = pd.concat([df_results, pd.DataFrame(dict(zip(list(df_results.columns),
                                                                  [prediction, y_true,
                                                                   prediction_cat_2, y_true_cat_2,
                                                                   prediction_cat_3, y_true_cat_3])),
                                                         index=range(1))], axis=0)
        print(current_subject, prediction, y_true, prediction_cat_2, y_true_cat_2, prediction_cat_3, y_true_cat_3)

    # Each subject's ID is set as a new column in the "df_results" DataFrame.
    df_results['Subject'] = x.Subject.unique()
    print(df_results.head(df_results.shape[0]))

    # The DataFrame's columns are transformed into categorical columns, with their respective order.
    df_results.Cat_True_2 = pd.Categorical(df_results.Cat_True_2, categories=[0, 1], ordered=True)
    df_results.Cat_Pred_2 = pd.Categorical(df_results.Cat_Pred_2, categories=[0, 1], ordered=True)
    df_results.Cat_True_3 = pd.Categorical(df_results.Cat_True_3, categories=[0, 1, 2], ordered=True)
    df_results.Cat_Pred_3 = pd.Categorical(df_results.Cat_Pred_3, categories=[0, 1, 2], ordered=True)

    print(accuracy_score(df_results.Cat_True_2, df_results.Cat_Pred_2))
    print(accuracy_score(df_results.Cat_True_3, df_results.Cat_Pred_3))
    print(r2_score(df_results.Cont_True, df_results.Cont_Pred))

    print(classification_report(df_results.Cat_True_2, df_results.Cat_Pred_2))
    print(classification_report(df_results.Cat_True_3, df_results.Cat_Pred_3))

    # A simple plot is generated, that related both continuous FAS predictions and true values.
    ax = df_results.loc[:, ['Cont_Pred', 'Cont_True']].sort_values('Cont_True', ascending=True).reset_index(
        drop=True).plot()

    # Moreover, a title is set, as well as the R-squared obtained using the LOO validation.
    ax.set_title('Predicted and true FAS values, using {} features'.format(name_plot))
    at = AnchoredText("Coefficient of determination $(R^2)$: {}".format(
        round(r2_score(df_results.Cont_True, df_results.Cont_Pred), 2)), loc='lower right')
    ax.add_artist(at)

    # The figure is obtained and thus saved as .pdf file.
    fig = ax.get_figure()
    plt.legend()

    fig.savefig('Empatica-Project-ALAS-main/Files/predictions_{}.pdf'.format(name_file), bbox_inches='tight')

    # A final Linear Regression model is fitted with all the data available, and thus exported as a .pkl file.
    model = LinearRegression().fit(x.drop('Subject', axis=1), y)

    l_features = list(x.drop('Subject', axis=1).columns)

    file = open("Empatica-Project-ALAS-main/Files/features_{}.txt".format(name_file), "w")
    for i in range(len(l_features)):
        if i == len(l_features) - 1:
            file.writelines(l_features[i])
        else:
            file.writelines(l_features[i] + ', ')
    file.close()

    dump(model, open('Empatica-Project-ALAS-main/Files/model_{}.pkl'.format(name_file), 'wb'))


df = pd.read_csv('pross/python_PSD_TAB_QM_norm.csv')
create_model(df)
