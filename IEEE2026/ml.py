import pandas as pd

def create_model(df, tuned=True, mean_plot=False, n_features=10):
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

    # Second and Repetition columns are removed, as only source features, FAS and Subject ID are required.
    x = comb_features().drop(['Second', 'Repetition'], axis=1)

    # If the "tuned" parameters is set to True, then subjects that are making noise to each model would be removed.
    if tuned:
        subjects_to_be_removed = remove_subjects['empatica'] if empatica else remove_subjects['non-empatica']
        x = x.drop(x[x.Subject.isin(subjects_to_be_removed)].index, axis=0)
    x = x.dropna(axis=1).reset_index(drop=True)

    print(len(x.Subject.unique()), x.Subject.unique())
    print(Counter(
        [1 if (x >= 22) else 0 for x in [x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()]]))
    print(Counter([1 if (35 > x >= 22) else 0 if x < 22 else 2 for x in
                   [x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()]]))
    print([x[x.Subject == sub].head(1).FAS.item() for sub in x.Subject.unique()])

    # Evaluacion para MLR Empatica & EEG (9 Feats y [16])
    # 12 0.9 0.8 0.50
    # 11 0.9 0.7 0.58
    # 10 0.9 0.8 0.46
    # 9 0.9 0.8 0.58
    # 8 0.6 0.4 0.09

    # Evaluacion para MLR EEG (10 Feats y [4, 20])
    #   ID       2       3        R
    #           0.79    0.5     -0.38
    # 4         0.69    0.38    -0.05
    # 4 13      0.58    0.25    -0.68
    # 4    20   0.83    0.5     0.072
    #   13      0.62    0.38    -0.027
    #   13 20   0.66    0.42    0.304
    #      20   0.77    0.53    -0.47
    # 4 13 20   0.63    0.27    -0.07

    # 2-Categories
    # 1 9
    # 0 5

    # 3-Categories
    # 2 4
    # 1 5
    # 0 5

    # Fatigue score is popped from the x DataFrame, as these would be the prediction and thus the y.
    y = x.pop('FAS')

    corr_matrix = x.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    x = x.drop([column for column in upper_tri.columns if any(upper_tri[column] > 0.95)], axis=1)

    #      Con              Sin
    # 0.64 0.42 -0.074  0.71 0.5 -0.031

    # If the "tuned" parameters is set to True, then the right number of features is selected for each model.
    if tuned:
        n_features = number_features['empatica'] if empatica else number_features['non-empatica']

    # RandomForestRegressor is used as a feature selection method, and thus the importance of each feature is sorted.
    s = pd.Series(index=x.drop(['Subject'], axis=1).columns,
                  data=RandomForestRegressor(random_state=50).fit(x.drop(['Subject'], axis=1),
                                                                  y).feature_importances_).sort_values(ascending=False)
    print(s.head(n_features))

    # The sorted list is used to obtain only the best "n_features" according to the integer selected.
    x = x.loc[:, list(s.index[:n_features]) + ['Subject']]

    # The following for loop generates two 2D plots, these plots would depend on the number of classes
    # that FAS score would be encoded, the first variables are colors and strings to keep the same format.
    name_plot = 'EEG & Empatica' if empatica else 'EEG'
    name_file = 'EEG_Empatica' if empatica else 'EEG'
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    cat_decoding = {2: ['No Fatigue', 'Substantial Fatigue'],
                    3: ['No Fatigue', 'Moderate Fatigue', 'Extreme Fatigue']}
    for n_cat in [2, 3]:

        fig = plt.figure()

        if mean_plot:
            # A scatter plot is generated usign the top 2 features, according to the s series.
            x['FAS'] = y
            l_fatigue, x_data, y_data, fatigue_enc = [], [], [], []
            for current_subject in x.Subject.unique():
                l_fatigue.append(x[x.Subject == current_subject].reset_index(drop=True).loc[0, 'FAS'])
                x_data.append(
                    x[x.Subject == current_subject].reset_index(drop=True).loc[0, list(s.index[:n_features])[0]])
                y_data.append(
                    x[x.Subject == current_subject].reset_index(drop=True).loc[0, list(s.index[:n_features])[1]])
                if n_cat == 2:
                    fatigue_enc.append(1 if (l_fatigue[-1] >= 22) else 0)
                elif n_cat == 3:
                    fatigue_enc.append(1 if (35 > l_fatigue[-1] >= 22) else 0 if l_fatigue[-1] < 22 else 2)

            df = pd.DataFrame({'X': x_data, 'Y': y_data, 'FAS': l_fatigue, 'FAS_Enc': fatigue_enc})

            for clas in range(n_cat):
                df_sub = df[df.FAS_Enc == clas]
                plt.scatter(df_sub.X, df_sub.Y, label=cat_decoding[n_cat][clas], color=colors[clas])

            plt.legend()
            y = x.pop('FAS')

        else:
            fas_cat = [1 if (fatigue >= 22) else 0 for fatigue in y] if n_cat == 2 else \
                [1 if (35 > fatigue >= 22) else 0 if fatigue < 22 else 2 for fatigue in y]
            fig = plt.figure()

            # A scatter plot is generated usign the top 2 features, according to the s series.
            plt.scatter(x.loc[:, list(s.index[:n_features])[0]], x.loc[:, list(s.index[:n_features])[1]],
                        color=[colors[cat] for cat in fas_cat])

            # A legend is manually generated using patches of colors, according to default matplotlib's colors.
            plt.legend(handles=[mpatches.Patch(color=c) for c in colors[:n_cat]], labels=cat_decoding[n_cat])

        # The axes object is used to set the title, x-label, and y-label.
        ax = fig.gca()
        ax.set_title('Best features visualization on {}-Class fatigue {}'.format(n_cat, name_plot))
        ax.set_xlabel(list(s.index[:n_features])[0])
        ax.set_ylabel(list(s.index[:n_features])[1])

        # Figure is saved in a .pdf file to mantain the best resolution when including in .tex files.
        fig.savefig('Empatica-Project-ALAS-main/Files/{}_2D_plot_{}.pdf'.format(n_cat, name_file),
                    bbox_inches='tight')
        plt.close()

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
    df_results = pd.DataFrame(
        columns=['Cont_Pred', 'Cont_True', 'Cat_Pred_2', 'Cat_True_2', 'Cat_Pred_3', 'Cat_True_3'])

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
