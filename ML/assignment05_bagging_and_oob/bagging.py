import numpy as np


class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        '''
        Generate indices for every bag and store in self.indices_list list
        Сгенерируйте индексы для каждой сумки и сохраните в списке self.indices_list
        '''
        self.indices_list = []
        data_length = len(data)
        for bag in range(self.num_bags):
            self.indices_list.append(np.random.randint(1, data_length, data_length))    
        
    def fit(self, model_constructor, data, target):
        '''
        Fit model on every bag.
        Model constructor with no parameters (and with no ()) is passed to this function.
        
        example:
        
        bagging_regressor = SimplifiedBaggingRegressor(num_bags=10, oob=True)
        bagging_regressor.fit(LinearRegression, X, y)
        '''
        self.data = None
        self.target = None
        self._generate_splits(data)
        assert len(set(list(map(len, self.indices_list)))) == 1, 'All bags should be of the same length!'
        assert list(map(len, self.indices_list))[0] == len(data), 'All bags should contain `len(data)` number of elements!'
        self.models_list = []
        for bag in range(self.num_bags):
            model = model_constructor()
            data_bag = data[self.indices_list[bag]]
            target_bag = target[self.indices_list[bag]] 
            self.models_list.append(model.fit(data_bag, target_bag)) # store fitted models here
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):
        '''
        Get average prediction for every object from passed dataset
        Получить средний прогноз для каждого объекта из переданного набора данных
        '''        
        revel = False
        if data.ndim == 1:
            data = data.reshape(1, -1)
            revel = True
        predicts = [model.predict((data)) for model in self.models_list]
        predicts = np.array(predicts)
        predict = np.average(predicts, axis = 0)
        if revel:
            predict = predict.ravel()
        return predict
        
    def _get_oob_predictions_from_every_model(self):
        '''
        Generates list of lists, where list i contains predictions for self.data[i] object
        from all models, which have not seen this object during training phase
        Генерирует список списков, где список i содержит прогнозы для объекта self.data[i]
        из всех моделей, которые не видели этот объект на этапе обучения
        '''
        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        missing_indexes = [np.setdiff1d(np.arange(self.data.shape[0]), self.indices_list[bag]) for bag in range(self.num_bags)]
        predicts_for_missing_indexes = [self.models_list[bag].predict(self.data[missing_indexes[bag]]) for bag in range(self.num_bags) ]
        for bag in range(self.num_bags):
            for num_pred, index in enumerate(missing_indexes[bag]):
                list_of_predictions_lists[index].append(predicts_for_missing_indexes[bag][num_pred])
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
        
    def _get_averaged_oob_predictions(self):
        '''
        Compute average prediction for every object from training set.
        If object has been used in all bags on training phase, return None instead of prediction
        
        Вычислите среднее предсказание для каждого объекта из обучающего набора.
        Если объект использовался во всех пакетах на этапе обучения, верните None вместо предсказания
        '''
        self._get_oob_predictions_from_every_model()
        self.oob_predictions = []
        for index in range(len(self.data)):
            if len(self.list_of_predictions_lists[index]) == 0:
                self.oob_predictions.append(None)
            else:
                self.oob_predictions.append(np.average(self.list_of_predictions_lists[index]))
        self.oob_predictions = np.array(self.oob_predictions)
        
    def OOB_score(self):
        '''
        Compute mean square error for all objects, which have at least one prediction
        
        Вычислите среднеквадратичную ошибку для всех объектов, которые имеют по крайней мере одно предсказание
        '''  
        self._get_averaged_oob_predictions()
        indexes = np.where(self.oob_predictions != None)[0]
        oob_score = (1 / indexes.shape[0]) * sum((self.oob_predictions[indexes] - self.target[indexes]) ** 2)
        return oob_score