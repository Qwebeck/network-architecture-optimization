import tensorflow as tf
import ray
from ray import tune
from ray.tune.schedulers import AsyncHyperBandScheduler
from ray.tune.integration.keras import TuneReportCallback

class HyperparameterOptimizer:
    def __init__(self, inputs, outputs, data):
        self.inputs = inputs
        self.outputs = outputs
        self.data = data

    def build_model(self, config):
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
        optimizer = tf.keras.optimizers.Adam(learning_rate=config["lr"])
        model = tf.keras.Model(inputs=self.inputs, outputs=self.outputs)
        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=['accuracy'])
        
        return model

    def train_model(self, config):
        batch_size = 128
        epochs = 12

        model = self.build_model(config)
        
        print(model.summary())
        print('===')
        print('===')
        print(self.data['x_train'])
        print(self.data['y_train'])
        print(self.data['x_test'])
        print(self.data['y_test'])
        
        model.fit(
            self.data['x_train'],
            self.data['y_train'],
            batch_size=batch_size,
            epochs=epochs,
            verbose=2,
            validation_data=[self.data['x_test'], self.data['y_test']],
            callbacks=[TuneReportCallback({
                'mean_accuracy': 'accuracy'
            })]
        )
    
    def tune_mnist(self, num_training_iterations):
        sched = AsyncHyperBandScheduler(time_attr="training_iteration", max_t=400, grace_period=20)

        analysis = tune.run(
            lambda cfg: self.train_model(cfg),
            name="exp",
            scheduler=sched,
            metric="mean_accuracy",
            mode="max",
            stop={
                "mean_accuracy": 0.99,
                "training_iteration": num_training_iterations
            },
            num_samples=3,
            resources_per_trial={
                "cpu": 2,
                "gpu": 0
            },
            config={
                "threads": 2,
                "lr": tune.uniform(0.001, 0.1),
            })
        print("Best hyperparameters found were: ", analysis.best_config)

        return analysis.best_config
    
    def get_model(self):
        config = self.tune_mnist(5)
        
        return self.build_model(config)