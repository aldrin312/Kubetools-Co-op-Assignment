import kfp
from kfp import dsl
from data_preprocessing import preprocess_data
from model_training import train_and_evaluate
from hyperparameter_tuning import hyperparameter_tuning

@dsl.pipeline(
    name="Diabetes Prediction Pipeline",
    description="A pipeline for training and optimizing a diabetes prediction model"
)
def diabetes_prediction_pipeline():

    preprocess_task = preprocess_data()

    # Train the model and return the model file path
    train_task = train_and_evaluate(
        X_scaled_csv=preprocess_task.outputs['output_X_scaled_csv'],
        y=preprocess_task.outputs['output_y'],
        model_output='/tmp/model.pkl'  # Model output path
    )

    # Hyperparameter tuning, use the model path output from train_and_evaluate
    hyperparameter_task = hyperparameter_tuning(
        X_scaled_csv=preprocess_task.outputs['output_X_scaled_csv'],
        y=preprocess_task.outputs['output_y'],
        model_input=train_task.output,  # Correct reference to the model file path
        best_model_output='/tmp/best_model.pkl'
    )

if __name__ == "__main__":
    output_directory = './pipeline_outputs'
    
    import os
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    yaml_file_path = os.path.join(output_directory, 'diabetes_prediction_pipeline.yaml')

    kfp.compiler.Compiler().compile(diabetes_prediction_pipeline, yaml_file_path)

    print(f"Pipeline YAML file saved to: {yaml_file_path}")
