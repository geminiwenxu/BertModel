from bin.prediction import prediction
import yaml
from pkg_resources import resource_filename


def get_config(path):
    with open(resource_filename(__name__, path), 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def main():
    config = get_config('/../config/config.yaml')
    model_path = resource_filename(__name__, config['model_path']['path'])
    file_path = resource_filename(__name__, config['file_path']['path'])
    prediction(model_path, file_path)


if __name__ == "__main__":
    main()
