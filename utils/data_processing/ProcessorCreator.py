import importlib


class ProcessorCreator:
    """
    class to create processing modules
    """

    @staticmethod
    def create(class_name, parameters):
        package, class_name = class_name.rsplit('.', 1)
        model_module = importlib.import_module(package)
        class_object = getattr(model_module, class_name)
        return class_object(**parameters)
